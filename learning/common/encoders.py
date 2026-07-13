from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(dims: Sequence[int], last_activation: bool = True) -> nn.Sequential:
    if len(dims) < 2:
        raise ValueError("MLP dims must contain at least input and output dimensions")
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        is_last = i == len(dims) - 2
        if (not is_last) or last_activation:
            layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


@dataclass(frozen=True)
class MultiLevelMAPSEncoderConfig:
    num_levels: int = 6
    in_channels_per_level: Union[int, Tuple[int, ...]] = 3
    conv_channels: Tuple[int, ...] = (16, 32)
    level_embed_dim: int = 64
    # "sgcnn" uses grouped conv blocks over stacked level maps.
    # "independent" keeps per-level independent CNN branches.
    mode: str = "sgcnn"
    # Input levels may have different H/W; SGCNN resizes to this target.
    sgcnn_target_hw: Tuple[int, int] = (7, 7)


@dataclass(frozen=True)
class RobotStateEncoderConfig:
    input_dim: int = 9
    hidden_dims: Tuple[int, ...] = (64, 64)


@dataclass(frozen=True)
class FusedMAPSStateEncoderConfig:
    maps: MultiLevelMAPSEncoderConfig = field(default_factory=MultiLevelMAPSEncoderConfig)
    robot_state: RobotStateEncoderConfig = field(default_factory=RobotStateEncoderConfig)
    fusion_hidden_dims: Tuple[int, ...] = (256, 256)


@dataclass(frozen=True)
class HybridLocalGlobalEncoderConfig:
    local_in_channels: int = 5
    global_in_channels: Union[int, Tuple[int, ...]] = 3
    global_sizes: Tuple[int, ...] = (64, 32, 16)
    conv_channels: Tuple[int, ...] = (16, 32)
    local_embed_dim: int = 64
    global_embed_dim: int = 64
    robot_state: RobotStateEncoderConfig = field(default_factory=RobotStateEncoderConfig)
    fusion_hidden_dims: Tuple[int, ...] = (256, 256)


class _LevelCNNBranch(nn.Module):
    def __init__(self, in_channels: int, conv_channels: Sequence[int], out_dim: int):
        super().__init__()
        if len(conv_channels) == 0:
            raise ValueError("conv_channels must contain at least one channel size")

        conv_layers = []
        ch_in = in_channels
        for ch_out in conv_channels:
            conv_layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1))
            conv_layers.append(nn.ReLU(inplace=True))
            ch_in = ch_out

        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(ch_in, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Level tensor must be 4D [B,C,H,W], got shape={tuple(x.shape)}")
        h = self.conv(x)
        h = self.pool(h).flatten(1)
        return self.proj(h)


def _normalize_level_channels(
    in_channels_per_level: Union[int, Sequence[int]],
    num_levels: int,
) -> Tuple[int, ...]:
    if isinstance(in_channels_per_level, int):
        channels = (int(in_channels_per_level),) * int(num_levels)
    else:
        channels = tuple(int(c) for c in in_channels_per_level)
        if len(channels) != int(num_levels):
            raise ValueError(
                "in_channels_per_level must be an int or contain one value per level "
                f"(expected {num_levels}, got {len(channels)})"
            )
    if any(c <= 0 for c in channels):
        raise ValueError("in_channels_per_level values must be positive")
    return channels


class _SGCNNGroupedEncoder(nn.Module):
    """
    Scale-Grouped CNN encoder:
    - stack all levels along channels
    - grouped conv with groups=num_levels (each scale processed separately)
    - per-scale projection heads to preserve scale-specific semantics
    """

    def __init__(
        self,
        num_levels: int,
        in_channels_per_level: Union[int, Sequence[int]],
        conv_channels: Sequence[int],
        level_embed_dim: int,
        target_hw: Tuple[int, int],
        allow_implicit_channel_padding: bool = False,
    ):
        super().__init__()
        if num_levels <= 0:
            raise ValueError("num_levels must be positive")
        if len(conv_channels) == 0:
            raise ValueError("conv_channels must contain at least one channel size")
        if target_hw[0] <= 0 or target_hw[1] <= 0:
            raise ValueError("sgcnn_target_hw values must be positive")

        self.num_levels = int(num_levels)
        self.level_in_channels = _normalize_level_channels(in_channels_per_level, self.num_levels)
        self.allow_implicit_channel_padding = bool(allow_implicit_channel_padding)
        # Grouped conv requires equal input channels for each group. Levels with
        # fewer observation channels are zero-padded before stacking.
        self.in_channels_per_level = max(self.level_in_channels)
        self.target_hw = (int(target_hw[0]), int(target_hw[1]))

        layers = []
        ch_per_level_in = self.in_channels_per_level
        for ch_per_level_out in conv_channels:
            if ch_per_level_out <= 0:
                raise ValueError("conv_channels must be positive")
            in_total = ch_per_level_in * self.num_levels
            out_total = int(ch_per_level_out) * self.num_levels
            layers.append(
                nn.Conv2d(
                    in_channels=in_total,
                    out_channels=out_total,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=self.num_levels,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            ch_per_level_in = int(ch_per_level_out)
        self.conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.ModuleList(
            [nn.Linear(ch_per_level_in, level_embed_dim) for _ in range(self.num_levels)]
        )
        self.output_dim = self.num_levels * level_embed_dim

    def forward(self, levels: Mapping[int, torch.Tensor], level_ids: Tuple[int, ...]) -> torch.Tensor:
        xs = []
        batch_size: Optional[int] = None
        th, tw = self.target_hw

        for i, lv in enumerate(level_ids):
            x = levels[lv]
            if x.ndim != 4:
                raise ValueError(f"Level {lv} tensor must be 4D [B,C,H,W], got shape={tuple(x.shape)}")
            observed_c = int(x.shape[1])
            if observed_c > self.in_channels_per_level:
                raise ValueError(
                    f"Level {lv} expected at most C={self.in_channels_per_level}, got C={x.shape[1]}"
                )
            expected_c = self.level_in_channels[i]
            if observed_c != expected_c and not self.allow_implicit_channel_padding:
                raise ValueError(f"Level {lv} expected C={expected_c}, got C={observed_c}")
            if batch_size is None:
                batch_size = int(x.shape[0])
            elif batch_size != int(x.shape[0]):
                raise ValueError("All levels must have the same batch size")

            if (int(x.shape[2]), int(x.shape[3])) != (th, tw):
                x = F.interpolate(x, size=(th, tw), mode="nearest")
            pad_c = self.in_channels_per_level - int(x.shape[1])
            if pad_c > 0:
                pad = torch.zeros(
                    (x.shape[0], pad_c, x.shape[2], x.shape[3]),
                    dtype=x.dtype,
                    device=x.device,
                )
                x = torch.cat([x, pad], dim=1)
            xs.append(x)

        stacked = torch.cat(xs, dim=1)  # [B, L*C, H, W]
        h = self.conv(stacked)
        h = self.pool(h).flatten(1)  # [B, L*C_final]
        h = h.view(h.shape[0], self.num_levels, -1)  # [B, L, C_final]
        emb = [self.proj[i](h[:, i, :]) for i in range(self.num_levels)]
        return torch.cat(emb, dim=1)


class MultiLevelMAPSEncoder(nn.Module):
    """
    Encodes MAPS levels with independent CNN branches per level.

    Input:
    - mapping level_id -> tensor [B, C, H, W]
    Output:
    - concatenated embedding [B, num_levels * level_embed_dim]
    """

    def __init__(self, config: Optional[MultiLevelMAPSEncoderConfig] = None):
        super().__init__()
        self.config = config or MultiLevelMAPSEncoderConfig()
        self.level_ids = tuple(range(self.config.num_levels))
        input_channels_was_uniform = isinstance(self.config.in_channels_per_level, int)
        self.level_in_channels = _normalize_level_channels(
            self.config.in_channels_per_level,
            self.config.num_levels,
        )
        mode = self.config.mode.lower().strip()
        if mode not in {"sgcnn", "independent"}:
            raise ValueError("maps encoder mode must be one of {'sgcnn', 'independent'}")
        self.mode = mode

        if self.mode == "independent":
            self.branches = nn.ModuleDict(
                {
                    str(lv): _LevelCNNBranch(
                        in_channels=self.level_in_channels[lv],
                        conv_channels=self.config.conv_channels,
                        out_dim=self.config.level_embed_dim,
                    )
                    for lv in self.level_ids
                }
            )
            self.sgcnn = None
            self.output_dim = self.config.num_levels * self.config.level_embed_dim
        else:
            self.branches = None
            self.sgcnn = _SGCNNGroupedEncoder(
                num_levels=self.config.num_levels,
                in_channels_per_level=self.level_in_channels,
                conv_channels=self.config.conv_channels,
                level_embed_dim=self.config.level_embed_dim,
                target_hw=self.config.sgcnn_target_hw,
                allow_implicit_channel_padding=input_channels_was_uniform,
            )
            self.output_dim = self.sgcnn.output_dim

    def forward(self, levels: Mapping[int, torch.Tensor]) -> torch.Tensor:
        missing = [lv for lv in self.level_ids if lv not in levels]
        if missing:
            raise KeyError(f"Missing MAPS levels: {missing}")

        if self.mode == "sgcnn":
            if self.sgcnn is None:
                raise RuntimeError("SGCNN encoder is not initialized")
            return self.sgcnn(levels, self.level_ids)

        if self.branches is None:
            raise RuntimeError("Independent branches are not initialized")

        batch_size: Optional[int] = None
        encoded = []
        for lv in self.level_ids:
            x = levels[lv]
            if x.ndim != 4:
                raise ValueError(f"Level {lv} tensor must be 4D [B,C,H,W], got shape={tuple(x.shape)}")
            expected_c = self.level_in_channels[lv]
            if x.shape[1] != expected_c:
                raise ValueError(
                    f"Level {lv} expected C={expected_c}, got C={x.shape[1]}"
                )
            if batch_size is None:
                batch_size = int(x.shape[0])
            elif batch_size != int(x.shape[0]):
                raise ValueError("All levels must have the same batch size")
            encoded.append(self.branches[str(lv)](x))
        return torch.cat(encoded, dim=1)


class RobotStateEncoder(nn.Module):
    def __init__(self, config: Optional[RobotStateEncoderConfig] = None):
        super().__init__()
        self.config = config or RobotStateEncoderConfig()
        if len(self.config.hidden_dims) == 0:
            raise ValueError("RobotStateEncoder hidden_dims must not be empty")
        dims = (self.config.input_dim, *self.config.hidden_dims)
        self.net = _make_mlp(dims, last_activation=True)
        self.output_dim = self.config.hidden_dims[-1]

    def forward(self, robot_state: torch.Tensor) -> torch.Tensor:
        if robot_state.ndim != 2:
            raise ValueError(
                f"robot_state tensor must be 2D [B, F], got shape={tuple(robot_state.shape)}"
            )
        if robot_state.shape[1] != self.config.input_dim:
            raise ValueError(
                f"robot_state expected F={self.config.input_dim}, got F={robot_state.shape[1]}"
            )
        return self.net(robot_state)


class FusedMAPSStateEncoder(nn.Module):
    """
    Shared encoder for BC and PPO:
    - MAPS multi-level CNN features
    - robot-state MLP features
    - fusion MLP output
    """

    def __init__(self, config: Optional[FusedMAPSStateEncoderConfig] = None):
        super().__init__()
        self.config = config or FusedMAPSStateEncoderConfig()

        self.maps_encoder = MultiLevelMAPSEncoder(self.config.maps)
        self.state_encoder = RobotStateEncoder(self.config.robot_state)

        if len(self.config.fusion_hidden_dims) == 0:
            raise ValueError("fusion_hidden_dims must not be empty")
        fusion_in = self.maps_encoder.output_dim + self.state_encoder.output_dim
        fusion_dims = (fusion_in, *self.config.fusion_hidden_dims)
        self.fusion = _make_mlp(fusion_dims, last_activation=True)
        self.output_dim = self.config.fusion_hidden_dims[-1]

    def forward(self, levels: Mapping[int, torch.Tensor], robot_state: torch.Tensor) -> torch.Tensor:
        maps_feat = self.maps_encoder(levels)
        state_feat = self.state_encoder(robot_state)
        fused = torch.cat([maps_feat, state_feat], dim=1)
        return self.fusion(fused)


class HybridLocalGlobalEncoder(nn.Module):
    """
    Encoder for hybrid CPP observation:
    - high-resolution local crop CNN
    - full-map coarse global CNN
    - robot-state MLP
    - fusion MLP
    """

    def __init__(self, config: Optional[HybridLocalGlobalEncoderConfig] = None):
        super().__init__()
        self.config = config or HybridLocalGlobalEncoderConfig()
        if int(self.config.local_in_channels) <= 0:
            raise ValueError("local_in_channels must be positive")
        self.global_sizes = tuple(int(s) for s in self.config.global_sizes)
        if len(self.global_sizes) == 0:
            raise ValueError("global_sizes must not be empty")
        if any(s <= 0 for s in self.global_sizes):
            raise ValueError("global_sizes values must be positive")
        self.global_in_channels = _normalize_level_channels(
            self.config.global_in_channels,
            len(self.global_sizes),
        )
        self.local_encoder = _LevelCNNBranch(
            in_channels=int(self.config.local_in_channels),
            conv_channels=self.config.conv_channels,
            out_dim=int(self.config.local_embed_dim),
        )
        self.global_encoders = nn.ModuleDict(
            {
                str(size): _LevelCNNBranch(
                    in_channels=int(channels),
                    conv_channels=self.config.conv_channels,
                    out_dim=int(self.config.global_embed_dim),
                )
                for size, channels in zip(self.global_sizes, self.global_in_channels)
            }
        )
        self.state_encoder = RobotStateEncoder(self.config.robot_state)
        if len(self.config.fusion_hidden_dims) == 0:
            raise ValueError("fusion_hidden_dims must not be empty")
        fusion_in = (
            int(self.config.local_embed_dim)
            + len(self.global_sizes) * int(self.config.global_embed_dim)
            + int(self.state_encoder.output_dim)
        )
        fusion_dims = (fusion_in, *self.config.fusion_hidden_dims)
        self.fusion = _make_mlp(fusion_dims, last_activation=True)
        self.output_dim = self.config.fusion_hidden_dims[-1]

    def forward(
        self,
        local_map: torch.Tensor,
        global_maps: Mapping[int, torch.Tensor],
        robot_state: torch.Tensor,
    ) -> torch.Tensor:
        if local_map.ndim != 4:
            raise ValueError(f"local_map must be 4D [B,C,H,W], got shape={tuple(local_map.shape)}")
        local_feat = self.local_encoder(local_map)
        global_feats = []
        for size, expected_c in zip(self.global_sizes, self.global_in_channels):
            if size not in global_maps:
                raise KeyError(f"Missing global map size {size}")
            global_map = global_maps[size]
            if global_map.ndim != 4:
                raise ValueError(
                    f"global_map_{size} must be 4D [B,C,H,W], got shape={tuple(global_map.shape)}"
                )
            if int(global_map.shape[1]) != int(expected_c):
                raise ValueError(
                    f"global_map_{size} expected C={int(expected_c)}, got C={int(global_map.shape[1])}"
                )
            global_feats.append(self.global_encoders[str(size)](global_map))
        state_feat = self.state_encoder(robot_state)
        fused = torch.cat([local_feat, *global_feats, state_feat], dim=1)
        return self.fusion(fused)

import torch

from learning.common import HybridLocalGlobalEncoder, HybridLocalGlobalEncoderConfig, RobotStateEncoderConfig


def _inputs(global_channels: int):
    return (
        torch.randn(2, 5, 41, 41),
        {
            64: torch.randn(2, global_channels, 64, 64),
            32: torch.randn(2, global_channels, 32, 32),
            16: torch.randn(2, global_channels, 16, 16),
        },
        torch.randn(2, 50),
    )


def _config(**kwargs) -> HybridLocalGlobalEncoderConfig:
    base = dict(
        local_in_channels=5,
        global_in_channels=(10, 10, 10),
        global_sizes=(64, 32, 16),
        conv_channels=(8, 8),
        local_embed_dim=12,
        global_embed_dim=12,
        robot_state=RobotStateEncoderConfig(input_dim=50, hidden_dims=(16,)),
        fusion_hidden_dims=(32,),
    )
    base.update(kwargs)
    return HybridLocalGlobalEncoderConfig(**base)


def test_hybrid_encoder_independent_conv1x1_dtm_forward_shape() -> None:
    encoder = HybridLocalGlobalEncoder(
        _config(dtm_channels=6, dtm_embed_mode="conv1x1", dtm_embed_channels=3)
    )
    local_map, global_maps, robot_state = _inputs(global_channels=10)

    out = encoder(local_map, global_maps, robot_state)

    assert out.shape == (2, 32)
    assert encoder.global_encoder_in_channels == (7, 7, 7)
    assert set(encoder.global_dtm_projectors.keys()) == {"64", "32", "16"}


def test_hybrid_encoder_sgcnn_conv1x1_dtm_forward_shape() -> None:
    encoder = HybridLocalGlobalEncoder(
        _config(
            global_encoder_mode="sgcnn",
            sgcnn_target_hw=(16, 16),
            dtm_channels=6,
            dtm_embed_mode="conv1x1",
            dtm_embed_channels=3,
        )
    )
    local_map, global_maps, robot_state = _inputs(global_channels=10)

    out = encoder(local_map, global_maps, robot_state)

    assert out.shape == (2, 32)
    assert encoder.global_encoder_in_channels == (7, 7, 7)
    assert encoder.global_sgcnn is not None


def test_hybrid_encoder_sgcnn_without_dtm_keeps_baseline_channels() -> None:
    encoder = HybridLocalGlobalEncoder(
        _config(
            global_in_channels=(4, 4, 4),
            global_encoder_mode="sgcnn",
            dtm_channels=0,
            dtm_embed_mode="conv1x1",
        )
    )
    local_map, global_maps, robot_state = _inputs(global_channels=4)

    out = encoder(local_map, global_maps, robot_state)

    assert out.shape == (2, 32)
    assert encoder.global_encoder_in_channels == (4, 4, 4)
    assert len(encoder.global_dtm_projectors) == 0


def test_hybrid_encoder_local_spatial_pool_preserves_larger_feature_before_projection() -> None:
    encoder = HybridLocalGlobalEncoder(
        _config(
            local_conv_channels=(6, 8),
            global_conv_channels=(4, 4),
            local_pool_hw=(3, 3),
            local_embed_dim=20,
        )
    )
    local_map, global_maps, robot_state = _inputs(global_channels=10)

    out = encoder(local_map, global_maps, robot_state)

    assert out.shape == (2, 32)
    assert encoder.local_encoder.pool_hw == (3, 3)
    assert encoder.local_encoder.proj.in_features == 8 * 3 * 3


def test_hybrid_encoder_paper41_stride_local_branch_outputs_9x9_feature() -> None:
    encoder = HybridLocalGlobalEncoder(
        _config(
            local_encoder_mode="paper41_stride",
            local_conv_channels=(4, 6, 8),
            local_input_hw=(41, 41),
            local_embed_dim=20,
        )
    )
    local_map, global_maps, robot_state = _inputs(global_channels=10)

    out = encoder(local_map, global_maps, robot_state)

    assert out.shape == (2, 32)
    assert encoder.local_encoder_mode == "paper41_stride"
    assert encoder.local_encoder.spatial_hw == (9, 9)
    assert encoder.local_encoder.proj.in_features == 8 * 9 * 9

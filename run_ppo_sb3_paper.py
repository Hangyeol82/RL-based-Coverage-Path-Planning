import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# Ensure local package imports work even on embeddable Python setups.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.common import (
    FusedMAPSStateEncoderConfig,
    HybridLocalGlobalEncoderConfig,
    MultiLevelMAPSEncoderConfig,
    RobotStateEncoderConfig,
)
from learning.observation import (
    HybridLocalGlobalCPPObservationConfig,
    MultiScaleCPPObservationConfig,
)
from paper_training.robot_state_observation import (
    RobotStateObservationBuilder,
    RobotStateObservationConfig,
)
import learning.reinforcement.cpp_env as cpp_env_module

cpp_env_module.RobotStateObservationBuilder = RobotStateObservationBuilder
from learning.reinforcement.cpp_env import CPPDiscreteEnvConfig
from learning.reinforcement.reward import CPPRewardConfig
from learning.reinforcement.sb3_policy import (
    HybridLocalGlobalFeaturesExtractor,
    MAPSStateFeaturesExtractor,
)
from paper_training.callbacks import PaperMetricsCallback
from paper_training.cpp_env import PaperCPPDiscreteEnv, PaperCPPDiscreteGymEnv
from paper_training.offline_cpp_env import OfflinePaperCPPDiscreteEnv, OfflinePaperCPPDiscreteGymEnv
from MapGenerator import MapGenerator
from run_cstar_custom_map import CUSTOM_MAP_TEXT, parse_custom_map


GridPos = Tuple[int, int]


def _argv_has_flag(*flags: str) -> bool:
    return any(
        arg == flag or arg.startswith(flag + "=")
        for arg in sys.argv[1:]
        for flag in flags
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SB3 PPO on CPP discrete online environment.")
    p.add_argument("--total-timesteps", type=int, default=20000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments. >1 enables multiprocessing rollout collection.",
    )
    p.add_argument(
        "--vec-env",
        type=str,
        default="auto",
        choices=["auto", "dummy", "subproc"],
        help="Vectorized env backend. auto: dummy for 1 env, subproc for >1 envs.",
    )
    p.add_argument(
        "--subproc-start-method",
        type=str,
        default="auto",
        choices=["auto", "spawn", "fork", "forkserver"],
        help="Start method for SubprocVecEnv.",
    )

    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="SB3 PPO verbosity level.",
    )

    p.add_argument("--sensor-range", type=int, default=2, help="2 -> 5x5 sensing window")
    p.add_argument(
        "--full-map-observation",
        action="store_true",
        help=(
            "Offline/full-known setting: reveal the complete true map as known_map "
            "at every step while keeping the PPO loop, reward, and metrics unchanged."
        ),
    )
    p.add_argument(
        "--local-blocks",
        type=str,
        default="",
        help="Optional comma list overriding MAPS local block sizes, e.g. 1,2,4,8",
    )
    p.add_argument("--max-episode-steps", type=int, default=1500)
    rs_pos_group = p.add_mutually_exclusive_group()
    rs_pos_group.add_argument("--robot-state-position", dest="robot_state_position", action="store_true")
    rs_pos_group.add_argument("--no-robot-state-position", dest="robot_state_position", action="store_false")
    rs_action_group = p.add_mutually_exclusive_group()
    rs_action_group.add_argument("--robot-state-action-history", dest="robot_state_action_history", action="store_true")
    rs_action_group.add_argument("--no-robot-state-action-history", dest="robot_state_action_history", action="store_false")
    rs_progress_group = p.add_mutually_exclusive_group()
    rs_progress_group.add_argument("--robot-state-progress", dest="robot_state_progress", action="store_true")
    rs_progress_group.add_argument("--no-robot-state-progress", dest="robot_state_progress", action="store_false")
    rs_stag_group = p.add_mutually_exclusive_group()
    rs_stag_group.add_argument("--robot-state-stagnation", dest="robot_state_stagnation", action="store_true")
    rs_stag_group.add_argument("--no-robot-state-stagnation", dest="robot_state_stagnation", action="store_false")
    p.add_argument("--robot-state-action-history-len", type=int, default=5)
    hole_group = p.add_mutually_exclusive_group()
    hole_group.add_argument("--hole-signals", dest="hole_signals", action="store_true")
    hole_group.add_argument("--no-hole-signals", dest="hole_signals", action="store_false")
    p.add_argument(
        "--coverage-hole-penalty-scale",
        type=float,
        default=0.0,
        help="Penalty applied when the executed action has confirmed hole risk.",
    )
    burden_group = p.add_mutually_exclusive_group()
    burden_group.add_argument(
        "--revisit-burden-shaping",
        dest="revisit_burden_shaping",
        action="store_true",
        help="Enable potential-based shaping on known-free revisit burden.",
    )
    burden_group.add_argument(
        "--no-revisit-burden-shaping",
        dest="revisit_burden_shaping",
        action="store_false",
    )
    p.add_argument(
        "--revisit-burden-scale",
        type=float,
        default=0.0,
        help="Scale for gamma*phi(s') - phi(s) revisit-burden shaping.",
    )
    p.add_argument(
        "--revisit-burden-normalizer",
        type=float,
        default=0.0,
        help="Normalizer for revisit-burden potential. 0 uses max(map height, width).",
    )
    p.add_argument(
        "--revisit-burden-unreachable-cost",
        type=float,
        default=0.0,
        help="Finite cost assigned to known-free targets unreachable in the known map. 0 uses max(map height, width).",
    )
    p.add_argument(
        "--metric-stagnation-threshold",
        type=int,
        default=30,
        help="Paper metric only: no-new-coverage steps before stagnation is counted.",
    )
    p.add_argument(
        "--metric-loop-window",
        type=int,
        default=12,
        help="Paper metric only: recent position window used for loop detection.",
    )
    boundary_group = p.add_mutually_exclusive_group()
    boundary_group.add_argument(
        "--boundary-exit-features",
        dest="boundary_exit_features",
        action="store_true",
        help="Append per-level DTM boundary-exit features to robot_state.",
    )
    boundary_group.add_argument(
        "--no-boundary-exit-features",
        dest="boundary_exit_features",
        action="store_false",
        help="Disable DTM boundary-exit features in robot_state.",
    )
    p.add_argument(
        "--boundary-exit-threshold",
        type=float,
        default=0.0,
        help="Threshold in [0,1] to binarize per-level exit scores.",
    )
    milestone_group = p.add_mutually_exclusive_group()
    milestone_group.add_argument(
        "--milestone-reward",
        dest="milestone_reward",
        action="store_true",
        help="Enable one-time milestone reward bonuses at coverage thresholds.",
    )
    milestone_group.add_argument(
        "--no-milestone-reward",
        dest="milestone_reward",
        action="store_false",
        help="Disable milestone reward bonuses.",
    )
    p.add_argument("--milestone-threshold-90", type=float, default=0.90)
    p.add_argument("--milestone-threshold-99", type=float, default=0.99)
    p.add_argument("--milestone-lambda-90", type=float, default=0.2)
    p.add_argument("--milestone-lambda-99", type=float, default=4.0)
    overlap_group = p.add_mutually_exclusive_group()
    overlap_group.add_argument(
        "--overlap-streak-penalty",
        dest="overlap_streak_penalty",
        action="store_true",
        help="Increase overlap penalty when overlap repeats for many consecutive steps.",
    )
    overlap_group.add_argument(
        "--no-overlap-streak-penalty",
        dest="overlap_streak_penalty",
        action="store_false",
        help="Disable overlap-streak penalty acceleration.",
    )
    p.add_argument("--overlap-streak-grace", type=int, default=2)
    p.add_argument("--overlap-streak-increment", type=float, default=0.05)
    p.add_argument(
        "--overlap-streak-max-abs",
        type=float,
        default=0.4,
        help="Maximum absolute overlap penalty after streak acceleration.",
    )
    p.add_argument("--include-dtm", action="store_true")
    p.add_argument(
        "--obs-unknown-policy",
        type=str,
        default="keep",
        choices=["keep", "as_free", "as_obstacle"],
        help="How to handle unknown cells in map-observation channels.",
    )
    phase_group = p.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--cell-phase-channels",
        dest="cell_phase_channels",
        action="store_true",
        help="Append per-level robot-in-cell phase features to robot_state.",
    )
    phase_group.add_argument(
        "--no-cell-phase-channels",
        dest="cell_phase_channels",
        action="store_false",
        help="Disable per-level robot-in-cell phase features.",
    )
    p.add_argument(
        "--dtm-coarse-mode",
        type=str,
        default="bfs",
        choices=["bfs", "aggregate", "aggregate_transfer"],
        help=(
            "bfs: compute DTM at every scale; "
            "aggregate: fine-scale BFS + max-pool upward; "
            "aggregate_transfer: fine-scale BFS + transfer-graph coarse composition."
        ),
    )
    p.add_argument(
        "--dtm-output-mode",
        type=str,
        default="six",
        choices=["six", "extent6", "two", "axis2", "axis2km", "four", "port6", "port12"],
        help=(
            "DTM output channels: six(side-pair), extent6, two/axis2(LR/UD), "
            "axis2km(pass+known for LR/UD), four(legacy), "
            "port6(undirected side-to-side), or port12(side-to-side)."
        ),
    )
    p.add_argument(
        "--dtm-connectivity",
        type=int,
        default=4,
        choices=[4, 8],
        help="Connectivity used inside DTM connectivity checks. 4 matches the environment action space.",
    )
    p.add_argument(
        "--maps-encoder-mode",
        type=str,
        default="sgcnn",
        choices=["sgcnn", "independent", "hybrid_local_global"],
    )
    p.add_argument("--local-crop-size", type=int, default=41)
    p.add_argument("--global-coarse-size", type=int, default=16, help="Deprecated; use --global-coarse-sizes.")
    p.add_argument("--global-coarse-sizes", type=str, default="64,32,16")
    p.add_argument(
        "--model-size",
        type=str,
        default="xlarge",
        choices=["small", "large", "xlarge"],
        help="Policy encoder size preset.",
    )
    mask_group = p.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--action-mask",
        dest="action_mask",
        action="store_true",
        help=(
            "Enable known-map action masking. Uses MaskablePPO when available; "
            "otherwise applies env-side safety masking."
        ),
    )
    mask_group.add_argument(
        "--no-action-mask",
        dest="action_mask",
        action="store_false",
        help="Disable action masking.",
    )
    heuristic_group = p.add_mutually_exclusive_group()
    heuristic_group.add_argument(
        "--heuristic-assist",
        dest="heuristic_assist",
        action="store_true",
        help=(
            "After repeated no-coverage steps, override the policy with an "
            "online-safe A* move toward the nearest observed uncovered free cell."
        ),
    )
    heuristic_group.add_argument(
        "--no-heuristic-assist",
        dest="heuristic_assist",
        action="store_false",
        help="Disable A* heuristic action assist.",
    )
    p.add_argument(
        "--heuristic-no-progress-threshold",
        type=int,
        default=50,
        help="No-new-coverage steps before heuristic assist may activate.",
    )
    p.add_argument(
        "--heuristic-min-coverage",
        type=float,
        default=0.0,
        help="Minimum current coverage ratio before heuristic assist may activate.",
    )
    p.add_argument(
        "--heuristic-max-astar-expansions",
        type=int,
        default=0,
        help="Maximum A* node expansions per assist call. 0 uses the full map size.",
    )

    p.add_argument("--map-source", type=str, default="random", choices=["random", "custom", "file"])
    p.add_argument("--map-size", type=int, default=32)
    p.add_argument("--map-stage", type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--num-obstacles", type=int, default=None)
    p.add_argument("--map-file", type=str, default=None)
    p.add_argument(
        "--map-pool-dir",
        type=str,
        default="",
        help="Optional directory of map .txt files used for episode-level map refresh.",
    )
    p.add_argument(
        "--episode-map-refresh",
        action="store_true",
        help="Use a new map from --map-pool-dir on every episode reset.",
    )
    p.add_argument(
        "--map-refresh-mode",
        type=str,
        default="cycle",
        choices=["cycle", "random"],
        help="Map-pool sampling mode for episode-level map refresh.",
    )
    p.add_argument(
        "--episode-success-threshold",
        type=float,
        default=1.0,
        help="End an episode early when coverage reaches this threshold. 1.0 preserves full-coverage behavior.",
    )

    p.add_argument("--save-model", type=str, default="learning/checkpoints/rl/ppo_sb3_latest")
    p.add_argument("--save-breakdown-json", type=str, default="learning/checkpoints/rl/logs/ppo_breakdown.json")
    p.add_argument("--save-breakdown-csv", type=str, default="learning/checkpoints/rl/logs/ppo_breakdown.csv")
    p.add_argument(
        "--load-model",
        type=str,
        default="",
        help="Resume PPO from existing SB3 .zip model. If set, PPO hyper-params come from checkpoint.",
    )
    p.add_argument(
        "--init-from-bc",
        type=str,
        default="",
        help="Warm-start PPO feature encoder from BC checkpoint (.pt). Applied only when not resuming PPO.",
    )
    p.add_argument(
        "--init-from-bc-strict",
        action="store_true",
        help="Require exact BC->PPO encoder key/shape match. Default loads shape-compatible subset.",
    )
    p.set_defaults(
        action_mask=True,
        milestone_reward=False,
        overlap_streak_penalty=False,
        boundary_exit_features=False,
        robot_state_position=True,
        robot_state_action_history=True,
        robot_state_progress=False,
        robot_state_stagnation=False,
        cell_phase_channels=True,
        hole_signals=False,
        revisit_burden_shaping=False,
        heuristic_assist=False,
    )
    args = p.parse_args()
    args.robot_state_progress = False
    args.robot_state_stagnation = False
    if (
        str(args.maps_encoder_mode).strip().lower() == "hybrid_local_global"
        and not _argv_has_flag("--cell-phase-channels", "--no-cell-phase-channels")
    ):
        args.cell_phase_channels = False
    return args


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_local_blocks(raw: str) -> Optional[Tuple[int, ...]]:
    s = str(raw).strip()
    if not s:
        return None
    vals = []
    for tok in s.split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        return None
    if any(v <= 0 for v in vals):
        raise ValueError("--local-blocks values must be positive")
    if sorted(vals) != vals:
        raise ValueError("--local-blocks must be in increasing order")
    if len(set(vals)) != len(vals):
        raise ValueError("--local-blocks must not contain duplicates")
    return tuple(vals)


def _parse_global_coarse_sizes(raw: str) -> Tuple[int, ...]:
    vals = []
    for tok in str(raw).split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        raise ValueError("--global-coarse-sizes must contain at least one size")
    if any(v <= 0 for v in vals):
        raise ValueError("--global-coarse-sizes values must be positive")
    return tuple(vals)


def _model_preset(model_size: str) -> Dict[str, object]:
    if model_size == "xlarge":
        return dict(
            conv_channels=(64, 128),
            level_embed_dim=256,
            state_hidden_dims=(256, 256),
            fusion_hidden_dims=(1024, 1024),
        )
    if model_size == "large":
        return dict(
            conv_channels=(32, 64),
            level_embed_dim=128,
            state_hidden_dims=(128, 128),
            fusion_hidden_dims=(512, 512),
        )
    # Legacy default.
    return dict(
        conv_channels=(16, 32),
        level_embed_dim=64,
        state_hidden_dims=(64, 64),
        fusion_hidden_dims=(256, 256),
    )


def _select_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prepare_square_map(base_map: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        raise ValueError("map-size must be positive")
    out = np.zeros((size, size), dtype=np.int32)
    h, w = base_map.shape
    rh = min(h, size)
    rw = min(w, size)
    out[:rh, :rw] = base_map[:rh, :rw]
    return out


def _parse_map_text(text: str) -> np.ndarray:
    rows = [list(map(int, line.split())) for line in text.strip().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Map text is empty")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"Inconsistent row widths: {sorted(widths)}")
    grid = np.asarray(rows, dtype=np.int32)
    if not np.isin(grid, [0, 1]).all():
        raise ValueError("Map must contain only 0 and 1")
    return grid


def _build_map(args: argparse.Namespace) -> np.ndarray:
    if args.map_source == "custom":
        base = parse_custom_map(CUSTOM_MAP_TEXT).astype(np.int32)
        if base[0, 0] == 1:
            base[0, 0] = 0
        return _prepare_square_map(base, args.map_size)

    if args.map_source == "file":
        if not args.map_file:
            raise ValueError("--map-file is required with --map-source file")
        p = Path(args.map_file)
        if not p.exists():
            raise FileNotFoundError(f"Map file not found: {p}")
        base = _parse_map_text(p.read_text(encoding="utf-8")).astype(np.int32)
        if base[0, 0] == 1:
            base[0, 0] = 0
        return _prepare_square_map(base, args.map_size)

    gen = MapGenerator(height=args.map_size, width=args.map_size, seed=args.seed)
    grid = np.asarray(
        gen.generate_map(stage=args.map_stage, num_obstacles=args.num_obstacles),
        dtype=np.int32,
    )
    if grid.shape != (args.map_size, args.map_size):
        grid = _prepare_square_map(grid, args.map_size)
    if grid[0, 0] == 1:
        grid[0, 0] = 0
    return grid


def _load_map_pool(
    args: argparse.Namespace,
    first_grid: np.ndarray,
) -> Tuple[Tuple[np.ndarray, ...], Tuple[Dict[str, Any], ...]]:
    raw = str(args.map_pool_dir or "").strip()
    if not raw:
        return (first_grid,), ({"map_index": 0},)
    pool_dir = Path(raw)
    if not pool_dir.exists() or not pool_dir.is_dir():
        raise FileNotFoundError(f"Map pool directory not found: {pool_dir}")
    maps = []
    paths = sorted(pool_dir.glob("*.txt"))
    for path in paths:
        grid = _parse_map_text(path.read_text(encoding="utf-8")).astype(np.int32)
        if grid[0, 0] == 1:
            grid[0, 0] = 0
        grid = _prepare_square_map(grid, args.map_size)
        if tuple(grid.shape) != tuple(first_grid.shape):
            raise ValueError(
                f"Map pool shape mismatch for {path}: {grid.shape} != {first_grid.shape}"
            )
        maps.append(grid)
    if not maps:
        raise ValueError(f"Map pool directory contains no .txt maps: {pool_dir}")

    manifest_by_name: Dict[str, Dict[str, Any]] = {}
    manifest_path = pool_dir / "manifest.jsonl"
    if manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            path_raw = str(row.get("path", ""))
            key = Path(path_raw).name if path_raw else str(row.get("file", ""))
            if key:
                manifest_by_name[key] = dict(row)
    metadata: List[Dict[str, Any]] = []
    for idx, path in enumerate(paths):
        row = dict(manifest_by_name.get(path.name, {}))
        row.setdefault("map_index", int(idx))
        row.setdefault("path", str(path))
        metadata.append(row)
    return tuple(maps), tuple(metadata)


def _pick_start(grid: np.ndarray) -> GridPos:
    free = np.argwhere(grid == 0)
    if free.size == 0:
        raise RuntimeError("No free cell exists in map")
    r, c = free[0]
    return int(r), int(c)


def _get_checkpoint_state_dict(payload: object) -> Dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a dictionary")

    raw = None
    if "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
        raw = payload["model_state_dict"]
    elif "state_dict" in payload and isinstance(payload["state_dict"], dict):
        raw = payload["state_dict"]
    elif all(isinstance(v, torch.Tensor) for v in payload.values()):
        raw = payload

    if raw is None:
        keys = list(payload.keys())[:8]
        raise ValueError(
            "Unsupported checkpoint format. Expected keys like model_state_dict/state_dict. "
            f"Top-level keys sample: {keys}"
        )
    return raw


def _extract_encoder_state_dict(raw_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    encoder_state: Dict[str, torch.Tensor] = {}
    prefixes = (
        "encoder.",
        "features_extractor.encoder.",
        "policy.features_extractor.encoder.",
    )
    direct_prefixes = ("maps_encoder.", "state_encoder.", "fusion.")

    for key, value in raw_state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue
        mapped = None
        for prefix in prefixes:
            if key.startswith(prefix):
                mapped = key[len(prefix) :]
                break
        if mapped is None and key.startswith(direct_prefixes):
            mapped = key
        if mapped is not None:
            encoder_state[mapped] = value
    return encoder_state


def _warm_start_encoder_from_bc(
    model,
    checkpoint_path: str,
    *,
    strict: bool = False,
) -> Dict[str, int]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    raw_state = _get_checkpoint_state_dict(ckpt)
    source_state = _extract_encoder_state_dict(raw_state)
    if len(source_state) == 0:
        raise ValueError(
            "No encoder parameters found in BC checkpoint. "
            "Expected keys starting with encoder. or policy.features_extractor.encoder."
        )

    target_encoder = model.policy.features_extractor.encoder
    target_state = target_encoder.state_dict()

    loadable: Dict[str, torch.Tensor] = {}
    shape_mismatch = []
    unexpected = []
    for key, value in source_state.items():
        if key not in target_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target_state[key].shape):
            shape_mismatch.append((key, tuple(value.shape), tuple(target_state[key].shape)))
            continue
        loadable[key] = value

    missing = [key for key in target_state.keys() if key not in loadable]
    if strict and (len(unexpected) > 0 or len(shape_mismatch) > 0 or len(missing) > 0):
        msg = (
            "BC strict warm-start failed: "
            f"loadable={len(loadable)}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}, shape_mismatch={len(shape_mismatch)}"
        )
        raise ValueError(msg)

    merged = dict(target_state)
    merged.update(loadable)
    target_encoder.load_state_dict(merged, strict=False)

    return {
        "source_keys": len(source_state),
        "loaded_keys": len(loadable),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "shape_mismatch": len(shape_mismatch),
    }


def main():
    args = _parse_args()
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.load_model and args.init_from_bc:
        raise ValueError("--load-model and --init-from-bc are mutually exclusive")
    _set_seed(args.seed)
    device = _select_device(args.device)

    try:
        from stable_baselines3 import PPO as SB3PPO
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except Exception as e:
        raise RuntimeError(
            "stable_baselines3 is not installed in this environment. "
            "Install it first, then rerun."
        ) from e

    AlgoClass = SB3PPO
    algo_name = "PPO"
    if args.action_mask:
        try:
            from sb3_contrib import MaskablePPO  # type: ignore
        except Exception:
            print(
                "[WARN] sb3_contrib not found. Falling back to PPO with env-side action masking.",
                flush=True,
            )
        else:
            AlgoClass = MaskablePPO
            algo_name = "MaskablePPO"

    grid = _build_map(args)
    grid_pool, grid_meta_pool = _load_map_pool(args, grid)
    episode_map_refresh = bool(args.episode_map_refresh) and len(grid_pool) > 1
    start = _pick_start(grid)
    local_blocks = _parse_local_blocks(args.local_blocks)
    global_coarse_sizes = _parse_global_coarse_sizes(args.global_coarse_sizes)

    reward_cfg = CPPRewardConfig(
        new_cell_reward=0.7,
        local_tv_reward_scale=0.0,
        local_tv_reward_max=5.0,
        local_tv_normalizer=2.5,
        global_tv_reward_scale=0.0,
        global_tv_reward_max=5.0,
        global_tv_normalizer=4.0,
        collision_reward=-10.0,
        constant_reward=-0.1,
        constant_reward_always=True,
        overlap_streak_enabled=bool(args.overlap_streak_penalty),
        overlap_streak_grace=int(args.overlap_streak_grace),
        overlap_streak_increment=float(args.overlap_streak_increment),
        overlap_streak_max_abs=float(args.overlap_streak_max_abs),
        milestone_reward_enabled=bool(args.milestone_reward),
        milestone_threshold_90=float(args.milestone_threshold_90),
        milestone_threshold_99=float(args.milestone_threshold_99),
        milestone_lambda_90=float(args.milestone_lambda_90),
        milestone_lambda_99=float(args.milestone_lambda_99),
    )
    env_cfg = CPPDiscreteEnvConfig(
        sensor_range=args.sensor_range,
        max_steps=args.max_episode_steps,
        collision_ends_episode=False,
        stop_on_full_coverage=True,
        include_dtm=args.include_dtm,
        maps_observation_mode=(
            "hybrid_local_global"
            if str(args.maps_encoder_mode).strip().lower() == "hybrid_local_global"
            else "multiscale"
        ),
        use_boundary_exit_features=bool(args.boundary_exit_features),
        boundary_exit_threshold=float(args.boundary_exit_threshold),
        observation=MultiScaleCPPObservationConfig(
            local_blocks=local_blocks or MultiScaleCPPObservationConfig().local_blocks,
            unknown_policy=str(args.obs_unknown_policy),
            dtm_coarse_mode=str(args.dtm_coarse_mode),
            dtm_output_mode=str(args.dtm_output_mode),
            dtm_connectivity=int(args.dtm_connectivity),
            include_cell_phase_channels=False,
        ),
        hybrid_observation=HybridLocalGlobalCPPObservationConfig(
            local_crop_size=int(args.local_crop_size),
            global_coarse_size=int(args.global_coarse_size),
            global_coarse_sizes=global_coarse_sizes,
            unknown_policy=str(args.obs_unknown_policy),
            dtm_patch_size=7,
            dtm_connectivity=int(args.dtm_connectivity),
            dtm_require_fully_known_patch=False,
            dtm_min_known_ratio=0.6,
            dtm_patch_min_known_ratio=0.6,
            dtm_uncertain_fill=-1.0,
            dtm_unknown_fill=-1.0,
            dtm_output_mode=str(args.dtm_output_mode),
        ),
        robot_state=RobotStateObservationConfig(
            action_history_len=int(args.robot_state_action_history_len),
            include_position=bool(args.robot_state_position),
            include_action_history=bool(args.robot_state_action_history),
            include_progress=bool(args.robot_state_progress),
            include_stagnation=bool(args.robot_state_stagnation),
        ),
        use_cell_phase_features=bool(args.cell_phase_channels),
        use_action_mask=bool(args.action_mask),
        heuristic_assist_enabled=bool(args.heuristic_assist),
        heuristic_no_progress_threshold=int(args.heuristic_no_progress_threshold),
        heuristic_min_coverage=float(args.heuristic_min_coverage),
        heuristic_max_astar_expansions=int(args.heuristic_max_astar_expansions),
        reward=reward_cfg,
    )

    CoreEnvClass = OfflinePaperCPPDiscreteEnv if bool(args.full_map_observation) else PaperCPPDiscreteEnv
    GymEnvClass = OfflinePaperCPPDiscreteGymEnv if bool(args.full_map_observation) else PaperCPPDiscreteGymEnv

    probe = CoreEnvClass(
        grid_map=grid,
        start_pos=start,
        config=env_cfg,
        include_hole_signals=bool(args.hole_signals),
        hole_penalty_scale=float(args.coverage_hole_penalty_scale),
        revisit_burden_shaping=bool(args.revisit_burden_shaping),
        revisit_burden_scale=float(args.revisit_burden_scale),
        revisit_burden_gamma=float(args.gamma),
        revisit_burden_normalizer=float(args.revisit_burden_normalizer),
        revisit_burden_unreachable_cost=float(args.revisit_burden_unreachable_cost),
        metric_stagnation_threshold=int(args.metric_stagnation_threshold),
        metric_loop_window=int(args.metric_loop_window),
        grid_map_pool=grid_pool,
        grid_map_metadata_pool=grid_meta_pool,
        episode_map_refresh=episode_map_refresh,
        map_refresh_mode=str(args.map_refresh_mode),
        map_refresh_seed=int(args.seed),
        episode_success_threshold=float(args.episode_success_threshold),
    )
    probe_obs = probe.reset()
    robot_state_dim = int(np.asarray(probe_obs["robot_state"], dtype=np.float32).shape[0])
    model_cfg = _model_preset(str(args.model_size))
    use_hybrid_maps = "hybrid_maps" in probe_obs
    level_channels: Tuple[int, ...] = ()
    local_channels = 0
    global_channels: Tuple[int, ...] = ()
    if use_hybrid_maps:
        hybrid_obs = probe_obs["hybrid_maps"]
        local_channels = int(np.asarray(hybrid_obs["local"], dtype=np.float32).shape[0])
        global_channels = tuple(
            int(np.asarray(hybrid_obs[f"global_{size}"], dtype=np.float32).shape[0])
            for size in global_coarse_sizes
        )
        encoder_cfg = HybridLocalGlobalEncoderConfig(
            local_in_channels=local_channels,
            global_in_channels=global_channels,
            global_sizes=global_coarse_sizes,
            conv_channels=model_cfg["conv_channels"],
            local_embed_dim=int(model_cfg["level_embed_dim"]),
            global_embed_dim=int(model_cfg["level_embed_dim"]),
            robot_state=RobotStateEncoderConfig(
                input_dim=robot_state_dim,
                hidden_dims=model_cfg["state_hidden_dims"],
            ),
            fusion_hidden_dims=model_cfg["fusion_hidden_dims"],
        )
        features_extractor_class = HybridLocalGlobalFeaturesExtractor
    else:
        level_ids = tuple(sorted(probe_obs["levels"].keys()))
        level_channels = tuple(
            int(np.asarray(probe_obs["levels"][lv], dtype=np.float32).shape[0])
            for lv in level_ids
        )
        maps_cfg = MultiLevelMAPSEncoderConfig(
            num_levels=probe.maps_builder.num_levels,
            in_channels_per_level=level_channels,
            conv_channels=model_cfg["conv_channels"],
            level_embed_dim=int(model_cfg["level_embed_dim"]),
            mode=args.maps_encoder_mode,
        )
        encoder_cfg = FusedMAPSStateEncoderConfig(
            maps=maps_cfg,
            robot_state=RobotStateEncoderConfig(
                input_dim=robot_state_dim,
                hidden_dims=model_cfg["state_hidden_dims"],
            ),
            fusion_hidden_dims=model_cfg["fusion_hidden_dims"],
        )
        features_extractor_class = MAPSStateFeaturesExtractor

    if args.vec_env == "auto":
        vec_env_mode = "dummy" if args.num_envs == 1 else "subproc"
    else:
        vec_env_mode = args.vec_env
    if vec_env_mode == "subproc" and args.num_envs == 1:
        # Subproc with one env is valid but unnecessary; use dummy for lower overhead.
        vec_env_mode = "dummy"

    if args.subproc_start_method == "auto":
        # Windows requires spawn. Linux/macOS can use forkserver.
        start_method = "spawn" if os.name == "nt" else "forkserver"
    else:
        start_method = args.subproc_start_method

    if vec_env_mode == "dummy":
        vec_cls = DummyVecEnv
        vec_kwargs = None
    else:
        vec_cls = SubprocVecEnv
        vec_kwargs = {"start_method": start_method}

    vec_env = make_vec_env(
        GymEnvClass,
        n_envs=args.num_envs,
        seed=args.seed,
        env_kwargs=dict(
            grid_map=grid,
            start_pos=start,
            config=env_cfg,
            include_hole_signals=bool(args.hole_signals),
            hole_penalty_scale=float(args.coverage_hole_penalty_scale),
            revisit_burden_shaping=bool(args.revisit_burden_shaping),
            revisit_burden_scale=float(args.revisit_burden_scale),
            revisit_burden_gamma=float(args.gamma),
            revisit_burden_normalizer=float(args.revisit_burden_normalizer),
            revisit_burden_unreachable_cost=float(args.revisit_burden_unreachable_cost),
            metric_stagnation_threshold=int(args.metric_stagnation_threshold),
            metric_loop_window=int(args.metric_loop_window),
            grid_map_pool=grid_pool,
            grid_map_metadata_pool=grid_meta_pool,
            episode_map_refresh=episode_map_refresh,
            map_refresh_mode=str(args.map_refresh_mode),
            map_refresh_seed=int(args.seed),
            episode_success_threshold=float(args.episode_success_threshold),
        ),
        vec_env_cls=vec_cls,
        vec_env_kwargs=vec_kwargs,
    )
    if algo_name == "MaskablePPO" and bool(args.action_mask):
        # sb3-contrib checks mask support at every rollout. Some versions call
        # VecEnv.has_attr(), and older versions call SubprocVecEnv.get_attr()
        # directly. Both routes can ask workers for the bound action_masks
        # method, which pickles the whole env object. DTM envs carry large
        # observation caches, so that support check can dominate runtime.
        original_has_attr = getattr(vec_env, "has_attr", None)

        def _has_attr_without_pickling_mask_method(attr_name: str) -> bool:
            if str(attr_name) == "action_masks":
                return True
            if original_has_attr is not None:
                return bool(original_has_attr(attr_name))
            try:
                vec_env.get_attr(attr_name)
                return True
            except AttributeError:
                return False

        vec_env.has_attr = _has_attr_without_pickling_mask_method  # type: ignore[method-assign]
        try:
            import sb3_contrib.common.maskable.utils as mask_utils  # type: ignore
            import sb3_contrib.ppo_mask.ppo_mask as ppo_mask_module  # type: ignore

            def _fast_is_masking_supported(_env: Any) -> bool:
                return True

            mask_utils.is_masking_supported = _fast_is_masking_supported
            ppo_mask_module.is_masking_supported = _fast_is_masking_supported
        except Exception as patch_err:
            print(f"[WARN] Could not patch MaskablePPO mask support check: {patch_err}", flush=True)

    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=dict(encoder_config=encoder_cfg),
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    if args.load_model:
        try:
            model = AlgoClass.load(args.load_model, env=vec_env, device=device)
        except Exception as load_err:
            # Old checkpoints may have been saved with plain PPO; recover gracefully.
            if algo_name == "MaskablePPO":
                print(
                    f"[WARN] MaskablePPO load failed ({load_err}). Retrying with PPO loader.",
                    flush=True,
                )
                AlgoClass = SB3PPO
                algo_name = "PPO"
                model = AlgoClass.load(args.load_model, env=vec_env, device=device)
            else:
                raise
        model.set_random_seed(args.seed)
    else:
        model = AlgoClass(
            policy="MultiInputPolicy",
            env=vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            policy_kwargs=policy_kwargs,
            seed=args.seed,
            device=device,
            verbose=int(args.verbose),
        )

    bc_warm_start_stats = None
    if args.init_from_bc:
        bc_warm_start_stats = _warm_start_encoder_from_bc(
            model,
            args.init_from_bc,
            strict=bool(args.init_from_bc_strict),
        )

    callback = PaperMetricsCallback(verbose=0)

    print(f"Device: {device}")
    print(f"Map: source={args.map_source}, shape={grid.shape}, include_dtm={args.include_dtm}")
    if use_hybrid_maps:
        global_desc = ", ".join(
            f"{size}x{size}:C{channels}"
            for size, channels in zip(global_coarse_sizes, global_channels)
        )
        print(
            "Hybrid observation:"
            f" local_channels={local_channels}, local_crop={int(args.local_crop_size)}x{int(args.local_crop_size)},"
            f" global_scales=[{global_desc}]"
        )
    else:
        print(f"Map observation channels by level: {level_channels}")
    print(
        "Observation setting:"
        f" {'offline_full_map' if bool(args.full_map_observation) else 'online_partial'},"
        f" sensor_range={int(args.sensor_range)}"
    )
    print(
        "Map pool:"
        f" size={len(grid_pool)},"
        f" episode_refresh={episode_map_refresh},"
        f" mode={args.map_refresh_mode},"
        f" success_threshold={float(args.episode_success_threshold):.3f}"
    )
    print(f"Obs unknown policy: {args.obs_unknown_policy}")
    print(f"Cell phase features: {bool(args.cell_phase_channels)}")
    print(f"Start: {start}")
    print(
        f"Action mask: {bool(args.action_mask)} | algo={algo_name} | "
        f"env_shield={bool(env_cfg.use_action_mask)}"
    )
    print(
        "PPO cfg:"
        f" total_timesteps={args.total_timesteps}, num_envs={args.num_envs},"
        f" vec_env={vec_env_mode}, n_steps={args.n_steps},"
        f" rollout_batch={args.n_steps * args.num_envs},"
        f" batch_size={args.batch_size}, n_epochs={args.n_epochs}"
    )
    print(
        "Reward shaping:"
        f" new_cell={float(reward_cfg.new_cell_reward):.3f},"
        f" local_tv_scale={float(reward_cfg.local_tv_reward_scale):.3f},"
        f" global_tv_scale={float(reward_cfg.global_tv_reward_scale):.3f},"
        f" collision={float(reward_cfg.collision_reward):.3f},"
        f" step={float(reward_cfg.constant_reward):.3f},"
        f" turn={float(reward_cfg.turn_change_penalty):.3f},"
        f" revisit={float(reward_cfg.revisit_penalty):.3f}"
    )
    print(
        "Milestone reward:"
        f" enabled={bool(args.milestone_reward)},"
        f" t90={float(args.milestone_threshold_90):.3f},"
        f" t99={float(args.milestone_threshold_99):.3f},"
        f" lambda90={float(args.milestone_lambda_90):.3f},"
        f" lambda99={float(args.milestone_lambda_99):.3f}"
    )
    print(
        "Boundary-exit features:"
        f" enabled={bool(args.boundary_exit_features)},"
        f" threshold={float(args.boundary_exit_threshold):.3f},"
        f" robot_state_dim={robot_state_dim}"
    )
    print(
        "Robot-state features:"
        f" position={bool(args.robot_state_position)},"
        f" action_history={bool(args.robot_state_action_history)},"
        f" progress={bool(args.robot_state_progress)},"
        f" stagnation={bool(args.robot_state_stagnation)},"
        f" action_history_len={int(args.robot_state_action_history_len)}"
    )
    print(
        "Hole features:"
        f" signals={bool(args.hole_signals)},"
        f" penalty_scale={float(args.coverage_hole_penalty_scale):.3f}"
    )
    print(
        "Revisit-burden shaping:"
        f" enabled={bool(args.revisit_burden_shaping) or float(args.revisit_burden_scale) > 0.0},"
        f" scale={float(args.revisit_burden_scale):.4f},"
        f" gamma={float(args.gamma):.4f},"
        f" normalizer={float(args.revisit_burden_normalizer):.1f},"
        f" unreachable_cost={float(args.revisit_burden_unreachable_cost):.1f}"
    )
    print(
        "Heuristic assist:"
        f" enabled={bool(args.heuristic_assist)},"
        f" threshold={int(args.heuristic_no_progress_threshold)},"
        f" min_coverage={float(args.heuristic_min_coverage):.3f},"
        f" max_astar_expansions={int(args.heuristic_max_astar_expansions)}"
    )
    print(
        "Paper metrics:"
        f" stagnation_threshold={int(args.metric_stagnation_threshold)},"
        f" loop_window={int(args.metric_loop_window)}"
    )
    print(
        f"Model size: {args.model_size} | conv={model_cfg['conv_channels']} | "
        f"level_embed={model_cfg['level_embed_dim']} | "
        f"state={model_cfg['state_hidden_dims']} | fusion={model_cfg['fusion_hidden_dims']}"
    )
    if args.load_model:
        print(f"Init mode: resume PPO from {args.load_model}")
    elif bc_warm_start_stats is not None:
        print(
            "Init mode: BC warm-start | "
            f"loaded={bc_warm_start_stats['loaded_keys']}/{bc_warm_start_stats['source_keys']} "
            f"(missing={bc_warm_start_stats['missing_keys']}, "
            f"unexpected={bc_warm_start_stats['unexpected_keys']}, "
            f"shape_mismatch={bc_warm_start_stats['shape_mismatch']})"
        )
        print(f"BC checkpoint: {args.init_from_bc}")
    else:
        print("Init mode: random init")
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    if args.save_model:
        out = Path(args.save_model)
        out.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(out))
        print(f"Saved model: {out}")

    if args.save_breakdown_json:
        callback.save_json(args.save_breakdown_json)
        print(f"Saved breakdown json: {args.save_breakdown_json}")
    if args.save_breakdown_csv:
        callback.save_csv(args.save_breakdown_csv)
        print(f"Saved breakdown csv: {args.save_breakdown_csv}")

    if callback.history:
        last = callback.history[-1]
        print("Last rollout breakdown:")
        for key in sorted(last.keys()):
            print(f"  {key}: {last[key]}")

    vec_env.close()


if __name__ == "__main__":
    main()

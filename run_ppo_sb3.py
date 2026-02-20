import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

# Ensure local package imports work even on embeddable Python setups.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.common import (
    FusedMAPSStateEncoderConfig,
    MultiLevelMAPSEncoderConfig,
    RobotStateEncoderConfig,
)
from learning.reinforcement.cpp_env import CPPDiscreteEnv, CPPDiscreteEnvConfig
from learning.reinforcement.reward import CPPRewardConfig
from learning.reinforcement.sb3_callbacks import RewardBreakdownCallback
from learning.reinforcement.sb3_env import CPPDiscreteGymEnv
from learning.reinforcement.sb3_policy import MAPSStateFeaturesExtractor
from MapGenerator import MapGenerator
from run_cstar_custom_map import CUSTOM_MAP_TEXT, parse_custom_map


GridPos = Tuple[int, int]


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

    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)

    p.add_argument("--sensor-range", type=int, default=2, help="2 -> 5x5 sensing window")
    p.add_argument("--max-episode-steps", type=int, default=1500)
    p.add_argument("--include-dtm", action="store_true")
    p.add_argument("--maps-encoder-mode", type=str, default="sgcnn", choices=["sgcnn", "independent"])
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

    p.add_argument("--map-source", type=str, default="random", choices=["random", "custom", "file"])
    p.add_argument("--map-size", type=int, default=32)
    p.add_argument("--map-stage", type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--num-obstacles", type=int, default=None)
    p.add_argument("--map-file", type=str, default=None)

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
    p.set_defaults(action_mask=True)
    return p.parse_args()


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
    start = _pick_start(grid)

    reward_cfg = CPPRewardConfig(
        newly_visited_reward_scale=1.0,
        newly_visited_reward_max=2.0,
        local_tv_reward_scale=1.0,
        local_tv_reward_max=5.0,
        local_tv_normalizer=2.5,
        global_tv_reward_scale=0.0,
        global_tv_reward_max=5.0,
        global_tv_normalizer=4.0,
        collision_reward=-10.0,
        constant_reward=-0.1,
        constant_reward_always=True,
    )
    env_cfg = CPPDiscreteEnvConfig(
        sensor_range=args.sensor_range,
        max_steps=args.max_episode_steps,
        collision_ends_episode=False,
        stop_on_full_coverage=True,
        include_dtm=args.include_dtm,
        use_action_mask=bool(args.action_mask),
        reward=reward_cfg,
    )

    probe = CPPDiscreteEnv(grid_map=grid, start_pos=start, config=env_cfg)
    maps_cfg = MultiLevelMAPSEncoderConfig(
        num_levels=probe.maps_builder.num_levels,
        in_channels_per_level=probe.maps_builder.channels_per_level,
        conv_channels=(16, 32),
        level_embed_dim=64,
        mode=args.maps_encoder_mode,
    )
    encoder_cfg = FusedMAPSStateEncoderConfig(
        maps=maps_cfg,
        robot_state=RobotStateEncoderConfig(input_dim=9, hidden_dims=(64, 64)),
        fusion_hidden_dims=(256, 256),
    )

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
        CPPDiscreteGymEnv,
        n_envs=args.num_envs,
        seed=args.seed,
        env_kwargs=dict(grid_map=grid, start_pos=start, config=env_cfg),
        vec_env_cls=vec_cls,
        vec_env_kwargs=vec_kwargs,
    )

    policy_kwargs = dict(
        features_extractor_class=MAPSStateFeaturesExtractor,
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
            verbose=1,
        )

    bc_warm_start_stats = None
    if args.init_from_bc:
        bc_warm_start_stats = _warm_start_encoder_from_bc(
            model,
            args.init_from_bc,
            strict=bool(args.init_from_bc_strict),
        )

    callback = RewardBreakdownCallback(verbose=0)

    print(f"Device: {device}")
    print(f"Map: source={args.map_source}, shape={grid.shape}, include_dtm={args.include_dtm}")
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

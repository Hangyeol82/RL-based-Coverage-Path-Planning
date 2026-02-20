import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

# Keep eval script robust in constrained/sandboxed environments.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
if "NUMEXPR_NUM_THREADS" not in os.environ:
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

_THIS_DIR = Path(__file__).resolve().parent
_MPL_CACHE_DIR = _THIS_DIR / ".mplconfig"
if "MPLCONFIGDIR" not in os.environ:
    _MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_MPL_CACHE_DIR)

import matplotlib.pyplot as plt
import numpy as np

from learning.reinforcement.cpp_env import CPPDiscreteEnvConfig
from learning.reinforcement.reward import CPPRewardConfig
from learning.reinforcement.sb3_env import CPPDiscreteGymEnv
from MapGenerator import MapGenerator
from run_cstar_custom_map import CUSTOM_MAP_TEXT, parse_custom_map


GridPos = Tuple[int, int]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained SB3 PPO model and visualize trajectory on a map.",
    )
    p.add_argument("--model", type=str, required=True, help="SB3 model .zip path or extracted model directory.")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--map-source", type=str, default="file", choices=["file", "random", "custom"])
    p.add_argument("--map-file", type=str, default="map/indoor_seed101.txt")
    p.add_argument("--map-size", type=int, default=64)
    p.add_argument("--map-stage", type=int, default=3, choices=[1, 2, 3, 4])
    p.add_argument("--num-obstacles", type=int, default=None)
    p.add_argument("--seed", type=int, default=101)

    p.add_argument("--sensor-range", type=int, default=2)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--include-dtm", action="store_true")
    mask_group = p.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--action-mask",
        dest="action_mask",
        action="store_true",
        help=(
            "Enable known-map action masking in eval. Uses MaskablePPO when available; "
            "otherwise applies env-side safety masking."
        ),
    )
    mask_group.add_argument(
        "--no-action-mask",
        dest="action_mask",
        action="store_false",
        help="Disable action masking in eval.",
    )
    det_group = p.add_mutually_exclusive_group()
    det_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        help="Use deterministic policy (argmax action).",
    )
    det_group.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Sample actions from policy distribution.",
    )
    p.set_defaults(action_mask=True)
    p.set_defaults(deterministic=True)

    p.add_argument("--save-path-json", type=str, default="")
    p.add_argument("--save-plot", type=str, default="")
    p.add_argument("--show-plot", action="store_true")
    p.add_argument("--title", type=str, default="PPO Coverage Trajectory")
    return p.parse_args()


def _select_device(device_arg: str) -> str:
    import torch

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


def _zip_if_directory(model_path: Path) -> Tuple[Path, Optional[tempfile.TemporaryDirectory]]:
    if model_path.is_file():
        return model_path, None
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    required = [
        "data",
        "policy.pth",
        "policy.optimizer.pth",
        "pytorch_variables.pth",
        "_stable_baselines3_version",
    ]
    for name in required:
        if not (model_path / name).exists():
            raise FileNotFoundError(
                f"Missing required SB3 file in model directory: {model_path / name}"
            )

    tmpdir = tempfile.TemporaryDirectory(prefix="sb3_model_")
    zip_path = Path(tmpdir.name) / "model.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for child in model_path.iterdir():
            if child.is_file():
                zf.write(child, arcname=child.name)
    return zip_path, tmpdir


def _plot_trajectory(
    grid: np.ndarray,
    path: np.ndarray,
    title: str,
    save_plot: str,
    show_plot: bool,
):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid == 1, cmap="gray_r", origin="upper")
    if len(path) > 1:
        ax.plot(path[:, 1], path[:, 0], color="tab:red", linewidth=1.8, label="Trajectory")
    ax.scatter(path[0, 1], path[0, 0], color="tab:blue", s=35, label="Start")
    ax.scatter(path[-1, 1], path[-1, 0], color="tab:green", s=35, label="End")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)
    ax.grid(False)
    ax.legend(loc="best")
    fig.tight_layout()

    if save_plot:
        out = Path(save_plot)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        print(f"[DONE] saved trajectory plot: {out}")

    if show_plot:
        plt.show()
    plt.close(fig)


def main():
    args = _parse_args()
    device = _select_device(args.device)

    try:
        from stable_baselines3 import PPO as SB3PPO
    except Exception as exc:
        raise RuntimeError("stable_baselines3 is required to run model evaluation.") from exc

    AlgoLoader = SB3PPO
    algo_name = "PPO"
    use_maskable_predict = False
    if args.action_mask:
        try:
            from sb3_contrib import MaskablePPO  # type: ignore
        except Exception:
            print(
                "[WARN] sb3_contrib not found. Falling back to PPO with env-side action masking.",
                flush=True,
            )
        else:
            AlgoLoader = MaskablePPO
            algo_name = "MaskablePPO"
            use_maskable_predict = True

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
    env = CPPDiscreteGymEnv(grid_map=grid, start_pos=start, config=env_cfg)

    model_input = Path(args.model)
    model_zip, temp_holder = _zip_if_directory(model_input)
    try:
        try:
            model = AlgoLoader.load(str(model_zip), device=device)
        except Exception as load_err:
            if algo_name == "MaskablePPO":
                print(
                    f"[WARN] MaskablePPO load failed ({load_err}). Retrying with PPO loader.",
                    flush=True,
                )
                model = SB3PPO.load(str(model_zip), device=device)
                algo_name = "PPO"
                use_maskable_predict = False
            else:
                raise
        obs, _ = env.reset()

        path: list[GridPos] = [start]
        total_reward = 0.0
        collision_count = 0
        action_overridden_count = 0
        final_info: Dict = {}

        for _ in range(int(args.max_episode_steps)):
            if use_maskable_predict:
                action, _ = model.predict(
                    obs,
                    action_masks=env.action_masks(),
                    deterministic=bool(args.deterministic),
                )
            else:
                action, _ = model.predict(obs, deterministic=bool(args.deterministic))
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            if bool(info.get("collision", False)):
                collision_count += 1
            if bool(info.get("action_overridden", False)):
                action_overridden_count += 1
            pos = info.get("position", path[-1])
            path.append((int(pos[0]), int(pos[1])))
            final_info = info
            if terminated or truncated:
                break

        path_arr = np.asarray(path, dtype=np.int32)
        coverage_ratio = float(final_info.get("coverage_ratio", np.nan))
        collision_last = float(final_info.get("collision", np.nan))
        steps = int(final_info.get("steps", len(path) - 1))
        done_reason = str(final_info.get("done_reason", ""))
        collision_rate = float(collision_count) / float(max(1, steps))
        action_override_rate = float(action_overridden_count) / float(max(1, steps))
        unique_positions = int(len({(int(r), int(c)) for r, c in path}))

        print("[EVAL RESULT]")
        print(f"  model: {model_input}")
        print(f"  algo: {algo_name}")
        print(f"  map shape: {grid.shape}, start: {start}")
        print(f"  steps: {steps}")
        print(f"  coverage_ratio: {coverage_ratio:.6f}")
        print(f"  collision_rate: {collision_rate:.6f} ({collision_count}/{steps})")
        print(f"  action_override_rate: {action_override_rate:.6f} ({action_overridden_count}/{steps})")
        print(f"  collision(last): {collision_last:.6f}")
        print(f"  unique_positions: {unique_positions}")
        print(f"  total_reward(sum): {total_reward:.6f}")
        print(f"  done_reason: {done_reason}")

        if args.save_path_json:
            out = Path(args.save_path_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "model": str(model_input),
                "map_source": args.map_source,
                "map_file": args.map_file if args.map_source == "file" else "",
                "map_shape": [int(grid.shape[0]), int(grid.shape[1])],
                "start": [int(start[0]), int(start[1])],
                "steps": steps,
                "coverage_ratio": coverage_ratio,
                "collision_last": collision_last,
                "collision_count": int(collision_count),
                "collision_rate": collision_rate,
                "action_overridden_count": int(action_overridden_count),
                "action_override_rate": action_override_rate,
                "unique_positions": unique_positions,
                "total_reward_sum": total_reward,
                "done_reason": done_reason,
                "path": [[int(r), int(c)] for r, c in path],
            }
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[DONE] saved path json: {out}")

        if args.save_plot or args.show_plot:
            _plot_trajectory(
                grid=grid,
                path=path_arr,
                title=args.title,
                save_plot=args.save_plot,
                show_plot=bool(args.show_plot),
            )
    finally:
        env.close()
        if temp_holder is not None:
            temp_holder.cleanup()


if __name__ == "__main__":
    main()

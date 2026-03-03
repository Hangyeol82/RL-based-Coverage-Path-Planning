import argparse
import csv
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Keep runtime stable in constrained environments.
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
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

from learning.observation import MultiScaleCPPObservationConfig
from learning.reinforcement.cpp_env import CPPDiscreteEnvConfig
from learning.reinforcement.reward import CPPRewardConfig
from learning.reinforcement.sb3_env import CPPDiscreteGymEnv
from map_generators.shape_grid import build_shape_grid_map


GridPos = Tuple[int, int]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare two PPO models on shape-grid level-4 maps and save "
            "trajectory plots + coverage/overlap metrics."
        )
    )
    p.add_argument("--baseline-model", type=str, required=True, help="Baseline model .zip or extracted SB3 dir")
    p.add_argument("--dtm-model", type=str, required=True, help="DTM model .zip or extracted SB3 dir")
    p.add_argument("--seeds", type=str, default="101,102,103,104,105,106,107,108,109,110")
    p.add_argument("--map-size", type=int, default=32)
    p.add_argument("--sensor-range", type=int, default=2)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    boundary_group = p.add_mutually_exclusive_group()
    boundary_group.add_argument("--boundary-exit-features", dest="boundary_exit_features", action="store_true")
    boundary_group.add_argument("--no-boundary-exit-features", dest="boundary_exit_features", action="store_false")
    p.add_argument("--boundary-exit-threshold", type=float, default=0.0)
    p.add_argument(
        "--dtm-output-mode",
        type=str,
        default="six",
        choices=["six", "extent6", "axis2", "axis2km", "four", "port12"],
        help="DTM output channel mode for env observation.",
    )
    p.add_argument(
        "--dtm-coarse-mode",
        type=str,
        default="bfs",
        choices=["bfs", "aggregate", "aggregate_transfer"],
        help="DTM coarse-level calculation mode for env observation.",
    )
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", type=str, default="log_analysis/logs/shapegrid_l4_compare")
    p.add_argument(
        "--shape-types",
        type=str,
        default="rect,circle",
        help="Comma list among rect,triangle,circle",
    )
    p.add_argument("--grid-step", type=int, default=8)
    p.add_argument("--spawn-prob", type=float, default=0.75)
    p.add_argument("--min-half-extent", type=int, default=1)
    p.add_argument("--max-half-extent", type=int, default=3)
    p.add_argument("--jitter", type=int, default=1)
    p.add_argument("--clearance", type=int, default=0)
    mask_group = p.add_mutually_exclusive_group()
    mask_group.add_argument("--action-mask", dest="action_mask", action="store_true")
    mask_group.add_argument("--no-action-mask", dest="action_mask", action="store_false")
    p.set_defaults(action_mask=True)
    det_group = p.add_mutually_exclusive_group()
    det_group.add_argument("--deterministic", dest="deterministic", action="store_true")
    det_group.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.set_defaults(deterministic=True, boundary_exit_features=False)
    return p.parse_args()


def _parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for tok in raw.split(","):
        s = tok.strip()
        if s:
            out.append(int(s))
    if not out:
        raise ValueError("No seeds provided")
    return out


def _parse_shape_types(raw: str) -> Tuple[str, ...]:
    allowed = {"rect", "triangle", "circle"}
    out: List[str] = []
    for tok in raw.split(","):
        t = tok.strip().lower()
        if not t:
            continue
        if t not in allowed:
            raise ValueError(f"Unsupported shape type: {t}")
        out.append(t)
    if not out:
        raise ValueError("shape-types is empty")
    return tuple(out)


def _select_device(device_arg: str) -> str:
    import torch

    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _pick_start(grid: np.ndarray) -> GridPos:
    free = np.argwhere(grid == 0)
    if free.size == 0:
        raise RuntimeError("No free cell in map")
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
            raise FileNotFoundError(f"Missing required SB3 file: {model_path / name}")

    tmpdir = tempfile.TemporaryDirectory(prefix="sb3_model_")
    zip_path = Path(tmpdir.name) / "model.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for child in model_path.iterdir():
            if child.is_file():
                zf.write(child, arcname=child.name)
    return zip_path, tmpdir


def _load_model(model_path: Path, device: str, action_mask: bool):
    try:
        from stable_baselines3 import PPO as SB3PPO
    except Exception as exc:
        raise RuntimeError("stable_baselines3 is required for evaluation.") from exc

    AlgoLoader = SB3PPO
    algo_name = "PPO"
    use_maskable_predict = False
    if action_mask:
        try:
            from sb3_contrib import MaskablePPO  # type: ignore
        except Exception:
            pass
        else:
            AlgoLoader = MaskablePPO
            algo_name = "MaskablePPO"
            use_maskable_predict = True

    model_zip, temp_holder = _zip_if_directory(model_path)
    try:
        try:
            model = AlgoLoader.load(str(model_zip), device=device)
        except Exception as load_err:
            if algo_name == "MaskablePPO":
                model = SB3PPO.load(str(model_zip), device=device)
                algo_name = "PPO"
                use_maskable_predict = False
            else:
                raise load_err
    finally:
        if temp_holder is not None:
            temp_holder.cleanup()
    return model, algo_name, use_maskable_predict


def _plot_trajectory_rainbow(grid: np.ndarray, path: np.ndarray, title: str, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(grid == 1, cmap="gray_r", origin="upper")

    if len(path) > 1:
        pts = np.stack([path[:, 1], path[:, 0]], axis=1)
        segments = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(
            segments,
            cmap="turbo",
            norm=Normalize(vmin=0, vmax=max(1, len(segments) - 1)),
            linewidths=1.8,
            alpha=0.95,
        )
        lc.set_array(np.arange(len(segments), dtype=np.float32))
        ax.add_collection(lc)

    ax.scatter(path[0, 1], path[0, 0], color="tab:blue", s=35, label="Start", zorder=3)
    ax.scatter(path[-1, 1], path[-1, 0], color="tab:green", s=35, label="End", zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _evaluate_one_map(
    *,
    model,
    use_maskable_predict: bool,
    deterministic: bool,
    grid: np.ndarray,
    start: GridPos,
    sensor_range: int,
    max_episode_steps: int,
    include_dtm: bool,
    boundary_exit_features: bool,
    boundary_exit_threshold: float,
    action_mask: bool,
    dtm_output_mode: str,
    dtm_coarse_mode: str,
) -> Dict[str, object]:
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
        sensor_range=int(sensor_range),
        max_steps=int(max_episode_steps),
        collision_ends_episode=False,
        stop_on_full_coverage=True,
        include_dtm=bool(include_dtm),
        use_boundary_exit_features=bool(boundary_exit_features),
        boundary_exit_threshold=float(boundary_exit_threshold),
        observation=MultiScaleCPPObservationConfig(
            dtm_output_mode=str(dtm_output_mode),
            dtm_coarse_mode=str(dtm_coarse_mode),
        ),
        use_action_mask=bool(action_mask),
        reward=reward_cfg,
    )
    env = CPPDiscreteGymEnv(grid_map=grid, start_pos=start, config=env_cfg)
    try:
        obs, _ = env.reset()

        path: List[GridPos] = [start]
        total_reward = 0.0
        collision_count = 0
        action_overridden_count = 0
        revisited_count = 0
        final_info: Dict[str, object] = {}

        for _ in range(int(max_episode_steps)):
            if use_maskable_predict:
                action, _ = model.predict(
                    obs,
                    action_masks=env.action_masks(),
                    deterministic=bool(deterministic),
                )
            else:
                action, _ = model.predict(obs, deterministic=bool(deterministic))
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            if bool(info.get("collision", False)):
                collision_count += 1
            if bool(info.get("action_overridden", False)):
                action_overridden_count += 1
            if bool(info.get("revisited_cell", False)):
                revisited_count += 1
            pos = info.get("position", path[-1])
            path.append((int(pos[0]), int(pos[1])))
            final_info = info
            if terminated or truncated:
                break

        steps = int(final_info.get("steps", len(path) - 1))
        unique_positions = len({(int(r), int(c)) for r, c in path})
        # Track overlap directly from env-level revisits for consistency.
        overlap_rate = float(revisited_count) / float(max(1, steps))

        return {
            "steps": int(steps),
            "coverage_ratio": float(final_info.get("coverage_ratio", np.nan)),
            "coverage_cells": int(final_info.get("coverage_cells", -1)),
            "free_cells": int(final_info.get("free_cells", -1)),
            "collision_count": int(collision_count),
            "collision_rate": float(collision_count) / float(max(1, steps)),
            "action_overridden_count": int(action_overridden_count),
            "action_override_rate": float(action_overridden_count) / float(max(1, steps)),
            "revisited_count": int(revisited_count),
            "overlap_rate": float(overlap_rate),
            "unique_positions": int(unique_positions),
            "path_length": int(len(path)),
            "path_overlap_count": int(max(0, len(path) - unique_positions)),
            "path_overlap_rate": float(max(0, len(path) - unique_positions)) / float(max(1, len(path) - 1)),
            "total_reward_sum": float(total_reward),
            "done_reason": str(final_info.get("done_reason", "")),
            "path": [[int(r), int(c)] for r, c in path],
        }
    finally:
        env.close()


def _write_rows_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _safe_delta(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b):
        return float("nan")
    return float(b - a)


def main() -> None:
    args = _parse_args()
    device = _select_device(args.device)
    seeds = _parse_int_list(args.seeds)
    shape_types = _parse_shape_types(args.shape_types)

    out_dir = Path(args.out_dir)
    map_dir = out_dir / "maps"
    traj_dir = out_dir / "trajectories"
    path_dir = out_dir / "paths"
    map_dir.mkdir(parents=True, exist_ok=True)
    traj_dir.mkdir(parents=True, exist_ok=True)
    path_dir.mkdir(parents=True, exist_ok=True)
    (traj_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (traj_dir / "dtm").mkdir(parents=True, exist_ok=True)
    (path_dir / "baseline").mkdir(parents=True, exist_ok=True)
    (path_dir / "dtm").mkdir(parents=True, exist_ok=True)

    baseline_model_path = Path(args.baseline_model).expanduser().resolve()
    dtm_model_path = Path(args.dtm_model).expanduser().resolve()
    if not baseline_model_path.exists():
        raise FileNotFoundError(f"Baseline model not found: {baseline_model_path}")
    if not dtm_model_path.exists():
        raise FileNotFoundError(f"DTM model not found: {dtm_model_path}")

    baseline_model, baseline_algo, baseline_maskable = _load_model(
        baseline_model_path,
        device=device,
        action_mask=bool(args.action_mask),
    )
    dtm_model, dtm_algo, dtm_maskable = _load_model(
        dtm_model_path,
        device=device,
        action_mask=bool(args.action_mask),
    )

    print(f"[INFO] device={device}")
    print(f"[INFO] baseline model={baseline_model_path} algo={baseline_algo}")
    print(f"[INFO] dtm model={dtm_model_path} algo={dtm_algo}")
    print(f"[INFO] seeds={seeds}")

    long_rows: List[Dict[str, object]] = []
    per_map_compare_rows: List[Dict[str, object]] = []
    baseline_by_seed: Dict[int, Dict[str, object]] = {}
    dtm_by_seed: Dict[int, Dict[str, object]] = {}

    for seed in seeds:
        grid = build_shape_grid_map(
            size=int(args.map_size),
            seed=int(seed),
            grid_step=int(args.grid_step),
            spawn_prob=float(args.spawn_prob),
            min_half_extent=int(args.min_half_extent),
            max_half_extent=int(args.max_half_extent),
            jitter=int(args.jitter),
            clearance=int(args.clearance),
            shape_types=shape_types,
            ensure_start_clear=True,
        )
        start = _pick_start(grid)
        map_txt = map_dir / f"shapegrid_l4_seed{int(seed)}.txt"
        np.savetxt(map_txt, np.asarray(grid, dtype=np.int32), fmt="%d")

        for mode, model, maskable, include_dtm in (
            ("baseline", baseline_model, baseline_maskable, False),
            ("dtm", dtm_model, dtm_maskable, True),
        ):
            result = _evaluate_one_map(
                model=model,
                use_maskable_predict=bool(maskable),
                deterministic=bool(args.deterministic),
                grid=grid,
                start=start,
                sensor_range=int(args.sensor_range),
                max_episode_steps=int(args.max_episode_steps),
                include_dtm=bool(include_dtm),
                boundary_exit_features=bool(args.boundary_exit_features),
                boundary_exit_threshold=float(args.boundary_exit_threshold),
                action_mask=bool(args.action_mask),
                dtm_output_mode=str(args.dtm_output_mode),
                dtm_coarse_mode=str(args.dtm_coarse_mode),
            )
            row = {
                "mode": mode,
                "seed": int(seed),
                "map_file": str(map_txt),
                "model": str(baseline_model_path if mode == "baseline" else dtm_model_path),
                "include_dtm": bool(include_dtm),
                "steps": int(result["steps"]),
                "coverage_ratio": float(result["coverage_ratio"]),
                "coverage_cells": int(result["coverage_cells"]),
                "free_cells": int(result["free_cells"]),
                "collision_rate": float(result["collision_rate"]),
                "collision_count": int(result["collision_count"]),
                "overlap_rate": float(result["overlap_rate"]),
                "revisited_count": int(result["revisited_count"]),
                "path_overlap_rate": float(result["path_overlap_rate"]),
                "path_overlap_count": int(result["path_overlap_count"]),
                "action_override_rate": float(result["action_override_rate"]),
                "action_overridden_count": int(result["action_overridden_count"]),
                "unique_positions": int(result["unique_positions"]),
                "path_length": int(result["path_length"]),
                "total_reward_sum": float(result["total_reward_sum"]),
                "done_reason": str(result["done_reason"]),
            }
            long_rows.append(row)

            mode_path_dir = path_dir / mode
            mode_traj_dir = traj_dir / mode
            path_json = mode_path_dir / f"seed{int(seed)}.json"
            plot_png = mode_traj_dir / f"seed{int(seed)}.png"
            payload = dict(row)
            payload["path"] = result["path"]
            path_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            _plot_trajectory_rainbow(
                grid=grid,
                path=np.asarray(result["path"], dtype=np.int32),
                title=f"{mode} | shapegrid lv4 | seed={int(seed)}",
                out_png=plot_png,
            )

            if mode == "baseline":
                baseline_by_seed[int(seed)] = row
            else:
                dtm_by_seed[int(seed)] = row

            print(
                f"[DONE] seed={seed} mode={mode} "
                f"cov={row['coverage_ratio']:.4f} overlap={row['overlap_rate']:.4f} "
                f"coll={row['collision_rate']:.4f} steps={row['steps']}"
            )

        b = baseline_by_seed[int(seed)]
        d = dtm_by_seed[int(seed)]
        cmp_row = {
            "seed": int(seed),
            "baseline_coverage_ratio": float(b["coverage_ratio"]),
            "dtm_coverage_ratio": float(d["coverage_ratio"]),
            "delta_coverage_ratio": _safe_delta(float(b["coverage_ratio"]), float(d["coverage_ratio"])),
            "baseline_overlap_rate": float(b["overlap_rate"]),
            "dtm_overlap_rate": float(d["overlap_rate"]),
            "delta_overlap_rate": _safe_delta(float(b["overlap_rate"]), float(d["overlap_rate"])),
            "baseline_collision_rate": float(b["collision_rate"]),
            "dtm_collision_rate": float(d["collision_rate"]),
            "delta_collision_rate": _safe_delta(float(b["collision_rate"]), float(d["collision_rate"])),
            "baseline_reward_total": float(b["total_reward_sum"]),
            "dtm_reward_total": float(d["total_reward_sum"]),
            "delta_reward_total": _safe_delta(float(b["total_reward_sum"]), float(d["total_reward_sum"])),
            "baseline_steps": int(b["steps"]),
            "dtm_steps": int(d["steps"]),
            "delta_steps": int(d["steps"]) - int(b["steps"]),
        }
        per_map_compare_rows.append(cmp_row)

    long_fieldnames = [
        "mode",
        "seed",
        "map_file",
        "model",
        "include_dtm",
        "steps",
        "coverage_ratio",
        "coverage_cells",
        "free_cells",
        "overlap_rate",
        "revisited_count",
        "path_overlap_rate",
        "path_overlap_count",
        "collision_rate",
        "collision_count",
        "action_override_rate",
        "action_overridden_count",
        "unique_positions",
        "path_length",
        "total_reward_sum",
        "done_reason",
    ]
    _write_rows_csv(out_dir / "metrics_long.csv", long_rows, long_fieldnames)

    cmp_fieldnames = [
        "seed",
        "baseline_coverage_ratio",
        "dtm_coverage_ratio",
        "delta_coverage_ratio",
        "baseline_overlap_rate",
        "dtm_overlap_rate",
        "delta_overlap_rate",
        "baseline_collision_rate",
        "dtm_collision_rate",
        "delta_collision_rate",
        "baseline_reward_total",
        "dtm_reward_total",
        "delta_reward_total",
        "baseline_steps",
        "dtm_steps",
        "delta_steps",
    ]
    _write_rows_csv(out_dir / "comparison_per_map.csv", per_map_compare_rows, cmp_fieldnames)

    def _mean(rows: List[Dict[str, object]], key: str) -> float:
        vals = [float(r[key]) for r in rows]
        return float(np.mean(np.asarray(vals, dtype=np.float64)))

    baseline_rows = [r for r in long_rows if r["mode"] == "baseline"]
    dtm_rows = [r for r in long_rows if r["mode"] == "dtm"]
    aggregate = {
        "seeds": seeds,
        "baseline_model": str(baseline_model_path),
        "dtm_model": str(dtm_model_path),
        "mean_baseline_coverage_ratio": _mean(baseline_rows, "coverage_ratio"),
        "mean_dtm_coverage_ratio": _mean(dtm_rows, "coverage_ratio"),
        "delta_coverage_ratio": _mean(dtm_rows, "coverage_ratio") - _mean(baseline_rows, "coverage_ratio"),
        "mean_baseline_overlap_rate": _mean(baseline_rows, "overlap_rate"),
        "mean_dtm_overlap_rate": _mean(dtm_rows, "overlap_rate"),
        "delta_overlap_rate": _mean(dtm_rows, "overlap_rate") - _mean(baseline_rows, "overlap_rate"),
        "mean_baseline_collision_rate": _mean(baseline_rows, "collision_rate"),
        "mean_dtm_collision_rate": _mean(dtm_rows, "collision_rate"),
        "delta_collision_rate": _mean(dtm_rows, "collision_rate") - _mean(baseline_rows, "collision_rate"),
        "mean_baseline_reward_total": _mean(baseline_rows, "total_reward_sum"),
        "mean_dtm_reward_total": _mean(dtm_rows, "total_reward_sum"),
        "delta_reward_total": _mean(dtm_rows, "total_reward_sum") - _mean(baseline_rows, "total_reward_sum"),
        "mean_baseline_steps": _mean(baseline_rows, "steps"),
        "mean_dtm_steps": _mean(dtm_rows, "steps"),
        "delta_steps": _mean(dtm_rows, "steps") - _mean(baseline_rows, "steps"),
    }
    (out_dir / "comparison_aggregate.json").write_text(
        json.dumps(aggregate, indent=2),
        encoding="utf-8",
    )

    print("\n[SUMMARY]")
    print(f"  out_dir: {out_dir}")
    print(f"  mean coverage: baseline={aggregate['mean_baseline_coverage_ratio']:.4f}, dtm={aggregate['mean_dtm_coverage_ratio']:.4f}, delta={aggregate['delta_coverage_ratio']:+.4f}")
    print(f"  mean overlap : baseline={aggregate['mean_baseline_overlap_rate']:.4f}, dtm={aggregate['mean_dtm_overlap_rate']:.4f}, delta={aggregate['delta_overlap_rate']:+.4f}")
    print(f"  mean collision: baseline={aggregate['mean_baseline_collision_rate']:.4f}, dtm={aggregate['mean_dtm_collision_rate']:.4f}, delta={aggregate['delta_collision_rate']:+.4f}")
    print(f"  mean reward: baseline={aggregate['mean_baseline_reward_total']:.4f}, dtm={aggregate['mean_dtm_reward_total']:.4f}, delta={aggregate['delta_reward_total']:+.4f}")


if __name__ == "__main__":
    main()

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from map_generators.shape_grid import build_shape_grid_map


@dataclass(frozen=True)
class ShapeGridPreset:
    level: int
    name: str
    grid_step: int
    spawn_prob: float
    min_half_extent: int
    max_half_extent: int
    jitter: int
    clearance: int
    shape_types: tuple


PRESETS: Dict[int, ShapeGridPreset] = {
    1: ShapeGridPreset(
        level=1,
        name="easy",
        grid_step=14,
        spawn_prob=0.55,
        min_half_extent=1,
        max_half_extent=2,
        jitter=1,
        clearance=1,
        shape_types=("rect", "circle"),
    ),
    2: ShapeGridPreset(
        level=2,
        name="easy_plus",
        grid_step=12,
        spawn_prob=0.60,
        min_half_extent=1,
        max_half_extent=2,
        jitter=1,
        clearance=1,
        shape_types=("rect", "circle"),
    ),
    3: ShapeGridPreset(
        level=3,
        name="medium",
        grid_step=10,
        spawn_prob=0.65,
        min_half_extent=1,
        max_half_extent=3,
        jitter=1,
        clearance=1,
        shape_types=("rect", "circle"),
    ),
    4: ShapeGridPreset(
        level=4,
        name="hard",
        grid_step=8,
        spawn_prob=0.75,
        min_half_extent=1,
        max_half_extent=3,
        jitter=1,
        clearance=0,
        shape_types=("rect", "circle"),
    ),
}


def _parse_int_list(raw: str, *, name: str) -> List[int]:
    vals: List[int] = []
    for tok in raw.split(","):
        s = tok.strip()
        if not s:
            continue
        vals.append(int(s))
    if not vals:
        raise ValueError(f"{name} must not be empty")
    return vals


def _parse_float_list(raw: str, *, name: str) -> List[float]:
    vals: List[float] = []
    for tok in raw.split(","):
        s = tok.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError(f"{name} must not be empty")
    return vals


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Shape-grid curriculum PPO runner. "
            "Runs 100k chunks, prints cumulative logs, and steps difficulty every stage interval."
        )
    )
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--chunk-timesteps", type=int, default=100_000, help="Log/report interval")
    p.add_argument("--stage-timesteps", type=int, default=200_000, help="Difficulty switch interval")
    p.add_argument(
        "--phase-levels",
        type=str,
        default="2,3,4,4",
        help="Comma list of difficulty levels per stage block.",
    )
    p.add_argument(
        "--curriculum-mode",
        type=str,
        default="fixed",
        choices=["fixed", "adaptive"],
        help="fixed: switch by stage-timesteps, adaptive: promote by chunk metrics.",
    )
    p.add_argument(
        "--adaptive-min-chunks",
        type=int,
        default=3,
        help="Minimum chunks to stay before adaptive promotion checks are active.",
    )
    p.add_argument(
        "--adaptive-window",
        type=int,
        default=3,
        help="Recent chunk window used for adaptive promotion checks.",
    )
    p.add_argument(
        "--adaptive-max-chunks-per-level",
        type=int,
        default=0,
        help="Force promotion after this many chunks at one level (0 disables).",
    )
    p.add_argument(
        "--adaptive-cov-thresholds",
        type=str,
        default="0.58,0.54,0.50,0.48",
        help="Per-level mean coverage thresholds for promotion.",
    )
    p.add_argument(
        "--adaptive-done-cov-thresholds",
        type=str,
        default="",
        help="Optional per-level done-episode final coverage thresholds. Empty disables this gate.",
    )
    p.add_argument(
        "--adaptive-cov-slope-min",
        type=float,
        default=0.0005,
        help="Minimum slope of recent chunk mean coverage required for promotion.",
    )
    p.add_argument(
        "--adaptive-collision-max",
        type=float,
        default=0.005,
        help="Maximum recent chunk mean collision allowed for promotion.",
    )
    p.add_argument("--seed", type=int, default=101)
    p.add_argument(
        "--map-seed-mode",
        type=str,
        default="chunk",
        choices=["fixed", "chunk"],
        help="fixed: same map seed always, chunk: seed+chunk_idx",
    )
    p.add_argument("--map-size", type=int, default=32)
    p.add_argument("--sensor-range", type=int, default=2)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    boundary_group = p.add_mutually_exclusive_group()
    boundary_group.add_argument("--boundary-exit-features", dest="boundary_exit_features", action="store_true")
    boundary_group.add_argument("--no-boundary-exit-features", dest="boundary_exit_features", action="store_false")
    p.add_argument("--boundary-exit-threshold", type=float, default=0.0)
    milestone_group = p.add_mutually_exclusive_group()
    milestone_group.add_argument("--milestone-reward", dest="milestone_reward", action="store_true")
    milestone_group.add_argument("--no-milestone-reward", dest="milestone_reward", action="store_false")
    p.add_argument("--milestone-threshold-90", type=float, default=0.90)
    p.add_argument("--milestone-threshold-99", type=float, default=0.99)
    p.add_argument("--milestone-lambda-90", type=float, default=0.2)
    p.add_argument("--milestone-lambda-99", type=float, default=4.0)
    p.add_argument("--maps-encoder-mode", type=str, default="sgcnn", choices=["sgcnn", "independent"])
    p.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "large", "xlarge"],
        help="Forwarded to run_ppo_sb3.py encoder size preset.",
    )
    p.add_argument(
        "--dtm-coarse-mode",
        type=str,
        default="bfs",
        choices=["bfs", "aggregate", "aggregate_transfer"],
        help="DTM multi-scale mode forwarded to run_ppo_sb3.py.",
    )
    p.add_argument(
        "--dtm-output-mode",
        type=str,
        default="six",
        choices=["six", "extent6", "axis2", "axis2km", "four", "port12"],
        help="DTM output channels forwarded to run_ppo_sb3.py.",
    )
    p.add_argument(
        "--obs-unknown-policy",
        type=str,
        default="keep",
        choices=["keep", "as_free", "as_obstacle"],
        help="Unknown-cell handling forwarded to run_ppo_sb3.py map-observation builder.",
    )
    p.add_argument("--include-dtm", action="store_true")

    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-envs", type=int, default=12)
    p.add_argument("--vec-env", type=str, default="subproc", choices=["auto", "dummy", "subproc"])
    p.add_argument(
        "--subproc-start-method",
        type=str,
        default="forkserver",
        choices=["auto", "spawn", "fork", "forkserver"],
    )
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)

    mask_group = p.add_mutually_exclusive_group()
    mask_group.add_argument("--action-mask", dest="action_mask", action="store_true")
    mask_group.add_argument("--no-action-mask", dest="action_mask", action="store_false")
    p.set_defaults(action_mask=True, milestone_reward=False, boundary_exit_features=False)

    p.add_argument(
        "--init-from-bc",
        type=str,
        default="learning/checkpoints/bc/bc_zigzag_empty_latest.pt",
        help="BC checkpoint for first chunk only. Empty string disables BC init.",
    )
    p.add_argument("--init-from-bc-strict", action="store_true")

    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def _resolve_out_dir(args: argparse.Namespace, repo_root: Path) -> Path:
    if args.out_dir:
        p = Path(args.out_dir)
        return p if p.is_absolute() else (repo_root / p)
    tag = args.run_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    return repo_root / "learning" / "checkpoints" / "rl" / f"shapegrid_curriculum_{tag}"


def _load_rollouts(json_path: Path) -> List[Dict]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rollouts = payload.get("rollouts", [])
    if not isinstance(rollouts, list):
        raise ValueError(f"Invalid rollout json: {json_path}")
    return rollouts


def _safe_mean(rows: List[Dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _safe_mean_vals(vals: List[float]) -> float:
    clean = [float(v) for v in vals if not np.isnan(float(v))]
    if len(clean) == 0:
        return float("nan")
    return float(np.mean(np.asarray(clean, dtype=np.float64)))


def _trend_slope(vals: List[float]) -> float:
    clean = [float(v) for v in vals if not np.isnan(float(v))]
    n = len(clean)
    if n < 2:
        return 0.0
    xs = np.arange(1, n + 1, dtype=np.float64)
    ys = np.asarray(clean, dtype=np.float64)
    mx = float(np.mean(xs))
    my = float(np.mean(ys))
    num = float(np.sum((xs - mx) * (ys - my)))
    den = float(np.sum((xs - mx) ** 2))
    if den <= 1e-12:
        return 0.0
    return num / den


def _preset_for_chunk(
    phase_levels: List[int],
    stage_timesteps: int,
    chunk_start_t: int,
) -> ShapeGridPreset:
    stage_idx = int(chunk_start_t // stage_timesteps)
    level = int(phase_levels[min(stage_idx, len(phase_levels) - 1)])
    if level not in PRESETS:
        raise ValueError(f"Unsupported shape-grid level in phase-levels: {level}")
    return PRESETS[level]


def _write_map_txt(path: Path, grid: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(grid, dtype=np.int32), fmt="%d")


def _build_run_cmd(
    args: argparse.Namespace,
    *,
    runner: Path,
    chunk_timesteps: int,
    run_seed: int,
    map_file: Path,
    save_model: Path,
    save_json: Path,
    save_csv: Path,
    init_from_bc: str,
    load_model: str,
) -> List[str]:
    cmd = [
        args.python_bin,
        str(runner),
        "--total-timesteps",
        str(int(chunk_timesteps)),
        "--seed",
        str(int(run_seed)),
        "--device",
        args.device,
        "--num-envs",
        str(int(args.num_envs)),
        "--vec-env",
        args.vec_env,
        "--subproc-start-method",
        args.subproc_start_method,
        "--n-steps",
        str(int(args.n_steps)),
        "--batch-size",
        str(int(args.batch_size)),
        "--n-epochs",
        str(int(args.n_epochs)),
        "--verbose",
        "0",
        "--sensor-range",
        str(int(args.sensor_range)),
        "--max-episode-steps",
        str(int(args.max_episode_steps)),
        "--boundary-exit-threshold",
        str(float(args.boundary_exit_threshold)),
        "--milestone-threshold-90",
        str(float(args.milestone_threshold_90)),
        "--milestone-threshold-99",
        str(float(args.milestone_threshold_99)),
        "--milestone-lambda-90",
        str(float(args.milestone_lambda_90)),
        "--milestone-lambda-99",
        str(float(args.milestone_lambda_99)),
        "--maps-encoder-mode",
        args.maps_encoder_mode,
        "--model-size",
        args.model_size,
        "--dtm-coarse-mode",
        str(args.dtm_coarse_mode),
        "--dtm-output-mode",
        str(args.dtm_output_mode),
        "--obs-unknown-policy",
        str(args.obs_unknown_policy),
        "--map-source",
        "file",
        "--map-file",
        str(map_file),
        "--map-size",
        str(int(args.map_size)),
        "--save-model",
        str(save_model),
        "--save-breakdown-json",
        str(save_json),
        "--save-breakdown-csv",
        str(save_csv),
    ]
    if args.include_dtm:
        cmd.append("--include-dtm")
    if args.action_mask:
        cmd.append("--action-mask")
    else:
        cmd.append("--no-action-mask")
    if args.boundary_exit_features:
        cmd.append("--boundary-exit-features")
    else:
        cmd.append("--no-boundary-exit-features")
    if args.milestone_reward:
        cmd.append("--milestone-reward")
    else:
        cmd.append("--no-milestone-reward")
    if init_from_bc:
        cmd += ["--init-from-bc", init_from_bc]
        if args.init_from_bc_strict:
            cmd.append("--init-from-bc-strict")
    if load_model:
        cmd += ["--load-model", load_model]
    return cmd


def main():
    args = _parse_args()
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.chunk_timesteps <= 0:
        raise ValueError("--chunk-timesteps must be positive")
    if args.curriculum_mode == "fixed" and args.stage_timesteps <= 0:
        raise ValueError("--stage-timesteps must be positive")
    if args.adaptive_min_chunks <= 0:
        raise ValueError("--adaptive-min-chunks must be positive")
    if args.adaptive_window <= 0:
        raise ValueError("--adaptive-window must be positive")
    if args.adaptive_max_chunks_per_level < 0:
        raise ValueError("--adaptive-max-chunks-per-level must be >= 0")
    if not (0.0 <= float(args.adaptive_collision_max) <= 1.0):
        raise ValueError("--adaptive-collision-max must be in [0, 1]")
    if args.map_size <= 1:
        raise ValueError("--map-size must be >= 2")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")

    phase_levels = _parse_int_list(args.phase_levels, name="phase-levels")
    adaptive_cov_thresholds = _parse_float_list(
        args.adaptive_cov_thresholds,
        name="adaptive-cov-thresholds",
    )
    adaptive_done_cov_thresholds: List[float] = []
    if args.adaptive_done_cov_thresholds.strip():
        adaptive_done_cov_thresholds = _parse_float_list(
            args.adaptive_done_cov_thresholds,
            name="adaptive-done-cov-thresholds",
        )
    repo_root = Path(__file__).resolve().parent
    runner = repo_root / "run_ppo_sb3.py"
    if not runner.exists():
        raise FileNotFoundError(f"Runner not found: {runner}")

    out_dir = _resolve_out_dir(args, repo_root)
    models_dir = out_dir / "models"
    logs_dir = out_dir / "logs"
    maps_dir = out_dir / "maps"
    report_dir = out_dir / "reports"
    for d in (models_dir, logs_dir, maps_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for p in list(models_dir.glob("*.zip")) + list(logs_dir.glob("*")) + list(maps_dir.glob("*.txt")):
            p.unlink()

    bc_ckpt = ""
    if args.init_from_bc.strip():
        p = Path(args.init_from_bc)
        if not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            raise FileNotFoundError(f"BC checkpoint not found: {p}")
        bc_ckpt = str(p)

    total = int(args.total_timesteps)
    chunk = int(args.chunk_timesteps)
    num_chunks = int(math.ceil(total / float(chunk)))

    print(
        f"[INFO] total={total} chunk={chunk} mode={args.curriculum_mode} "
        f"stage={args.stage_timesteps} chunks={num_chunks} "
        f"num_envs={args.num_envs} vec={args.vec_env}"
    , flush=True)
    print(f"[INFO] out_dir={out_dir}", flush=True)
    if args.curriculum_mode == "adaptive":
        print(
            "[INFO] adaptive gates:"
            f" min_chunks={args.adaptive_min_chunks},"
            f" window={args.adaptive_window},"
            f" max_chunks_per_level={args.adaptive_max_chunks_per_level},"
            f" cov_thresholds={adaptive_cov_thresholds},"
            f" done_cov_thresholds={adaptive_done_cov_thresholds if adaptive_done_cov_thresholds else 'disabled'},"
            f" cov_slope_min={args.adaptive_cov_slope_min},"
            f" collision_max={args.adaptive_collision_max}",
            flush=True,
        )

    all_rollouts: List[Dict] = []
    progress_rows: List[Dict] = []
    prev_model_zip = ""
    done_steps = 0
    failed = False
    adaptive_stage_idx = 0
    adaptive_stage_chunks: List[Dict] = []

    for i in range(num_chunks):
        chunk_start = done_steps
        chunk_t = int(min(chunk, total - done_steps))
        if args.curriculum_mode == "fixed":
            preset = _preset_for_chunk(phase_levels, int(args.stage_timesteps), chunk_start)
            stage_idx = int(chunk_start // int(args.stage_timesteps))
        else:
            stage_idx = int(adaptive_stage_idx)
            level = int(phase_levels[min(stage_idx, len(phase_levels) - 1)])
            if level not in PRESETS:
                raise ValueError(f"Unsupported shape-grid level in phase-levels: {level}")
            preset = PRESETS[level]
        map_seed = int(args.seed if args.map_seed_mode == "fixed" else (args.seed + i))
        run_seed = int(args.seed + i)

        grid = build_shape_grid_map(
            size=int(args.map_size),
            seed=map_seed,
            grid_step=preset.grid_step,
            spawn_prob=preset.spawn_prob,
            min_half_extent=preset.min_half_extent,
            max_half_extent=preset.max_half_extent,
            jitter=preset.jitter,
            clearance=preset.clearance,
            shape_types=preset.shape_types,
            ensure_start_clear=True,
        )
        map_txt = maps_dir / f"chunk{i+1:02d}_L{preset.level}_{preset.name}_seed{map_seed}.txt"
        _write_map_txt(map_txt, grid)
        obs_cells = int(np.count_nonzero(grid == 1))
        obs_ratio = float(obs_cells) / float(grid.size)

        save_model_base = models_dir / f"chunk{i+1:02d}"
        save_model_zip = save_model_base.with_suffix(".zip")
        save_json = logs_dir / f"chunk{i+1:02d}.json"
        save_csv = logs_dir / f"chunk{i+1:02d}.csv"

        init_bc = bc_ckpt if i == 0 and bc_ckpt else ""
        load_model = prev_model_zip if i > 0 else ""
        cmd = _build_run_cmd(
            args,
            runner=runner,
            chunk_timesteps=chunk_t,
            run_seed=run_seed,
            map_file=map_txt,
            save_model=save_model_base,
            save_json=save_json,
            save_csv=save_csv,
            init_from_bc=init_bc,
            load_model=load_model,
        )

        print(
            f"\n[CHUNK {i+1}/{num_chunks}] steps={chunk_t} "
            f"curriculum=L{preset.level}({preset.name}) map_seed={map_seed} "
            f"obs_ratio={obs_ratio:.3f}"
        , flush=True)
        print(f"[RUN] {' '.join(cmd)}", flush=True)

        try:
            subprocess.run(cmd, cwd=str(repo_root), check=True)
        except subprocess.CalledProcessError as e:
            failed = True
            rec = {
                "chunk": i + 1,
                "status": "failed",
                "returncode": int(e.returncode),
                "steps_done": int(done_steps),
            }
            progress_rows.append(rec)
            if not args.continue_on_error:
                break
            continue

        if not save_model_zip.exists():
            failed = True
            rec = {
                "chunk": i + 1,
                "status": "failed_no_model",
                "steps_done": int(done_steps),
            }
            progress_rows.append(rec)
            if not args.continue_on_error:
                break
            continue

        rollouts = _load_rollouts(save_json)
        all_rollouts.extend(rollouts)
        last = rollouts[-1] if rollouts else {}
        done_steps += chunk_t
        prev_model_zip = str(save_model_zip)

        rec = {
            "chunk": i + 1,
            "status": "ok",
            "chunk_timesteps": int(chunk_t),
            "timesteps_done": int(done_steps),
            "curriculum_mode": str(args.curriculum_mode),
            "curriculum_stage_index": int(stage_idx),
            "curriculum_level": int(preset.level),
            "curriculum_name": preset.name,
            "map_seed": int(map_seed),
            "run_seed": int(run_seed),
            "map_file": str(map_txt),
            "map_obstacle_cells": int(obs_cells),
            "map_obstacle_ratio": float(obs_ratio),
            "chunk_last_reward_total": float(last.get("reward_total", float("nan"))),
            "chunk_last_coverage_ratio": float(last.get("coverage_ratio", float("nan"))),
            "chunk_last_collision": float(last.get("collision", float("nan"))),
            "cum_mean_reward_total": _safe_mean(all_rollouts, "reward_total"),
            "cum_mean_coverage_ratio": _safe_mean(all_rollouts, "coverage_ratio"),
            "cum_mean_collision": _safe_mean(all_rollouts, "collision"),
            "model_zip": str(save_model_zip),
            "rollouts_seen": int(len(all_rollouts)),
        }
        rec["chunk_mean_coverage_ratio"] = _safe_mean(rollouts, "coverage_ratio")
        rec["chunk_mean_collision"] = _safe_mean(rollouts, "collision")
        rec["chunk_mean_done_final_coverage_ratio"] = _safe_mean(rollouts, "episode_final_coverage_ratio")
        progress_rows.append(rec)

        print(
            f"[PROGRESS] {done_steps}/{total} | "
            f"chunk(last): cov={rec['chunk_last_coverage_ratio']:.4f}, "
            f"rew={rec['chunk_last_reward_total']:.4f}, coll={rec['chunk_last_collision']:.4f} | "
            f"cumulative(mean): cov={rec['cum_mean_coverage_ratio']:.4f}, "
            f"rew={rec['cum_mean_reward_total']:.4f}, coll={rec['cum_mean_collision']:.4f}"
        , flush=True)

        if args.curriculum_mode == "adaptive":
            adaptive_stage_chunks.append(rec)
            if adaptive_stage_idx < (len(phase_levels) - 1):
                chunks_here = len(adaptive_stage_chunks)
                win = int(min(args.adaptive_window, chunks_here))
                recent = adaptive_stage_chunks[-win:]
                recent_cov = [float(r.get("chunk_mean_coverage_ratio", float("nan"))) for r in recent]
                recent_coll = [float(r.get("chunk_mean_collision", float("nan"))) for r in recent]
                recent_done_cov = [
                    float(r.get("chunk_mean_done_final_coverage_ratio", float("nan")))
                    for r in recent
                ]
                cov_mean = _safe_mean_vals(recent_cov)
                coll_mean = _safe_mean_vals(recent_coll)
                done_cov_mean = _safe_mean_vals(recent_done_cov)
                cov_slope = _trend_slope(recent_cov)

                thr_idx = min(adaptive_stage_idx, len(adaptive_cov_thresholds) - 1)
                cov_thr = float(adaptive_cov_thresholds[thr_idx])
                done_cov_thr = None
                if len(adaptive_done_cov_thresholds) > 0:
                    done_cov_thr = float(
                        adaptive_done_cov_thresholds[
                            min(adaptive_stage_idx, len(adaptive_done_cov_thresholds) - 1)
                        ]
                    )

                cond_ready = chunks_here >= int(args.adaptive_min_chunks)
                cond_cov = (not np.isnan(cov_mean)) and (cov_mean >= cov_thr)
                cond_slope = float(cov_slope) >= float(args.adaptive_cov_slope_min)
                cond_coll = (not np.isnan(coll_mean)) and (coll_mean <= float(args.adaptive_collision_max))
                cond_done_cov = True
                if done_cov_thr is not None:
                    cond_done_cov = (not np.isnan(done_cov_mean)) and (done_cov_mean >= done_cov_thr)

                converged = bool(cond_cov and cond_slope and cond_coll and cond_done_cov)
                promote = bool(cond_ready and converged)
                force_promote = bool(
                    int(args.adaptive_max_chunks_per_level) > 0
                    and chunks_here >= int(args.adaptive_max_chunks_per_level)
                )
                decision = "promote" if (promote or force_promote) else "hold"
                reason = (
                    "converged"
                    if promote
                    else (
                        f"max_chunks={args.adaptive_max_chunks_per_level}"
                        if force_promote
                        else "not_ready_or_not_converged"
                    )
                )
                rec["adaptive_window_chunks"] = int(win)
                rec["adaptive_recent_cov_mean"] = float(cov_mean)
                rec["adaptive_recent_cov_slope"] = float(cov_slope)
                rec["adaptive_recent_collision_mean"] = float(coll_mean)
                rec["adaptive_recent_done_cov_mean"] = float(done_cov_mean)
                rec["adaptive_cov_threshold"] = float(cov_thr)
                rec["adaptive_done_cov_threshold"] = (
                    float(done_cov_thr) if done_cov_thr is not None else float("nan")
                )
                rec["adaptive_cond_ready"] = bool(cond_ready)
                rec["adaptive_cond_cov"] = bool(cond_cov)
                rec["adaptive_cond_slope"] = bool(cond_slope)
                rec["adaptive_cond_collision"] = bool(cond_coll)
                rec["adaptive_cond_done_cov"] = bool(cond_done_cov)
                rec["adaptive_converged"] = bool(converged)
                rec["adaptive_decision"] = str(decision)
                rec["adaptive_reason"] = str(reason)

                print(
                    f"[ADAPTIVE] stage={adaptive_stage_idx} "
                    f"recent_cov={cov_mean:.4f} (thr={cov_thr:.4f}) "
                    f"slope={cov_slope:.5f} (min={args.adaptive_cov_slope_min:.5f}) "
                    f"coll={coll_mean:.4f} (max={args.adaptive_collision_max:.4f}) "
                    f"done_cov={done_cov_mean:.4f} (thr={done_cov_thr if done_cov_thr is not None else 'off'}) "
                    f"converged={converged} ready={cond_ready} "
                    f"decision={decision} reason={reason}",
                    flush=True,
                )
                if promote or force_promote:
                    print(
                        f"[PROMOTE] stage={adaptive_stage_idx} -> {adaptive_stage_idx + 1} "
                        f"reason={reason} "
                        f"(cov_mean={cov_mean:.4f}, cov_thr={cov_thr:.4f}, "
                        f"slope={cov_slope:.5f}, coll_mean={coll_mean:.4f}, "
                        f"done_cov_mean={done_cov_mean:.4f})",
                        flush=True,
                    )
                    adaptive_stage_idx += 1
                    adaptive_stage_chunks = []
            else:
                rec["adaptive_decision"] = "hold"
                rec["adaptive_reason"] = "max_stage_reached"
                rec["adaptive_converged"] = True
                print(
                    f"[ADAPTIVE] stage={adaptive_stage_idx} decision=hold reason=max_stage_reached",
                    flush=True,
                )

        progress_jsonl = report_dir / "progress.jsonl"
        with progress_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "config": vars(args),
        "completed_steps": int(done_steps),
        "requested_steps": int(total),
        "num_chunks": int(num_chunks),
        "failed": bool(failed),
        "final_model_zip": prev_model_zip,
        "progress": progress_rows,
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[INFO] summary={summary_path}", flush=True)
    if prev_model_zip:
        print(f"[INFO] final_model={prev_model_zip}", flush=True)


if __name__ == "__main__":
    main()

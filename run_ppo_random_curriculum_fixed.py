import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from MapGenerator import MapGenerator
from map_generators.curriculum_profiles import (
    available_curriculum_profiles,
    default_stages_for_profile,
    stage_difficulty_label,
    stage_token,
)


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
            "Fixed random-map curriculum PPO runner. "
            "Uses equal or user-provided stage ratios and applies short entropy boosts on promotion."
        )
    )
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--chunk-timesteps", type=int, default=100_000, help="Log/report interval")
    p.add_argument(
        "--phase-levels",
        type=str,
        default="2,3,4",
        help="Comma list of random-map stages to use, e.g. 2,3,4.",
    )
    p.add_argument(
        "--curriculum-profile",
        type=str,
        default="legacy4",
        choices=available_curriculum_profiles(),
        help="Random-map curriculum profile name.",
    )
    p.add_argument(
        "--phase-ratios",
        type=str,
        default="",
        help="Optional comma list of stage weights. Empty => equal ratios across phases.",
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
        default="large",
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
        default="axis2km",
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
    p.add_argument(
        "--ent-coef-base",
        type=float,
        default=0.01,
        help="Base PPO entropy coefficient passed to run_ppo_sb3.py.",
    )
    p.add_argument(
        "--ent-coef-boost",
        type=float,
        default=0.02,
        help="Entropy coefficient used for a few chunks right after curriculum promotion.",
    )
    p.add_argument(
        "--ent-coef-boost-chunks",
        type=int,
        default=2,
        help="Number of chunks to apply boosted entropy after each promotion.",
    )

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
    p.add_argument(
        "--init-from-model",
        type=str,
        default="",
        help="PPO .zip checkpoint for first chunk only. If set, overrides --init-from-bc.",
    )

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
    return repo_root / "learning" / "checkpoints" / "rl" / f"random_curriculum_fixed_{tag}"


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


def _write_map_txt(path: Path, grid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(grid, dtype=np.int32), fmt="%d")


def _write_merged_rollouts_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        path.write_text("", encoding="utf-8")
        return

    preferred = [
        "global_rollout",
        "chunk",
        "rollout",
        "timesteps",
        "reward_total",
        "coverage_ratio",
        "collision",
        "episode_final_coverage_ratio",
        "episode_final_collision",
        "episode_final_steps",
        "chunk_level",
        "chunk_name",
        "map_seed",
        "run_seed",
        "chunk_timesteps_target",
    ]
    keys = sorted({k for r in rows for k in r.keys()})
    fieldnames = preferred + [k for k in keys if k not in preferred]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            out = dict(row)
            out["global_rollout"] = int(idx)
            writer.writerow({k: out.get(k, "") for k in fieldnames})


def _phase_weights(levels: List[int], ratios_raw: str) -> List[float]:
    if not ratios_raw.strip():
        return [1.0 for _ in levels]
    weights = _parse_float_list(ratios_raw, name="phase-ratios")
    if len(weights) != len(levels):
        raise ValueError("phase-ratios must match phase-levels length")
    if any(float(w) <= 0.0 for w in weights):
        raise ValueError("phase-ratios must be positive")
    return [float(w) for w in weights]


def _stage_for_timestep(
    chunk_start_t: int,
    total_timesteps: int,
    levels: List[int],
    weights: List[float],
) -> Tuple[int, int]:
    progress = float(chunk_start_t) / float(max(1, total_timesteps))
    total_w = float(sum(weights))
    acc = 0.0
    for idx, (level, weight) in enumerate(zip(levels, weights)):
        acc += float(weight) / total_w
        if progress < acc or idx == len(levels) - 1:
            return idx, int(level)
    return len(levels) - 1, int(levels[-1])


def _build_run_cmd(
    args: argparse.Namespace,
    *,
    runner: Path,
    chunk_timesteps: int,
    ent_coef: float,
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
        "--ent-coef",
        str(float(ent_coef)),
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


def main() -> None:
    args = _parse_args()
    if args.total_timesteps <= 0:
        raise ValueError("--total-timesteps must be positive")
    if args.chunk_timesteps <= 0:
        raise ValueError("--chunk-timesteps must be positive")
    if args.map_size <= 1:
        raise ValueError("--map-size must be >= 2")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if float(args.ent_coef_base) < 0.0:
        raise ValueError("--ent-coef-base must be >= 0")
    if float(args.ent_coef_boost) < 0.0:
        raise ValueError("--ent-coef-boost must be >= 0")
    if int(args.ent_coef_boost_chunks) < 0:
        raise ValueError("--ent-coef-boost-chunks must be >= 0")

    phase_levels = _parse_int_list(args.phase_levels, name="phase-levels")
    allowed_levels = set(default_stages_for_profile(args.curriculum_profile))
    if any(level not in allowed_levels for level in phase_levels):
        raise ValueError(
            f"phase-levels must be a subset of {sorted(allowed_levels)} "
            f"for profile {args.curriculum_profile}"
        )
    phase_ratios = _phase_weights(phase_levels, args.phase_ratios)

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
        for p in list(report_dir.glob("*.json")) + list(report_dir.glob("*.jsonl")):
            p.unlink()

    bc_ckpt = ""
    if args.init_from_bc.strip():
        p = Path(args.init_from_bc)
        if not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            raise FileNotFoundError(f"BC checkpoint not found: {p}")
        bc_ckpt = str(p)

    init_model_ckpt = ""
    if args.init_from_model.strip():
        p = Path(args.init_from_model)
        if not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            raise FileNotFoundError(f"Init model checkpoint not found: {p}")
        init_model_ckpt = str(p)

    if bc_ckpt and init_model_ckpt:
        print(
            "[WARN] Both --init-from-bc and --init-from-model were provided. "
            "Using --init-from-model for chunk 1 and ignoring BC init.",
            flush=True,
        )

    total = int(args.total_timesteps)
    chunk = int(args.chunk_timesteps)
    num_chunks = int(math.ceil(total / float(chunk)))

    print(
        f"[INFO] total={total} chunk={chunk} chunks={num_chunks} "
        f"num_envs={args.num_envs} vec={args.vec_env}",
        flush=True,
    )
    print(f"[INFO] out_dir={out_dir}", flush=True)
    print(
        f"[INFO] profile={args.curriculum_profile} phases={phase_levels} ratios={phase_ratios} "
        f"(equal={'yes' if not args.phase_ratios.strip() else 'no'})",
        flush=True,
    )
    print(
        "[INFO] entropy schedule:"
        f" base={float(args.ent_coef_base):.6f},"
        f" boost={float(args.ent_coef_boost):.6f},"
        f" boost_chunks={int(args.ent_coef_boost_chunks)}",
        flush=True,
    )

    all_rollouts: List[Dict] = []
    progress_rows: List[Dict] = []
    prev_model_zip = ""
    done_steps = 0
    failed = False
    prev_stage_idx = -1
    ent_boost_remaining = 0

    for i in range(num_chunks):
        chunk_start = done_steps
        chunk_t = int(min(chunk, total - done_steps))
        stage_idx, stage_level = _stage_for_timestep(
            chunk_start_t=chunk_start,
            total_timesteps=total,
            levels=phase_levels,
            weights=phase_ratios,
        )
        if prev_stage_idx >= 0 and stage_idx > prev_stage_idx:
            ent_boost_remaining = int(args.ent_coef_boost_chunks)
        if ent_boost_remaining > 0:
            ent_coef = float(args.ent_coef_boost)
            ent_boost_remaining -= 1
        else:
            ent_coef = float(args.ent_coef_base)
        prev_stage_idx = int(stage_idx)

        map_seed = int(args.seed if args.map_seed_mode == "fixed" else (args.seed + i))
        run_seed = int(args.seed + i)
        gen = MapGenerator(
            height=args.map_size,
            width=args.map_size,
            seed=map_seed,
            curriculum_profile=args.curriculum_profile,
        )
        grid, meta = gen.generate_map(stage=stage_level, return_metadata=True)
        grid = np.asarray(grid, dtype=np.int32)
        obs_cells = int(np.count_nonzero(grid == 1))
        obs_ratio = float(obs_cells) / float(grid.size)
        stage_label = stage_difficulty_label(args.curriculum_profile, stage_level)
        stage_name_token = stage_token(args.curriculum_profile, stage_level)

        map_txt = maps_dir / f"chunk{i+1:03d}_S{stage_name_token}_seed{map_seed}.txt"
        _write_map_txt(map_txt, grid)

        save_model_base = models_dir / f"chunk{i+1:03d}"
        save_model_zip = save_model_base.with_suffix(".zip")
        save_json = logs_dir / f"chunk{i+1:03d}.json"
        save_csv = logs_dir / f"chunk{i+1:03d}.csv"

        init_bc = bc_ckpt if i == 0 and bc_ckpt and not init_model_ckpt else ""
        load_model = init_model_ckpt if i == 0 else prev_model_zip
        cmd = _build_run_cmd(
            args,
            runner=runner,
            chunk_timesteps=chunk_t,
            ent_coef=ent_coef,
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
            f"stage={stage_level}({stage_label}) map_seed={map_seed} "
            f"obs_ratio={obs_ratio:.3f} obs_inst={len(meta)} ent_coef={ent_coef:.6f}",
            flush=True,
        )
        print(f"[RUN] {' '.join(cmd)}", flush=True)

        try:
            subprocess.run(cmd, cwd=str(repo_root), check=True)
        except subprocess.CalledProcessError as exc:
            failed = True
            rec = {
                "chunk": i + 1,
                "status": "failed",
                "returncode": int(exc.returncode),
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
        merged_rows = []
        for row in rollouts:
            merged = dict(row)
            merged["chunk"] = int(i + 1)
            merged["chunk_level"] = int(stage_level)
            merged["chunk_name"] = f"{args.curriculum_profile}_stage{stage_name_token}"
            merged["chunk_difficulty_label"] = str(stage_label)
            merged["curriculum_profile"] = str(args.curriculum_profile)
            merged["map_seed"] = int(map_seed)
            merged["run_seed"] = int(run_seed)
            merged["chunk_timesteps_target"] = int(chunk_t)
            merged_rows.append(merged)
        all_rollouts.extend(merged_rows)

        last = rollouts[-1] if rollouts else {}
        done_steps += chunk_t
        prev_model_zip = str(save_model_zip)

        rec = {
            "chunk": i + 1,
            "status": "ok",
            "chunk_timesteps": int(chunk_t),
            "timesteps_done": int(done_steps),
            "curriculum_stage_index": int(stage_idx),
            "curriculum_level": int(stage_level),
            "curriculum_name": f"{args.curriculum_profile}_stage{stage_name_token}",
            "curriculum_profile": str(args.curriculum_profile),
            "curriculum_difficulty_label": str(stage_label),
            "map_seed": int(map_seed),
            "run_seed": int(run_seed),
            "map_file": str(map_txt),
            "map_obstacle_cells": int(obs_cells),
            "map_obstacle_ratio": float(obs_ratio),
            "map_obstacle_instances": int(len(meta)),
            "chunk_last_reward_total": float(last.get("reward_total", float("nan"))),
            "chunk_last_coverage_ratio": float(last.get("coverage_ratio", float("nan"))),
            "chunk_last_collision": float(last.get("collision", float("nan"))),
            "cum_mean_reward_total": _safe_mean(all_rollouts, "reward_total"),
            "cum_mean_coverage_ratio": _safe_mean(all_rollouts, "coverage_ratio"),
            "cum_mean_collision": _safe_mean(all_rollouts, "collision"),
            "model_zip": str(save_model_zip),
            "rollouts_seen": int(len(all_rollouts)),
            "ent_coef": float(ent_coef),
        }
        rec["chunk_mean_coverage_ratio"] = _safe_mean(rollouts, "coverage_ratio")
        rec["chunk_mean_collision"] = _safe_mean(rollouts, "collision")
        rec["chunk_mean_done_final_coverage_ratio"] = _safe_mean(
            rollouts,
            "episode_final_coverage_ratio",
        )
        progress_rows.append(rec)

        print(
            f"[PROGRESS] {done_steps}/{total} | "
            f"chunk(last): cov={rec['chunk_last_coverage_ratio']:.4f}, "
            f"rew={rec['chunk_last_reward_total']:.4f}, coll={rec['chunk_last_collision']:.4f} | "
            f"cumulative(mean): cov={rec['cum_mean_coverage_ratio']:.4f}, "
            f"rew={rec['cum_mean_reward_total']:.4f}, coll={rec['cum_mean_collision']:.4f}",
            flush=True,
        )

        progress_jsonl = report_dir / "progress.jsonl"
        with progress_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    merged_rollouts_csv = logs_dir / "all_rollouts_merged.csv"
    _write_merged_rollouts_csv(merged_rollouts_csv, all_rollouts)
    print(f"[INFO] merged_rollouts={merged_rollouts_csv}", flush=True)

    summary = {
        "status": "failed" if failed else "ok",
        "total_timesteps": int(total),
        "chunk_timesteps": int(chunk),
        "num_chunks": int(num_chunks),
        "phase_levels": [int(v) for v in phase_levels],
        "phase_ratios": [float(v) for v in phase_ratios],
        "seed": int(args.seed),
        "map_seed_mode": str(args.map_seed_mode),
        "include_dtm": bool(args.include_dtm),
        "final_model_zip": str(prev_model_zip),
        "merged_rollouts_csv": str(merged_rollouts_csv),
        "chunks": progress_rows,
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[DONE] summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()

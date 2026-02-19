import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np


METRICS = ("reward_total", "coverage_ratio", "collision", "reward_area", "reward_tv_i")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage-1 overfit benchmark: run PPO baseline vs DTM on fixed indoor seeds "
            "and generate comparison + convergence diagnostics."
        )
    )
    p.add_argument("--manifest", type=str, default="map/manifest.json")
    p.add_argument("--group", type=str, default="indoor")
    p.add_argument("--map-seeds", type=str, default="101,202")

    p.add_argument("--total-timesteps", type=int, default=500000)
    p.add_argument("--device", type=str, default="cpu", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-envs", type=int, default=24)
    p.add_argument("--vec-env", type=str, default="subproc", choices=["auto", "dummy", "subproc"])
    p.add_argument(
        "--subproc-start-method",
        type=str,
        default="spawn",
        choices=["auto", "spawn", "fork", "forkserver"],
    )
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--sensor-range", type=int, default=2)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--map-size", type=int, default=64)

    p.add_argument(
        "--seed-mode",
        type=str,
        default="map",
        choices=["map", "index", "fixed"],
        help="map: map seed, index: seed-base+index, fixed: always seed-base",
    )
    p.add_argument("--seed-base", type=int, default=100)
    p.add_argument("--python-bin", type=str, default=sys.executable)

    p.add_argument(
        "--init-from-bc",
        type=str,
        default="learning/checkpoints/bc/bc_smoke_latest.pt",
        help="BC checkpoint for PPO warm-start. Set empty string to disable.",
    )
    p.add_argument("--init-from-bc-strict", action="store_true")

    p.add_argument("--conv-window", type=int, default=4, help="Window size for early/late means.")
    p.add_argument("--coverage-plateau-delta", type=float, default=0.0005)
    p.add_argument("--coverage-slope-eps", type=float, default=0.00005)
    p.add_argument("--reward-plateau-delta", type=float, default=0.05)
    p.add_argument("--reward-slope-eps", type=float, default=0.005)
    p.add_argument("--collision-plateau-delta", type=float, default=0.01)
    p.add_argument("--collision-slope-eps", type=float, default=0.001)

    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()


def _parse_seed_list(raw: str) -> List[int]:
    seeds: List[int] = []
    for tok in raw.split(","):
        s = tok.strip()
        if not s:
            continue
        seeds.append(int(s))
    if not seeds:
        raise ValueError("No map seeds selected")
    return seeds


def _load_manifest_entries(manifest: Path, group: str, seeds: List[int]) -> List[Dict]:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if group not in raw:
        raise KeyError(f"group '{group}' not found in {manifest}")
    arr = raw[group]
    if not isinstance(arr, list):
        raise ValueError(f"manifest group '{group}' is not a list")

    by_seed: Dict[int, Dict] = {}
    for item in arr:
        if "seed" not in item or "txt" not in item:
            continue
        by_seed[int(item["seed"])] = dict(item)

    missing = [s for s in seeds if s not in by_seed]
    if missing:
        raise KeyError(f"seed(s) not found in manifest group '{group}': {missing}")
    return [by_seed[s] for s in seeds]


def _run_seed(args: argparse.Namespace, map_seed: int, idx: int) -> int:
    if args.seed_mode == "map":
        return int(map_seed)
    if args.seed_mode == "index":
        return int(args.seed_base + idx)
    return int(args.seed_base)


def _build_cmd(
    args: argparse.Namespace,
    *,
    map_file: str,
    run_seed: int,
    include_dtm: bool,
    save_model: Path,
    save_json: Path,
    save_csv: Path,
) -> List[str]:
    runner = Path(__file__).resolve().parent / "run_ppo_sb3.py"
    cmd = [
        args.python_bin,
        str(runner),
        "--total-timesteps",
        str(args.total_timesteps),
        "--device",
        args.device,
        "--num-envs",
        str(args.num_envs),
        "--vec-env",
        args.vec_env,
        "--subproc-start-method",
        args.subproc_start_method,
        "--map-source",
        "file",
        "--map-file",
        map_file,
        "--map-size",
        str(args.map_size),
        "--sensor-range",
        str(args.sensor_range),
        "--max-episode-steps",
        str(args.max_episode_steps),
        "--n-steps",
        str(args.n_steps),
        "--batch-size",
        str(args.batch_size),
        "--n-epochs",
        str(args.n_epochs),
        "--seed",
        str(run_seed),
        "--save-model",
        str(save_model),
        "--save-breakdown-json",
        str(save_json),
        "--save-breakdown-csv",
        str(save_csv),
    ]
    if include_dtm:
        cmd.append("--include-dtm")
    if args.init_from_bc:
        cmd += ["--init-from-bc", args.init_from_bc]
    if args.init_from_bc_strict:
        cmd.append("--init-from-bc-strict")
    return cmd


def _resolve_input_path(path_str: str, repo_root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return repo_root / p


def _load_rollouts(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rollouts = payload.get("rollouts", [])
    if not isinstance(rollouts, list) or len(rollouts) == 0:
        raise ValueError(f"empty rollouts in {path}")
    return rollouts


def _linear_slope(y: np.ndarray) -> float:
    if y.size < 2:
        return float("nan")
    x = np.arange(y.size, dtype=np.float64)
    return float(np.polyfit(x, y, 1)[0])


def _metric_stats(values: np.ndarray, window: int) -> Dict[str, float]:
    if values.size == 0:
        return {
            "first": float("nan"),
            "last": float("nan"),
            "mean": float("nan"),
            "early_mean": float("nan"),
            "late_mean": float("nan"),
            "late_minus_early": float("nan"),
            "slope": float("nan"),
        }
    w = max(2, min(window, int(values.size)))
    early = float(np.mean(values[:w]))
    late = float(np.mean(values[-w:]))
    return {
        "first": float(values[0]),
        "last": float(values[-1]),
        "mean": float(np.mean(values)),
        "early_mean": early,
        "late_mean": late,
        "late_minus_early": float(late - early),
        "slope": _linear_slope(values),
    }


def _summarize_rollouts(rollouts: List[Dict], args: argparse.Namespace) -> Dict:
    n = len(rollouts)
    out = {
        "num_rollouts": int(n),
        "window": int(max(2, min(args.conv_window, n))),
        "last_timesteps": float(rollouts[-1].get("timesteps", float("nan"))),
        "metrics": {},
    }
    for key in METRICS:
        vals = np.asarray([float(r[key]) for r in rollouts if key in r], dtype=np.float64)
        out["metrics"][key] = _metric_stats(vals, args.conv_window)

    cov = out["metrics"]["coverage_ratio"]
    rew = out["metrics"]["reward_total"]
    col = out["metrics"]["collision"]
    conv = {
        "coverage_plateau_delta": abs(cov["late_minus_early"]) <= float(args.coverage_plateau_delta),
        "coverage_plateau_slope": abs(cov["slope"]) <= float(args.coverage_slope_eps),
        "reward_plateau_delta": abs(rew["late_minus_early"]) <= float(args.reward_plateau_delta),
        "reward_plateau_slope": abs(rew["slope"]) <= float(args.reward_slope_eps),
        "collision_plateau_delta": abs(col["late_minus_early"]) <= float(args.collision_plateau_delta),
        "collision_plateau_slope": abs(col["slope"]) <= float(args.collision_slope_eps),
    }
    conv["all_plateau"] = all(conv.values())
    out["convergence"] = conv
    return out


def _write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "map_id",
        "seed",
        "map_file",
        "baseline_last_reward_total",
        "dtm_last_reward_total",
        "delta_last_reward_total",
        "baseline_last_coverage_ratio",
        "dtm_last_coverage_ratio",
        "delta_last_coverage_ratio",
        "baseline_last_collision",
        "dtm_last_collision",
        "delta_last_collision",
        "baseline_cov_slope",
        "dtm_cov_slope",
        "baseline_cov_late_minus_early",
        "dtm_cov_late_minus_early",
        "baseline_all_plateau",
        "dtm_all_plateau",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _aggregate(rows: List[Dict]) -> Dict:
    if len(rows) == 0:
        return {}

    def arr(key: str):
        return np.asarray([float(x[key]) for x in rows], dtype=np.float64)

    reward = arr("delta_last_reward_total")
    cov = arr("delta_last_coverage_ratio")
    coll = arr("delta_last_collision")
    return {
        "num_maps": int(len(rows)),
        "delta_last": {
            "reward_total_mean": float(np.mean(reward)),
            "coverage_ratio_mean": float(np.mean(cov)),
            "collision_mean": float(np.mean(coll)),
            "reward_total_std": float(np.std(reward)),
            "coverage_ratio_std": float(np.std(cov)),
            "collision_std": float(np.std(coll)),
        },
        "wins": {
            "reward_total": int(np.count_nonzero(reward > 0.0)),
            "coverage_ratio": int(np.count_nonzero(cov > 0.0)),
            "collision": int(np.count_nonzero(coll < 0.0)),
        },
    }


def main():
    args = _parse_args()
    repo_root = Path(__file__).resolve().parent
    manifest = _resolve_input_path(args.manifest, repo_root)
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    if args.init_from_bc:
        bc_ckpt = _resolve_input_path(args.init_from_bc, repo_root)
        if not bc_ckpt.exists():
            raise FileNotFoundError(f"BC checkpoint not found: {bc_ckpt}")

    selected_seeds = _parse_seed_list(args.map_seeds)
    entries = _load_manifest_entries(manifest, args.group, selected_seeds)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        tag = args.run_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("learning/checkpoints/rl") / f"stage1_overfit_{tag}"

    models_dir = out_dir / "models"
    logs_dir = out_dir / "logs"
    reports_dir = out_dir / "reports"
    for d in (models_dir, logs_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] group={args.group}, seeds={selected_seeds}")
    print(f"[INFO] output dir: {out_dir}")

    per_map_rows: List[Dict] = []
    per_map_diag: List[Dict] = []
    failures: List[Dict] = []

    for idx, entry in enumerate(entries):
        map_seed = int(entry["seed"])
        map_file = str(entry["txt"])
        run_seed = _run_seed(args, map_seed, idx)
        map_id = f"{args.group}_seed{map_seed}"
        print(f"\n[MAP {idx+1}/{len(entries)}] {map_id} | run_seed={run_seed}")

        baseline_json = logs_dir / f"{map_id}_baseline.json"
        baseline_csv = logs_dir / f"{map_id}_baseline.csv"
        baseline_model = models_dir / f"{map_id}_baseline"
        dtm_json = logs_dir / f"{map_id}_dtm.json"
        dtm_csv = logs_dir / f"{map_id}_dtm.csv"
        dtm_model = models_dir / f"{map_id}_dtm"

        tasks = [
            ("baseline", False, baseline_model, baseline_json, baseline_csv),
            ("dtm", True, dtm_model, dtm_json, dtm_csv),
        ]

        for mode_name, include_dtm, model_path, json_path, csv_path in tasks:
            model_zip = model_path.with_suffix(".zip")
            if args.skip_existing and json_path.exists() and model_zip.exists():
                print(f"  [SKIP] {mode_name}: existing outputs")
                continue
            if (not args.overwrite) and (json_path.exists() or model_zip.exists()):
                raise FileExistsError(
                    f"existing outputs for {map_id} {mode_name}. "
                    "Use --overwrite or --skip-existing."
                )
            if args.overwrite:
                # Avoid stale artifacts being reused when a rerun crashes mid-training.
                if json_path.exists():
                    json_path.unlink()
                if csv_path.exists():
                    csv_path.unlink()
                if model_zip.exists():
                    model_zip.unlink()

            cmd = _build_cmd(
                args,
                map_file=map_file,
                run_seed=run_seed,
                include_dtm=include_dtm,
                save_model=model_path,
                save_json=json_path,
                save_csv=csv_path,
            )
            print(f"  [RUN] {mode_name}: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, cwd=str(repo_root))
            except subprocess.CalledProcessError as exc:
                fail = {
                    "map_id": map_id,
                    "mode": mode_name,
                    "returncode": int(exc.returncode),
                }
                failures.append(fail)
                print(f"  [FAIL] {fail}")
                if not args.continue_on_error:
                    raise

        if not baseline_json.exists() or not dtm_json.exists():
            print(f"  [WARN] missing baseline/dtm logs for {map_id}, skip report row")
            continue

        base_rollouts = _load_rollouts(baseline_json)
        dtm_rollouts = _load_rollouts(dtm_json)
        base_diag = _summarize_rollouts(base_rollouts, args)
        dtm_diag = _summarize_rollouts(dtm_rollouts, args)

        row = {
            "map_id": map_id,
            "seed": map_seed,
            "map_file": map_file,
            "baseline_last_reward_total": base_diag["metrics"]["reward_total"]["last"],
            "dtm_last_reward_total": dtm_diag["metrics"]["reward_total"]["last"],
            "delta_last_reward_total": (
                dtm_diag["metrics"]["reward_total"]["last"] - base_diag["metrics"]["reward_total"]["last"]
            ),
            "baseline_last_coverage_ratio": base_diag["metrics"]["coverage_ratio"]["last"],
            "dtm_last_coverage_ratio": dtm_diag["metrics"]["coverage_ratio"]["last"],
            "delta_last_coverage_ratio": (
                dtm_diag["metrics"]["coverage_ratio"]["last"] - base_diag["metrics"]["coverage_ratio"]["last"]
            ),
            "baseline_last_collision": base_diag["metrics"]["collision"]["last"],
            "dtm_last_collision": dtm_diag["metrics"]["collision"]["last"],
            "delta_last_collision": (
                dtm_diag["metrics"]["collision"]["last"] - base_diag["metrics"]["collision"]["last"]
            ),
            "baseline_cov_slope": base_diag["metrics"]["coverage_ratio"]["slope"],
            "dtm_cov_slope": dtm_diag["metrics"]["coverage_ratio"]["slope"],
            "baseline_cov_late_minus_early": base_diag["metrics"]["coverage_ratio"]["late_minus_early"],
            "dtm_cov_late_minus_early": dtm_diag["metrics"]["coverage_ratio"]["late_minus_early"],
            "baseline_all_plateau": int(base_diag["convergence"]["all_plateau"]),
            "dtm_all_plateau": int(dtm_diag["convergence"]["all_plateau"]),
        }
        per_map_rows.append(row)
        per_map_diag.append(
            {
                "map_id": map_id,
                "baseline": base_diag,
                "dtm": dtm_diag,
            }
        )
        print(
            "  [RESULT] "
            f"delta_last(reward={row['delta_last_reward_total']:+.4f}, "
            f"cov={row['delta_last_coverage_ratio']:+.4f}, "
            f"collision={row['delta_last_collision']:+.4f}) "
            f"| plateau(base={bool(row['baseline_all_plateau'])}, dtm={bool(row['dtm_all_plateau'])})"
        )

    per_map_csv = reports_dir / "per_map_comparison.csv"
    _write_csv(per_map_csv, per_map_rows)

    summary = {
        "config": {
            "manifest": str(manifest),
            "group": args.group,
            "map_seeds": selected_seeds,
            "total_timesteps": int(args.total_timesteps),
            "device": args.device,
            "num_envs": int(args.num_envs),
            "vec_env": args.vec_env,
            "subproc_start_method": args.subproc_start_method,
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
            "sensor_range": int(args.sensor_range),
            "max_episode_steps": int(args.max_episode_steps),
            "map_size": int(args.map_size),
            "seed_mode": args.seed_mode,
            "seed_base": int(args.seed_base),
            "init_from_bc": args.init_from_bc,
            "init_from_bc_strict": bool(args.init_from_bc_strict),
            "convergence_thresholds": {
                "conv_window": int(args.conv_window),
                "coverage_plateau_delta": float(args.coverage_plateau_delta),
                "coverage_slope_eps": float(args.coverage_slope_eps),
                "reward_plateau_delta": float(args.reward_plateau_delta),
                "reward_slope_eps": float(args.reward_slope_eps),
                "collision_plateau_delta": float(args.collision_plateau_delta),
                "collision_slope_eps": float(args.collision_slope_eps),
            },
        },
        "num_selected_maps": int(len(entries)),
        "num_reported_maps": int(len(per_map_rows)),
        "failures": failures,
        "aggregate": _aggregate(per_map_rows),
        "per_map_diagnostics": per_map_diag,
    }
    summary_json = reports_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[DONE]")
    print(f"  per-map csv: {per_map_csv}")
    print(f"  summary json: {summary_json}")
    if summary["aggregate"]:
        agg = summary["aggregate"]
        print(
            "  aggregate delta_last mean:"
            f" reward={agg['delta_last']['reward_total_mean']:+.6f},"
            f" coverage={agg['delta_last']['coverage_ratio_mean']:+.6f},"
            f" collision={agg['delta_last']['collision_mean']:+.6f}"
        )
        print(f"  wins: {agg['wins']}")


if __name__ == "__main__":
    main()

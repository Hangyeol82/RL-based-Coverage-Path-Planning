import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


METRICS = ("reward_total", "coverage_ratio", "collision", "reward_area", "reward_tv_i")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run baseline+DTM PPO across map manifest entries and report per-map + aggregate comparison."
        )
    )
    p.add_argument("--manifest", type=str, default="map/manifest.json")
    p.add_argument("--groups", type=str, default="indoor,random")
    p.add_argument("--max-maps", type=int, default=0, help="0 means use all selected maps")

    p.add_argument("--total-timesteps", type=int, default=50000)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--num-envs", type=int, default=6)
    p.add_argument("--vec-env", type=str, default="subproc", choices=["auto", "dummy", "subproc"])
    p.add_argument(
        "--subproc-start-method",
        type=str,
        default="spawn",
        choices=["auto", "spawn", "fork", "forkserver"],
    )
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--sensor-range", type=int, default=2)
    p.add_argument("--max-episode-steps", type=int, default=2000)
    p.add_argument("--map-size", type=int, default=64)

    p.add_argument(
        "--seed-mode",
        type=str,
        default="map",
        choices=["map", "index", "fixed"],
        help="map: use seed from manifest, index: base+idx, fixed: always seed-base",
    )
    p.add_argument("--seed-base", type=int, default=100)

    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def _load_manifest(path: Path, groups: List[str], max_maps: int) -> List[Dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries: List[Dict] = []
    for group in groups:
        if group not in raw:
            raise KeyError(f"group '{group}' not found in manifest: {path}")
        arr = raw[group]
        if not isinstance(arr, list):
            raise ValueError(f"manifest group '{group}' is not a list")
        for item in arr:
            if "txt" not in item or "seed" not in item:
                raise ValueError(f"invalid manifest entry in group '{group}': {item}")
            e = dict(item)
            e["group"] = group
            entries.append(e)
    if max_maps > 0:
        entries = entries[:max_maps]
    return entries


def _run_seed(args: argparse.Namespace, entry_seed: int, idx: int) -> int:
    if args.seed_mode == "map":
        return int(entry_seed)
    if args.seed_mode == "index":
        return int(args.seed_base + idx)
    return int(args.seed_base)


def _build_run_cmd(
    args: argparse.Namespace,
    *,
    map_file: str,
    run_seed: int,
    include_dtm: bool,
    save_model: Path,
    save_json: Path,
    save_csv: Path,
) -> List[str]:
    cmd = [
        args.python_bin,
        "run_ppo_sb3.py",
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
    return cmd


def _load_breakdown(path: Path) -> Dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    rollouts = data.get("rollouts", [])
    if not isinstance(rollouts, list) or len(rollouts) == 0:
        raise ValueError(f"empty rollouts in {path}")
    return {"rollouts": rollouts}


def _metric_summary(rollouts: List[Dict]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for m in METRICS:
        vals = np.asarray([float(r[m]) for r in rollouts if m in r], dtype=np.float64)
        if vals.size == 0:
            out[m] = {"last": float("nan"), "mean": float("nan")}
        else:
            out[m] = {"last": float(vals[-1]), "mean": float(vals.mean())}
    return out


def _delta(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    # delta = b - a
    out: Dict[str, Dict[str, float]] = {}
    for m in METRICS:
        out[m] = {
            "last": float(b[m]["last"] - a[m]["last"]),
            "mean": float(b[m]["mean"] - a[m]["mean"]),
        }
    return out


def _write_per_map_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "map_id",
        "group",
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
        "delta_mean_reward_total",
        "delta_mean_coverage_ratio",
        "delta_mean_collision",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _aggregate(per_map: List[Dict]) -> Dict:
    if len(per_map) == 0:
        return {}

    def arr(key: str):
        return np.asarray([float(x[key]) for x in per_map], dtype=np.float64)

    deltas = {
        "reward_total_last": arr("delta_last_reward_total"),
        "coverage_ratio_last": arr("delta_last_coverage_ratio"),
        "collision_last": arr("delta_last_collision"),
        "reward_total_mean": arr("delta_mean_reward_total"),
        "coverage_ratio_mean": arr("delta_mean_coverage_ratio"),
        "collision_mean": arr("delta_mean_collision"),
    }

    agg = {
        "num_maps": int(len(per_map)),
        "delta_stats": {},
        "wins": {},
    }
    for k, v in deltas.items():
        agg["delta_stats"][k] = {
            "mean": float(np.mean(v)),
            "std": float(np.std(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        }

    # win definition:
    # - reward/coverage: dtm > baseline
    # - collision: dtm < baseline
    agg["wins"]["reward_total_last"] = int(np.count_nonzero(deltas["reward_total_last"] > 0.0))
    agg["wins"]["coverage_ratio_last"] = int(np.count_nonzero(deltas["coverage_ratio_last"] > 0.0))
    agg["wins"]["collision_last"] = int(np.count_nonzero(deltas["collision_last"] < 0.0))
    agg["wins"]["reward_total_mean"] = int(np.count_nonzero(deltas["reward_total_mean"] > 0.0))
    agg["wins"]["coverage_ratio_mean"] = int(np.count_nonzero(deltas["coverage_ratio_mean"] > 0.0))
    agg["wins"]["collision_mean"] = int(np.count_nonzero(deltas["collision_mean"] < 0.0))
    return agg


def main():
    args = _parse_args()
    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    if not groups:
        raise ValueError("No groups selected")

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")
    entries = _load_manifest(manifest, groups, args.max_maps)
    if len(entries) == 0:
        raise RuntimeError("No map entries selected")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        tag = args.run_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("learning/checkpoints/rl") / f"benchmark10_{tag}"

    models_dir = out_dir / "models"
    logs_dir = out_dir / "logs"
    report_dir = out_dir / "reports"
    for d in (models_dir, logs_dir, report_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] selected maps: {len(entries)} from groups={groups}")
    print(f"[INFO] output dir: {out_dir}")

    per_map_rows: List[Dict] = []
    failures: List[Dict] = []

    for i, entry in enumerate(entries):
        map_file = str(entry["txt"])
        map_seed = int(entry["seed"])
        group = str(entry["group"])
        run_seed = _run_seed(args, map_seed, i)
        map_id = f"{group}_seed{map_seed}"
        print(f"\n[MAP {i+1}/{len(entries)}] {map_id} | run_seed={run_seed}")

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
            if args.skip_existing and json_path.exists() and model_path.with_suffix(".zip").exists():
                print(f"  [SKIP] {mode_name}: existing outputs found")
                continue
            if (not args.overwrite) and (json_path.exists() or model_path.with_suffix(".zip").exists()):
                raise FileExistsError(
                    f"Output already exists for {map_id} {mode_name}. "
                    f"Use --overwrite or --skip-existing."
                )

            cmd = _build_run_cmd(
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
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                fail = {"map_id": map_id, "mode": mode_name, "returncode": int(e.returncode)}
                failures.append(fail)
                print(f"  [FAIL] {fail}")
                if not args.continue_on_error:
                    raise
                continue

        if not baseline_json.exists() or not dtm_json.exists():
            print(f"  [WARN] missing logs for {map_id}, skip report row")
            continue

        baseline_rollouts = _load_breakdown(baseline_json)["rollouts"]
        dtm_rollouts = _load_breakdown(dtm_json)["rollouts"]
        base_s = _metric_summary(baseline_rollouts)
        dtm_s = _metric_summary(dtm_rollouts)
        dlt = _delta(base_s, dtm_s)

        row = {
            "map_id": map_id,
            "group": group,
            "seed": map_seed,
            "map_file": map_file,
            "baseline_last_reward_total": base_s["reward_total"]["last"],
            "dtm_last_reward_total": dtm_s["reward_total"]["last"],
            "delta_last_reward_total": dlt["reward_total"]["last"],
            "baseline_last_coverage_ratio": base_s["coverage_ratio"]["last"],
            "dtm_last_coverage_ratio": dtm_s["coverage_ratio"]["last"],
            "delta_last_coverage_ratio": dlt["coverage_ratio"]["last"],
            "baseline_last_collision": base_s["collision"]["last"],
            "dtm_last_collision": dtm_s["collision"]["last"],
            "delta_last_collision": dlt["collision"]["last"],
            "delta_mean_reward_total": dlt["reward_total"]["mean"],
            "delta_mean_coverage_ratio": dlt["coverage_ratio"]["mean"],
            "delta_mean_collision": dlt["collision"]["mean"],
        }
        per_map_rows.append(row)
        print(
            "  [RESULT] "
            f"delta_last(reward={row['delta_last_reward_total']:+.4f}, "
            f"cov={row['delta_last_coverage_ratio']:+.4f}, "
            f"collision={row['delta_last_collision']:+.4f})"
        )

    per_map_csv = report_dir / "per_map_comparison.csv"
    _write_per_map_csv(per_map_csv, per_map_rows)

    summary = {
        "config": {
            "manifest": str(manifest),
            "groups": groups,
            "max_maps": int(args.max_maps),
            "total_timesteps": int(args.total_timesteps),
            "num_envs": int(args.num_envs),
            "vec_env": args.vec_env,
            "subproc_start_method": args.subproc_start_method,
            "seed_mode": args.seed_mode,
            "seed_base": int(args.seed_base),
            "map_size": int(args.map_size),
            "sensor_range": int(args.sensor_range),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
        },
        "num_selected_maps": int(len(entries)),
        "num_reported_maps": int(len(per_map_rows)),
        "failures": failures,
        "aggregate": _aggregate(per_map_rows),
    }
    summary_json = report_dir / "summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[DONE]")
    print(f"  per-map csv: {per_map_csv}")
    print(f"  summary json: {summary_json}")
    if summary["aggregate"]:
        agg = summary["aggregate"]
        print("  aggregate delta means:")
        for k, v in agg["delta_stats"].items():
            print(f"    {k}: {v['mean']:+.6f} (std={v['std']:.6f})")
        print("  wins:", agg["wins"])


if __name__ == "__main__":
    main()

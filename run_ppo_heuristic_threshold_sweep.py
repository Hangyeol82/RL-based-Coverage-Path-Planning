import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Keep each worker from oversubscribing BLAS/OpenMP threads.
if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
if "NUMEXPR_NUM_THREADS" not in os.environ:
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from map_generators.validation import parse_map_txt
from run_ppo_eval_shapegrid_compare import _evaluate_one_map, _load_model, _pick_start


GridPos = Tuple[int, int]

_WORKER_MODEL = None
_WORKER_MASKABLE = False
_WORKER_CONFIG: Dict[str, object] = {}


def _parse_csv_ints(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(","):
        s = tok.strip().lower()
        if not s:
            continue
        if s in {"off", "none", "no", "disabled"}:
            out.append(0)
            continue
        out.append(int(s))
    if not out:
        raise ValueError("empty integer list")
    if any(v < 0 for v in out):
        raise ValueError("thresholds must be non-negative; use 0/off for no heuristic")
    return out


def _parse_map_files(raw: str) -> List[Path]:
    out: List[Path] = []
    for tok in str(raw).split(","):
        s = tok.strip()
        if s:
            out.append(Path(s).expanduser().resolve())
    return out


def _discover_maps(
    map_files: str,
    map_dir: str,
    map_glob: str,
    exclude_substrings: Sequence[str],
) -> List[Path]:
    explicit = _parse_map_files(map_files)
    if explicit:
        paths = explicit
    else:
        base = Path(map_dir).expanduser()
        paths = sorted(base.glob(str(map_glob)))
    excludes = [str(s).strip() for s in exclude_substrings if str(s).strip()]
    if excludes:
        paths = [
            p
            for p in paths
            if not any(excluded in p.name for excluded in excludes)
        ]
    if not paths:
        raise FileNotFoundError("No evaluation maps found. Use --map-files or --map-dir/--map-glob.")
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing map files: {missing}")
    return [p.resolve() for p in paths]


def _select_worker_count(device: str, requested: int, task_count: int) -> int:
    if requested > 0:
        return max(1, int(requested))
    if str(device).lower() == "cuda":
        return 1
    cpu = os.cpu_count() or 1
    return max(1, min(int(task_count), max(1, min(cpu // 2, 8))))


def _coverage_auc(grid: np.ndarray, path: Sequence[Sequence[int]]) -> float:
    arr = np.asarray(grid, dtype=np.int32)
    free_mask = arr == 0
    free_cells = int(np.count_nonzero(free_mask))
    if free_cells <= 0:
        return 0.0
    visited: set[GridPos] = set()
    vals: List[float] = []
    rows, cols = arr.shape
    for raw_pos in path:
        if len(raw_pos) < 2:
            continue
        r, c = int(raw_pos[0]), int(raw_pos[1])
        if 0 <= r < rows and 0 <= c < cols and bool(free_mask[r, c]):
            visited.add((r, c))
        vals.append(float(len(visited)) / float(free_cells))
    if not vals:
        return 0.0
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _mean(rows: Sequence[Dict[str, object]], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _rate(rows: Sequence[Dict[str, object]], key: str) -> float:
    if not rows:
        return float("nan")
    vals = [1.0 if bool(r.get(key, False)) else 0.0 for r in rows]
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _mean_optional_penalized(rows: Sequence[Dict[str, object]], key: str, max_steps: int) -> float:
    vals: List[float] = []
    for row in rows:
        value = row.get(key)
        vals.append(float(max_steps if value in (None, "") else value))
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _init_worker(
    model_path: str,
    device: str,
    action_mask: bool,
    config: Dict[str, object],
) -> None:
    global _WORKER_MODEL, _WORKER_MASKABLE, _WORKER_CONFIG
    model, _algo, maskable = _load_model(
        Path(model_path).expanduser().resolve(),
        device=str(device),
        action_mask=bool(action_mask),
    )
    _WORKER_MODEL = model
    _WORKER_MASKABLE = bool(maskable)
    _WORKER_CONFIG = dict(config)


def _evaluate_task(task: Dict[str, object]) -> Dict[str, object]:
    if _WORKER_MODEL is None:
        raise RuntimeError("Worker model was not initialised")

    map_file = Path(str(task["map_file"]))
    grid = parse_map_txt(map_file)
    start = _pick_start(grid)
    threshold = int(task["threshold"])
    heuristic_enabled = threshold > 0

    result = _evaluate_one_map(
        model=_WORKER_MODEL,
        use_maskable_predict=bool(_WORKER_MASKABLE),
        deterministic=bool(_WORKER_CONFIG["deterministic"]),
        grid=grid,
        start=start,
        sensor_range=int(_WORKER_CONFIG["sensor_range"]),
        max_episode_steps=int(_WORKER_CONFIG["max_episode_steps"]),
        include_dtm=bool(_WORKER_CONFIG["include_dtm"]),
        boundary_exit_features=bool(_WORKER_CONFIG["boundary_exit_features"]),
        boundary_exit_threshold=float(_WORKER_CONFIG["boundary_exit_threshold"]),
        action_mask=bool(_WORKER_CONFIG["action_mask"]),
        heuristic_assist=bool(heuristic_enabled),
        heuristic_no_progress_threshold=int(threshold),
        heuristic_min_coverage=float(_WORKER_CONFIG["heuristic_min_coverage"]),
        heuristic_max_astar_expansions=int(_WORKER_CONFIG["heuristic_max_astar_expansions"]),
        robot_state_position=bool(_WORKER_CONFIG["robot_state_position"]),
        robot_state_action_history=bool(_WORKER_CONFIG["robot_state_action_history"]),
        robot_state_progress=bool(_WORKER_CONFIG["robot_state_progress"]),
        robot_state_stagnation=bool(_WORKER_CONFIG["robot_state_stagnation"]),
        robot_state_action_history_len=int(_WORKER_CONFIG["robot_state_action_history_len"]),
        dtm_output_mode=str(_WORKER_CONFIG["dtm_output_mode"]),
        dtm_coarse_mode=str(_WORKER_CONFIG["dtm_coarse_mode"]),
    )
    path = result["path"]
    row = {
        "threshold": int(threshold),
        "heuristic_assist_enabled": bool(heuristic_enabled),
        "map_id": map_file.stem,
        "map_file": str(map_file),
        "repeat": int(task["repeat"]),
        "steps": int(result["steps"]),
        "coverage_ratio": float(result["coverage_ratio"]),
        "coverage_auc": _coverage_auc(grid, path),
        "coverage_cells": int(result["coverage_cells"]),
        "free_cells": int(result["free_cells"]),
        "success_90": bool(result["success_90"]),
        "success_95": bool(result["success_95"]),
        "success_99": bool(result["success_99"]),
        "step_to_90": result["step_to_90"],
        "step_to_95": result["step_to_95"],
        "step_to_99": result["step_to_99"],
        "overlap_rate": float(result["overlap_rate"]),
        "revisit_ratio": float(result["revisit_ratio"]),
        "path_overlap_rate": float(result["path_overlap_rate"]),
        "heuristic_active_rate": float(result["heuristic_active_rate"]),
        "heuristic_active_count": int(result["heuristic_active_count"]),
        "heuristic_selected_rate": float(result["heuristic_selected_rate"]),
        "heuristic_selected_count": int(result["heuristic_selected_count"]),
        "heuristic_frontier_selected_rate": float(result.get("heuristic_frontier_selected_rate", 0.0)),
        "heuristic_frontier_selected_count": int(result.get("heuristic_frontier_selected_count", 0)),
        "heuristic_known_uncovered_selected_rate": float(
            result.get("heuristic_known_uncovered_selected_rate", 0.0),
        ),
        "heuristic_known_uncovered_selected_count": int(
            result.get("heuristic_known_uncovered_selected_count", 0),
        ),
        "heuristic_cached_selected_rate": float(result.get("heuristic_cached_selected_rate", 0.0)),
        "heuristic_cached_selected_count": int(result.get("heuristic_cached_selected_count", 0)),
        "heuristic_override_rate": float(result["heuristic_override_rate"]),
        "heuristic_overridden_count": int(result["heuristic_overridden_count"]),
        "action_override_rate": float(result["action_override_rate"]),
        "unique_positions": int(result["unique_positions"]),
        "total_reward_sum": float(result["total_reward_sum"]),
        "done_reason": str(result["done_reason"]),
    }
    if bool(_WORKER_CONFIG["save_paths"]):
        row["path"] = path
    return row


def _aggregate_thresholds(
    rows: Sequence[Dict[str, object]],
    *,
    max_episode_steps: int,
    max_heuristic_selected_rate: float,
    heuristic_penalty_weight: float,
) -> List[Dict[str, object]]:
    thresholds = sorted({int(r["threshold"]) for r in rows})
    out: List[Dict[str, object]] = []
    for threshold in thresholds:
        group = [r for r in rows if int(r["threshold"]) == int(threshold)]
        mean_cov = _mean(group, "coverage_ratio")
        mean_auc = _mean(group, "coverage_auc")
        success_95 = _rate(group, "success_95")
        success_99 = _rate(group, "success_99")
        overlap = _mean(group, "overlap_rate")
        selected = _mean(group, "heuristic_selected_rate")
        step95_pen = _mean_optional_penalized(group, "step_to_95", int(max_episode_steps))
        score = (
            float(mean_cov)
            + 0.30 * float(mean_auc)
            + 0.25 * float(success_95)
            + 0.15 * float(success_99)
            - 0.20 * float(overlap)
            - float(heuristic_penalty_weight) * float(selected)
            - 0.00001 * float(step95_pen)
        )
        out.append(
            {
                "threshold": int(threshold),
                "heuristic_assist_enabled": bool(threshold > 0),
                "episodes": int(len(group)),
                "mean_coverage_ratio": mean_cov,
                "mean_coverage_auc": mean_auc,
                "success_90_rate": _rate(group, "success_90"),
                "success_95_rate": success_95,
                "success_99_rate": success_99,
                "mean_step_to_90_penalized": _mean_optional_penalized(
                    group,
                    "step_to_90",
                    int(max_episode_steps),
                ),
                "mean_step_to_95_penalized": step95_pen,
                "mean_step_to_99_penalized": _mean_optional_penalized(
                    group,
                    "step_to_99",
                    int(max_episode_steps),
                ),
                "mean_steps": _mean(group, "steps"),
                "mean_overlap_rate": overlap,
                "mean_revisit_ratio": _mean(group, "revisit_ratio"),
                "mean_heuristic_active_rate": _mean(group, "heuristic_active_rate"),
                "mean_heuristic_selected_rate": selected,
                "mean_heuristic_frontier_selected_rate": _mean(group, "heuristic_frontier_selected_rate"),
                "mean_heuristic_known_uncovered_selected_rate": _mean(
                    group,
                    "heuristic_known_uncovered_selected_rate",
                ),
                "mean_heuristic_cached_selected_rate": _mean(group, "heuristic_cached_selected_rate"),
                "mean_heuristic_override_rate": _mean(group, "heuristic_override_rate"),
                "mean_action_override_rate": _mean(group, "action_override_rate"),
                "score": float(score),
                "within_assist_budget": bool(float(selected) <= float(max_heuristic_selected_rate)),
            }
        )
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep heuristic no-progress thresholds for a fixed PPO policy over "
            "a set of evaluation maps."
        )
    )
    p.add_argument("--model", type=str, required=True, help="SB3 model .zip or extracted model directory.")
    p.add_argument("--mode-name", type=str, default="", help="Optional label, e.g. baseline or dtm_six.")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--max-workers", type=int, default=0, help="0 chooses CPU/GPU-safe default.")
    p.add_argument(
        "--thresholds",
        type=str,
        default="off,10,20,30,50,75,100,150,200,300",
        help="Comma list of n values. Use off/0 for no heuristic.",
    )
    p.add_argument("--repeats", type=int, default=1)

    p.add_argument("--map-files", type=str, default="")
    p.add_argument("--map-dir", type=str, default="map/custom_test_maps")
    p.add_argument("--map-glob", type=str, default="handcrafted_*_128.txt")
    p.add_argument(
        "--exclude-map-substrings",
        type=str,
        default="urban_plaza",
        help="Comma list of filename substrings to skip when discovering maps from --map-dir.",
    )
    p.add_argument("--out-dir", type=str, default="log_analysis/logs/heuristic_threshold_sweep")

    p.add_argument("--sensor-range", type=int, default=3)
    p.add_argument("--max-episode-steps", type=int, default=30000)
    p.add_argument("--include-dtm", action="store_true")
    p.add_argument(
        "--dtm-output-mode",
        type=str,
        default="six",
        choices=["six", "extent6", "two", "axis2", "axis2km", "four", "port12"],
    )
    p.add_argument(
        "--dtm-coarse-mode",
        type=str,
        default="bfs",
        choices=["bfs", "aggregate", "aggregate_transfer"],
    )
    p.add_argument("--boundary-exit-features", action="store_true")
    p.add_argument("--boundary-exit-threshold", type=float, default=0.0)
    mask_group = p.add_mutually_exclusive_group()
    mask_group.add_argument("--action-mask", dest="action_mask", action="store_true")
    mask_group.add_argument("--no-action-mask", dest="action_mask", action="store_false")
    p.set_defaults(action_mask=True)

    p.add_argument("--heuristic-min-coverage", type=float, default=0.0)
    p.add_argument("--heuristic-max-astar-expansions", type=int, default=0)
    p.add_argument(
        "--max-heuristic-selected-rate",
        type=float,
        default=0.15,
        help="Budget used only for recommending n; all thresholds are still evaluated.",
    )
    p.add_argument("--heuristic-penalty-weight", type=float, default=0.75)

    robot_pos_group = p.add_mutually_exclusive_group()
    robot_pos_group.add_argument("--robot-state-position", dest="robot_state_position", action="store_true")
    robot_pos_group.add_argument("--no-robot-state-position", dest="robot_state_position", action="store_false")
    robot_hist_group = p.add_mutually_exclusive_group()
    robot_hist_group.add_argument(
        "--robot-state-action-history",
        dest="robot_state_action_history",
        action="store_true",
    )
    robot_hist_group.add_argument(
        "--no-robot-state-action-history",
        dest="robot_state_action_history",
        action="store_false",
    )
    robot_prog_group = p.add_mutually_exclusive_group()
    robot_prog_group.add_argument("--robot-state-progress", dest="robot_state_progress", action="store_true")
    robot_prog_group.add_argument("--no-robot-state-progress", dest="robot_state_progress", action="store_false")
    robot_stag_group = p.add_mutually_exclusive_group()
    robot_stag_group.add_argument("--robot-state-stagnation", dest="robot_state_stagnation", action="store_true")
    robot_stag_group.add_argument("--no-robot-state-stagnation", dest="robot_state_stagnation", action="store_false")
    p.add_argument("--robot-state-action-history-len", type=int, default=5)
    det_group = p.add_mutually_exclusive_group()
    det_group.add_argument("--deterministic", dest="deterministic", action="store_true")
    det_group.add_argument("--stochastic", dest="deterministic", action="store_false")
    p.set_defaults(
        deterministic=True,
        robot_state_position=False,
        robot_state_action_history=True,
        robot_state_progress=False,
        robot_state_stagnation=False,
    )
    p.add_argument("--save-paths", action="store_true", help="Store full paths in metrics_long.jsonl.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    thresholds = _parse_csv_ints(args.thresholds)
    exclude_substrings = [
        tok.strip()
        for tok in str(args.exclude_map_substrings).split(",")
        if tok.strip()
    ]
    map_paths = _discover_maps(
        args.map_files,
        args.map_dir,
        args.map_glob,
        exclude_substrings,
    )
    repeats = max(1, int(args.repeats))
    tasks = [
        {"threshold": int(threshold), "map_file": str(path), "repeat": int(rep)}
        for threshold in thresholds
        for path in map_paths
        for rep in range(repeats)
    ]
    workers = _select_worker_count(str(args.device), int(args.max_workers), len(tasks))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "deterministic": bool(args.deterministic),
        "sensor_range": int(args.sensor_range),
        "max_episode_steps": int(args.max_episode_steps),
        "include_dtm": bool(args.include_dtm),
        "boundary_exit_features": bool(args.boundary_exit_features),
        "boundary_exit_threshold": float(args.boundary_exit_threshold),
        "action_mask": bool(args.action_mask),
        "heuristic_min_coverage": float(args.heuristic_min_coverage),
        "heuristic_max_astar_expansions": int(args.heuristic_max_astar_expansions),
        "robot_state_position": bool(args.robot_state_position),
        "robot_state_action_history": bool(args.robot_state_action_history),
        "robot_state_progress": bool(args.robot_state_progress),
        "robot_state_stagnation": bool(args.robot_state_stagnation),
        "robot_state_action_history_len": int(args.robot_state_action_history_len),
        "dtm_output_mode": str(args.dtm_output_mode),
        "dtm_coarse_mode": str(args.dtm_coarse_mode),
        "save_paths": bool(args.save_paths),
    }
    run_config = {
        "model": str(model_path),
        "mode_name": str(args.mode_name),
        "device": str(args.device),
        "max_workers": int(workers),
        "thresholds": thresholds,
        "map_files": [str(p) for p in map_paths],
        "exclude_map_substrings": exclude_substrings,
        "repeats": int(repeats),
        "config": config,
        "max_heuristic_selected_rate": float(args.max_heuristic_selected_rate),
        "heuristic_penalty_weight": float(args.heuristic_penalty_weight),
    }
    (out_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print(f"[INFO] model={model_path}")
    print(f"[INFO] mode={args.mode_name or 'model'} include_dtm={bool(args.include_dtm)}")
    print(f"[INFO] maps={len(map_paths)} thresholds={thresholds} repeats={repeats}")
    print(f"[INFO] device={args.device} workers={workers}")
    if str(args.device) == "cuda" and workers > 1:
        print("[WARN] CUDA with multiple workers loads one model copy per worker; monitor GPU memory.")

    rows: List[Dict[str, object]] = []

    def _record_row(idx: int, row: Dict[str, object]) -> None:
        row["mode_name"] = str(args.mode_name)
        rows.append(row)
        print(
            "[DONE]"
            f" {idx}/{len(tasks)}"
            f" n={row['threshold']}"
            f" map={row['map_id']}"
            f" cov={float(row['coverage_ratio']):.4f}"
            f" auc={float(row['coverage_auc']):.4f}"
            f" succ95={int(bool(row['success_95']))}"
            f" overlap={float(row['overlap_rate']):.4f}"
            f" hsel={float(row['heuristic_selected_rate']):.4f}"
            f" steps={int(row['steps'])}",
            flush=True,
        )

    if int(workers) == 1:
        _init_worker(
            str(model_path),
            str(args.device),
            bool(args.action_mask),
            config,
        )
        for idx, task in enumerate(tasks, start=1):
            _record_row(idx, _evaluate_task(task))
    else:
        with ProcessPoolExecutor(
            max_workers=int(workers),
            initializer=_init_worker,
            initargs=(
                str(model_path),
                str(args.device),
                bool(args.action_mask),
                config,
            ),
        ) as pool:
            futures = [pool.submit(_evaluate_task, task) for task in tasks]
            for idx, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                _record_row(idx, row)

    rows.sort(key=lambda r: (int(r["threshold"]), str(r["map_id"]), int(r["repeat"])))
    summary = _aggregate_thresholds(
        rows,
        max_episode_steps=int(args.max_episode_steps),
        max_heuristic_selected_rate=float(args.max_heuristic_selected_rate),
        heuristic_penalty_weight=float(args.heuristic_penalty_weight),
    )
    eligible = [r for r in summary if bool(r["within_assist_budget"])]
    recommendation_pool = eligible if eligible else summary
    best = max(recommendation_pool, key=lambda r: float(r["score"])) if recommendation_pool else None

    long_fields = [
        "mode_name",
        "threshold",
        "heuristic_assist_enabled",
        "map_id",
        "map_file",
        "repeat",
        "steps",
        "coverage_ratio",
        "coverage_auc",
        "coverage_cells",
        "free_cells",
        "success_90",
        "success_95",
        "success_99",
        "step_to_90",
        "step_to_95",
        "step_to_99",
        "overlap_rate",
        "revisit_ratio",
        "path_overlap_rate",
        "heuristic_active_rate",
        "heuristic_active_count",
        "heuristic_selected_rate",
        "heuristic_selected_count",
        "heuristic_frontier_selected_rate",
        "heuristic_frontier_selected_count",
        "heuristic_known_uncovered_selected_rate",
        "heuristic_known_uncovered_selected_count",
        "heuristic_cached_selected_rate",
        "heuristic_cached_selected_count",
        "heuristic_override_rate",
        "heuristic_overridden_count",
        "action_override_rate",
        "unique_positions",
        "total_reward_sum",
        "done_reason",
    ]
    summary_fields = [
        "threshold",
        "heuristic_assist_enabled",
        "episodes",
        "mean_coverage_ratio",
        "mean_coverage_auc",
        "success_90_rate",
        "success_95_rate",
        "success_99_rate",
        "mean_step_to_90_penalized",
        "mean_step_to_95_penalized",
        "mean_step_to_99_penalized",
        "mean_steps",
        "mean_overlap_rate",
        "mean_revisit_ratio",
        "mean_heuristic_active_rate",
        "mean_heuristic_selected_rate",
        "mean_heuristic_frontier_selected_rate",
        "mean_heuristic_known_uncovered_selected_rate",
        "mean_heuristic_cached_selected_rate",
        "mean_heuristic_override_rate",
        "mean_action_override_rate",
        "score",
        "within_assist_budget",
    ]
    _write_csv(out_dir / "metrics_long.csv", rows, long_fields)
    _write_csv(out_dir / "threshold_summary.csv", summary, summary_fields)
    (out_dir / "metrics_long.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / "threshold_summary.json").write_text(
        json.dumps(
            {
                "recommended": best,
                "summary": summary,
                "selection_note": (
                    "Recommended threshold maximizes the heuristic-adjusted score. "
                    "If any threshold satisfies the selected-rate budget, thresholds outside "
                    "that budget are excluded from recommendation but still reported."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n[SUMMARY]")
    print(f"  out_dir: {out_dir}")
    if best is not None:
        print(
            "  recommended n:"
            f" {int(best['threshold'])}"
            f" | score={float(best['score']):.4f}"
            f" | cov={float(best['mean_coverage_ratio']):.4f}"
            f" | auc={float(best['mean_coverage_auc']):.4f}"
            f" | succ95={float(best['success_95_rate']):.4f}"
            f" | succ99={float(best['success_99_rate']):.4f}"
            f" | hsel={float(best['mean_heuristic_selected_rate']):.4f}"
        )
    print(f"  wrote: {out_dir / 'metrics_long.csv'}")
    print(f"  wrote: {out_dir / 'threshold_summary.csv'}")


if __name__ == "__main__":
    main()

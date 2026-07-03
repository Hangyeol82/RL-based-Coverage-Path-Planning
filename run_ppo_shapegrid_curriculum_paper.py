import argparse
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

from map_generators.macro_detail_grid import build_macro_detail_grid_map
from map_generators.shape_grid_presets import (
    PAPER_SHAPE_GRID_PRESETS as PRESETS,
    ShapeGridPreset,
    build_validated_shape_grid_map,
)
from map_generators.structured import build_room_corridor_map
from map_generators.trail_grid import build_trail_grid_map
from map_generators.validation import MapValidationStats, analyze_grid_map, map_passes_paper_checks


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
        "--generator-curriculum",
        type=str,
        default="shapegrid",
        choices=["shapegrid", "mixed_paper"],
        help=(
            "shapegrid: original paper shape-grid curriculum. "
            "mixed_paper: object-placement, trail-grid, and room-corridor curriculum."
        ),
    )
    p.add_argument(
        "--phase-timesteps",
        type=str,
        default="5000000,10000000,15000000,20000000",
        help=(
            "Mixed-paper fixed curriculum phase durations. Default gives "
            "0-5M, 5-15M, 15-30M, 30-50M for a 50M run."
        ),
    )
    p.add_argument(
        "--phase-level-probs",
        type=str,
        default="1:1.0;1:0.2,2:0.8;2:0.2,3:0.8;3:0.2,4:0.8",
        help=(
            "Mixed-paper level mixture per phase. Phases use ';', entries use ','. "
            "Example: '1:1;1:0.2,2:0.8;2:0.2,3:0.8;3:0.2,4:0.8'."
        ),
    )
    p.add_argument(
        "--family-weights",
        type=str,
        default="object:1,trail_grid:1,room_corridor:1",
        help=(
            "Mixed-paper map-family weights. 'object' internally samples "
            "shape_grid and macro_detail."
        ),
    )
    p.add_argument(
        "--object-generator-weights",
        type=str,
        default="shape_grid:1,macro_detail:1",
        help="Generator weights used inside the object-placement family.",
    )
    p.add_argument("--mixed-map-max-retries", type=int, default=64)
    p.add_argument("--mixed-min-start-component-ratio", type=float, default=0.995)
    p.add_argument("--mixed-min-free-ratio", type=float, default=0.05)
    p.add_argument("--mixed-max-obstacle-ratio", type=float, default=0.95)
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
    p.add_argument(
        "--maps-per-chunk",
        type=int,
        default=1,
        help="Generate this many maps per chunk and refresh maps on episode reset.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate map manifests and print runner commands without launching PPO.",
    )
    p.add_argument(
        "--dry-run-chunks",
        type=int,
        default=1,
        help="Number of chunks to materialize when --dry-run is set.",
    )
    p.add_argument(
        "--map-refresh-mode",
        type=str,
        default="cycle",
        choices=["cycle", "random"],
        help="Episode-level map-pool sampling mode when maps-per-chunk > 1.",
    )
    p.add_argument(
        "--episode-success-threshold",
        type=float,
        default=1.0,
        help="End episodes at this coverage threshold. 1.0 keeps full-coverage-only termination.",
    )
    p.add_argument("--sensor-range", type=int, default=2)
    p.add_argument(
        "--full-map-observation",
        action="store_true",
        help=(
            "Offline/full-known setting forwarded to run_ppo_sb3_paper.py. "
            "The true map is revealed as known_map at every step."
        ),
    )
    p.add_argument("--max-episode-steps", type=int, default=2000)
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
    p.add_argument("--coverage-hole-penalty-scale", type=float, default=0.0)
    burden_group = p.add_mutually_exclusive_group()
    burden_group.add_argument(
        "--revisit-burden-shaping",
        dest="revisit_burden_shaping",
        action="store_true",
    )
    burden_group.add_argument(
        "--no-revisit-burden-shaping",
        dest="revisit_burden_shaping",
        action="store_false",
    )
    p.add_argument("--revisit-burden-scale", type=float, default=0.0)
    p.add_argument("--revisit-burden-normalizer", type=float, default=0.0)
    p.add_argument("--revisit-burden-unreachable-cost", type=float, default=0.0)
    p.add_argument("--metric-stagnation-threshold", type=int, default=30)
    p.add_argument("--metric-loop-window", type=int, default=12)
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
        default="xlarge",
        choices=["small", "large", "xlarge"],
        help="Forwarded to run_ppo_sb3_paper.py encoder size preset.",
    )
    p.add_argument(
        "--dtm-coarse-mode",
        type=str,
        default="bfs",
        choices=["bfs", "aggregate", "aggregate_transfer"],
        help="DTM multi-scale mode forwarded to run_ppo_sb3_paper.py.",
    )
    p.add_argument(
        "--dtm-output-mode",
        type=str,
        default="six",
        choices=["six", "extent6", "two", "axis2", "axis2km", "four", "port12"],
        help="DTM output channels forwarded to run_ppo_sb3_paper.py.",
    )
    p.add_argument(
        "--obs-unknown-policy",
        type=str,
        default="keep",
        choices=["keep", "as_free", "as_obstacle"],
        help="Unknown-cell handling forwarded to run_ppo_sb3_paper.py map-observation builder.",
    )
    phase_group = p.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--cell-phase-channels",
        dest="cell_phase_channels",
        action="store_true",
        help="Forward per-level robot-in-cell phase features in robot_state.",
    )
    phase_group.add_argument(
        "--no-cell-phase-channels",
        dest="cell_phase_channels",
        action="store_false",
        help="Disable per-level robot-in-cell phase features.",
    )
    p.add_argument("--include-dtm", action="store_true")
    p.add_argument("--dtm-connectivity", type=int, default=4, choices=[4, 8])

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
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-4)

    mask_group = p.add_mutually_exclusive_group()
    mask_group.add_argument("--action-mask", dest="action_mask", action="store_true")
    mask_group.add_argument("--no-action-mask", dest="action_mask", action="store_false")
    heuristic_group = p.add_mutually_exclusive_group()
    heuristic_group.add_argument("--heuristic-assist", dest="heuristic_assist", action="store_true")
    heuristic_group.add_argument("--no-heuristic-assist", dest="heuristic_assist", action="store_false")
    p.add_argument("--heuristic-no-progress-threshold", type=int, default=50)
    p.add_argument("--heuristic-min-coverage", type=float, default=0.0)
    p.add_argument("--heuristic-max-astar-expansions", type=int, default=0)
    p.set_defaults(
        action_mask=True,
        milestone_reward=False,
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
    p.add_argument(
        "--resume-from-model",
        type=str,
        default="",
        help="Existing SB3 model zip to load before the first chunk of this run.",
    )
    p.add_argument(
        "--start-chunk-index",
        type=int,
        default=0,
        help=(
            "Number of chunks already completed before this run. "
            "Used for output chunk numbering, map seeds, and run seeds."
        ),
    )
    p.add_argument(
        "--start-timesteps",
        type=int,
        default=0,
        help="Timesteps already completed before this run, used for curriculum phase accounting.",
    )
    args = p.parse_args()
    args.robot_state_progress = False
    args.robot_state_stagnation = False
    return args


def _resolve_out_dir(args: argparse.Namespace, repo_root: Path) -> Path:
    if args.out_dir:
        p = Path(args.out_dir)
        return p if p.is_absolute() else (repo_root / p)
    tag = args.run_tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "mixed_curriculum" if args.generator_curriculum == "mixed_paper" else "shapegrid_curriculum"
    return repo_root / "learning" / "checkpoints" / "rl" / f"{prefix}_{tag}"


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


def _safe_mean_finite(rows: List[Dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and np.isfinite(float(r[key]))]
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


def _parse_weight_spec(
    raw: str,
    *,
    name: str,
    allowed: Sequence[str] | None = None,
) -> List[Tuple[str, float]]:
    pairs: List[Tuple[str, float]] = []
    allowed_set = set(allowed) if allowed is not None else None
    for tok in raw.split(","):
        s = tok.strip()
        if not s:
            continue
        sep = ":" if ":" in s else "=" if "=" in s else None
        if sep is None:
            raise ValueError(f"{name} entries must be key:weight or key=weight: {s}")
        key, weight_s = s.split(sep, 1)
        key = key.strip()
        if allowed_set is not None and key not in allowed_set:
            raise ValueError(f"Unsupported {name} key: {key}; allowed={sorted(allowed_set)}")
        weight = float(weight_s.strip())
        if weight < 0:
            raise ValueError(f"{name} weights must be non-negative")
        pairs.append((key, weight))
    if not pairs or sum(w for _k, w in pairs) <= 0.0:
        raise ValueError(f"{name} must contain at least one positive weight")
    return pairs


def _parse_level_prob_phases(raw: str) -> List[List[Tuple[int, float]]]:
    phases: List[List[Tuple[int, float]]] = []
    for phase_raw in raw.split(";"):
        s = phase_raw.strip()
        if not s:
            continue
        level_pairs: List[Tuple[int, float]] = []
        for tok in s.split(","):
            entry = tok.strip()
            if not entry:
                continue
            sep = ":" if ":" in entry else "=" if "=" in entry else None
            if sep is None:
                raise ValueError(f"phase-level-probs entries must be level:prob: {entry}")
            level_s, weight_s = entry.split(sep, 1)
            level = int(level_s.strip())
            if level not in PRESETS:
                raise ValueError(f"Unsupported mixed-paper level: {level}")
            weight = float(weight_s.strip())
            if weight < 0:
                raise ValueError("phase-level-probs weights must be non-negative")
            level_pairs.append((level, weight))
        if not level_pairs or sum(w for _level, w in level_pairs) <= 0.0:
            raise ValueError("Every phase in phase-level-probs must have a positive weight")
        phases.append(level_pairs)
    if not phases:
        raise ValueError("phase-level-probs must not be empty")
    return phases


def _phase_for_mixed_chunk(
    phase_timesteps: List[int],
    phase_level_probs: List[List[Tuple[int, float]]],
    chunk_start_t: int,
) -> Tuple[int, List[Tuple[int, float]]]:
    elapsed = 0
    phase_idx = 0
    for idx, duration in enumerate(phase_timesteps):
        elapsed += int(duration)
        if int(chunk_start_t) < elapsed:
            phase_idx = idx
            break
    else:
        phase_idx = len(phase_timesteps) - 1
    return phase_idx, phase_level_probs[min(phase_idx, len(phase_level_probs) - 1)]


def _normalised_prob_dict(pairs: Sequence[Tuple[object, float]]) -> Dict[str, float]:
    total = float(sum(float(w) for _k, w in pairs))
    return {str(k): float(w) / total for k, w in pairs}


def _jsonable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def _count_by(rows: List[Dict], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        val = str(row.get(key, ""))
        counts[val] = int(counts.get(val, 0) + 1)
    return counts


def _rollout_group_summary(rows: List[Dict], kind: str) -> Dict[str, Dict[str, float]]:
    prefix = f"{kind}_"
    out: Dict[str, Dict[str, float]] = {}
    metric_suffixes = (
        "episode_final_coverage_ratio",
        "episode_final_steps",
        "episode_revisit_ratio",
        "episode_overlap_ratio",
        "episode_coverage_auc",
        "success_90",
        "success_95",
        "success_99",
        "step_to_90",
        "step_to_95",
        "step_to_99",
    )
    for row in rows:
        group_counts: Dict[str, float] = {}
        count_prefix = f"episode_count_{kind}_"
        for key, val in row.items():
            if key.startswith(count_prefix):
                group_counts[key[len(count_prefix) :]] = float(val)
        for key, val in row.items():
            if key.startswith(count_prefix):
                group = key[len(count_prefix) :]
                out.setdefault(group, {})
                out[group]["episode_count"] = out[group].get("episode_count", 0.0) + float(val)
                continue
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix) :]
            metric = next((suffix for suffix in metric_suffixes if rest.endswith("_" + suffix)), "")
            if not metric:
                continue
            group = rest[: -(len(metric) + 1)]
            weight = float(group_counts.get(group, 1.0))
            out.setdefault(group, {}).setdefault(metric, 0.0)
            count_key = f"__count_{metric}"
            out[group][metric] += float(val) * weight
            out[group][count_key] = out[group].get(count_key, 0.0) + weight
    for group, metrics in out.items():
        for key in list(metrics.keys()):
            if key.startswith("__count_"):
                continue
            count = metrics.get(f"__count_{key}", 0.0)
            if count > 0.0:
                metrics[key] = float(metrics[key]) / float(count)
        for key in [k for k in metrics if k.startswith("__count_")]:
            metrics.pop(key, None)
    return out


def _allocate_weighted_counts(total: int, pairs: Sequence[Tuple[object, float]]) -> Dict[object, int]:
    n = max(0, int(total))
    weights = np.asarray([float(w) for _k, w in pairs], dtype=np.float64)
    probs = weights / float(np.sum(weights))
    raw = probs * float(n)
    floors = np.floor(raw).astype(np.int64)
    counts = {pairs[idx][0]: int(floors[idx]) for idx in range(len(pairs))}
    remaining = int(n - int(np.sum(floors)))
    if remaining > 0:
        remainders = raw - floors
        order = np.argsort(-remainders)
        for idx in order[:remaining]:
            key = pairs[int(idx)][0]
            counts[key] = int(counts.get(key, 0) + 1)
    return counts


def _expand_weighted_plan(
    total: int,
    pairs: Sequence[Tuple[object, float]],
    *,
    rng: np.random.Generator,
) -> List[object]:
    counts = _allocate_weighted_counts(total, pairs)
    items: List[object] = []
    for key, _weight in pairs:
        items.extend([key] * int(counts.get(key, 0)))
    if items:
        order = rng.permutation(len(items))
        items = [items[int(idx)] for idx in order]
    return items


def _build_balanced_mixed_spec_plan(
    *,
    maps_per_chunk: int,
    level_probs: List[Tuple[int, float]],
    family_weights: List[Tuple[str, float]],
    object_generator_weights: List[Tuple[str, float]],
    rng: np.random.Generator,
) -> List[Tuple[str, str, int]]:
    families = _expand_weighted_plan(maps_per_chunk, family_weights, rng=rng)
    levels = [int(v) for v in _expand_weighted_plan(maps_per_chunk, level_probs, rng=rng)]
    family_generators: List[Tuple[str, str]] = []
    object_count = int(sum(1 for family in families if str(family) == "object"))
    object_generators = _expand_weighted_plan(
        object_count,
        object_generator_weights,
        rng=rng,
    )
    object_cursor = 0
    for family_raw in families:
        family = str(family_raw)
        if family == "object":
            generator = str(object_generators[object_cursor])
            object_cursor += 1
            family_generators.append(("object_placement", generator))
        else:
            family_generators.append((family, family))
    if len(family_generators) != len(levels):
        raise RuntimeError("Internal mixed curriculum plan size mismatch")
    return [
        (family, generator, int(level))
        for (family, generator), level in zip(family_generators, levels)
    ]


def _build_non_shape_grid_map(
    *,
    generator: str,
    size: int,
    seed: int,
    level: int,
):
    if generator == "macro_detail":
        return build_macro_detail_grid_map(
            size=int(size),
            seed=int(seed),
            level=int(level),
            return_metadata=True,
        )
    if generator == "trail_grid":
        return build_trail_grid_map(
            size=int(size),
            seed=int(seed),
            level=int(level),
            return_metadata=True,
        )
    if generator == "room_corridor":
        return build_room_corridor_map(
            size=int(size),
            seed=int(seed),
            level=int(level),
            return_metadata=True,
        )
    raise ValueError(f"Unsupported mixed-paper generator: {generator}")


def _build_validated_curriculum_map(
    *,
    generator: str,
    family: str,
    size: int,
    seed: int,
    level: int,
    max_retries: int,
    min_start_component_ratio: float,
    min_free_ratio: float,
    max_obstacle_ratio: float,
) -> Tuple[np.ndarray, MapValidationStats, int, int, Dict]:
    if generator == "shape_grid":
        grid, stats, used_seed, attempt = build_validated_shape_grid_map(
            size=int(size),
            seed=int(seed),
            level=int(level),
            max_retries=int(max_retries),
            min_start_component_ratio=float(min_start_component_ratio),
            min_free_ratio=float(min_free_ratio),
            max_obstacle_ratio=float(max_obstacle_ratio),
        )
        meta = {
            "family": str(family),
            "generator": "shape_grid",
            "level": int(level),
            "shape_grid_preset": PRESETS[int(level)].name,
        }
        return grid, stats, int(used_seed), int(attempt), meta

    last_stats = None
    last_meta = None
    for attempt in range(max(1, int(max_retries))):
        candidate_seed = int(seed) + attempt * 1_000_003
        grid, source_meta = _build_non_shape_grid_map(
            generator=str(generator),
            size=int(size),
            seed=int(candidate_seed),
            level=int(level),
        )
        stats = analyze_grid_map(grid, start=(0, 0))
        if map_passes_paper_checks(
            stats,
            min_start_component_ratio=float(min_start_component_ratio),
            min_free_ratio=float(min_free_ratio),
            max_obstacle_ratio=float(max_obstacle_ratio),
        ):
            meta = {
                "family": str(family),
                "generator": str(generator),
                "level": int(level),
                "source": _jsonable(source_meta),
            }
            return np.asarray(grid, dtype=np.int32), stats, int(candidate_seed), int(attempt), meta
        last_stats = stats
        last_meta = source_meta

    raise RuntimeError(
        "Failed to generate a valid mixed-paper map "
        f"family={family} generator={generator} level={level} seed={seed} "
        f"after {max_retries} retries; "
        f"last_stats={last_stats.as_dict() if last_stats is not None else None}; "
        f"last_meta={_jsonable(last_meta)}"
    )


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
    map_pool_dir: Path | None,
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
        "--gamma",
        str(float(args.gamma)),
        "--gae-lambda",
        str(float(args.gae_lambda)),
        "--clip-range",
        str(float(args.clip_range)),
        "--ent-coef",
        str(float(args.ent_coef)),
        "--vf-coef",
        str(float(args.vf_coef)),
        "--lr",
        str(float(args.lr)),
        "--verbose",
        "0",
        "--sensor-range",
        str(int(args.sensor_range)),
        "--max-episode-steps",
        str(int(args.max_episode_steps)),
        "--robot-state-action-history-len",
        str(int(args.robot_state_action_history_len)),
        "--coverage-hole-penalty-scale",
        str(float(args.coverage_hole_penalty_scale)),
        "--revisit-burden-scale",
        str(float(args.revisit_burden_scale)),
        "--revisit-burden-normalizer",
        str(float(args.revisit_burden_normalizer)),
        "--revisit-burden-unreachable-cost",
        str(float(args.revisit_burden_unreachable_cost)),
        "--metric-stagnation-threshold",
        str(int(args.metric_stagnation_threshold)),
        "--metric-loop-window",
        str(int(args.metric_loop_window)),
        "--heuristic-no-progress-threshold",
        str(int(args.heuristic_no_progress_threshold)),
        "--heuristic-min-coverage",
        str(float(args.heuristic_min_coverage)),
        "--heuristic-max-astar-expansions",
        str(int(args.heuristic_max_astar_expansions)),
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
        "--dtm-connectivity",
        str(int(args.dtm_connectivity)),
        "--map-source",
        "file",
        "--map-file",
        str(map_file),
        "--map-refresh-mode",
        str(args.map_refresh_mode),
        "--episode-success-threshold",
        str(float(args.episode_success_threshold)),
        "--map-size",
        str(int(args.map_size)),
        "--save-model",
        str(save_model),
        "--save-breakdown-json",
        str(save_json),
        "--save-breakdown-csv",
        str(save_csv),
    ]
    if args.full_map_observation:
        cmd.append("--full-map-observation")
    if args.include_dtm:
        cmd.append("--include-dtm")
    if int(args.maps_per_chunk) > 1:
        if map_pool_dir is None:
            raise ValueError("map_pool_dir is required when maps-per-chunk > 1")
        cmd += ["--map-pool-dir", str(map_pool_dir)]
        cmd.append("--episode-map-refresh")
    if args.cell_phase_channels:
        cmd.append("--cell-phase-channels")
    else:
        cmd.append("--no-cell-phase-channels")
    if args.action_mask:
        cmd.append("--action-mask")
    else:
        cmd.append("--no-action-mask")
    if args.heuristic_assist:
        cmd.append("--heuristic-assist")
    else:
        cmd.append("--no-heuristic-assist")
    if args.robot_state_position:
        cmd.append("--robot-state-position")
    else:
        cmd.append("--no-robot-state-position")
    if args.robot_state_action_history:
        cmd.append("--robot-state-action-history")
    else:
        cmd.append("--no-robot-state-action-history")
    if args.robot_state_progress:
        cmd.append("--robot-state-progress")
    else:
        cmd.append("--no-robot-state-progress")
    if args.robot_state_stagnation:
        cmd.append("--robot-state-stagnation")
    else:
        cmd.append("--no-robot-state-stagnation")
    if args.hole_signals:
        cmd.append("--hole-signals")
    else:
        cmd.append("--no-hole-signals")
    if args.revisit_burden_shaping:
        cmd.append("--revisit-burden-shaping")
    else:
        cmd.append("--no-revisit-burden-shaping")
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
    if args.generator_curriculum == "mixed_paper" and args.curriculum_mode != "fixed":
        raise ValueError("--generator-curriculum mixed_paper currently supports --curriculum-mode fixed only")
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
    if args.maps_per_chunk <= 0:
        raise ValueError("--maps-per-chunk must be positive")
    if args.mixed_map_max_retries <= 0:
        raise ValueError("--mixed-map-max-retries must be positive")
    if args.dry_run_chunks <= 0:
        raise ValueError("--dry-run-chunks must be positive")
    if not (0.0 < float(args.episode_success_threshold) <= 1.0):
        raise ValueError("--episode-success-threshold must be in (0, 1]")
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.start_chunk_index < 0:
        raise ValueError("--start-chunk-index must be non-negative")
    if args.start_timesteps < 0:
        raise ValueError("--start-timesteps must be non-negative")
    if args.resume_from_model.strip() and args.init_from_bc.strip():
        raise ValueError("--resume-from-model and --init-from-bc cannot be used together")

    phase_levels = _parse_int_list(args.phase_levels, name="phase-levels")
    phase_timesteps = _parse_int_list(args.phase_timesteps, name="phase-timesteps")
    if any(int(v) <= 0 for v in phase_timesteps):
        raise ValueError("--phase-timesteps entries must be positive")
    phase_level_probs = _parse_level_prob_phases(args.phase_level_probs)
    family_weights = _parse_weight_spec(
        args.family_weights,
        name="family-weights",
        allowed=("object", "trail_grid", "room_corridor"),
    )
    object_generator_weights = _parse_weight_spec(
        args.object_generator_weights,
        name="object-generator-weights",
        allowed=("shape_grid", "macro_detail"),
    )
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
    runner = repo_root / "run_ppo_sb3_paper.py"
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
        for p in (
            list(models_dir.glob("*.zip"))
            + list(logs_dir.glob("*"))
            + list(maps_dir.glob("*.txt"))
            + list(report_dir.glob("*.json"))
            + list(report_dir.glob("*.jsonl"))
        ):
            p.unlink()

    bc_ckpt = ""
    if args.init_from_bc.strip():
        p = Path(args.init_from_bc)
        if not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            raise FileNotFoundError(f"BC checkpoint not found: {p}")
        bc_ckpt = str(p)

    resume_model_zip = ""
    if args.resume_from_model.strip():
        p = Path(args.resume_from_model)
        if not p.is_absolute():
            p = repo_root / p
        if not p.exists():
            raise FileNotFoundError(f"Resume model not found: {p}")
        resume_model_zip = str(p)

    total = int(args.total_timesteps)
    chunk = int(args.chunk_timesteps)
    num_chunks = int(math.ceil(total / float(chunk)))
    if args.dry_run:
        num_chunks = int(min(num_chunks, max(1, int(args.dry_run_chunks))))
    rollout_quantum = max(1, int(args.num_envs) * int(args.n_steps))
    actual_full_chunk_steps = int(math.ceil(chunk / float(rollout_quantum)) * rollout_quantum)
    per_env_steps_per_full_chunk = actual_full_chunk_steps / float(max(1, int(args.num_envs)))
    if int(args.max_episode_steps) > per_env_steps_per_full_chunk:
        print(
            "[WARN] max_episode_steps exceeds the rollout steps available to one environment "
            f"inside a full chunk ({int(args.max_episode_steps)} > "
            f"{per_env_steps_per_full_chunk:.0f}). Episodes that do not finish early will be "
            "discarded at the chunk process boundary, so terminal episode metrics may be absent. "
            "Increase --chunk-timesteps or lower --max-episode-steps.",
            flush=True,
        )
    progress_jsonl = report_dir / "progress.jsonl"
    manifest_jsonl = report_dir / "map_manifest.jsonl"

    print(
        f"[INFO] total={total} chunk={chunk} mode={args.curriculum_mode} "
        f"generator_curriculum={args.generator_curriculum} "
        f"stage={args.stage_timesteps} chunks={num_chunks} "
        f"num_envs={args.num_envs} vec={args.vec_env} "
        f"observation={'offline_full_map' if args.full_map_observation else 'online_partial'}"
    , flush=True)
    print(f"[INFO] out_dir={out_dir}", flush=True)
    if args.generator_curriculum == "mixed_paper":
        print(
            "[INFO] mixed-paper schedule:"
            f" phase_timesteps={phase_timesteps},"
            f" phase_level_probs={[ _normalised_prob_dict(pairs) for pairs in phase_level_probs ]},"
            f" family_probs={_normalised_prob_dict(family_weights)},"
            f" object_generator_probs={_normalised_prob_dict(object_generator_weights)}",
            flush=True,
        )
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
    prev_model_zip = resume_model_zip

    for i in range(num_chunks):
        global_chunk_idx = int(args.start_chunk_index) + int(i)
        display_chunk = int(global_chunk_idx + 1)
        chunk_start = int(args.start_timesteps) + int(done_steps)
        chunk_t = int(min(chunk, total - done_steps))
        maps_per_chunk = max(1, int(args.maps_per_chunk))
        map_seed = int(
            args.seed
            if args.map_seed_mode == "fixed"
            else (args.seed + global_chunk_idx * maps_per_chunk)
        )
        run_seed = int(args.seed + global_chunk_idx)

        map_pool_dir = maps_dir / f"chunk{display_chunk:02d}_pool"
        map_pool_dir.mkdir(parents=True, exist_ok=True)
        for old_map in map_pool_dir.glob("*.txt"):
            old_map.unlink()
        map_files: List[Path] = []
        obs_ratios: List[float] = []
        used_map_seeds: List[int] = []
        start_component_ratios: List[float] = []
        map_records: List[Dict] = []
        level_prob_dict: Dict[str, float] = {}

        if args.generator_curriculum == "shapegrid":
            if args.curriculum_mode == "fixed":
                preset = _preset_for_chunk(phase_levels, int(args.stage_timesteps), chunk_start)
                stage_idx = int(chunk_start // int(args.stage_timesteps))
            else:
                stage_idx = int(adaptive_stage_idx)
                level = int(phase_levels[min(stage_idx, len(phase_levels) - 1)])
                if level not in PRESETS:
                    raise ValueError(f"Unsupported shape-grid level in phase-levels: {level}")
                preset = PRESETS[level]
            curriculum_desc = f"L{preset.level}({preset.name})"
            curriculum_name = preset.name
            curriculum_level = int(preset.level)
            level_prob_dict = {str(preset.level): 1.0}

            for j in range(maps_per_chunk):
                this_seed = int(map_seed if args.map_seed_mode == "fixed" else (map_seed + j))
                grid, stats, used_seed, attempt = build_validated_shape_grid_map(
                    size=int(args.map_size),
                    seed=this_seed,
                    level=int(preset.level),
                )
                if maps_per_chunk == 1:
                    map_txt = maps_dir / f"chunk{display_chunk:02d}_L{preset.level}_{preset.name}_seed{this_seed}.txt"
                else:
                    map_txt = map_pool_dir / (
                        f"map{j+1:03d}_L{preset.level}_{preset.name}_seed{this_seed}.txt"
                    )
                _write_map_txt(map_txt, grid)
                map_files.append(map_txt)
                obs_ratios.append(float(stats.obstacle_ratio))
                used_map_seeds.append(int(used_seed))
                start_component_ratios.append(float(stats.start_component_ratio))
                map_records.append(
                    {
                        "chunk": int(i + 1),
                        "global_chunk": int(display_chunk),
                        "map_index": int(j + 1),
                        "family": "object_placement",
                        "generator": "shape_grid",
                        "level": int(preset.level),
                        "seed_requested": int(this_seed),
                        "seed_used": int(used_seed),
                        "attempt": int(attempt),
                        "path": str(map_txt),
                        "free_ratio": float(stats.free_ratio),
                        "obstacle_ratio": float(stats.obstacle_ratio),
                        "start_component_ratio": float(stats.start_component_ratio),
                    }
                )
                if int(used_seed) != int(this_seed):
                    print(
                        f"[MAP] requested_seed={this_seed} replaced_by={used_seed} "
                        f"for validated L{preset.level} map",
                        flush=True,
                    )
        else:
            stage_idx, level_probs = _phase_for_mixed_chunk(
                phase_timesteps,
                phase_level_probs,
                chunk_start,
            )
            level_prob_dict = _normalised_prob_dict(level_probs)
            curriculum_desc = f"mixed_paper_phase{stage_idx + 1}_levels={level_prob_dict}"
            curriculum_name = "mixed_paper"
            curriculum_level = -1
            spec_plan = _build_balanced_mixed_spec_plan(
                maps_per_chunk=maps_per_chunk,
                level_probs=level_probs,
                family_weights=family_weights,
                object_generator_weights=object_generator_weights,
                rng=np.random.default_rng(map_seed),
            )

            for j in range(maps_per_chunk):
                this_seed = int(map_seed if args.map_seed_mode == "fixed" else (map_seed + j))
                family, generator, level = spec_plan[j]
                generator_seed = int(this_seed + 7919)
                grid, stats, used_seed, attempt, meta = _build_validated_curriculum_map(
                    generator=generator,
                    family=family,
                    size=int(args.map_size),
                    seed=generator_seed,
                    level=int(level),
                    max_retries=int(args.mixed_map_max_retries),
                    min_start_component_ratio=float(args.mixed_min_start_component_ratio),
                    min_free_ratio=float(args.mixed_min_free_ratio),
                    max_obstacle_ratio=float(args.mixed_max_obstacle_ratio),
                )
                stem = f"map{j+1:03d}_P{stage_idx+1}_L{level}_{generator}_seed{this_seed}"
                map_txt = maps_dir / f"chunk{display_chunk:02d}_{stem}.txt" if maps_per_chunk == 1 else map_pool_dir / f"{stem}.txt"
                _write_map_txt(map_txt, grid)
                map_files.append(map_txt)
                obs_ratios.append(float(stats.obstacle_ratio))
                used_map_seeds.append(int(used_seed))
                start_component_ratios.append(float(stats.start_component_ratio))
                map_records.append(
                    {
                        "chunk": int(i + 1),
                        "global_chunk": int(display_chunk),
                        "map_index": int(j + 1),
                        "family": str(family),
                        "generator": str(generator),
                        "level": int(level),
                        "phase_index": int(stage_idx),
                        "phase_level_probs": level_prob_dict,
                        "seed_requested": int(this_seed),
                        "seed_generator": int(generator_seed),
                        "seed_used": int(used_seed),
                        "attempt": int(attempt),
                        "path": str(map_txt),
                        "free_ratio": float(stats.free_ratio),
                        "obstacle_ratio": float(stats.obstacle_ratio),
                        "start_component_ratio": float(stats.start_component_ratio),
                        "meta": _jsonable(meta),
                    }
                )
                if int(used_seed) != int(generator_seed):
                    print(
                        f"[MAP] requested_seed={generator_seed} replaced_by={used_seed} "
                        f"for validated {generator} L{level} map",
                        flush=True,
                    )
        map_txt = map_files[0]
        obs_ratio = float(np.mean(np.asarray(obs_ratios, dtype=np.float64)))
        family_counts = _count_by(map_records, "family")
        generator_counts = _count_by(map_records, "generator")
        level_counts = _count_by(map_records, "level")
        with manifest_jsonl.open("a", encoding="utf-8") as f:
            for row in map_records:
                f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")
        pool_manifest_jsonl = map_pool_dir / "manifest.jsonl"
        with pool_manifest_jsonl.open("w", encoding="utf-8") as f:
            for row in map_records:
                f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")

        save_model_base = models_dir / f"chunk{display_chunk:02d}"
        save_model_zip = save_model_base.with_suffix(".zip")
        save_json = logs_dir / f"chunk{display_chunk:02d}.json"
        save_csv = logs_dir / f"chunk{display_chunk:02d}.csv"

        init_bc = bc_ckpt if i == 0 and bc_ckpt and not resume_model_zip else ""
        load_model = prev_model_zip if (i > 0 or resume_model_zip) else ""
        cmd = _build_run_cmd(
            args,
            runner=runner,
            chunk_timesteps=chunk_t,
            run_seed=run_seed,
            map_file=map_txt,
            map_pool_dir=map_pool_dir if maps_per_chunk > 1 else None,
            save_model=save_model_base,
            save_json=save_json,
            save_csv=save_csv,
            init_from_bc=init_bc,
            load_model=load_model,
        )

        print(
            f"\n[CHUNK {display_chunk} local={i+1}/{num_chunks}] steps={chunk_t} "
            f"curriculum={curriculum_desc} map_seed={map_seed} "
            f"maps={maps_per_chunk} obs_ratio_mean={obs_ratio:.3f} "
            f"family_counts={family_counts} generator_counts={generator_counts} "
            f"level_counts={level_counts}"
        , flush=True)
        print(f"[RUN] {' '.join(cmd)}", flush=True)

        if args.dry_run:
            done_steps += chunk_t
            rec = {
                "chunk": i + 1,
                "global_chunk": int(display_chunk),
                "status": "dry_run",
                "chunk_timesteps": int(chunk_t),
                "timesteps_done": int(args.start_timesteps + done_steps),
                "generator_curriculum": str(args.generator_curriculum),
                "curriculum_mode": str(args.curriculum_mode),
                "curriculum_stage_index": int(stage_idx),
                "curriculum_level": int(curriculum_level),
                "curriculum_name": curriculum_name,
                "curriculum_level_probs": level_prob_dict,
                "map_seed": int(map_seed),
                "used_map_seeds": used_map_seeds,
                "maps_per_chunk": int(maps_per_chunk),
                "episode_map_refresh": bool(maps_per_chunk > 1),
                "run_seed": int(run_seed),
                "resume_from_model": str(resume_model_zip),
                "map_file": str(map_txt),
                "map_obstacle_ratio": float(obs_ratio),
                "map_start_component_ratio_mean": float(
                    np.mean(np.asarray(start_component_ratios, dtype=np.float64))
                ),
                "map_family_counts": family_counts,
                "map_generator_counts": generator_counts,
                "map_level_counts": level_counts,
            }
            progress_rows.append(rec)
            with progress_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(_jsonable(rec), ensure_ascii=False) + "\n")
            continue

        train_t0 = perf_counter()
        try:
            subprocess.run(cmd, cwd=str(repo_root), check=True)
        except subprocess.CalledProcessError as e:
            train_wall_time_sec = float(perf_counter() - train_t0)
            failed = True
            rec = {
                "chunk": i + 1,
                "global_chunk": int(display_chunk),
                "status": "failed",
                "returncode": int(e.returncode),
                "steps_done": int(args.start_timesteps + done_steps),
                "chunk_wall_time_sec": float(train_wall_time_sec),
            }
            progress_rows.append(rec)
            if not args.continue_on_error:
                break
            continue
        train_wall_time_sec = float(perf_counter() - train_t0)

        if not save_model_zip.exists():
            failed = True
            rec = {
                "chunk": i + 1,
                "global_chunk": int(display_chunk),
                "status": "failed_no_model",
                "steps_done": int(args.start_timesteps + done_steps),
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
            "global_chunk": int(display_chunk),
            "status": "ok",
            "chunk_timesteps": int(chunk_t),
            "timesteps_done": int(args.start_timesteps + done_steps),
            "generator_curriculum": str(args.generator_curriculum),
            "curriculum_mode": str(args.curriculum_mode),
            "curriculum_stage_index": int(stage_idx),
            "curriculum_level": int(curriculum_level),
            "curriculum_name": curriculum_name,
            "curriculum_level_probs": level_prob_dict,
            "map_seed": int(map_seed),
            "used_map_seeds": used_map_seeds,
            "maps_per_chunk": int(maps_per_chunk),
            "episode_map_refresh": bool(maps_per_chunk > 1),
            "run_seed": int(run_seed),
            "resume_from_model": str(resume_model_zip),
            "chunk_wall_time_sec": float(train_wall_time_sec),
            "chunk_env_steps_per_sec": float(chunk_t) / float(max(train_wall_time_sec, 1e-9)),
            "map_file": str(map_txt),
            "map_obstacle_cells": int(round(float(obs_ratio) * float(args.map_size * args.map_size))),
            "map_obstacle_ratio": float(obs_ratio),
            "map_start_component_ratio_mean": float(
                np.mean(np.asarray(start_component_ratios, dtype=np.float64))
            ),
            "map_family_counts": family_counts,
            "map_generator_counts": generator_counts,
            "map_level_counts": level_counts,
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
        rec["chunk_mean_episode_final_steps"] = _safe_mean_finite(rollouts, "episode_final_steps")
        rec["chunk_mean_episode_revisit_ratio"] = _safe_mean_finite(rollouts, "episode_revisit_ratio")
        rec["chunk_mean_episode_overlap_ratio"] = _safe_mean_finite(rollouts, "episode_overlap_ratio")
        rec["chunk_mean_episode_coverage_auc"] = _safe_mean_finite(rollouts, "episode_coverage_auc")
        for suffix in ("90", "95", "99"):
            rec[f"chunk_mean_success_{suffix}"] = _safe_mean_finite(rollouts, f"success_{suffix}")
            rec[f"chunk_mean_step_to_{suffix}"] = _safe_mean_finite(rollouts, f"step_to_{suffix}")
            rec[f"chunk_step_to_{suffix}_sample_count"] = int(
                sum(
                    1
                    for row in rollouts
                    if f"step_to_{suffix}" in row and np.isfinite(float(row[f"step_to_{suffix}"]))
                )
            )
        rec["cum_mean_episode_final_steps"] = _safe_mean_finite(all_rollouts, "episode_final_steps")
        rec["cum_mean_episode_revisit_ratio"] = _safe_mean_finite(all_rollouts, "episode_revisit_ratio")
        rec["cum_mean_episode_overlap_ratio"] = _safe_mean_finite(all_rollouts, "episode_overlap_ratio")
        rec["cum_mean_episode_coverage_auc"] = _safe_mean_finite(all_rollouts, "episode_coverage_auc")
        for suffix in ("90", "95", "99"):
            rec[f"cum_mean_success_{suffix}"] = _safe_mean_finite(all_rollouts, f"success_{suffix}")
            rec[f"cum_mean_step_to_{suffix}"] = _safe_mean_finite(all_rollouts, f"step_to_{suffix}")
        rec["chunk_episode_by_family"] = _rollout_group_summary(rollouts, "family")
        rec["chunk_episode_by_generator"] = _rollout_group_summary(rollouts, "generator")
        rec["chunk_episode_by_level"] = _rollout_group_summary(rollouts, "level")
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

        with progress_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(rec), ensure_ascii=False) + "\n")

    summary = {
        "config": _jsonable(vars(args)),
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

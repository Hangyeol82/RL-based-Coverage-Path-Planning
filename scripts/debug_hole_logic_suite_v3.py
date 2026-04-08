from __future__ import annotations

import importlib.util
import json
import sys
import types
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path("/Users/ihangyeol/Desktop/졸업 작품/RL based online CPP")
HEUR_ROOT = Path("/Users/ihangyeol/Desktop/졸업 작품/RL based online CPP.heuristic")
OUT_DIR = WORKSPACE / "log_analysis" / "logs" / "_tmp_hole_logic_suite_v3_20260407"

# Display encoding:
# -1: obstacle
#  0: known free and uncovered
#  1: covered
#  2: unknown
#  9: robot
Grid = np.ndarray
ACTION_TO_DELTA = {
    0: (-1, 0),  # up
    1: (0, 1),  # right
    2: (1, 0),  # down
    3: (0, -1),  # left
}
ACTION_NAMES = ["up", "right", "down", "left"]


@dataclass(frozen=True)
class HoleCase:
    name: str
    note: str
    grid: Grid
    expected_risk: Tuple[int, int, int, int]
    expected_hole_count: int


def _load_cpp_env():
    if "learning.reinforcement" not in sys.modules:
        pkg = types.ModuleType("learning.reinforcement")
        pkg.__path__ = [str(HEUR_ROOT / "learning" / "reinforcement")]
        sys.modules["learning.reinforcement"] = pkg

    obs_pkg = types.ModuleType("learning.observation")

    class DummyBuilder:
        def __init__(self, *args, **kwargs):
            pass

    class DummyConfig:
        def __init__(self, *args, **kwargs):
            pass

    obs_pkg.MultiScaleCPPObservationBuilder = DummyBuilder
    obs_pkg.MultiScaleCPPObservationConfig = DummyConfig
    obs_pkg.RobotStateObservationBuilder = DummyBuilder
    obs_pkg.RobotStateObservationConfig = DummyConfig
    sys.modules["learning.observation"] = obs_pkg

    reward_spec = importlib.util.spec_from_file_location(
        "learning.reinforcement.reward",
        str(HEUR_ROOT / "learning" / "reinforcement" / "reward.py"),
    )
    reward_mod = importlib.util.module_from_spec(reward_spec)
    sys.modules["learning.reinforcement.reward"] = reward_mod
    assert reward_spec.loader is not None
    reward_spec.loader.exec_module(reward_mod)

    env_spec = importlib.util.spec_from_file_location(
        "learning.reinforcement.cpp_env",
        str(HEUR_ROOT / "learning" / "reinforcement" / "cpp_env.py"),
    )
    env_mod = importlib.util.module_from_spec(env_spec)
    sys.modules["learning.reinforcement.cpp_env"] = env_mod
    assert env_spec.loader is not None
    env_spec.loader.exec_module(env_mod)
    return env_mod.CPPDiscreteEnv


CPPDiscreteEnv = _load_cpp_env()


def _display_to_state(grid: Grid):
    robot_pos = tuple(map(int, np.argwhere(grid == 9)[0]))
    known_map = np.where(grid == -1, 1, np.where(grid == 2, -1, 0)).astype(np.int8)
    explored = np.isin(grid, [1, 9])
    return robot_pos, known_map, explored


def _make_env(grid: Grid):
    robot_pos, known_map, explored = _display_to_state(grid)
    env = object.__new__(CPPDiscreteEnv)
    env.rows, env.cols = known_map.shape
    env.known_map = known_map
    env.explored = explored
    env.current_pos = robot_pos
    env._hole_component_data = None
    env._hole_stats_cache = None
    env._hole_stats_pos = None
    env._refresh_hole_cache(robot_pos)
    return env


def _reference_component_data(known_map: np.ndarray, explored: np.ndarray):
    rows, cols = known_map.shape
    known_free = known_map == 0
    uncovered_known = known_free & (~explored)
    unknown = known_map == -1
    open_base = unknown | uncovered_known

    comp_id = np.full((rows, cols), -1, dtype=np.int32)
    comp_known_mass: List[int] = []
    comp_open_mass: List[int] = []
    comp_touches_boundary: List[bool] = []
    next_id = 0

    for sr, sc in np.argwhere(open_base):
        sr = int(sr)
        sc = int(sc)
        if comp_id[sr, sc] >= 0:
            continue
        q = deque([(sr, sc)])
        comp_id[sr, sc] = next_id
        known_mass = 0
        open_mass = 0
        touches_boundary = sr == 0 or sr == rows - 1 or sc == 0 or sc == cols - 1
        while q:
            cr, cc = q.popleft()
            open_mass += 1
            if uncovered_known[cr, cc]:
                known_mass += 1
            for dr, dc in ACTION_TO_DELTA.values():
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < rows and 0 <= nc < cols):
                    continue
                if not open_base[nr, nc] or comp_id[nr, nc] >= 0:
                    continue
                comp_id[nr, nc] = next_id
                if nr == 0 or nr == rows - 1 or nc == 0 or nc == cols - 1:
                    touches_boundary = True
                q.append((nr, nc))

        comp_known_mass.append(known_mass)
        comp_open_mass.append(open_mass)
        comp_touches_boundary.append(touches_boundary)
        next_id += 1

    return {
        "open_base": open_base,
        "comp_id": comp_id,
        "comp_known_mass": np.asarray(comp_known_mass, dtype=np.int32),
        "comp_open_mass": np.asarray(comp_open_mass, dtype=np.int32),
        "comp_touches_boundary": np.asarray(comp_touches_boundary, dtype=bool),
    }


def _reference_hole_stats(
    robot_pos: Tuple[int, int], known_map: np.ndarray, explored: np.ndarray
) -> Dict[str, float]:
    rows, cols = known_map.shape
    data = _reference_component_data(known_map, explored)
    open_base = data["open_base"]
    comp_id = data["comp_id"]
    comp_known_mass = data["comp_known_mass"]
    comp_open_mass = data["comp_open_mass"]
    comp_touches_boundary = data["comp_touches_boundary"]

    rr, cc = robot_pos
    active = set()
    if 0 <= rr < rows and 0 <= cc < cols and open_base[rr, cc]:
        cid = int(comp_id[rr, cc])
        if cid >= 0:
            active.add(cid)
    for dr, dc in ACTION_TO_DELTA.values():
        nr, nc = rr + dr, cc + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        if not open_base[nr, nc]:
            continue
        cid = int(comp_id[nr, nc])
        if cid >= 0:
            active.add(cid)

    hole_count = 0.0
    hole_known_mass = 0.0
    hole_open_mass = 0.0
    for cid in range(len(comp_known_mass)):
        if comp_known_mass[cid] <= 0:
            continue
        if cid in active:
            continue
        if bool(comp_touches_boundary[cid]):
            continue
        hole_count += 1.0
        hole_known_mass += float(comp_known_mass[cid])
        hole_open_mass += float(comp_open_mass[cid])

    return {
        "coverage_hole_count": hole_count,
        "coverage_hole_known_mass": hole_known_mass,
        "coverage_hole_open_mass": hole_open_mass,
    }


def _reference_hole_signal(robot_pos: Tuple[int, int], known_map: np.ndarray, explored: np.ndarray):
    rows, cols = known_map.shape
    current = _reference_hole_stats(robot_pos, known_map, explored)
    current_count = float(current["coverage_hole_count"])
    risk = np.zeros(4, dtype=np.float32)
    rr, cc = robot_pos
    for action, (dr, dc) in ACTION_TO_DELTA.items():
        nr, nc = rr + dr, cc + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        if known_map[nr, nc] == 1:
            continue
        next_stats = _reference_hole_stats((nr, nc), known_map, explored)
        risk[action] = 1.0 if float(next_stats["coverage_hole_count"]) > current_count else 0.0
    return risk


def _curated_cases() -> List[HoleCase]:
    return [
        HoleCase(
            name="enlarged_original_like",
            note="Still open to exterior. No warning yet.",
            grid=np.array(
                [
                    [2, 2, 2, 2, 2, 2, 2],
                    [1, 1, 1, 1, 1, 1, 2],
                    [0, 0, 0, 0, 1, 1, 2],
                    [0, 0, 9, 0, 1, 1, 2],
                    [0, 0, 0, 0, 1, 1, 2],
                    [0, 1, 1, 1, 1, 1, 2],
                    [2, 2, 2, 2, 2, 2, 2],
                ],
                dtype=np.int16,
            ),
            expected_risk=(0, 0, 0, 0),
            expected_hole_count=0,
        ),
        HoleCase(
            name="bridge_mouth_exterior_open",
            note="Exterior left, interior right. Only leaving left should warn.",
            grid=np.array(
                [
                    [2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [1, 1, 1, 1, 1, 1, 1, 1, 2],
                    [0, 0, -1, -1, -1, -1, -1, 1, 2],
                    [0, 0, 9, 0, 0, 0, -1, 1, 2],
                    [0, 0, -1, 0, 0, 0, -1, 1, 2],
                    [0, 0, -1, 0, 0, 0, -1, 1, 2],
                    [0, 0, -1, -1, -1, -1, -1, 1, 2],
                    [0, 0, 0, 0, 0, 0, 0, 1, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2],
                ],
                dtype=np.int16,
            ),
            expected_risk=(0, 0, 0, 1),
            expected_hole_count=0,
        ),
        HoleCase(
            name="sealed_known_pocket_leave_left",
            note="Known-only pocket. Leaving left seals it.",
            grid=np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, -1, -1, -1, -1, -1, 1, 1],
                    [0, 0, 9, 0, 0, 0, 0, 1, 1],
                    [0, 0, -1, 0, 0, 0, 0, 1, 1],
                    [0, 0, -1, 0, 0, 0, 0, 1, 1],
                    [0, 0, -1, -1, -1, -1, -1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=np.int16,
            ),
            expected_risk=(0, 0, 0, 1),
            expected_hole_count=0,
        ),
        HoleCase(
            name="sealed_unknown_pocket_leave_left",
            note="Pocket with unknown inside still counts when sealed.",
            grid=np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, -1, -1, 2, 2, -1, 1, 1],
                    [0, 0, 9, 0, 0, 2, -1, 1, 1],
                    [0, 0, -1, 0, 0, 2, -1, 1, 1],
                    [0, 0, -1, 0, 0, 2, -1, 1, 1],
                    [0, 0, -1, -1, -1, -1, -1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=np.int16,
            ),
            expected_risk=(0, 0, 0, 1),
            expected_hole_count=0,
        ),
        HoleCase(
            name="already_sealed_pocket",
            note="Already sealed pocket should count, but no new risk should appear.",
            grid=np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 9, 1, -1, -1, -1, -1, 1, 1],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1],
                    [0, 0, 1, 0, 0, 0, 0, 1, 1],
                    [0, 0, 1, -1, -1, -1, -1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=np.int16,
            ),
            expected_risk=(0, 0, 0, 0),
            expected_hole_count=1,
        ),
        HoleCase(
            name="two_entrances_open",
            note="Pocket with two entrances must not trigger warning.",
            grid=np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, -1, -1, -1, -1, -1, 0, 0],
                    [0, 0, 9, 0, 0, 0, 0, 0, 0],
                    [0, 0, -1, 0, 0, 0, 0, 0, 0],
                    [0, 0, -1, 0, 0, 0, 0, 0, 0],
                    [0, 0, -1, -1, -1, -1, -1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=np.int16,
            ),
            expected_risk=(0, 0, 0, 0),
            expected_hole_count=0,
        ),
        HoleCase(
            name="two_sealed_pockets",
            note="Two independent sealed pockets should both count.",
            grid=np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 9, 0, 0, 1, 0, 0, 0, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0, 1, 0, 1, -1, 1],
                    [1, 0, 1, -1, 1, 0, 1, -1, 1, -1, 1],
                    [1, 0, 1, -1, 1, 0, 1, -1, 1, 0, 1],
                    [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=np.int16,
            ),
            expected_risk=(0, 0, 0, 0),
            expected_hole_count=2,
        ),
        HoleCase(
            name="diagonal_not_open",
            note="Diagonal gap is not an entrance under 4-connectivity.",
            grid=np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1, 9, 1],
                    [1, 0, -1, 1, 1, 0, 1],
                    [1, 0, 1, -1, -1, 0, 1],
                    [1, 0, 1, -1, -1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=np.int16,
            ),
            expected_risk=(0, 0, 0, 0),
            expected_hole_count=1,
        ),
        HoleCase(
            name="robot_adjacent_hole_not_counted",
            note="Pocket exists but robot still touches its mouth, so count stays zero.",
            grid=np.array(
                [
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 0, -1, -1, -1, 1, 1],
                    [1, 0, 9, 0, -1, 1, 1],
                    [1, 0, -1, 0, -1, 1, 1],
                    [1, 0, -1, -1, -1, 1, 1],
                    [1, 0, 0, 0, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                ],
                dtype=np.int16,
            ),
            expected_risk=(0, 0, 0, 1),
            expected_hole_count=0,
        ),
    ]


def _curated_result(case: HoleCase) -> Dict[str, object]:
    env = _make_env(case.grid)
    robot_pos, known_map, explored = _display_to_state(case.grid)

    env_stats = env._current_hole_stats()
    env_risk = tuple(int(v) for v in env._hole_signal_vector().tolist())
    ref_stats = _reference_hole_stats(robot_pos, known_map, explored)
    ref_risk = tuple(int(v) for v in _reference_hole_signal(robot_pos, known_map, explored).tolist())

    return {
        "expected_risk": list(case.expected_risk),
        "env_risk": list(env_risk),
        "ref_risk": list(ref_risk),
        "expected_hole_count": case.expected_hole_count,
        "env_hole_count": int(env_stats["coverage_hole_count"]),
        "ref_hole_count": int(ref_stats["coverage_hole_count"]),
        "matches_expected": bool(
            env_risk == case.expected_risk and int(env_stats["coverage_hole_count"]) == case.expected_hole_count
        ),
        "matches_reference": bool(
            env_risk == ref_risk and int(env_stats["coverage_hole_count"]) == int(ref_stats["coverage_hole_count"])
        ),
        "env_stats": env_stats,
        "ref_stats": ref_stats,
        "note": case.note,
    }


def _draw_case(ax, case: HoleCase, result: Dict[str, object]) -> None:
    colors = {-1: "#222222", 0: "#f5f5f5", 1: "#75c58d", 2: "#bfc7d5", 9: "#75c58d"}
    grid = case.grid
    h, w = grid.shape
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_aspect("equal")
    ax.set_xticks(range(w + 1))
    ax.set_yticks(range(h + 1))
    ax.grid(color="#666666", linewidth=1)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)

    status = "PASS" if (result["matches_expected"] and result["matches_reference"]) else "FAIL"
    risk = result["env_risk"]
    ax.set_title(f"{case.name}\\n{status} risk={risk}", fontsize=10)

    rr, cc = tuple(map(int, np.argwhere(grid == 9)[0]))
    for r in range(h):
        for c in range(w):
            ax.add_patch(
                plt.Rectangle((c, r), 1, 1, facecolor=colors[int(grid[r, c])], edgecolor="none")
            )
            if int(grid[r, c]) == 9:
                ax.text(c + 0.5, r + 0.52, "R", ha="center", va="center", color="crimson", fontsize=16, weight="bold")

    for label, dr, dc, rv in [
        ("U", -1, 0, risk[0]),
        ("R", 0, 1, risk[1]),
        ("D", 1, 0, risk[2]),
        ("L", 0, -1, risk[3]),
    ]:
        ax.text(
            cc + 0.5 + 0.68 * dc,
            rr + 0.52 + 0.68 * dr,
            label,
            ha="center",
            va="center",
            color=("crimson" if rv else "#2e8b57"),
            fontsize=14,
            weight="bold",
        )

    ax.text(0.01, -0.13, case.note, transform=ax.transAxes, fontsize=8)


def _random_grid(rng: np.random.Generator, rows: int, cols: int) -> Grid:
    probs = np.array([0.16, 0.36, 0.24, 0.24])  # obstacle, uncovered, covered, unknown
    vals = np.array([-1, 0, 1, 2], dtype=np.int16)
    grid = vals[rng.choice(len(vals), size=(rows, cols), p=probs)]

    # Bias boundary toward covered/unknown/obstacle so edge cases appear.
    for r in [0, rows - 1]:
        for c in range(cols):
            grid[r, c] = vals[rng.choice(len(vals), p=[0.25, 0.15, 0.35, 0.25])]
    for c in [0, cols - 1]:
        for r in range(rows):
            grid[r, c] = vals[rng.choice(len(vals), p=[0.25, 0.15, 0.35, 0.25])]

    free_cells = np.argwhere(grid != -1)
    if free_cells.size == 0:
        r = int(rng.integers(rows))
        c = int(rng.integers(cols))
        grid[r, c] = 0
        free_cells = np.argwhere(grid != -1)
    rr, cc = map(int, free_cells[int(rng.integers(len(free_cells)))])
    grid[rr, cc] = 9
    return grid


def _random_case_summary(grid: Grid) -> Dict[str, object]:
    env = _make_env(grid)
    robot_pos, known_map, explored = _display_to_state(grid)
    env_stats = env._current_hole_stats()
    env_risk = tuple(int(v) for v in env._hole_signal_vector().tolist())
    ref_stats = _reference_hole_stats(robot_pos, known_map, explored)
    ref_risk = tuple(int(v) for v in _reference_hole_signal(robot_pos, known_map, explored).tolist())
    return {
        "env_risk": env_risk,
        "ref_risk": ref_risk,
        "env_hole_count": int(env_stats["coverage_hole_count"]),
        "ref_hole_count": int(ref_stats["coverage_hole_count"]),
        "env_hole_known_mass": int(env_stats["coverage_hole_known_mass"]),
        "ref_hole_known_mass": int(ref_stats["coverage_hole_known_mass"]),
        "env_hole_open_mass": int(env_stats["coverage_hole_open_mass"]),
        "ref_hole_open_mass": int(ref_stats["coverage_hole_open_mass"]),
        "risk_match": env_risk == ref_risk,
        "count_match": int(env_stats["coverage_hole_count"]) == int(ref_stats["coverage_hole_count"]),
        "known_mass_match": int(env_stats["coverage_hole_known_mass"]) == int(ref_stats["coverage_hole_known_mass"]),
        "open_mass_match": int(env_stats["coverage_hole_open_mass"]) == int(ref_stats["coverage_hole_open_mass"]),
    }


def _run_random_differential(seed: int = 20260407, trials: int = 5000) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    failures = []
    stats = {"risk": 0, "count": 0, "known_mass": 0, "open_mass": 0}
    size_hist: Dict[str, int] = {}

    for idx in range(trials):
        rows = int(rng.integers(5, 8))
        cols = int(rng.integers(5, 8))
        size_hist[f"{rows}x{cols}"] = size_hist.get(f"{rows}x{cols}", 0) + 1
        grid = _random_grid(rng, rows, cols)
        res = _random_case_summary(grid)

        stats["risk"] += int(res["risk_match"])
        stats["count"] += int(res["count_match"])
        stats["known_mass"] += int(res["known_mass_match"])
        stats["open_mass"] += int(res["open_mass_match"])

        if not all([res["risk_match"], res["count_match"], res["known_mass_match"], res["open_mass_match"]]):
            failures.append(
                {
                    "trial": idx,
                    "grid": grid.tolist(),
                    **res,
                }
            )
            if len(failures) >= 20:
                break

    return {
        "seed": seed,
        "trials": trials,
        "size_histogram": size_hist,
        "risk_match_trials": stats["risk"],
        "count_match_trials": stats["count"],
        "known_mass_match_trials": stats["known_mass"],
        "open_mass_match_trials": stats["open_mass"],
        "failures": failures,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    curated = {}
    cases = _curated_cases()
    cols = 3
    rows = (len(cases) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.8 * cols, 5.3 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, case in zip(axes, cases):
        result = _curated_result(case)
        curated[case.name] = result
        _draw_case(ax, case, result)
    for ax in axes[len(cases) :]:
        ax.axis("off")
    img_path = OUT_DIR / "hole_logic_suite_v3.png"
    fig.savefig(img_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    random_summary = _run_random_differential()

    summary = {
        "curated_cases": curated,
        "random_differential": random_summary,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = ["# Hole logic suite v3", "", "## Curated cases", ""]
    for case in cases:
        result = curated[case.name]
        md_lines.append(
            f"- `{case.name}`: expected risk={result['expected_risk']} / env risk={result['env_risk']} / "
            f"ref risk={result['ref_risk']} / expected holes={result['expected_hole_count']} / "
            f"env holes={result['env_hole_count']} / ref holes={result['ref_hole_count']} / "
            f"pass={result['matches_expected'] and result['matches_reference']}"
        )

    rd = random_summary
    md_lines.extend(
        [
            "",
            "## Random differential test",
            "",
            f"- seed: `{rd['seed']}`",
            f"- trials: `{rd['trials']}`",
            f"- risk match: `{rd['risk_match_trials']}/{rd['trials']}`",
            f"- hole count match: `{rd['count_match_trials']}/{rd['trials']}`",
            f"- hole known mass match: `{rd['known_mass_match_trials']}/{rd['trials']}`",
            f"- hole open mass match: `{rd['open_mass_match_trials']}/{rd['trials']}`",
            f"- failures captured: `{len(rd['failures'])}`",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"image": str(img_path), "summary_path": str(OUT_DIR / "summary.json")}, indent=2))


if __name__ == "__main__":
    main()

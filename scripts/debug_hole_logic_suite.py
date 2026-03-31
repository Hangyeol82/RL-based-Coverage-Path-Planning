from __future__ import annotations

import importlib.util
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path("/Users/ihangyeol/Desktop/졸업 작품/RL based online CPP")
HEUR_ROOT = Path("/tmp/rlcpp-heuristic")
OUT_DIR = WORKSPACE / "log_analysis" / "logs" / "_tmp_hole_logic_suite_20260331"


# Display-only encoding:
# -1: obstacle
#  0: known free and uncovered
#  1: covered
#  2: unknown
#  9: robot
Grid = np.ndarray


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

    # The diagnostic only needs hole/risk logic. Stub observation imports.
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
    return env_mod.CPPDiscreteEnv, env_mod.ACTION_TO_DELTA


CPPDiscreteEnv, ACTION_TO_DELTA = _load_cpp_env()


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


def _action_name(action: int) -> str:
    return ["up", "right", "down", "left"][action]


def _collect_case(case: HoleCase) -> Dict[str, object]:
    env = _make_env(case.grid)
    current_stats = env._current_hole_stats()
    risk_vec = tuple(int(v) for v in env._hole_signal_vector().tolist())
    per_action: Dict[str, Dict[str, float]] = {}
    rr, cc = env.current_pos
    for action, (dr, dc) in ACTION_TO_DELTA.items():
        nr, nc = rr + dr, cc + dc
        if 0 <= nr < env.rows and 0 <= nc < env.cols and env.known_map[nr, nc] != 1:
            per_action[_action_name(action)] = env._coverage_hole_stats(
                (nr, nc), env._hole_component_data
            )

    return {
        "note": case.note,
        "expected_risk": list(case.expected_risk),
        "actual_risk": list(risk_vec),
        "risk_match": bool(risk_vec == case.expected_risk),
        "expected_hole_count": case.expected_hole_count,
        "actual_hole_count": int(current_stats["coverage_hole_count"]),
        "hole_count_match": bool(int(current_stats["coverage_hole_count"]) == case.expected_hole_count),
        "current_stats": current_stats,
        "next_stats_by_action": per_action,
    }


def _draw_case(ax, case: HoleCase, result: Dict[str, object]) -> None:
    colors = {
        -1: "#222222",
        0: "#f5f5f5",
        1: "#75c58d",
        2: "#bfc7d5",
        9: "#75c58d",
    }
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

    risk = result["actual_risk"]
    status = "PASS" if (result["risk_match"] and result["hole_count_match"]) else "FAIL"
    ax.set_title(f"{case.name}\n{status} risk={risk}", fontsize=10)

    robot_pos = tuple(map(int, np.argwhere(grid == 9)[0]))
    for r in range(h):
        for c in range(w):
            ax.add_patch(
                plt.Rectangle((c, r), 1, 1, facecolor=colors[int(grid[r, c])], edgecolor="none")
            )
            if int(grid[r, c]) == 9:
                ax.text(
                    c + 0.5,
                    r + 0.52,
                    "R",
                    ha="center",
                    va="center",
                    color="crimson",
                    fontsize=18,
                    weight="bold",
                )

    rr, cc = robot_pos
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
            fontsize=16,
            weight="bold",
        )

    ax.text(0.02, -0.12, case.note, transform=ax.transAxes, fontsize=8)


def _cases() -> List[HoleCase]:
    return [
        HoleCase(
            name="enlarged_original_like",
            note="User-like enlarged shape. Should not warn yet.",
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
            note="Left is exterior open region. Right should be safe, left should be risky.",
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
            note="Known-only internal pocket. Leaving left should seal it.",
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
            note="Internal pocket with unknown inside. Leaving left should still seal it.",
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
            note="Pocket already sealed. Hole count should be 1 and no action should create a new one.",
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
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = _cases()
    summary: Dict[str, Dict[str, object]] = {}

    cols = min(3, len(cases))
    rows = (len(cases) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 5.4 * rows), constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    for ax, case in zip(axes, cases):
        result = _collect_case(case)
        summary[case.name] = result
        _draw_case(ax, case, result)

    for ax in axes[len(cases) :]:
        ax.axis("off")

    img_path = OUT_DIR / "hole_logic_suite.png"
    fig.savefig(img_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    json_path = OUT_DIR / "summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = ["# Hole logic suite", ""]
    for case in cases:
        result = summary[case.name]
        md_lines.append(
            f"- `{case.name}`: risk expected={result['expected_risk']} actual={result['actual_risk']} "
            f"/ hole_count expected={result['expected_hole_count']} actual={result['actual_hole_count']} "
            f"/ pass={result['risk_match'] and result['hole_count_match']}"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({"image": str(img_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()

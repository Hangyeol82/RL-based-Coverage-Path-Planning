from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple

import importlib.util
import numpy as np
import torch

from learning.observation.maps_observation import MAPSObservationBuilder
from learning.observation.robot_state_observation import RobotStateObservationBuilder


GridPos = Tuple[int, int]

# Action encoding for BC/PPO:
# 0: up, 1: right, 2: down, 3: left
ACTION_TO_DELTA = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}
DELTA_TO_ACTION = {v: k for k, v in ACTION_TO_DELTA.items()}


@dataclass
class RolloutTransition:
    occupancy: np.ndarray
    explored: np.ndarray
    robot_pos: GridPos
    prev_pos: Optional[GridPos]
    action: int
    potential_level0: Optional[np.ndarray] = None


@dataclass
class BCTensorDataset:
    levels: Dict[int, torch.Tensor]
    robot_state: torch.Tensor
    actions: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.actions.shape[0])


def _load_epsilon_star_class(epsilon_file: Path):
    # E* module imports matplotlib; ensure writable cache dir in sandboxed runs.
    if "MPLCONFIGDIR" not in os.environ:
        mpl_dir = Path(tempfile.gettempdir()) / "mplconfig"
        mpl_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    if "XDG_CACHE_HOME" not in os.environ:
        cache_dir = Path(tempfile.gettempdir()) / "xdg-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(cache_dir)

    spec = importlib.util.spec_from_file_location("epsilon_star_module", str(epsilon_file))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load E* module from {epsilon_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "EpsilonStarCPP"):
        raise AttributeError(f"EpsilonStarCPP not found in {epsilon_file}")
    return module.EpsilonStarCPP


def _action_from_nodes(src: GridPos, dst: GridPos) -> int:
    dr = int(dst[0] - src[0])
    dc = int(dst[1] - src[1])
    key = (dr, dc)
    if key not in DELTA_TO_ACTION:
        raise ValueError(f"Non-4connected move detected: src={src}, dst={dst}")
    return DELTA_TO_ACTION[key]


def collect_epsilon_rollout(
    grid_map: np.ndarray,
    start_node: GridPos = (0, 0),
    epsilon: float = 0.5,
    sensor_range: int = 5,
    max_steps: int = 4000,
    epsilon_file: Optional[Path] = None,
) -> Tuple[List[RolloutTransition], List[GridPos]]:
    if epsilon_file is None:
        epsilon_file = Path(__file__).resolve().parents[2] / "EStarOlineCpp.py"

    base_cls = _load_epsilon_star_class(epsilon_file)

    class LoggedEpsilonStar(base_cls):
        def __init__(self, *args, **kwargs):
            self.logged_transitions: List[RolloutTransition] = []
            super().__init__(*args, **kwargs)

        def _move_to(self, node):
            prev_pos = self.path[-2] if len(self.path) >= 2 else None
            action = _action_from_nodes(self.current_node, node)
            if not self.level_maps:
                raise RuntimeError("Teacher level maps are not ready at move time")
            self.logged_transitions.append(
                RolloutTransition(
                    occupancy=self.known_grid.copy(),
                    explored=self.explored.copy(),
                    robot_pos=tuple(self.current_node),
                    prev_pos=tuple(prev_pos) if prev_pos is not None else None,
                    action=int(action),
                    potential_level0=self.level_maps[0].copy(),
                )
            )
            super()._move_to(node)

    teacher = LoggedEpsilonStar(
        grid_map=grid_map,
        start_node=start_node,
        epsilon=epsilon,
        sensor_range=sensor_range,
    )
    path = teacher.run(max_steps=max_steps)
    return teacher.logged_transitions, path


def _sense_square(
    true_map: np.ndarray,
    known_map: np.ndarray,
    pos: GridPos,
    sensor_range: int,
):
    rr, cc = pos
    rows, cols = true_map.shape
    r0 = max(0, rr - sensor_range)
    r1 = min(rows - 1, rr + sensor_range)
    c0 = max(0, cc - sensor_range)
    c1 = min(cols - 1, cc + sensor_range)
    known_map[r0 : r1 + 1, c0 : c1 + 1] = true_map[r0 : r1 + 1, c0 : c1 + 1]


def build_serpentine_path(
    rows: int,
    cols: int,
    *,
    start_node: GridPos = (0, 0),
    sweep_axis: str = "row",
) -> List[GridPos]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")
    if start_node != (0, 0):
        raise ValueError("serpentine teacher currently supports start_node=(0, 0) only")
    if sweep_axis not in {"row", "col"}:
        raise ValueError("sweep_axis must be one of {'row', 'col'}")

    path: List[GridPos] = []
    if sweep_axis == "row":
        for r in range(rows):
            if (r % 2) == 0:
                c_iter = range(cols)
            else:
                c_iter = range(cols - 1, -1, -1)
            for c in c_iter:
                path.append((int(r), int(c)))
    else:
        for c in range(cols):
            if (c % 2) == 0:
                r_iter = range(rows)
            else:
                r_iter = range(rows - 1, -1, -1)
            for r in r_iter:
                path.append((int(r), int(c)))
    return path


def collect_serpentine_rollout(
    grid_map: np.ndarray,
    *,
    start_node: GridPos = (0, 0),
    sensor_range: int = 2,
    sweep_axis: str = "row",
    max_steps: Optional[int] = None,
    strict_empty: bool = True,
) -> Tuple[List[RolloutTransition], List[GridPos]]:
    """
    Collect deterministic zigzag(serpentine) teacher transitions on a grid map.

    The teacher path is generated without planner search and is intended for
    simple directional prior pretraining on easy maps (typically empty maps).
    """
    true_map = np.asarray(grid_map, dtype=np.int32)
    if true_map.ndim != 2:
        raise ValueError("grid_map must be 2D")
    if not np.isin(true_map, [0, 1]).all():
        raise ValueError("grid_map must contain only 0(free),1(obstacle)")
    rows, cols = true_map.shape

    if strict_empty and int(np.count_nonzero(true_map == 1)) > 0:
        raise ValueError("strict_empty=True requires an obstacle-free map")
    sr, sc = start_node
    if not (0 <= sr < rows and 0 <= sc < cols):
        raise ValueError(f"start_node out of bounds: {start_node}")
    if true_map[sr, sc] != 0:
        raise ValueError(f"start_node is blocked: {start_node}")

    path = build_serpentine_path(rows, cols, start_node=start_node, sweep_axis=sweep_axis)
    if max_steps is not None:
        max_steps_i = int(max_steps)
        if max_steps_i <= 0:
            raise ValueError("max_steps must be positive when provided")
        max_nodes = min(len(path), max_steps_i + 1)
        path = path[:max_nodes]

    known_map = np.full_like(true_map, fill_value=-1, dtype=np.int32)
    explored = np.zeros_like(true_map, dtype=bool)
    transitions: List[RolloutTransition] = []

    current = path[0]
    prev_pos: Optional[GridPos] = None
    _sense_square(true_map, known_map, current, int(sensor_range))
    explored[current] = True

    for nxt in path[1:]:
        action = _action_from_nodes(current, nxt)
        transitions.append(
            RolloutTransition(
                occupancy=known_map.copy(),
                explored=explored.copy(),
                robot_pos=tuple(current),
                prev_pos=tuple(prev_pos) if prev_pos is not None else None,
                action=int(action),
                potential_level0=None,
            )
        )

        prev_pos = current
        current = nxt
        _sense_square(true_map, known_map, current, int(sensor_range))
        explored[current] = True

    return transitions, path


def build_bc_tensors_from_rollout(
    transitions: Sequence[RolloutTransition],
    maps_builder: Optional[object] = None,
    robot_state_builder: Optional[RobotStateObservationBuilder] = None,
) -> BCTensorDataset:
    if len(transitions) == 0:
        raise ValueError("No rollout transitions were collected")

    maps_builder = maps_builder or MAPSObservationBuilder()
    robot_state_builder = robot_state_builder or RobotStateObservationBuilder()

    level_acc: Dict[int, List[np.ndarray]] = {}
    level_ids: Optional[List[int]] = None
    robot_state_acc: List[np.ndarray] = []
    actions_acc: List[int] = []

    explored_prev: Optional[int] = None
    gain_hist: List[float] = []

    for tr in transitions:
        explored_now = int(np.count_nonzero(tr.explored))
        if explored_prev is not None:
            gain_hist.append(float(max(0, explored_now - explored_prev)))
        explored_prev = explored_now

        # Legacy MAPS builder uses teacher level-0 potential as an input.
        if isinstance(maps_builder, MAPSObservationBuilder):
            build_kwargs = dict(
                occupancy=tr.occupancy,
                robot_pos=tr.robot_pos,
                explored=tr.explored,
            )
            if tr.potential_level0 is not None:
                build_kwargs["potential_level0"] = tr.potential_level0
            levels = maps_builder.build_levels(**build_kwargs)
        else:
            levels = maps_builder.build_levels(
                occupancy=tr.occupancy,
                robot_pos=tr.robot_pos,
                explored=tr.explored,
            )

        if level_ids is None:
            level_ids = sorted(levels.keys())
            level_acc = {lv: [] for lv in level_ids}

        for lv in level_ids:
            level_acc[lv].append(levels[lv].astype(np.float32))

        recent = gain_hist[-robot_state_builder.config.stagnation_window :] if gain_hist else None
        rs = robot_state_builder.build(
            occupancy=tr.occupancy,
            explored=tr.explored,
            robot_pos=tr.robot_pos,
            prev_pos=tr.prev_pos,
            recent_new_coverage=recent,
        )
        robot_state_acc.append(rs.astype(np.float32))
        actions_acc.append(int(tr.action))

    if level_ids is None:
        raise RuntimeError("No levels were produced by maps_builder")

    levels_t = {
        lv: torch.from_numpy(np.stack(level_acc[lv], axis=0)).float()
        for lv in level_ids
    }
    robot_state_t = torch.from_numpy(np.stack(robot_state_acc, axis=0)).float()
    actions_t = torch.tensor(actions_acc, dtype=torch.long)

    return BCTensorDataset(
        levels=levels_t,
        robot_state=robot_state_t,
        actions=actions_t,
    )

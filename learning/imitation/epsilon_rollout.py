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
    potential_level0: np.ndarray


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
            levels = maps_builder.build_levels(
                occupancy=tr.occupancy,
                robot_pos=tr.robot_pos,
                explored=tr.explored,
                potential_level0=tr.potential_level0,
            )
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

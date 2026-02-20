from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


GridPos = Tuple[int, int]


@dataclass(frozen=True)
class RobotStateObservationConfig:
    # Number of recent steps used to measure stagnation.
    stagnation_window: int = 20


class RobotStateObservationBuilder:
    """
    Builds a robot-state observation vector for imitation/RL.

    Feature groups:
    1) normalized position: [row_norm, col_norm]
    2) previous move direction (one-hot): [stay, up, right, down, left]
    3) coverage progress ratio over free cells: [progress]
    4) stagnation index over recent window: [stagnation]
    """

    _DIR_INDEX = {"stay": 0, "up": 1, "right": 2, "down": 3, "left": 4}

    def __init__(self, config: Optional[RobotStateObservationConfig] = None):
        self.config = config or RobotStateObservationConfig()

    @staticmethod
    def feature_names() -> Tuple[str, ...]:
        return (
            "pos_row_norm",
            "pos_col_norm",
            "dir_stay",
            "dir_up",
            "dir_right",
            "dir_down",
            "dir_left",
            "coverage_progress",
            "stagnation_index",
        )

    def _normalize_position(self, robot_pos: GridPos, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        rr, cc = robot_pos
        row_norm = 0.0 if h <= 1 else float(rr) / float(h - 1)
        col_norm = 0.0 if w <= 1 else float(cc) / float(w - 1)
        return np.array([row_norm, col_norm], dtype=np.float32)

    def _direction_one_hot(self, prev_pos: Optional[GridPos], robot_pos: GridPos) -> np.ndarray:
        vec = np.zeros(5, dtype=np.float32)
        if prev_pos is None:
            vec[self._DIR_INDEX["stay"]] = 1.0
            return vec

        pr, pc = prev_pos
        rr, cc = robot_pos
        dr = rr - pr
        dc = cc - pc

        direction = "stay"
        if dr < 0 and dc == 0:
            direction = "up"
        elif dr > 0 and dc == 0:
            direction = "down"
        elif dc > 0 and dr == 0:
            direction = "right"
        elif dc < 0 and dr == 0:
            direction = "left"

        vec[self._DIR_INDEX[direction]] = 1.0
        return vec

    def _coverage_progress(self, occupancy: np.ndarray, explored: np.ndarray) -> float:
        # Online setting: unknown (-1) must not be treated as free.
        known_free = occupancy == 0
        free_total = int(np.count_nonzero(known_free))
        if free_total == 0:
            return 0.0
        covered_free = int(np.count_nonzero(known_free & explored.astype(bool)))
        return float(covered_free) / float(free_total)

    def _stagnation_index(self, recent_new_coverage: Optional[Sequence[float]]) -> float:
        if recent_new_coverage is None:
            return 0.0
        hist = np.asarray(list(recent_new_coverage), dtype=np.float32)
        if hist.size == 0:
            return 0.0
        window = max(1, int(self.config.stagnation_window))
        hist = hist[-window:]
        progress_steps = int(np.count_nonzero(hist > 0.0))
        stagnation = 1.0 - (float(progress_steps) / float(hist.size))
        return float(np.clip(stagnation, 0.0, 1.0))

    def build(
        self,
        occupancy: np.ndarray,
        explored: np.ndarray,
        robot_pos: GridPos,
        prev_pos: Optional[GridPos] = None,
        recent_new_coverage: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        if occupancy.ndim != 2:
            raise ValueError("occupancy must be 2D")
        if explored.shape != occupancy.shape:
            raise ValueError("explored shape must match occupancy shape")

        h, w = occupancy.shape
        rr, cc = robot_pos
        if not (0 <= rr < h and 0 <= cc < w):
            raise ValueError(f"robot_pos {robot_pos} is out of bounds {(h, w)}")
        if prev_pos is not None:
            pr, pc = prev_pos
            if not (0 <= pr < h and 0 <= pc < w):
                raise ValueError(f"prev_pos {prev_pos} is out of bounds {(h, w)}")

        pos = self._normalize_position(robot_pos, (h, w))
        direction = self._direction_one_hot(prev_pos, robot_pos)
        progress = np.array([self._coverage_progress(occupancy, explored)], dtype=np.float32)
        stagnation = np.array([self._stagnation_index(recent_new_coverage)], dtype=np.float32)
        return np.concatenate([pos, direction, progress, stagnation], axis=0).astype(np.float32)

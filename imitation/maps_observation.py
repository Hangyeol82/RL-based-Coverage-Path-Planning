from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


GridPos = Tuple[int, int]


@dataclass(frozen=True)
class MAPSObservationConfig:
    # level -> (block_size_on_level0, output_window_size)
    level_specs: Tuple[Tuple[int, int], ...] = (
        (1, 11),   # level 0 (robot-centered)
        (2, 7),    # level 1 (robot-centered)
        (3, 7),    # level 2 (robot-centered, non-log scale)
        (4, 7),    # level 3 (robot-centered)
        (8, 5),    # level 4 (robot-centered)
        (16, 4),   # level 5 (global, not robot-centered)
    )
    # Left->right potential decay for zigzag bias.
    left_weight: float = 1.0
    right_weight: float = 0.1
    # Occupancy code for unknown/unobserved cells.
    unknown_value: int = -1


class MAPSObservationBuilder:
    """
    Builds MAPS observation tensors for imitation/RL models.

    Output per level:
    - shape: (3, H, W)
    - channel 0: potential
    - channel 1: obstacle ratio over known cells
    - channel 2: known ratio
    """

    def __init__(self, config: Optional[MAPSObservationConfig] = None):
        self.config = config or MAPSObservationConfig()

    def _col_decay(self, width: int) -> np.ndarray:
        if width <= 1:
            return np.array([self.config.left_weight], dtype=np.float32)
        return np.linspace(
            self.config.left_weight,
            self.config.right_weight,
            width,
            dtype=np.float32,
        )

    def _center_crop_with_pad(
        self,
        arr: np.ndarray,
        center: GridPos,
        out_h: int,
        out_w: int,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        h, w = arr.shape
        cr, cc = center
        out = np.full((out_h, out_w), pad_value, dtype=np.float32)
        r0 = cr - out_h // 2
        c0 = cc - out_w // 2

        for rr in range(out_h):
            sr = r0 + rr
            if sr < 0 or sr >= h:
                continue
            for cc_out in range(out_w):
                sc = c0 + cc_out
                if sc < 0 or sc >= w:
                    continue
                out[rr, cc_out] = float(arr[sr, sc])
        return out

    def _to_fixed_global(
        self,
        arr: np.ndarray,
        out_h: int,
        out_w: int,
        pad_value: float = 0.0,
    ) -> np.ndarray:
        out = np.full((out_h, out_w), pad_value, dtype=np.float32)
        h, w = arr.shape
        rh = min(out_h, h)
        rw = min(out_w, w)
        out[:rh, :rw] = arr[:rh, :rw].astype(np.float32)
        return out

    def _coarsen_obstacle_known(
        self,
        obstacle: np.ndarray,
        known: np.ndarray,
        block: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = obstacle.shape
        ch = (h + block - 1) // block
        cw = (w + block - 1) // block
        obs_ratio = np.zeros((ch, cw), dtype=np.float32)
        known_ratio = np.zeros((ch, cw), dtype=np.float32)
        for r in range(ch):
            rs = r * block
            re = min(h, rs + block)
            for c in range(cw):
                cs = c * block
                ce = min(w, cs + block)
                obs_blk = obstacle[rs:re, cs:ce]
                known_blk = known[rs:re, cs:ce]
                cell_count = (re - rs) * (ce - cs)
                known_count = int(np.count_nonzero(known_blk))
                if known_count == 0:
                    obs_ratio[r, c] = 0.0
                else:
                    obs_ratio[r, c] = float(np.count_nonzero(obs_blk & known_blk)) / float(known_count)
                known_ratio[r, c] = float(known_count) / float(cell_count)
        return obs_ratio, known_ratio

    def _coarsen_potential(
        self,
        obstacle: np.ndarray,
        explored: np.ndarray,
        block: int,
    ) -> np.ndarray:
        h, w = obstacle.shape
        ch = (h + block - 1) // block
        cw = (w + block - 1) // block
        out = np.zeros((ch, cw), dtype=np.float32)
        col_decay = self._col_decay(cw)

        for r in range(ch):
            rs = r * block
            re = min(h, rs + block)
            for c in range(cw):
                cs = c * block
                ce = min(w, cs + block)
                obs_blk = obstacle[rs:re, cs:ce]
                exp_blk = explored[rs:re, cs:ce]
                non_obs = ~obs_blk
                free_count = int(np.count_nonzero(non_obs))
                if free_count == 0:
                    out[r, c] = -1.0
                    continue
                unexplored_ratio = float(np.count_nonzero(non_obs & (~exp_blk))) / float(free_count)
                out[r, c] = unexplored_ratio * col_decay[c]
        return out

    def _build_level0_potential(
        self,
        obstacle: np.ndarray,
        explored: np.ndarray,
    ) -> np.ndarray:
        h, w = obstacle.shape
        col_decay = self._col_decay(w)[None, :]
        pot = np.zeros((h, w), dtype=np.float32)
        pot[obstacle] = -1.0
        pot[(~obstacle) & explored] = 0.0
        pot[(~obstacle) & (~explored)] = col_decay.repeat(h, axis=0)[(~obstacle) & (~explored)]
        return pot

    def build_levels(
        self,
        occupancy: np.ndarray,
        robot_pos: GridPos,
        explored: Optional[np.ndarray] = None,
        potential_level0: Optional[np.ndarray] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Parameters
        ----------
        occupancy:
            HxW grid with obstacle coding.
            - obstacle is value 1
            - unknown is value == config.unknown_value
            - known non-obstacle is every value not in {1, unknown}
        robot_pos:
            (row, col) robot coordinate.
        explored:
            Optional HxW bool mask. If None, defaults to all False.
        potential_level0:
            Optional HxW potential map for level-0. If None, generated from
            obstacle/explored with left->right zigzag decay.
        """
        if occupancy.ndim != 2:
            raise ValueError("occupancy must be 2D")
        h, w = occupancy.shape
        rr, cc = robot_pos
        if not (0 <= rr < h and 0 <= cc < w):
            raise ValueError(f"robot_pos {robot_pos} is out of bounds {(h, w)}")

        obstacle = occupancy == 1
        known = occupancy != self.config.unknown_value
        if explored is None:
            explored_bool = np.zeros_like(obstacle, dtype=bool)
        else:
            if explored.shape != occupancy.shape:
                raise ValueError("explored shape must match occupancy shape")
            explored_bool = explored.astype(bool)

        if potential_level0 is None:
            pot0_map = self._build_level0_potential(obstacle, explored_bool)
        else:
            if potential_level0.shape != occupancy.shape:
                raise ValueError("potential_level0 shape must match occupancy shape")
            pot0_map = potential_level0.astype(np.float32)

        levels: Dict[int, np.ndarray] = {}
        obs0_map = np.zeros_like(occupancy, dtype=np.float32)
        obs0_map[obstacle & known] = 1.0
        known0_map = known.astype(np.float32)

        for lv, (block, out_size) in enumerate(self.config.level_specs):
            if lv == 0:
                pot_lv = self._center_crop_with_pad(pot0_map, robot_pos, out_size, out_size, pad_value=-1.0)
                obs_lv = self._center_crop_with_pad(obs0_map, robot_pos, out_size, out_size, pad_value=1.0)
                known_lv = self._center_crop_with_pad(known0_map, robot_pos, out_size, out_size, pad_value=1.0)
            else:
                pot_coarse = self._coarsen_potential(obstacle, explored_bool, block)
                obs_coarse, known_coarse = self._coarsen_obstacle_known(obstacle, known, block)

                if lv < 5:
                    center_coarse = (rr // block, cc // block)
                    pot_lv = self._center_crop_with_pad(
                        pot_coarse, center_coarse, out_size, out_size, pad_value=-1.0
                    )
                    obs_lv = self._center_crop_with_pad(
                        obs_coarse, center_coarse, out_size, out_size, pad_value=1.0
                    )
                    known_lv = self._center_crop_with_pad(
                        known_coarse, center_coarse, out_size, out_size, pad_value=1.0
                    )
                else:
                    # Level-5 is global (not robot-centered), fixed 4x4.
                    # Outside-map padding is blocked/non-traversable.
                    pot_lv = self._to_fixed_global(pot_coarse, out_size, out_size, pad_value=-1.0)
                    obs_lv = self._to_fixed_global(obs_coarse, out_size, out_size, pad_value=1.0)
                    known_lv = self._to_fixed_global(known_coarse, out_size, out_size, pad_value=1.0)

            levels[lv] = np.stack([pot_lv, obs_lv, known_lv], axis=0).astype(np.float32)

        return levels

    def build_cnn_input(
        self,
        occupancy: np.ndarray,
        robot_pos: GridPos,
        explored: Optional[np.ndarray] = None,
        potential_level0: Optional[np.ndarray] = None,
        target_hw: Tuple[int, int] = (11, 11),
    ) -> np.ndarray:
        """
        Returns a single tensor for CNN:
        - shape: (18, target_h, target_w) from 6 levels x 3 channels.
        - each level map is centered in the target canvas with blocked pad.
        """
        levels = self.build_levels(occupancy, robot_pos, explored=explored, potential_level0=potential_level0)
        th, tw = target_hw
        stacked: List[np.ndarray] = []
        for lv in range(len(self.config.level_specs)):
            x = levels[lv]  # (3, h, w)
            _, h, w = x.shape
            # Outside-map canvas is blocked by default to avoid learning fake free-space.
            canvas = np.empty((3, th, tw), dtype=np.float32)
            canvas[0, :, :] = -1.0  # potential channel
            canvas[1, :, :] = 1.0   # obstacle ratio channel
            canvas[2, :, :] = 1.0   # known ratio channel
            rs = max(0, (th - h) // 2)
            cs = max(0, (tw - w) // 2)
            re = min(th, rs + h)
            ce = min(tw, cs + w)
            canvas[:, rs:re, cs:ce] = x[:, : re - rs, : ce - cs]
            stacked.append(canvas)
        return np.concatenate(stacked, axis=0)

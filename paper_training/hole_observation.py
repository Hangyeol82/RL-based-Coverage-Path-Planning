from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict, Optional, Tuple

import numpy as np

try:
    from scipy import ndimage as scipy_ndimage
except Exception:  # pragma: no cover - handled at runtime.
    scipy_ndimage = None


GridPos = Tuple[int, int]

ACTION_TO_DELTA: Dict[int, GridPos] = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}

_LABEL_STRUCTURE_4 = np.asarray(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.int8,
)


@dataclass(frozen=True)
class HoleStats:
    count: float
    known_mass: float
    open_mass: float


@dataclass(frozen=True)
class HoleComponentCache:
    open_base: np.ndarray
    comp_id: np.ndarray
    known_mass: np.ndarray
    open_mass: np.ndarray
    touches_boundary: np.ndarray
    eligible: np.ndarray
    eligible_count_total: int
    eligible_known_mass_total: int
    eligible_open_mass_total: int
    refresh_ms: float


class HoleObservationCalculator:
    """
    Computes action-conditioned coverage-hole risk from the current known map.

    The expensive operation is one connected-component labeling pass over
    open_base = unknown OR known-free-uncovered. After that, hole stats for a
    robot position only inspect the component ids at the position and its four
    neighbors, so the four-action risk vector is O(1) after the O(HW) label pass.
    """

    def __init__(self, *, require_scipy: bool = True):
        if require_scipy and scipy_ndimage is None:
            raise RuntimeError(
                "scipy is required for paper hole observation. "
                "Install scipy or disable hole signals/reward."
            )
        self.cache: Optional[HoleComponentCache] = None
        self.last_risk_ms: float = 0.0

    def refresh(self, known_map: np.ndarray, explored: np.ndarray) -> HoleComponentCache:
        t0 = perf_counter()
        known = np.asarray(known_map)
        exp = np.asarray(explored, dtype=bool)
        if known.ndim != 2:
            raise ValueError("known_map must be 2D")
        if exp.shape != known.shape:
            raise ValueError("explored shape must match known_map shape")

        known_free = known == 0
        uncovered_known = known_free & (~exp)
        unknown = known == -1
        open_base = np.logical_or(unknown, uncovered_known)

        if scipy_ndimage is None:
            raise RuntimeError("scipy.ndimage is unavailable; hole labeling cannot run.")

        labels, num_labels = scipy_ndimage.label(open_base, structure=_LABEL_STRUCTURE_4)
        if num_labels <= 0:
            comp_id = np.full(known.shape, -1, dtype=np.int32)
            cache = HoleComponentCache(
                open_base=open_base,
                comp_id=comp_id,
                known_mass=np.zeros((0,), dtype=np.int32),
                open_mass=np.zeros((0,), dtype=np.int32),
                touches_boundary=np.zeros((0,), dtype=bool),
                eligible=np.zeros((0,), dtype=bool),
                eligible_count_total=0,
                eligible_known_mass_total=0,
                eligible_open_mass_total=0,
                refresh_ms=(perf_counter() - t0) * 1000.0,
            )
            self.cache = cache
            return cache

        labels = labels.astype(np.int32, copy=False)
        comp_id = np.where(labels > 0, labels - 1, -1).astype(np.int32, copy=False)
        open_counts = np.bincount(labels.ravel(), minlength=int(num_labels) + 1)
        known_counts = np.bincount(labels[uncovered_known], minlength=int(num_labels) + 1)
        boundary_labels = np.unique(
            np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
        )
        touches = np.zeros((int(num_labels) + 1,), dtype=bool)
        touches[boundary_labels] = True

        known_mass = known_counts[1:].astype(np.int32, copy=False)
        open_mass = open_counts[1:].astype(np.int32, copy=False)
        touches_boundary = touches[1:].astype(bool, copy=False)
        eligible = np.logical_and(known_mass > 0, ~touches_boundary)

        cache = HoleComponentCache(
            open_base=open_base,
            comp_id=comp_id,
            known_mass=known_mass,
            open_mass=open_mass,
            touches_boundary=touches_boundary,
            eligible=eligible,
            eligible_count_total=int(np.count_nonzero(eligible)),
            eligible_known_mass_total=int(np.sum(known_mass[eligible], dtype=np.int64)),
            eligible_open_mass_total=int(np.sum(open_mass[eligible], dtype=np.int64)),
            refresh_ms=(perf_counter() - t0) * 1000.0,
        )
        self.cache = cache
        return cache

    def _require_cache(self) -> HoleComponentCache:
        if self.cache is None:
            raise RuntimeError("HoleObservationCalculator.refresh() must be called first")
        return self.cache

    def stats_for_pos(self, robot_pos: GridPos) -> HoleStats:
        cache = self._require_cache()
        rows, cols = cache.open_base.shape
        rr, cc = int(robot_pos[0]), int(robot_pos[1])
        active = set()
        for dr, dc in ((0, 0), *ACTION_TO_DELTA.values()):
            nr, nc = rr + dr, cc + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if not bool(cache.open_base[nr, nc]):
                continue
            cid = int(cache.comp_id[nr, nc])
            if cid >= 0 and bool(cache.eligible[cid]):
                active.add(cid)

        active_count = len(active)
        if active:
            active_ids = np.fromiter(active, dtype=np.int32)
            active_known = int(np.sum(cache.known_mass[active_ids], dtype=np.int64))
            active_open = int(np.sum(cache.open_mass[active_ids], dtype=np.int64))
        else:
            active_known = 0
            active_open = 0

        return HoleStats(
            count=float(cache.eligible_count_total - active_count),
            known_mass=float(cache.eligible_known_mass_total - active_known),
            open_mass=float(cache.eligible_open_mass_total - active_open),
        )

    def risk_vector(self, robot_pos: GridPos, known_map: np.ndarray) -> np.ndarray:
        t0 = perf_counter()
        known = np.asarray(known_map)
        rows, cols = known.shape
        rr, cc = int(robot_pos[0]), int(robot_pos[1])
        current = self.stats_for_pos((rr, cc)).count
        risk = np.zeros(4, dtype=np.float32)
        for action, (dr, dc) in ACTION_TO_DELTA.items():
            nr, nc = rr + dr, cc + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if int(known[nr, nc]) == 1:
                continue
            next_count = self.stats_for_pos((int(nr), int(nc))).count
            risk[action] = 1.0 if next_count > current else 0.0
        self.last_risk_ms = (perf_counter() - t0) * 1000.0
        return risk

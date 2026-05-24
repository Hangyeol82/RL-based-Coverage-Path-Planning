from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from time import perf_counter
from typing import Tuple

import numpy as np


GridPos = Tuple[int, int]


@dataclass(frozen=True)
class RevisitBurdenStats:
    """Known-map potential used for revisit-burden reward shaping."""

    phi: float
    burden: float
    target_count: int
    reachable_target_count: int
    unreachable_target_count: int
    max_revisit_cost: float
    compute_ms: float


class RevisitBurdenPotential:
    """
    Potential over the current known map.

    Unknown cells are excluded from the shaping graph. The graph contains only
    known-free cells; entering an already covered cell has cost 1, while entering
    an uncovered known-free cell has cost 0. The resulting potential is higher
    when remaining known-free uncovered cells require less revisit cost.
    """

    def __init__(self, *, normalizer: float = 0.0, unreachable_cost: float = 0.0):
        self.normalizer = max(0.0, float(normalizer))
        self.unreachable_cost = max(0.0, float(unreachable_cost))
        self.last_stats = RevisitBurdenStats(
            phi=0.0,
            burden=0.0,
            target_count=0,
            reachable_target_count=0,
            unreachable_target_count=0,
            max_revisit_cost=0.0,
            compute_ms=0.0,
        )

    def compute(
        self,
        known_map: np.ndarray,
        explored: np.ndarray,
        robot_pos: GridPos,
    ) -> RevisitBurdenStats:
        t0 = perf_counter()
        known = np.asarray(known_map)
        covered = np.asarray(explored, dtype=bool)
        if known.ndim != 2:
            raise ValueError("known_map must be a 2D array")
        if covered.shape != known.shape:
            raise ValueError("explored must have the same shape as known_map")

        rows, cols = known.shape
        passable = known == 0
        targets = passable & (~covered)
        target_count = int(np.count_nonzero(targets))
        if target_count <= 0:
            stats = RevisitBurdenStats(
                phi=0.0,
                burden=0.0,
                target_count=0,
                reachable_target_count=0,
                unreachable_target_count=0,
                max_revisit_cost=0.0,
                compute_ms=(perf_counter() - t0) * 1000.0,
            )
            self.last_stats = stats
            return stats

        rr, cc = int(robot_pos[0]), int(robot_pos[1])
        if not (0 <= rr < rows and 0 <= cc < cols) or not bool(passable[rr, cc]):
            stats = self._all_unreachable(target_count, rows, cols, t0)
            self.last_stats = stats
            return stats

        inf = np.iinfo(np.int32).max
        dist = np.full((rows, cols), inf, dtype=np.int32)
        dist[rr, cc] = 0
        q: deque[GridPos] = deque([(rr, cc)])
        while q:
            r, c = q.popleft()
            base = int(dist[r, c])
            nr = r - 1
            if nr >= 0 and passable[nr, c]:
                self._relax(dist, covered, q, base, nr, c)
            nr = r + 1
            if nr < rows and passable[nr, c]:
                self._relax(dist, covered, q, base, nr, c)
            nc = c - 1
            if nc >= 0 and passable[r, nc]:
                self._relax(dist, covered, q, base, r, nc)
            nc = c + 1
            if nc < cols and passable[r, nc]:
                self._relax(dist, covered, q, base, r, nc)

        target_dist = dist[targets]
        reachable = target_dist < inf
        reachable_count = int(np.count_nonzero(reachable))
        unreachable_count = int(target_count - reachable_count)
        unreachable_cost = self._unreachable_cost(rows, cols)

        if reachable_count > 0:
            reachable_dist = target_dist[reachable].astype(np.float64)
            dist_sum = float(np.sum(reachable_dist))
            max_revisit_cost = float(np.max(reachable_dist))
        else:
            dist_sum = 0.0
            max_revisit_cost = 0.0
        burden = (dist_sum + float(unreachable_count) * unreachable_cost) / float(target_count)
        if unreachable_count > 0:
            max_revisit_cost = max(max_revisit_cost, unreachable_cost)

        normalizer = self._normalizer(rows, cols)
        burden_norm = min(1.0, max(0.0, burden / normalizer))
        stats = RevisitBurdenStats(
            phi=-float(burden_norm),
            burden=float(burden),
            target_count=int(target_count),
            reachable_target_count=int(reachable_count),
            unreachable_target_count=int(unreachable_count),
            max_revisit_cost=float(max_revisit_cost),
            compute_ms=(perf_counter() - t0) * 1000.0,
        )
        self.last_stats = stats
        return stats

    @staticmethod
    def _relax(
        dist: np.ndarray,
        covered: np.ndarray,
        q: deque[GridPos],
        base_cost: int,
        nr: int,
        nc: int,
    ) -> None:
        edge_cost = 1 if bool(covered[nr, nc]) else 0
        next_cost = int(base_cost + edge_cost)
        if next_cost >= int(dist[nr, nc]):
            return
        dist[nr, nc] = next_cost
        if edge_cost == 0:
            q.appendleft((int(nr), int(nc)))
        else:
            q.append((int(nr), int(nc)))

    def _normalizer(self, rows: int, cols: int) -> float:
        if self.normalizer > 0.0:
            return self.normalizer
        return float(max(1, max(int(rows), int(cols))))

    def _unreachable_cost(self, rows: int, cols: int) -> float:
        if self.unreachable_cost > 0.0:
            return self.unreachable_cost
        return float(max(1, max(int(rows), int(cols))))

    def _all_unreachable(self, target_count: int, rows: int, cols: int, t0: float) -> RevisitBurdenStats:
        cost = self._unreachable_cost(rows, cols)
        normalizer = self._normalizer(rows, cols)
        phi = -min(1.0, max(0.0, cost / normalizer))
        return RevisitBurdenStats(
            phi=float(phi),
            burden=float(cost),
            target_count=int(target_count),
            reachable_target_count=0,
            unreachable_target_count=int(target_count),
            max_revisit_cost=float(cost),
            compute_ms=(perf_counter() - t0) * 1000.0,
        )

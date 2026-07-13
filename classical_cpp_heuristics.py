from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


GridPos = Tuple[int, int]

# Same action order as learning.reinforcement.cpp_env:
# 0: up, 1: right, 2: down, 3: left
DIRS: Tuple[GridPos, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class HeuristicResult:
    path: List[GridPos]
    stats: Dict[str, object]


class OnlineGridCPPHeuristics:
    """
    Deterministic online CPP baselines for grid maps.

    The planner only expands its known map through local sensing. Movement attempts
    into true obstacles are represented like the RL env: the path repeats the
    current cell and the obstacle becomes known.
    """

    def __init__(
        self,
        grid_map: np.ndarray,
        *,
        start: GridPos,
        sensor_range: int = 3,
    ) -> None:
        grid = np.asarray(grid_map, dtype=np.int32)
        if grid.ndim != 2:
            raise ValueError("grid_map must be 2D")
        if not np.isin(grid, [0, 1]).all():
            raise ValueError("grid_map must contain only 0(free) and 1(obstacle)")
        sr, sc = start
        if not (0 <= sr < grid.shape[0] and 0 <= sc < grid.shape[1]):
            raise ValueError(f"start {start} is out of bounds")
        if grid[sr, sc] != 0:
            raise ValueError(f"start {start} is not free")

        self.true_grid = grid
        self.rows, self.cols = grid.shape
        self.start = (int(sr), int(sc))
        self.sensor_range = max(0, int(sensor_range))
        self.free_total = int(np.count_nonzero(self.true_grid == 0))
        self.reset()

    def reset(self) -> None:
        self.known_grid = np.full_like(self.true_grid, fill_value=-1, dtype=np.int32)
        self.explored: Set[GridPos] = set()
        self.current = self.start
        self.heading = 1
        self.path: List[GridPos] = [self.current]
        self.collision_count = 0
        self.frontier_handoff_count = 0
        self.replan_count = 0
        self._sense()
        self._mark_current()

    def _in_bounds(self, pos: GridPos) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _true_free(self, pos: GridPos) -> bool:
        r, c = pos
        return self._in_bounds(pos) and self.true_grid[r, c] == 0

    def _known_free(self, pos: GridPos) -> bool:
        r, c = pos
        return self._in_bounds(pos) and self.known_grid[r, c] == 0

    def _sense(self) -> None:
        r, c = self.current
        rng = self.sensor_range
        r0 = max(0, r - rng)
        r1 = min(self.rows - 1, r + rng)
        c0 = max(0, c - rng)
        c1 = min(self.cols - 1, c + rng)
        self.known_grid[r0 : r1 + 1, c0 : c1 + 1] = self.true_grid[r0 : r1 + 1, c0 : c1 + 1]

    def _mark_current(self) -> None:
        if self._true_free(self.current):
            self.explored.add(self.current)

    def _neighbors(self, pos: GridPos) -> Iterable[GridPos]:
        r, c = pos
        for dr, dc in DIRS:
            nxt = (r + dr, c + dc)
            if self._in_bounds(nxt):
                yield nxt

    def _known_free_neighbors(self, pos: GridPos) -> List[GridPos]:
        return [n for n in self._neighbors(pos) if self._known_free(n)]

    def _unknown_gain(self, pos: GridPos) -> int:
        return sum(1 for n in self._neighbors(pos) if self.known_grid[n[0], n[1]] == -1)

    def _frontier_cells(self) -> List[GridPos]:
        cells: List[GridPos] = []
        for r, c in np.argwhere(self.known_grid == 0):
            pos = (int(r), int(c))
            if any(self.known_grid[n[0], n[1]] == -1 for n in self._neighbors(pos)):
                cells.append(pos)
        return cells

    def _known_unvisited_cells(self) -> List[GridPos]:
        return [
            (int(r), int(c))
            for r, c in np.argwhere(self.known_grid == 0)
            if (int(r), int(c)) not in self.explored
        ]

    def _bfs_to_targets(
        self,
        targets: Set[GridPos],
        *,
        allow_unknown: bool = False,
    ) -> List[GridPos]:
        if self.current in targets:
            return [self.current]
        q = deque([self.current])
        parent: Dict[GridPos, GridPos] = {}
        seen = {self.current}
        while q:
            cur = q.popleft()
            for nxt in self._neighbors(cur):
                if nxt in seen:
                    continue
                known = int(self.known_grid[nxt[0], nxt[1]])
                if known == 1 or (known == -1 and not allow_unknown):
                    continue
                seen.add(nxt)
                parent[nxt] = cur
                if nxt in targets:
                    path = [nxt]
                    while path[-1] != self.current:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                q.append(nxt)
        return []

    def _move_to(self, nxt: GridPos) -> bool:
        cr, cc = self.current
        nr, nc = nxt
        if abs(cr - nr) + abs(cc - nc) != 1:
            raise ValueError(f"Non-adjacent move requested: {self.current} -> {nxt}")

        for idx, (dr, dc) in enumerate(DIRS):
            if (cr + dr, cc + dc) == nxt:
                self.heading = idx
                break

        if not self._in_bounds(nxt):
            self.collision_count += 1
            self.path.append(self.current)
            return False
        if self.true_grid[nr, nc] != 0:
            self.collision_count += 1
            self.known_grid[nr, nc] = 1
            self.path.append(self.current)
            self._sense()
            self._mark_current()
            return False

        self.current = (int(nr), int(nc))
        self.path.append(self.current)
        self._sense()
        self._mark_current()
        return True

    def _done(self) -> bool:
        return len(self.explored) >= self.free_total

    def _result(self, algorithm: str) -> HeuristicResult:
        return HeuristicResult(
            path=[(int(r), int(c)) for r, c in self.path],
            stats={
                "algorithm": algorithm,
                "collision_count": int(self.collision_count),
                "frontier_handoff_count": int(self.frontier_handoff_count),
                "replan_count": int(self.replan_count),
                "known_free_cells": int(np.count_nonzero(self.known_grid == 0)),
                "known_obstacle_cells": int(np.count_nonzero(self.known_grid == 1)),
                "explored_cells_internal": int(len(self.explored)),
            },
        )

    def run_nearest_unvisited(self, max_steps: int) -> HeuristicResult:
        self.reset()
        for _ in range(int(max_steps)):
            if self._done():
                break
            targets = set(self._known_unvisited_cells())
            route = self._bfs_to_targets(targets) if targets else []
            if len(route) <= 1:
                frontiers = set(self._frontier_cells())
                route = self._bfs_to_targets(frontiers) if frontiers else []
                if len(route) <= 1:
                    break
                self.frontier_handoff_count += 1
            self.replan_count += 1
            self._move_to(route[1])
        return self._result("nearest_unvisited")

    def run_frontier_greedy(self, max_steps: int) -> HeuristicResult:
        self.reset()
        for _ in range(int(max_steps)):
            if self._done():
                break

            local = [n for n in self._known_free_neighbors(self.current) if n not in self.explored]
            if local:
                local.sort(key=lambda p: (self._unknown_gain(p), -_turn_cost(self.current, p, self.heading)), reverse=True)
                self._move_to(local[0])
                continue

            candidates = self._frontier_cells()
            if not candidates:
                candidates = self._known_unvisited_cells()
            if not candidates:
                break

            best_route: List[GridPos] = []
            best_score: Optional[Tuple[float, int, int]] = None
            for target in candidates:
                route = self._bfs_to_targets({target})
                if len(route) <= 1:
                    continue
                score = (
                    float(self._unknown_gain(target)) / float(max(1, len(route) - 1)),
                    -len(route),
                    -_turn_cost(self.current, route[1], self.heading),
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_route = route

            if len(best_route) <= 1:
                break
            self.frontier_handoff_count += 1
            self.replan_count += 1
            self._move_to(best_route[1])
        return self._result("frontier_greedy")

    def run_wall_follow(self, max_steps: int) -> HeuristicResult:
        self.reset()
        stale_steps = 0
        for _ in range(int(max_steps)):
            if self._done():
                break
            before = len(self.explored)
            moved = False
            # Right-hand rule, with a small coverage bias among currently known free cells.
            order = ((self.heading + 1) % 4, self.heading, (self.heading - 1) % 4, (self.heading + 2) % 4)
            choices: List[Tuple[int, GridPos]] = []
            cr, cc = self.current
            for direction in order:
                dr, dc = DIRS[direction]
                nxt = (cr + dr, cc + dc)
                if self._known_free(nxt):
                    choices.append((direction, nxt))
            unvisited = [(d, p) for d, p in choices if p not in self.explored]
            for direction, nxt in unvisited or choices:
                self.heading = direction
                self._move_to(nxt)
                moved = True
                break

            if not moved:
                frontiers = set(self._frontier_cells())
                route = self._bfs_to_targets(frontiers) if frontiers else []
                if len(route) <= 1:
                    break
                self.frontier_handoff_count += 1
                self.replan_count += 1
                self._move_to(route[1])

            stale_steps = stale_steps + 1 if len(self.explored) == before else 0
            if stale_steps > max(64, self.rows + self.cols) and not self._frontier_cells():
                break
        return self._result("wall_follow")

    def run_spiral_stc(self, max_steps: int) -> HeuristicResult:
        self.reset()
        stack: List[GridPos] = [self.current]
        parent: Dict[GridPos, GridPos] = {}

        for _ in range(int(max_steps)):
            if self._done():
                break

            neighbors = [n for n in self._known_free_neighbors(self.current) if n not in self.explored]
            if neighbors:
                neighbors.sort(key=lambda p: (_turn_cost(self.current, p, self.heading), -self._unknown_gain(p), p))
                nxt = neighbors[0]
                parent.setdefault(nxt, self.current)
                stack.append(nxt)
                self._move_to(nxt)
                continue

            targets = set(self._known_unvisited_cells())
            route = self._bfs_to_targets(targets) if targets else []
            if len(route) > 1:
                self.replan_count += 1
                self._move_to(route[1])
                continue

            frontiers = set(self._frontier_cells())
            route = self._bfs_to_targets(frontiers) if frontiers else []
            if len(route) > 1:
                self.frontier_handoff_count += 1
                self.replan_count += 1
                self._move_to(route[1])
                continue

            while stack and stack[-1] == self.current:
                stack.pop()
            if not stack:
                break
            route = self._bfs_to_targets({stack[-1]})
            if len(route) <= 1:
                break
            self._move_to(route[1])

        return self._result("spiral_stc")


def _turn_cost(current: GridPos, nxt: GridPos, heading: int) -> int:
    cr, cc = current
    nr, nc = nxt
    delta = (nr - cr, nc - cc)
    try:
        direction = DIRS.index(delta)
    except ValueError:
        return 2
    diff = abs(direction - int(heading)) % 4
    return min(diff, 4 - diff)


def run_online_heuristic(
    algorithm: str,
    grid: np.ndarray,
    *,
    start: GridPos,
    sensor_range: int,
    max_steps: int,
) -> HeuristicResult:
    planner = OnlineGridCPPHeuristics(grid, start=start, sensor_range=sensor_range)
    if algorithm == "nearest_unvisited":
        return planner.run_nearest_unvisited(max_steps)
    if algorithm == "frontier_greedy":
        return planner.run_frontier_greedy(max_steps)
    if algorithm == "wall_follow":
        return planner.run_wall_follow(max_steps)
    if algorithm == "spiral_stc":
        return planner.run_spiral_stc(max_steps)
    raise ValueError(f"Unknown online heuristic: {algorithm}")

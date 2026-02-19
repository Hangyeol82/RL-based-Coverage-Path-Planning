import heapq
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .types import NODE_CLOSED, NODE_OPEN, GridPos


class CStarControlMixin:
    def _movement_mask(self) -> np.ndarray:
        # Conservative online motion: move only through known free cells.
        return self.known_grid == 0

    def _node_is_open(self, nid: int) -> bool:
        if nid not in self.nodes:
            return False
        if self.node_state.get(nid) != NODE_OPEN:
            return False
        n = self.nodes[nid]
        if n.node_type == "link":
            return True
        if n.node_type == "frontier":
            return n.pos in self.active_frontier_positions
        if n.node_type == "gate":
            return n.pos in self.active_gate_positions
        return True

    def _nodes_at_position(self, pos: GridPos) -> List[int]:
        ids = [nid for nid, node in self.nodes.items() if node.pos == pos]
        if not ids:
            return []
        priority = {
            "frontier": 0,
            "link": 1,
            "gate": 2,
            "region": 3,
            "hole_candidate": 4,
            "hole": 5,
        }
        ids.sort(key=lambda nid: (priority.get(self.nodes[nid].node_type, 99), nid))
        return ids

    def _refresh_current_graph_node(self):
        ids = self._nodes_at_position(self.current_node)
        self.current_graph_node_id = ids[0] if ids else None

    def _nearest_graph_node(self) -> Optional[int]:
        if not self.nodes:
            return None
        candidates = [
            nid
            for nid, n in self.nodes.items()
            if n.node_type in ("frontier", "link", "gate", "region")
        ]
        if not candidates:
            return None
        best = min(
            candidates,
            key=lambda nid: abs(self.nodes[nid].pos[0] - self.current_node[0])
            + abs(self.nodes[nid].pos[1] - self.current_node[1]),
        )
        return best

    def _neighbor_direction(self, base_id: int, nbr_id: int) -> Optional[str]:
        if base_id not in self.nodes or nbr_id not in self.nodes:
            return None
        base = self.nodes[base_id]
        nbr = self.nodes[nbr_id]
        if nbr.lap < base.lap:
            return "L"
        if nbr.lap > base.lap:
            return "R"
        if nbr.pos[0] < base.pos[0]:
            return "U"
        if nbr.pos[0] > base.pos[0]:
            return "D"
        return None

    def _directional_neighbors(
        self,
        cur_id: int,
        direction: str,
        adj: Dict[int, Set[int]],
        open_only: bool = False,
    ) -> List[int]:
        options = []
        for nid in adj.get(cur_id, set()):
            if open_only and not self._node_is_open(nid):
                continue
            d = self._neighbor_direction(cur_id, nid)
            if d != direction:
                continue
            key = self._edge_key(cur_id, nid)
            if key in self.edges:
                options.append((self.edges[key].weight, nid))
        options.sort(key=lambda x: (x[0], x[1]))
        return [nid for _, nid in options]

    def _directional_open_neighbor(
        self,
        cur_id: int,
        direction: str,
        adj: Dict[int, Set[int]],
    ) -> Optional[int]:
        options = self._directional_neighbors(cur_id, direction, adj, open_only=True)
        if not options:
            return None
        # Algorithm 1: random pick when multiple open neighbors exist on left/right lap.
        if direction in ("L", "R") and len(options) > 1:
            return random.choice(options)
        return options[0]

    def _create_link_node(
        self,
        cur_id: int,
        direction: str,
        traversable: np.ndarray,
    ) -> Optional[int]:
        if cur_id not in self.nodes:
            return None
        if direction not in ("U", "D"):
            return None

        dr = -self.lap_width if direction == "U" else self.lap_width
        base = self.nodes[cur_id].pos
        target = (base[0] + dr, base[1])
        if not self._inside(target[0], target[1]):
            return None
        if not traversable[target[0], target[1]]:
            return None

        existing_ids = self._nodes_at_position(target)
        if existing_ids:
            link_id = existing_ids[0]
        else:
            link_id, _ = self._add_node(
                pos=target,
                node_type="link",
                iteration=self.iteration,
                meta={"link": 1.0},
                dedupe=True,
            )

        self._try_connect_nodes(cur_id, link_id, "same_lap_adjacent", traversable)
        self._connect_same_lap_adjacency(link_id, traversable)
        self._connect_adjacent_laps(link_id, traversable)
        return link_id

    def _update_state_algorithm2(self, goal_id: Optional[int], traversable: np.ndarray):
        # Algorithm 2: update q(n_i) only when one of the vertical neighbors is closed.
        cur_id = self.current_graph_node_id
        if cur_id is None or cur_id not in self.nodes:
            return
        if self.node_state.get(cur_id) != NODE_OPEN:
            return

        adj = self._build_adjacency()
        up_neighbors = self._directional_neighbors(cur_id, "U", adj, open_only=False)
        down_neighbors = self._directional_neighbors(cur_id, "D", adj, open_only=False)

        up_closed = any(self.node_state.get(nid) == NODE_CLOSED for nid in up_neighbors)
        down_closed = any(self.node_state.get(nid) == NODE_CLOSED for nid in down_neighbors)
        # Boundary/end-node adaptation: missing U/D neighbor is treated as closed.
        up_missing = len(up_neighbors) == 0
        down_missing = len(down_neighbors) == 0
        if not (up_closed or down_closed or up_missing or down_missing):
            return

        self.node_state[cur_id] = NODE_CLOSED
        self.closed_graph_node_ids.add(cur_id)

        if goal_id is None or goal_id not in self.nodes:
            return
        if self._neighbor_direction(cur_id, goal_id) != "L":
            return

        up_open = any(self._node_is_open(nid) for nid in up_neighbors)
        down_open = any(self._node_is_open(nid) for nid in down_neighbors)
        if up_open:
            self._create_link_node(cur_id, "U", traversable)
        if down_open:
            self._create_link_node(cur_id, "D", traversable)

    def _graph_shortest_path_ids(self, start: int, goal: int) -> List[int]:
        if start == goal:
            return [start]
        adj = self._build_adjacency()
        pq = [(0.0, start)]
        dist: Dict[int, float] = {start: 0.0}
        prev: Dict[int, int] = {}

        while pq:
            d, cur = heapq.heappop(pq)
            if d > dist.get(cur, float("inf")):
                continue
            if cur == goal:
                out = [cur]
                while cur in prev:
                    cur = prev[cur]
                    out.append(cur)
                return list(reversed(out))

            for nxt in adj.get(cur, set()):
                key = self._edge_key(cur, nxt)
                if key not in self.edges:
                    continue
                nd = d + self.edges[key].weight
                if nd < dist.get(nxt, float("inf")):
                    dist[nxt] = nd
                    prev[nxt] = cur
                    heapq.heappush(pq, (nd, nxt))

        return []

    def _retreat_nodes(self, adj: Dict[int, Set[int]]) -> Set[int]:
        retreat = set()
        for nid in self.nodes:
            if self.node_state.get(nid) != NODE_CLOSED:
                continue
            for d in ("L", "U", "D", "R"):
                if self._directional_open_neighbor(nid, d, adj) is not None:
                    retreat.add(nid)
                    break
        return retreat

    def _escape_dead_end(self, cur_id: int, retreat_nodes: Set[int]) -> Optional[int]:
        best_path = None
        best_len = float("inf")
        for rid in retreat_nodes:
            if rid == cur_id:
                continue
            path = self._graph_shortest_path_ids(cur_id, rid)
            if len(path) >= 2 and len(path) < best_len:
                best_len = len(path)
                best_path = path
        if best_path is None:
            return None
        return best_path[1]

    def _select_goal_node_algorithm1(self) -> Optional[int]:
        if not self.nodes:
            return None
        if self.current_graph_node_id is None or self.current_graph_node_id not in self.nodes:
            return None

        adj = self._build_adjacency()
        cur_id = self.current_graph_node_id

        # Algorithm 1 priority: Left -> Up -> Down -> Right
        for d in ("L", "U", "D", "R"):
            nxt = self._directional_open_neighbor(cur_id, d, adj)
            if nxt is not None:
                return nxt

        # Dead-end escape
        retreat = self._retreat_nodes(adj)
        if retreat:
            return self._escape_dead_end(cur_id, retreat)
        return None

    def _component_touches_unknown(self, comp_set: Set[GridPos]) -> bool:
        for r, c in comp_set:
            for nr, nc in self._neighbors((r, c), connectivity=4):
                if self.known_grid[nr, nc] == -1:
                    return True
        return False

    def _manhattan(self, a: GridPos, b: GridPos) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _augment_hole_cells_for_tcp(self, hole_cells: Set[GridPos], radius: int = 2) -> Set[GridPos]:
        augmented = set(hole_cells)
        frontier = set(hole_cells)
        for _ in range(max(1, radius)):
            nxt_frontier: Set[GridPos] = set()
            for r, c in frontier:
                for nr, nc in self._neighbors((r, c), connectivity=4):
                    if self.known_grid[nr, nc] != 0:
                        continue
                    if self.explored[nr, nc]:
                        continue
                    p = (nr, nc)
                    if p in augmented:
                        continue
                    augmented.add(p)
                    nxt_frontier.add(p)
            if not nxt_frontier:
                break
            frontier = nxt_frontier
        return augmented

    def _collect_outside_unexplored_targets(
        self,
        hole_cells: Set[GridPos],
        traversable: np.ndarray,
    ) -> Set[GridPos]:
        outside = {
            p
            for p in self.active_frontier_positions
            if p not in hole_cells and traversable[p[0], p[1]] and (not self.explored[p[0], p[1]])
        }
        if outside:
            return outside

        free_unexplored = np.argwhere((self.known_grid == 0) & (~self.explored))
        for r, c in free_unexplored:
            p = (int(r), int(c))
            if p in hole_cells:
                continue
            if not traversable[p[0], p[1]]:
                continue
            outside.add(p)
            if len(outside) >= 256:
                break
        return outside

    def _choose_hole_exit_hint(self, hole_cells: Set[GridPos], traversable: np.ndarray) -> Optional[GridPos]:
        if not hole_cells:
            return None

        outside_targets = self._collect_outside_unexplored_targets(hole_cells, traversable)
        if not outside_targets:
            return None

        best = None
        best_cost = float("inf")
        for p in hole_cells:
            d = min(self._manhattan(p, q) for q in outside_targets)
            if d < best_cost:
                best_cost = d
                best = p
        return best

    def _detect_confirmed_hole_components(self) -> List[Set[GridPos]]:
        """
        Coverage-hole criterion (user-defined):
        - Virtual obstacles = real obstacles + already covered path.
        - If an unvisited component is blocked by this virtual obstacle field and
          it does not touch unknown, it is a confirmed coverage hole.
        """
        candidate_mask = (self.known_grid == 0) & (~self.explored)
        if not np.any(candidate_mask):
            return []

        virtual_traversable = candidate_mask.copy()
        virtual_traversable[self.current_node[0], self.current_node[1]] = True

        components = self._connected_components_from_mask(candidate_mask)
        holes: List[Set[GridPos]] = []
        for comp in components:
            comp_set = set(comp)
            if not comp_set:
                continue

            # If a component touches unknown, it may still expand and is not a hole.
            if self._component_touches_unknown(comp_set):
                continue

            # If robot can reach the component without crossing covered path
            # (virtual obstacles), this is not a blocked hole.
            p_virtual = self._shortest_path_to_any(self.current_node, comp_set, virtual_traversable)
            if len(p_virtual) > 1:
                continue

            holes.append(comp_set)
        return holes

    def _predict_hole_if_visit_cell(
        self,
        visit_cell: GridPos,
        candidate_mask: np.ndarray,
    ) -> Optional[Set[GridPos]]:
        if not candidate_mask[visit_cell[0], visit_cell[1]]:
            return None

        # Current reachable unvisited space (current cell is kept traversable for planning).
        traversable_now = candidate_mask.copy()
        traversable_now[self.current_node[0], self.current_node[1]] = True
        if len(self._shortest_path_to_any(self.current_node, {visit_cell}, traversable_now)) <= 1:
            return None

        # Simulate that the next visited cell becomes a virtual obstacle.
        traversable_after = traversable_now.copy()
        traversable_after[visit_cell[0], visit_cell[1]] = False

        components_after = self._connected_components_from_mask(traversable_after)
        best_comp = None
        best_size = -1
        for comp in components_after:
            comp_set = set(comp)
            if not comp_set:
                continue
            if self._component_touches_unknown(comp_set):
                continue

            path_now = self._shortest_path_to_any(self.current_node, comp_set, traversable_now)
            if len(path_now) <= 1:
                continue
            path_after = self._shortest_path_to_any(self.current_node, comp_set, traversable_after)
            if len(path_after) > 1:
                continue

            if len(comp_set) > best_size:
                best_size = len(comp_set)
                best_comp = comp_set
        return best_comp

    def _predict_hole_component_from_waypoint(
        self,
        waypoint: Optional[GridPos],
        traversable: np.ndarray,
    ) -> Optional[Set[GridPos]]:
        if waypoint is None:
            return None

        plan = self._a_star(self.current_node, waypoint, traversable)
        if len(plan) <= 1:
            return None

        candidate_mask = (self.known_grid == 0) & (~self.explored)
        horizon = min(len(plan) - 1, max(3, self.sensor_range))
        best_comp = None
        best_size = -1

        for p in plan[1 : horizon + 1]:
            comp = self._predict_hole_if_visit_cell(p, candidate_mask)
            if comp is not None and len(comp) > best_size:
                best_comp = comp
                best_size = len(comp)

        if best_comp is not None:
            return best_comp

        # Extra local probe catches narrow-gate closures right around the robot.
        for p in self._neighbors(self.current_node, connectivity=4):
            if not traversable[p[0], p[1]]:
                continue
            if self.explored[p[0], p[1]]:
                continue
            comp = self._predict_hole_if_visit_cell(p, candidate_mask)
            if comp is not None and len(comp) > best_size:
                best_comp = comp
                best_size = len(comp)
        return best_comp

    def _grid_shortest_distances(
        self,
        start: GridPos,
        traversable: np.ndarray,
    ) -> Dict[GridPos, int]:
        if not traversable[start[0], start[1]]:
            return {}
        dist: Dict[GridPos, int] = {start: 0}
        q = deque([start])
        while q:
            cur = q.popleft()
            cur_d = dist[cur]
            for nxt in self._neighbors(cur, connectivity=4):
                if nxt in dist:
                    continue
                if not traversable[nxt[0], nxt[1]]:
                    continue
                dist[nxt] = cur_d + 1
                q.append(nxt)
        return dist

    def _pairwise_route_costs(
        self,
        nodes: List[GridPos],
        traversable: np.ndarray,
    ) -> List[List[float]]:
        n = len(nodes)
        inf = float("inf")
        costs: List[List[float]] = [[inf] * n for _ in range(n)]
        for i in range(n):
            costs[i][i] = 0.0

        for i, p in enumerate(nodes):
            dmap = self._grid_shortest_distances(p, traversable)
            for j in range(i + 1, n):
                q = nodes[j]
                d = dmap.get(q)
                if d is None:
                    continue
                costs[i][j] = float(d)
                costs[j][i] = float(d)
        return costs

    def _route_cost(self, route: List[int], costs: List[List[float]]) -> float:
        if len(route) <= 1:
            return 0.0
        total = 0.0
        for i in range(1, len(route)):
            w = costs[route[i - 1]][route[i]]
            if w == float("inf"):
                return float("inf")
            total += w
        return total

    def _tsp_held_karp_path(
        self,
        costs: List[List[float]],
        start_idx: int,
        end_idx: Optional[int],
    ) -> Optional[List[int]]:
        n = len(costs)
        if n == 0:
            return []
        if n == 1:
            return [0]

        full_mask = (1 << n) - 1
        dp: Dict[Tuple[int, int], float] = {(1 << start_idx, start_idx): 0.0}
        parent: Dict[Tuple[int, int], int] = {}

        for mask in range(1 << n):
            if not (mask & (1 << start_idx)):
                continue
            for j in range(n):
                key = (mask, j)
                if key not in dp:
                    continue
                cur_cost = dp[key]
                for k in range(n):
                    if mask & (1 << k):
                        continue
                    w = costs[j][k]
                    if w == float("inf"):
                        continue
                    nxt_mask = mask | (1 << k)
                    nxt_key = (nxt_mask, k)
                    cand = cur_cost + w
                    if cand < dp.get(nxt_key, float("inf")):
                        dp[nxt_key] = cand
                        parent[nxt_key] = j

        end_choice = None
        if end_idx is not None:
            if (full_mask, end_idx) not in dp:
                return None
            end_choice = end_idx
        else:
            best_cost = float("inf")
            for j in range(n):
                val = dp.get((full_mask, j))
                if val is not None and val < best_cost:
                    best_cost = val
                    end_choice = j
            if end_choice is None:
                return None

        route_rev = [end_choice]
        mask = full_mask
        cur = end_choice
        while not (mask == (1 << start_idx) and cur == start_idx):
            p_key = (mask, cur)
            if p_key not in parent:
                return None
            prev = parent[p_key]
            route_rev.append(prev)
            mask ^= 1 << cur
            cur = prev
        route_rev.reverse()
        return route_rev

    def _tsp_nearest_neighbor_path(
        self,
        costs: List[List[float]],
        start_idx: int,
        end_idx: Optional[int],
    ) -> List[int]:
        n = len(costs)
        if n <= 1:
            return [0] if n == 1 else []
        remaining = set(range(n))
        remaining.discard(start_idx)
        route = [start_idx]

        fixed_end = end_idx is not None and end_idx != start_idx and end_idx in remaining
        if fixed_end:
            remaining.discard(end_idx)

        while remaining:
            cur = route[-1]
            nxt = min(remaining, key=lambda idx: costs[cur][idx])
            route.append(nxt)
            remaining.remove(nxt)

        if fixed_end:
            route.append(end_idx)  # type: ignore[arg-type]
        return route

    def _tsp_two_opt(
        self,
        route: List[int],
        costs: List[List[float]],
        fixed_end: bool,
    ) -> List[int]:
        if len(route) < 4:
            return route

        start_i = 1
        end_limit = len(route) - 1 if fixed_end else len(route)
        improved = True
        best = route[:]
        best_cost = self._route_cost(best, costs)

        while improved:
            improved = False
            for i in range(start_i, end_limit - 2):
                for j in range(i + 1, end_limit - 1):
                    if i == 0:
                        continue
                    cand = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                    cand_cost = self._route_cost(cand, costs)
                    if cand_cost + 1e-9 < best_cost:
                        best = cand
                        best_cost = cand_cost
                        improved = True
                        break
                if improved:
                    break
        return best

    def _build_hole_tcp_route(
        self,
        hole_cells: Set[GridPos],
        entry: GridPos,
        traversable: np.ndarray,
        exit_hint: Optional[GridPos] = None,
    ) -> List[GridPos]:
        if not hole_cells:
            return []
        remaining = set(hole_cells)
        if entry not in remaining:
            entry = min(remaining, key=lambda p: self._manhattan(self.current_node, p))

        tail = None
        if (
            exit_hint is not None
            and exit_hint in remaining
            and len(remaining) > 1
            and exit_hint != entry
        ):
            tail = exit_hint
        nodes = list(remaining)
        if tail is not None and tail not in nodes:
            nodes.append(tail)
        if entry not in nodes:
            nodes.append(entry)

        start_idx = nodes.index(entry)
        end_idx = nodes.index(tail) if tail is not None else None
        costs = self._pairwise_route_costs(nodes, traversable)

        # Exact TSP path for small holes; fallback to improved heuristic for larger holes.
        exact_limit = 13
        route_idx: Optional[List[int]] = None
        if len(nodes) <= exact_limit:
            route_idx = self._tsp_held_karp_path(costs, start_idx, end_idx)

        if route_idx is None:
            route_idx = self._tsp_nearest_neighbor_path(costs, start_idx, end_idx)
            fixed_end = end_idx is not None and len(route_idx) > 1 and route_idx[-1] == end_idx
            route_idx = self._tsp_two_opt(route_idx, costs, fixed_end=fixed_end)

        return [nodes[i] for i in route_idx]

    def _load_hole_tcp_from_component(self, hole_comp: Set[GridPos], traversable: np.ndarray) -> bool:
        hole_cells = self._augment_hole_cells_for_tcp(hole_comp)
        entry_path = self._shortest_path_to_any(self.current_node, hole_cells, traversable)
        if len(entry_path) <= 1:
            return False

        entry = entry_path[-1]
        exit_hint = self._choose_hole_exit_hint(hole_cells, traversable)
        route = self._build_hole_tcp_route(hole_cells, entry, traversable, exit_hint)
        queue = [p for p in route if traversable[p[0], p[1]] and (not self.explored[p[0], p[1]])]
        queue_set = set(queue)

        outside_targets = self._collect_outside_unexplored_targets(hole_cells, traversable)
        if queue and outside_targets:
            exit_target = min(outside_targets, key=lambda p: self._manhattan(queue[-1], p))
            exit_path = self._a_star(queue[-1], exit_target, traversable)
            if len(exit_path) > 1:
                for p in exit_path[1:]:
                    if (
                        traversable[p[0], p[1]]
                        and (not self.explored[p[0], p[1]])
                        and p not in queue_set
                    ):
                        queue.append(p)
                        queue_set.add(p)

        if not queue:
            return False

        self.active_hole_cells = set(hole_cells)
        self.hole_tcp_queue = queue
        return True

    def _refresh_hole_tcp_queue(self, traversable: np.ndarray):
        while self.hole_tcp_queue:
            p = self.hole_tcp_queue[0]
            if p == self.current_node:
                self.hole_tcp_queue.pop(0)
                continue
            if not traversable[p[0], p[1]] or self.explored[p[0], p[1]]:
                self.hole_tcp_queue.pop(0)
                continue
            break

        if self.hole_tcp_queue:
            return

        self.active_hole_cells = set()
        hole_components = self._detect_confirmed_hole_components()
        if not hole_components:
            return

        # Pick closest reachable confirmed hole, then build TCP order inside it.
        best_comp = None
        best_len = float("inf")
        for comp_set in hole_components:
            hole_cells = self._augment_hole_cells_for_tcp(comp_set)
            p_real = self._shortest_path_to_any(self.current_node, hole_cells, traversable)
            if len(p_real) <= 1:
                continue
            if len(p_real) < best_len:
                best_len = len(p_real)
                best_comp = comp_set

        if best_comp is None:
            return

        self._load_hole_tcp_from_component(best_comp, traversable)

    def _next_hole_waypoint(self, traversable: np.ndarray) -> Optional[GridPos]:
        self._refresh_hole_tcp_queue(traversable)
        if not self.hole_tcp_queue:
            return None
        return self.hole_tcp_queue[0]

    def _select_non_hole_waypoint(self, traversable: np.ndarray) -> Optional[GridPos]:
        goal_id = self._select_goal_node_algorithm1()
        self._update_state_algorithm2(goal_id, traversable)
        if goal_id is not None and goal_id in self.nodes:
            goal_pos = self.nodes[goal_id].pos
            plan = self._a_star(self.current_node, goal_pos, traversable)
            if len(plan) > 1:
                self.current_goal_node_id = goal_id
                return goal_pos
            # Graph-level open node can be temporarily unreachable in online map.
            # Close it for now and try fallback targets.
            self.node_state[goal_id] = NODE_CLOSED
            self.closed_graph_node_ids.add(goal_id)
        self.current_goal_node_id = None

        # Keep zigzag continuity: finish unexplored cells of current lap before
        # jumping to global frontier targets.
        same_lap_wp = self._same_lap_waypoint(traversable)
        if same_lap_wp is not None:
            return same_lap_wp

        # Fallback 1: nearest active frontier/gate cell (may exist even if RCG node was pruned).
        active_targets = {
            p
            for p in (self.active_frontier_positions | self.active_gate_positions)
            if p != self.current_node and traversable[p[0], p[1]]
        }
        path_active = self._shortest_path_to_any(self.current_node, active_targets, traversable)
        if len(path_active) > 1:
            return path_active[-1]

        # Fallback: nearest known-free unexplored cell.
        free_unexplored = np.argwhere((self.known_grid == 0) & (~self.explored))
        if len(free_unexplored) == 0:
            return None
        unexplored_targets = {
            (int(r), int(c))
            for r, c in free_unexplored
            if (int(r), int(c)) != self.current_node
        }
        path_unexplored = self._shortest_path_to_any(self.current_node, unexplored_targets, traversable)
        if len(path_unexplored) > 1:
            return path_unexplored[-1]
        return None

    def _select_next_waypoint(self) -> Optional[GridPos]:
        self._refresh_current_graph_node()
        traversable = self._movement_mask()

        normal_wp = self._select_non_hole_waypoint(traversable)

        # Predictive hole handling: if continuing current route will force a hole,
        # switch immediately to hole TCP routing.
        predicted = self._predict_hole_component_from_waypoint(normal_wp, traversable)
        if predicted is not None:
            self.hole_tcp_queue = []
            self.active_hole_cells = set()
            self._load_hole_tcp_from_component(predicted, traversable)
            hole_wp = self._next_hole_waypoint(traversable)
            if hole_wp is not None:
                self.current_goal_node_id = None
                return hole_wp

        # Confirmed hole handling if any pending.
        hole_wp = self._next_hole_waypoint(traversable)
        if hole_wp is not None:
            self.current_goal_node_id = None
            return hole_wp

        return normal_wp

    def step(self) -> bool:
        prev_known = self.known_grid.copy()
        self._sense()
        self.explored[self.current_node[0], self.current_node[1]] = True

        newly_known_free = (self.known_grid == 0) & (prev_known == -1)
        self._expand_rcg(newly_known_free)
        self._refresh_current_graph_node()

        if not self.current_plan:
            self.current_waypoint = self._select_next_waypoint()
            if self.current_waypoint is None:
                return False
            plan = self._a_star(self.current_node, self.current_waypoint, self._movement_mask())
            if len(plan) <= 1:
                self.current_plan = []
            else:
                self.current_plan = plan[1:]

        if not self.current_plan:
            return False

        next_node = self.current_plan.pop(0)
        if self.known_grid[next_node[0], next_node[1]] != 0:
            self.current_plan = []
            self.current_goal_node_id = None
            return True

        self.current_node = next_node
        self.path.append(next_node)
        self._refresh_current_graph_node()
        if (
            self.current_goal_node_id is not None
            and self.current_goal_node_id in self.nodes
            and self.current_node == self.nodes[self.current_goal_node_id].pos
        ):
            self.current_graph_node_id = self.current_goal_node_id
            self.current_goal_node_id = None
        self.iteration += 1
        return True

    def run(self, max_steps: Optional[int] = None) -> List[GridPos]:
        if max_steps is None:
            max_steps = self.rows * self.cols * 8
        steps = 0
        while steps < max_steps and self.step():
            steps += 1
        return self.path

    def rcg_stats(self) -> Dict[str, int]:
        edge_types: Dict[str, int] = defaultdict(int)
        for e in self.edges.values():
            edge_types[e.edge_type] += 1
        open_count = sum(1 for nid in self.nodes if self.node_state.get(nid) == NODE_OPEN)
        closed_count = sum(1 for nid in self.nodes if self.node_state.get(nid) == NODE_CLOSED)
        return {
            "nodes_total": len(self.nodes),
            "edges_total": len(self.edges),
            "open_nodes": open_count,
            "closed_nodes": closed_count,
            "frontier_nodes": len(self.frontier_node_ids),
            "link_nodes": len(self.link_node_ids),
            "gate_nodes": len(self.gate_node_ids),
            "region_nodes": len(self.region_node_ids),
            "hole_candidate_nodes": len(self.hole_candidate_node_ids),
            "hole_nodes": len(self.hole_node_ids),
            "edge_same_lap_adjacent": edge_types.get("same_lap_adjacent", 0),
            "edge_adjacent_lap": edge_types.get("adjacent_lap", 0),
            "edge_region_adjacent": edge_types.get("region_adjacent", 0),
            "edge_gate_region": edge_types.get("gate_region", 0),
            "total_pruned_nodes": self.total_pruned_nodes,
            "total_pruned_edges": self.total_pruned_edges,
        }

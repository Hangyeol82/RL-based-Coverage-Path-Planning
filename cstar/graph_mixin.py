import heapq
import math
import random
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .types import NODE_CLOSED, NODE_OPEN, GridPos, RCGEdge, RCGNode


class CStarGraphMixin:
    def _inside(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _neighbors(self, node: GridPos, connectivity: int = 4) -> List[GridPos]:
        x, y = node
        if connectivity == 8:
            dirs = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
        else:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        out = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            if self._inside(nx, ny):
                out.append((nx, ny))
        return out

    def _lap_of(self, pos: GridPos) -> int:
        # Laps are vertical strips by column index.
        return pos[1] // self.lap_width

    def _orient(self, a: GridPos, b: GridPos, c: GridPos) -> int:
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        if val == 0:
            return 0
        return 1 if val > 0 else -1

    def _on_segment(self, a: GridPos, b: GridPos, c: GridPos) -> bool:
        return (
            min(a[0], c[0]) <= b[0] <= max(a[0], c[0])
            and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])
        )

    def _segments_cross(self, p1: GridPos, p2: GridPos, q1: GridPos, q2: GridPos) -> bool:
        o1 = self._orient(p1, p2, q1)
        o2 = self._orient(p1, p2, q2)
        o3 = self._orient(q1, q2, p1)
        o4 = self._orient(q1, q2, p2)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and self._on_segment(p1, q1, p2):
            return True
        if o2 == 0 and self._on_segment(p1, q2, p2):
            return True
        if o3 == 0 and self._on_segment(q1, p1, q2):
            return True
        if o4 == 0 and self._on_segment(q1, p2, q2):
            return True
        return False

    def _would_cross_existing_edge(self, u: int, v: int) -> bool:
        if u not in self.nodes or v not in self.nodes:
            return False
        p1 = self.nodes[u].pos
        p2 = self.nodes[v].pos
        for e in self.edges.values():
            a = e.u
            b = e.v
            if len({u, v, a, b}) < 4:
                continue
            q1 = self.nodes[a].pos
            q2 = self.nodes[b].pos
            if self._segments_cross(p1, p2, q1, q2):
                return True
        return False

    # ---------- RCG node / edge ----------
    def _add_node(
        self,
        pos: GridPos,
        node_type: str,
        iteration: int,
        meta: Optional[Dict[str, float]] = None,
        dedupe: bool = True,
    ) -> Tuple[int, bool]:
        key = (node_type, pos)
        if dedupe and key in self.node_key_to_id:
            return self.node_key_to_id[key], False

        node_id = self.next_node_id
        self.next_node_id += 1

        node = RCGNode(
            node_id=node_id,
            pos=pos,
            node_type=node_type,
            lap=self._lap_of(pos),
            iteration=iteration,
            meta=meta or {},
        )
        self.nodes[node_id] = node
        self.node_ids_by_lap[node.lap].add(node_id)
        self.node_key_to_id[key] = node_id
        self.node_state[node_id] = NODE_OPEN

        if node_type == "region":
            self.region_node_ids.add(node_id)
        elif node_type == "gate":
            self.gate_node_ids.add(node_id)
        elif node_type == "frontier":
            self.frontier_node_ids.add(node_id)
        elif node_type == "link":
            self.link_node_ids.add(node_id)
            self.frontier_node_ids.add(node_id)
        elif node_type == "hole_candidate":
            self.hole_candidate_node_ids.add(node_id)
        elif node_type == "hole":
            self.hole_node_ids.add(node_id)

        return node_id, True

    def _add_edge(self, u: int, v: int, weight: float, edge_type: str, tentative: bool = False):
        if u == v:
            return
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        if self._would_cross_existing_edge(a, b):
            return
        if key in self.edges:
            if weight < self.edges[key].weight:
                self.edges[key].weight = weight
                self.edges[key].edge_type = edge_type
                self.edges[key].tentative = tentative
            return
        self.edges[key] = RCGEdge(u=a, v=b, weight=weight, edge_type=edge_type, tentative=tentative)

    # ---------- Feasibility ----------
    def _build_traversable_mask(self) -> np.ndarray:
        # Unknown is treated as blocked for edge feasibility.
        blocked = self.known_grid != 0
        obs = self.known_grid == 1
        inflated = blocked.copy()

        eff_infl = self.obstacle_inflation + max(0, (self.min_traversable_width - 1) // 2)
        if eff_infl <= 0:
            return ~inflated

        obs_cells = np.argwhere(obs)
        for r, c in obs_cells:
            for dr in range(-eff_infl, eff_infl + 1):
                for dc in range(-eff_infl, eff_infl + 1):
                    if abs(dr) + abs(dc) > eff_infl:
                        continue
                    nr, nc = r + dr, c + dc
                    if self._inside(nr, nc):
                        inflated[nr, nc] = True

        return ~inflated

    def _line_of_sight_free(self, a: GridPos, b: GridPos, traversable: np.ndarray) -> bool:
        x0, y0 = a
        x1, y1 = b

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0
        while True:
            if not traversable[x, y]:
                return False
            if x == x1 and y == y1:
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    def _a_star(self, start: GridPos, goal: GridPos, traversable: np.ndarray) -> List[GridPos]:
        if not traversable[start[0], start[1]] or not traversable[goal[0], goal[1]]:
            return []
        if start == goal:
            return [start]

        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        came_from: Dict[GridPos, GridPos] = {}
        g_score: Dict[GridPos, int] = {start: 0}

        def h(p: GridPos) -> int:
            return abs(p[0] - goal[0]) + abs(p[1] - goal[1])

        while open_set:
            _, g, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for n in self._neighbors(current, connectivity=4):
                if not traversable[n[0], n[1]]:
                    continue
                tentative = g + 1
                if n not in g_score or tentative < g_score[n]:
                    g_score[n] = tentative
                    heapq.heappush(open_set, (tentative + h(n), tentative, n))
                    came_from[n] = current

        return []

    def _shortest_path_to_any(
        self,
        start: GridPos,
        targets: Set[GridPos],
        traversable: np.ndarray,
    ) -> List[GridPos]:
        # Uniform-cost grid motion, so BFS gives the nearest reachable target.
        if not targets:
            return []
        if not traversable[start[0], start[1]]:
            return []
        if start in targets:
            return [start]

        parent: Dict[GridPos, GridPos] = {}
        visited: Set[GridPos] = {start}
        q = deque([start])

        while q:
            cur = q.popleft()
            for nxt in self._neighbors(cur, connectivity=4):
                if nxt in visited:
                    continue
                if not traversable[nxt[0], nxt[1]]:
                    continue
                visited.add(nxt)
                parent[nxt] = cur
                if nxt in targets:
                    path = [nxt]
                    while path[-1] != start:
                        path.append(parent[path[-1]])
                    path.reverse()
                    return path
                q.append(nxt)
        return []

    def _last_move(self) -> Optional[Tuple[int, int]]:
        if len(self.path) < 2:
            return None
        a = self.path[-2]
        b = self.path[-1]
        return (b[0] - a[0], b[1] - a[1])

    def _same_lap_waypoint(self, traversable: np.ndarray) -> Optional[GridPos]:
        cur = self.current_node
        lap = self._lap_of(cur)
        c0 = lap * self.lap_width
        c1 = min(self.cols - 1, c0 + self.lap_width - 1)

        last = self._last_move()
        if last is not None and last[0] != 0:
            dr_order = [1 if last[0] > 0 else -1, -1 if last[0] > 0 else 1]
        else:
            # If there is no vertical momentum, prefer moving away from nearest boundary.
            dr_order = [1, -1] if cur[0] <= self.rows // 2 else [-1, 1]

        # Prefer local vertical continuation inside current lap.
        for dr in dr_order:
            nr, nc = cur[0] + dr, cur[1]
            if not self._inside(nr, nc):
                continue
            if not (c0 <= nc <= c1):
                continue
            if not traversable[nr, nc]:
                continue
            if self.explored[nr, nc]:
                continue
            return (nr, nc)

        same_lap_targets: Set[GridPos] = set()
        free_unexplored = np.argwhere((self.known_grid == 0) & (~self.explored))
        for r, c in free_unexplored:
            rr = int(r)
            cc = int(c)
            if (rr, cc) == cur:
                continue
            if c0 <= cc <= c1:
                same_lap_targets.add((rr, cc))

        if not same_lap_targets:
            return None
        path = self._shortest_path_to_any(cur, same_lap_targets, traversable)
        if len(path) > 1:
            return path[-1]
        return None

    def _edge_feasible_and_cost(
        self,
        pos_a: GridPos,
        pos_b: GridPos,
        traversable: np.ndarray,
    ) -> Tuple[bool, float]:
        if self._line_of_sight_free(pos_a, pos_b, traversable):
            return True, math.dist(pos_a, pos_b)
        path = self._a_star(pos_a, pos_b, traversable)
        if path:
            return True, float(max(1, len(path) - 1))
        return False, float("inf")

    # ---------- Frontier / region extraction ----------
    def _frontier_cells(self) -> Set[GridPos]:
        frontiers: Set[GridPos] = set()
        free_mask = self.known_grid == 0
        for r in range(self.rows):
            for c in range(self.cols):
                if not free_mask[r, c]:
                    continue
                for nr, nc in self._neighbors((r, c), connectivity=4):
                    if self.known_grid[nr, nc] == -1:
                        frontiers.add((r, c))
                        break
        return frontiers

    def _sampling_front(self, frontiers: Set[GridPos]) -> Set[GridPos]:
        cx, cy = self.current_node
        out = set()
        for r, c in frontiers:
            if abs(r - cx) <= self.sensor_range and abs(c - cy) <= self.sensor_range:
                out.add((r, c))
        return out

    def _cluster_points(self, points: Set[GridPos], connectivity: int = 8) -> List[List[GridPos]]:
        points_set = set(points)
        visited: Set[GridPos] = set()
        clusters: List[List[GridPos]] = []

        for p in points_set:
            if p in visited:
                continue
            q = deque([p])
            visited.add(p)
            cluster = []
            while q:
                cur = q.popleft()
                cluster.append(cur)
                for n in self._neighbors(cur, connectivity=connectivity):
                    if n in points_set and n not in visited:
                        visited.add(n)
                        q.append(n)
            clusters.append(cluster)
        return clusters

    def _progressive_samples(self, cluster: List[GridPos]) -> List[GridPos]:
        if not cluster:
            return []
        ordered = sorted(cluster)
        stride_samples = ordered[:: self.frontier_sample_stride]

        cr = int(round(sum(p[0] for p in cluster) / len(cluster)))
        cc = int(round(sum(p[1] for p in cluster) / len(cluster)))
        centroid_cell = min(cluster, key=lambda p: abs(p[0] - cr) + abs(p[1] - cc))

        sampled = set(stride_samples)
        sampled.add(centroid_cell)
        return sorted(sampled)

    def _connected_components_from_mask(self, mask: np.ndarray) -> List[List[GridPos]]:
        visited = np.zeros_like(mask, dtype=bool)
        components: List[List[GridPos]] = []

        for r in range(self.rows):
            for c in range(self.cols):
                if not mask[r, c] or visited[r, c]:
                    continue
                q = deque([(r, c)])
                visited[r, c] = True
                comp = []
                while q:
                    x, y = q.popleft()
                    comp.append((x, y))
                    for nx, ny in self._neighbors((x, y), connectivity=4):
                        if mask[nx, ny] and not visited[nx, ny]:
                            visited[nx, ny] = True
                            q.append((nx, ny))
                components.append(comp)

        return components

    def _component_boundary(self, comp_set: Set[GridPos]) -> List[GridPos]:
        boundary = []
        for p in comp_set:
            for n in self._neighbors(p, connectivity=4):
                if n not in comp_set:
                    boundary.append(p)
                    break
        return boundary

    def _is_gate_cluster(self, cluster: List[GridPos]) -> bool:
        if len(cluster) < self.gate_min_cluster_size:
            return False
        rows = [p[0] for p in cluster]
        cols = [p[1] for p in cluster]
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        neck = min(height, width)
        span = max(height, width)
        return neck <= self.gate_max_width and span >= neck * 2

    def _ensure_region_nodes(
        self,
        components: List[List[GridPos]],
        newly_known_free: np.ndarray,
    ) -> Tuple[List[int], List[Tuple[int, Set[GridPos], List[GridPos]]]]:
        new_region_node_ids = []
        region_infos: List[Tuple[int, Set[GridPos], List[GridPos]]] = []

        for comp in components:
            comp_set = set(comp)
            region_id = None
            for nid in self.region_node_ids:
                if self.nodes[nid].pos in comp_set:
                    region_id = nid
                    break

            if region_id is None:
                if not any(newly_known_free[r, c] for r, c in comp):
                    # Component existed before and no anchor in graph yet.
                    # Create one lazily at first observation.
                    pass

                cr = int(round(sum(p[0] for p in comp) / len(comp)))
                cc = int(round(sum(p[1] for p in comp) / len(comp)))
                centroid_cell = min(comp, key=lambda p: abs(p[0] - cr) + abs(p[1] - cc))
                region_id, created = self._add_node(
                    pos=centroid_cell,
                    node_type="region",
                    iteration=self.iteration,
                    meta={"size": float(len(comp))},
                    dedupe=True,
                )
                if created:
                    new_region_node_ids.append(region_id)

            boundary = self._component_boundary(comp_set)
            region_infos.append((region_id, comp_set, boundary))

        return new_region_node_ids, region_infos

    def _obstacle_boundary_cells(self) -> Set[GridPos]:
        # A_o: free cells adjacent to known obstacles.
        out: Set[GridPos] = set()
        for r in range(self.rows):
            for c in range(self.cols):
                if self.known_grid[r, c] != 0:
                    continue
                for nr, nc in self._neighbors((r, c), connectivity=4):
                    if self.known_grid[nr, nc] == 1:
                        out.add((r, c))
                        break
        return out

    def _distance_to_cell_set(self, p: Tuple[float, float], cells: Set[GridPos]) -> float:
        if not cells:
            return float("inf")
        return min(abs(p[0] - r) + abs(p[1] - c) for r, c in cells)

    def _sampling_front_boundary(self, sample_front: Set[GridPos]) -> Set[GridPos]:
        if not sample_front:
            return set()
        out: Set[GridPos] = set()
        for p in sample_front:
            for n in self._neighbors(p, connectivity=4):
                if n not in sample_front:
                    out.add(p)
                    break
        return out

    def _nodes_adjacent_to_boundary(
        self,
        node_ids: Set[int],
        boundary_cells: Set[GridPos],
    ) -> Set[int]:
        if not boundary_cells:
            return set()

        expanded = set(boundary_cells)
        for r, c in list(boundary_cells):
            for nr, nc in self._neighbors((r, c), connectivity=4):
                expanded.add((nr, nc))

        out: Set[int] = set()
        for nid in node_ids:
            if nid in self.nodes and self.nodes[nid].pos in expanded:
                out.add(nid)
        return out

    # ---------- RCG expansion ----------
    def _try_connect_nodes(
        self,
        u: int,
        v: int,
        edge_type: str,
        traversable: np.ndarray,
        tentative: bool = False,
    ):
        feasible, cost = self._edge_feasible_and_cost(self.nodes[u].pos, self.nodes[v].pos, traversable)
        if feasible:
            self._add_edge(u, v, cost, edge_type=edge_type, tentative=tentative)

    def _connect_same_lap_adjacency(self, nid: int, traversable: np.ndarray):
        node = self.nodes[nid]
        same = [x for x in self.node_ids_by_lap[node.lap] if x != nid]
        if not same:
            return

        # Same lap: connect nearest nodes above/below (row direction).
        r = node.pos[0]
        above = None
        below = None
        above_dist = float("inf")
        below_dist = float("inf")
        for oid in same:
            rr = self.nodes[oid].pos[0]
            d = abs(rr - r)
            if rr < r and d < above_dist:
                above = oid
                above_dist = d
            elif rr > r and d < below_dist:
                below = oid
                below_dist = d

        if above is not None:
            self._try_connect_nodes(nid, above, "same_lap_adjacent", traversable)
        if below is not None:
            self._try_connect_nodes(nid, below, "same_lap_adjacent", traversable)

    def _connect_adjacent_laps(self, nid: int, traversable: np.ndarray):
        node = self.nodes[nid]
        lap = node.lap
        threshold = math.sqrt(2.0) * self.lap_width

        for adj_lap in [lap - 1, lap + 1]:
            for oid in self.node_ids_by_lap.get(adj_lap, set()):
                if oid == nid:
                    continue
                d = math.dist(node.pos, self.nodes[oid].pos)
                if d <= threshold:
                    self._try_connect_nodes(nid, oid, "adjacent_lap", traversable)

    def _connect_region_adjacency(
        self,
        region_infos: List[Tuple[int, Set[GridPos], List[GridPos]]],
        traversable: np.ndarray,
    ):
        n = len(region_infos)
        for i in range(n):
            rid_a, _, bnd_a = region_infos[i]
            for j in range(i + 1, n):
                rid_b, _, bnd_b = region_infos[j]
                best = None
                best_d = float("inf")
                for pa in bnd_a:
                    for pb in bnd_b:
                        d = abs(pa[0] - pb[0]) + abs(pa[1] - pb[1])
                        if d < best_d:
                            best_d = d
                            best = (pa, pb)
                        if best_d <= self.adjacency_cell_dist:
                            break
                    if best_d <= self.adjacency_cell_dist:
                        break

                if best is not None and best_d <= self.adjacency_cell_dist:
                    self._try_connect_nodes(rid_a, rid_b, "region_adjacent", traversable)

    def _connect_gate_region_links(
        self,
        gate_ids: List[int],
        region_infos: List[Tuple[int, Set[GridPos], List[GridPos]]],
        traversable: np.ndarray,
    ):
        for gid in gate_ids:
            gp = self.nodes[gid].pos
            for rid, _, boundary in region_infos:
                close = any(abs(gp[0] - bp[0]) + abs(gp[1] - bp[1]) <= 1 for bp in boundary)
                if close:
                    self._try_connect_nodes(gid, rid, "gate_region", traversable)

    # ---------- RCG pruning ----------
    def _edge_key(self, u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    def _build_adjacency(self) -> Dict[int, Set[int]]:
        adj: Dict[int, Set[int]] = defaultdict(set)
        for e in self.edges.values():
            if e.u in self.nodes and e.v in self.nodes:
                adj[e.u].add(e.v)
                adj[e.v].add(e.u)
        return adj

    def _is_adjacent_to_set(self, pos: GridPos, cells: Set[GridPos]) -> bool:
        if pos in cells:
            return True
        return any(abs(pos[0] - r) + abs(pos[1] - c) == 1 for r, c in cells)

    def _lap_end_flags(self, frontier_ids: Set[int]) -> Dict[int, bool]:
        by_lap: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for nid in frontier_ids:
            if nid not in self.nodes:
                continue
            by_lap[self.nodes[nid].lap].append((self.nodes[nid].pos[0], nid))

        flags: Dict[int, bool] = {}
        for _, entries in by_lap.items():
            if not entries:
                continue
            if len(entries) <= 2:
                for _, nid in entries:
                    flags[nid] = True
                continue

            rows = [r for r, _ in entries]
            r_min = min(rows)
            r_max = max(rows)
            for r, nid in entries:
                flags[nid] = (r == r_min) or (r == r_max)

        return flags

    def _edge_boundary_score(
        self,
        edge_key: Tuple[int, int],
        ao_cells: Set[GridPos],
        au_cells: Set[GridPos],
    ) -> float:
        if edge_key not in self.edges:
            return float("inf")
        u, v = edge_key
        if u not in self.nodes or v not in self.nodes:
            return float("inf")
        p = self.nodes[u].pos
        q = self.nodes[v].pos
        mid = ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
        return min(self._distance_to_cell_set(mid, ao_cells), self._distance_to_cell_set(mid, au_cells))

    def _edge_is_closest_for_transition(
        self,
        end_node_id: int,
        target_node_id: int,
        target_lap: int,
        adj: Dict[int, Set[int]],
        ao_cells: Set[GridPos],
        au_cells: Set[GridPos],
    ) -> bool:
        neighbors_on_lap = []
        for oid in adj.get(end_node_id, set()):
            if oid not in self.nodes or oid not in self.frontier_node_ids:
                continue
            if self.nodes[oid].lap == target_lap:
                neighbors_on_lap.append(oid)

        if target_node_id not in neighbors_on_lap:
            return False

        target_key = self._edge_key(end_node_id, target_node_id)
        target_score = self._edge_boundary_score(target_key, ao_cells, au_cells)
        best = float("inf")
        for oid in neighbors_on_lap:
            key = self._edge_key(end_node_id, oid)
            score = self._edge_boundary_score(key, ao_cells, au_cells)
            if score < best:
                best = score

        return target_score <= best + 1e-9

    def _is_frontier_node_essential(
        self,
        nid: int,
        adj: Dict[int, Set[int]],
        end_flags: Dict[int, bool],
        ao_cells: Set[GridPos],
        au_cells: Set[GridPos],
    ) -> bool:
        if nid not in self.nodes:
            return False
        if nid not in self.frontier_node_ids:
            return True
        n = self.nodes[nid]

        # 1) n adjacent to A_u.
        if self._is_adjacent_to_set(n.pos, au_cells):
            return True

        # 2) n is an end node of its lap.
        if end_flags.get(nid, False):
            return True

        # 3) n is connected to end-node n_x on adjacent lap.
        for nx in adj.get(nid, set()):
            if nx not in self.nodes or nx not in self.frontier_node_ids:
                continue
            if abs(self.nodes[nx].lap - n.lap) != 1:
                continue
            if not end_flags.get(nx, False):
                continue

            neighbors_on_n_lap = [
                oid
                for oid in adj.get(nx, set())
                if oid in self.nodes and oid in self.frontier_node_ids and self.nodes[oid].lap == n.lap
            ]
            other_neighbors = [oid for oid in neighbors_on_n_lap if oid != nid]

            # 3a) n_x has no other neighbor on n's lap.
            if not other_neighbors:
                return True

            # 3b) others exist, all are non-end, and (n, n_x) is closest to A_o or A_u.
            if all(not end_flags.get(oid, False) for oid in other_neighbors):
                if self._edge_is_closest_for_transition(nx, nid, n.lap, adj, ao_cells, au_cells):
                    return True

        return False

    def _merge_same_lap_neighbors_for_pruned_node(self, nid: int, adj: Dict[int, Set[int]]):
        if nid not in self.nodes:
            return

        n = self.nodes[nid]
        same_lap_neighbors = []
        for oid in adj.get(nid, set()):
            if oid in self.nodes and self.nodes[oid].lap == n.lap:
                same_lap_neighbors.append(oid)

        if len(same_lap_neighbors) < 2:
            return

        above = None
        below = None
        above_dist = float("inf")
        below_dist = float("inf")
        for oid in same_lap_neighbors:
            rr = self.nodes[oid].pos[0]
            d = abs(rr - n.pos[0])
            if rr < n.pos[0] and d < above_dist:
                above = oid
                above_dist = d
            elif rr > n.pos[0] and d < below_dist:
                below = oid
                below_dist = d

        if above is None or below is None:
            return

        k1 = self._edge_key(nid, above)
        k2 = self._edge_key(nid, below)
        if k1 not in self.edges or k2 not in self.edges:
            return

        merged_weight = self.edges[k1].weight + self.edges[k2].weight
        self._add_edge(above, below, merged_weight, edge_type="merged_prune", tentative=False)

    def _remove_node(self, nid: int) -> int:
        if nid not in self.nodes:
            return 0
        node = self.nodes.pop(nid)
        self.node_ids_by_lap[node.lap].discard(nid)
        self.node_key_to_id.pop((node.node_type, node.pos), None)

        self.region_node_ids.discard(nid)
        self.gate_node_ids.discard(nid)
        self.frontier_node_ids.discard(nid)
        self.link_node_ids.discard(nid)
        self.hole_candidate_node_ids.discard(nid)
        self.hole_node_ids.discard(nid)
        self.node_state.pop(nid, None)

        removed_edges = 0
        drop_keys = [k for k, e in self.edges.items() if e.u == nid or e.v == nid]
        for k in drop_keys:
            del self.edges[k]
            removed_edges += 1

        if self.current_waypoint == node.pos:
            self.current_waypoint = None
            self.current_plan = []
        if self.current_graph_node_id == nid:
            self.current_graph_node_id = None
        if self.current_goal_node_id == nid:
            self.current_goal_node_id = None
        self.closed_graph_node_ids.discard(nid)

        return removed_edges

    def _is_essential_edge(
        self,
        edge_key: Tuple[int, int],
        adj: Dict[int, Set[int]],
        essential_frontier: Set[int],
        end_flags: Dict[int, bool],
        ao_cells: Set[GridPos],
        au_cells: Set[GridPos],
    ) -> bool:
        if edge_key not in self.edges:
            return False
        u, v = edge_key
        if u not in self.nodes or v not in self.nodes:
            return False

        # Keep structural edges for auxiliary node types in cell-based adaptation.
        if u not in self.frontier_node_ids or v not in self.frontier_node_ids:
            return True

        if u not in essential_frontier or v not in essential_frontier:
            return False

        lu = self.nodes[u].lap
        lv = self.nodes[v].lap

        # Def III.9-1
        if lu == lv:
            return True

        # Def III.9-2
        if abs(lu - lv) != 1:
            return False

        u_end = end_flags.get(u, False)
        v_end = end_flags.get(v, False)

        # 2a
        if u_end and v_end:
            return True

        # 2b
        if u_end ^ v_end:
            end_node = u if u_end else v
            non_end_node = v if u_end else u
            target_lap = self.nodes[non_end_node].lap

            neighbors_on_target_lap = [
                oid
                for oid in adj.get(end_node, set())
                if oid in self.nodes and oid in self.frontier_node_ids and self.nodes[oid].lap == target_lap
            ]
            others = [oid for oid in neighbors_on_target_lap if oid != non_end_node]

            # 2b-i
            if not others:
                return True

            # 2b-ii
            if all(not end_flags.get(oid, False) for oid in others):
                return self._edge_is_closest_for_transition(
                    end_node,
                    non_end_node,
                    target_lap,
                    adj,
                    ao_cells,
                    au_cells,
                )

        return False

    def _build_adjacency_from_edges(
        self,
        edge_keys: Set[Tuple[int, int]],
        skip_edge: Optional[Tuple[int, int]] = None,
    ) -> Dict[int, Set[int]]:
        out: Dict[int, Set[int]] = defaultdict(set)
        for key in edge_keys:
            if key not in self.edges:
                continue
            if skip_edge is not None and key == skip_edge:
                continue
            e = self.edges[key]
            if e.u not in self.nodes or e.v not in self.nodes:
                continue
            out[e.u].add(e.v)
            out[e.v].add(e.u)
        return out

    def _path_exists_between(
        self,
        start: int,
        goal: int,
        adj: Dict[int, Set[int]],
    ) -> bool:
        if start == goal:
            return True
        q = deque([start])
        seen = {start}
        while q:
            cur = q.popleft()
            for nxt in adj.get(cur, set()):
                if nxt == goal:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return False

    def _can_remove_edge_keep_connected(self, edge_key: Tuple[int, int]) -> bool:
        if edge_key not in self.edges:
            return False
        e = self.edges[edge_key]
        if e.u not in self.nodes or e.v not in self.nodes:
            return True
        adj_wo = self._build_adjacency_from_edges(set(self.edges.keys()), skip_edge=edge_key)
        return self._path_exists_between(e.u, e.v, adj_wo)

    def _connected_components(self) -> List[Set[int]]:
        nodes = set(self.nodes.keys())
        if not nodes:
            return []
        adj = self._build_adjacency()
        comps: List[Set[int]] = []
        seen: Set[int] = set()
        for nid in nodes:
            if nid in seen:
                continue
            q = deque([nid])
            seen.add(nid)
            comp = set()
            while q:
                cur = q.popleft()
                comp.add(cur)
                for nxt in adj.get(cur, set()):
                    if nxt in self.nodes and nxt not in seen:
                        seen.add(nxt)
                        q.append(nxt)
            comps.append(comp)
        return comps

    def _find_bridge_pair_between_components(
        self,
        comp_a: Set[int],
        comp_b: Set[int],
        traversable: np.ndarray,
    ) -> Optional[Tuple[int, int, float]]:
        list_a = list(comp_a)
        list_b = list(comp_b)
        if not list_a or not list_b:
            return None

        best = None
        best_cost = float("inf")

        product = len(list_a) * len(list_b)
        if product <= 6000:
            pairs = []
            for u in list_a:
                pu = self.nodes[u].pos
                for v in list_b:
                    pv = self.nodes[v].pos
                    md = abs(pu[0] - pv[0]) + abs(pu[1] - pv[1])
                    pairs.append((md, u, v))
            pairs.sort(key=lambda x: x[0])
            candidate_pairs = pairs[: min(len(pairs), 120)]
        else:
            candidate_pairs = []
            for u in list_a:
                pu = self.nodes[u].pos
                v = min(
                    list_b,
                    key=lambda x: abs(pu[0] - self.nodes[x].pos[0]) + abs(pu[1] - self.nodes[x].pos[1]),
                )
                md = abs(pu[0] - self.nodes[v].pos[0]) + abs(pu[1] - self.nodes[v].pos[1])
                candidate_pairs.append((md, u, v))
            candidate_pairs.sort(key=lambda x: x[0])
            candidate_pairs = candidate_pairs[: min(len(candidate_pairs), 120)]

        for _, u, v in candidate_pairs:
            feasible, cost = self._edge_feasible_and_cost(self.nodes[u].pos, self.nodes[v].pos, traversable)
            if feasible and cost < best_cost:
                best_cost = cost
                best = (u, v, cost)
        return best

    def _ensure_connected_graph(self, traversable: np.ndarray) -> int:
        added = 0
        comps = self._connected_components()
        if len(comps) <= 1:
            return added

        comps = sorted(comps, key=len, reverse=True)
        main = set(comps[0])
        rest = [set(c) for c in comps[1:]]

        for comp in rest:
            bridge = self._find_bridge_pair_between_components(main, comp, traversable)
            if bridge is None:
                continue
            u, v, cost = bridge
            prev_edges = len(self.edges)
            self._add_edge(u, v, cost, edge_type="connectivity_bridge", tentative=False)
            if len(self.edges) > prev_edges:
                main |= comp
                added += 1

        return added

    def _enforce_single_connected_component(self) -> Tuple[int, int]:
        comps = self._connected_components()
        if len(comps) <= 1:
            return (0, 0)

        keep_comp = None
        for comp in comps:
            if any(self.nodes[nid].pos == self.current_node for nid in comp if nid in self.nodes):
                keep_comp = comp
                break
        if keep_comp is None:
            keep_comp = max(comps, key=len)

        drop_nodes = set()
        for comp in comps:
            if comp is keep_comp:
                continue
            drop_nodes |= comp

        pruned_nodes = 0
        pruned_edges = 0
        for nid in list(drop_nodes):
            if nid not in self.nodes:
                continue
            pruned_edges += self._remove_node(nid)
            pruned_nodes += 1

        return (pruned_nodes, pruned_edges)

    def _enforce_scalability(
        self,
        adj: Dict[int, Set[int]],
        essential_frontier: Set[int],
        end_flags: Dict[int, bool],
        ao_cells: Set[GridPos],
        au_cells: Set[GridPos],
    ) -> int:
        n = len(self.nodes)
        if n <= 2:
            return 0
        max_edges = 3 * n - 6
        if len(self.edges) <= max_edges:
            return 0

        removed = 0
        edge_keys = sorted(
            list(self.edges.keys()),
            key=lambda k: self.edges[k].weight,
            reverse=True,
        )

        for key in edge_keys:
            if len(self.edges) <= max_edges:
                break
            if key not in self.edges:
                continue
            if self._is_essential_edge(key, adj, essential_frontier, end_flags, ao_cells, au_cells):
                continue
            if not self._can_remove_edge_keep_connected(key):
                continue
            del self.edges[key]
            removed += 1

        return removed

    def _prune_rcg(
        self,
        node_candidates: Set[int],
        edge_candidates: Set[Tuple[int, int]],
    ) -> Dict[str, int]:
        if not node_candidates and not edge_candidates:
            return {"nodes_pruned": 0, "edges_pruned": 0}

        au_cells = self._frontier_cells()  # A_u
        ao_cells = self._obstacle_boundary_cells()  # A_o
        adj = self._build_adjacency()
        end_flags = self._lap_end_flags(set(self.frontier_node_ids))
        essential_frontier = {
            nid
            for nid in set(self.frontier_node_ids)
            if nid in self.nodes and self._is_frontier_node_essential(nid, adj, end_flags, ao_cells, au_cells)
        }

        prune_candidates = []
        for nid in node_candidates:
            if nid not in self.nodes:
                continue
            if nid not in self.frontier_node_ids:
                continue
            if self.nodes[nid].pos == self.current_node:
                continue
            if nid in essential_frontier:
                continue
            prune_candidates.append(nid)

        pruned_nodes = 0
        pruned_edges = 0
        for nid in sorted(prune_candidates):
            if nid not in self.nodes:
                continue
            adj_now = self._build_adjacency()
            self._merge_same_lap_neighbors_for_pruned_node(nid, adj_now)
            pruned_edges += self._remove_node(nid)
            pruned_nodes += 1

        # Recompute after node pruning.
        au_cells = self._frontier_cells()
        ao_cells = self._obstacle_boundary_cells()
        adj = self._build_adjacency()
        end_flags = self._lap_end_flags(set(self.frontier_node_ids))
        essential_frontier = {
            nid
            for nid in set(self.frontier_node_ids)
            if nid in self.nodes and self._is_frontier_node_essential(nid, adj, end_flags, ao_cells, au_cells)
        }

        edge_prune_keys = []
        for key in edge_candidates:
            if key not in self.edges:
                continue
            if not self._is_essential_edge(key, adj, essential_frontier, end_flags, ao_cells, au_cells):
                edge_prune_keys.append(key)

        # Keep graph connected: remove only non-bridge inessential edges.
        for key in sorted(edge_prune_keys, key=lambda k: self.edges[k].weight if k in self.edges else 0.0, reverse=True):
            if key in self.edges:
                if not self._can_remove_edge_keep_connected(key):
                    continue
                del self.edges[key]
                pruned_edges += 1

        # Enforce P6 (Euler upper-bound) for sparse scalable graph.
        adj = self._build_adjacency()
        extra_removed = self._enforce_scalability(adj, essential_frontier, end_flags, ao_cells, au_cells)
        pruned_edges += extra_removed

        # Maintain connected graph (P3) after pruning.
        traversable = self._build_traversable_mask()
        added_back = self._ensure_connected_graph(traversable)
        if added_back > 0:
            # If bridges were added, re-apply scalability softly without
            # removing bridges that are required for connectivity.
            adj = self._build_adjacency()
            extra_removed_2 = self._enforce_scalability(adj, essential_frontier, end_flags, ao_cells, au_cells)
            pruned_edges += extra_removed_2

        # Final fallback for P3: keep only the component that contains robot.
        cut_nodes, cut_edges = self._enforce_single_connected_component()
        pruned_nodes += cut_nodes
        pruned_edges += cut_edges

        self.total_pruned_nodes += pruned_nodes
        self.total_pruned_edges += pruned_edges
        return {"nodes_pruned": pruned_nodes, "edges_pruned": pruned_edges}

    def _expand_rcg(self, newly_known_free: np.ndarray) -> Dict[str, int]:
        prev_frontier_ids = set(self.frontier_node_ids)
        prev_edge_keys = set(self.edges.keys())

        traversable = self._build_traversable_mask()
        frontiers = self._frontier_cells()
        self.active_frontier_positions = set(frontiers)
        sample_front = self._sampling_front(frontiers)
        sample_front_boundary = self._sampling_front_boundary(sample_front)
        boundary_node_ids = self._nodes_adjacent_to_boundary(prev_frontier_ids, sample_front_boundary)
        boundary_edge_keys = {
            key
            for key in prev_edge_keys
            if key in self.edges and (self.edges[key].u in boundary_node_ids or self.edges[key].v in boundary_node_ids)
        }
        frontier_clusters = self._cluster_points(sample_front, connectivity=self.frontier_connectivity)
        current_gate_positions: Set[GridPos] = set()

        new_frontier_node_ids = []
        new_gate_node_ids = []

        # Frontier sample nodes (paper RCG expansion into Fi)
        for cluster in frontier_clusters:
            for p in self._progressive_samples(cluster):
                nid, created = self._add_node(
                    pos=p,
                    node_type="frontier",
                    iteration=self.iteration,
                    meta={"cluster_size": float(len(cluster))},
                    dedupe=True,
                )
                if created:
                    new_frontier_node_ids.append(nid)

            if self._is_gate_cluster(cluster):
                rows = [p[0] for p in cluster]
                cols = [p[1] for p in cluster]
                cr = int(round(sum(rows) / len(rows)))
                cc = int(round(sum(cols) / len(cols)))
                gate_pos = min(cluster, key=lambda p: abs(p[0] - cr) + abs(p[1] - cc))
                gid, created = self._add_node(
                    pos=gate_pos,
                    node_type="gate",
                    iteration=self.iteration,
                    meta={"cluster_size": float(len(cluster))},
                    dedupe=True,
                )
                current_gate_positions.add(gate_pos)
                if created:
                    new_gate_node_ids.append(gid)

        # Region nodes from known free-space components
        free_mask = self.known_grid == 0
        components = self._connected_components_from_mask(free_mask)
        _, region_infos = self._ensure_region_nodes(components, newly_known_free)

        # Edge expansion from newly added frontier nodes
        for nid in new_frontier_node_ids:
            self._connect_same_lap_adjacency(nid, traversable)
            self._connect_adjacent_laps(nid, traversable)

        # Additional structural links for cell-based implementation
        self._connect_region_adjacency(region_infos, traversable)
        self._connect_gate_region_links(new_gate_node_ids, region_infos, traversable)
        self.active_gate_positions = current_gate_positions
        new_edge_keys = set(self.edges.keys()) - prev_edge_keys
        node_candidates = set(new_frontier_node_ids) | boundary_node_ids
        edge_candidates = set(boundary_edge_keys) | set(new_edge_keys)
        prune_stats = self._prune_rcg(node_candidates, edge_candidates)

        return {
            "frontier_nodes_added": len(new_frontier_node_ids),
            "gate_nodes_added": len(new_gate_node_ids),
            "regions_seen": len(region_infos),
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "nodes_pruned": prune_stats["nodes_pruned"],
            "edges_pruned": prune_stats["edges_pruned"],
        }

    # ---------- Coverage control ----------

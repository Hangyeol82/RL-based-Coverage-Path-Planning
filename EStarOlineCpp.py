import heapq
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from MapGenerator import MapGenerator


class EpsilonStarCPP:
    def __init__(
        self,
        grid_map,
        start_node,
        epsilon=0.5,
        sensor_range=5,
        zigzag_max=10.0,
        zigzag_min=1.0,
    ):
        self.true_grid = grid_map
        self.rows, self.cols = grid_map.shape
        self.current_node = start_node
        self.epsilon = epsilon
        self.sensor_range = sensor_range
        self.zigzag_max = zigzag_max
        self.zigzag_min = zigzag_min

        # known_grid: -1 unknown, 0 free, 1 obstacle
        self.known_grid = np.full_like(grid_map, fill_value=-1, dtype=int)
        self.explored = np.zeros_like(grid_map, dtype=bool)

        self.path = [start_node]
        self.level_maps = []
        self.current_level = 0
        self.etm_state = "ST"
        self.waypoint = None
        self.escape_path = []
        self.sweep_mode = None
        self.sweep_dir = 0
        self.sweep_start = None
        self.sweep_trace = []
        self.sweep_return_path = []

        self._sense()

    def _sense(self):
        cx, cy = self.current_node
        r0 = max(0, cx - self.sensor_range)
        r1 = min(self.rows - 1, cx + self.sensor_range)
        c0 = max(0, cy - self.sensor_range)
        c1 = min(self.cols - 1, cy + self.sensor_range)

        self.known_grid[r0 : r1 + 1, c0 : c1 + 1] = self.true_grid[r0 : r1 + 1, c0 : c1 + 1]

    def _zigzag_weight(self, col, width):
        if width <= 1:
            return self.zigzag_max
        t = 1.0 - (col / (width - 1))
        return self.zigzag_min + (self.zigzag_max - self.zigzag_min) * t

    def _compute_level0_potential(self, traversable, unexplored):
        pot = np.zeros((self.rows, self.cols), dtype=float)
        for r in range(self.rows):
            for c in range(self.cols):
                if not traversable[r, c]:
                    pot[r, c] = -1.0
                elif not unexplored[r, c]:
                    pot[r, c] = 0.0
                else:
                    pot[r, c] = self._zigzag_weight(c, self.cols)
        return pot

    def _compute_level_potential(self, trav_count, unexp_count):
        rows, cols = trav_count.shape
        pot = np.zeros((rows, cols), dtype=float)
        for r in range(rows):
            for c in range(cols):
                if trav_count[r, c] == 0:
                    pot[r, c] = -1.0
                else:
                    ratio = unexp_count[r, c] / trav_count[r, c]
                    pot[r, c] = ratio * self._zigzag_weight(c, cols)
        return pot

    def _coarsen_sum(self, arr):
        rows, cols = arr.shape
        # odd size is merged with a single leftover cell (ceil(n/2))
        new_rows = (rows + 1) // 2
        new_cols = (cols + 1) // 2
        out = np.zeros((new_rows, new_cols), dtype=int)
        for r in range(new_rows):
            for c in range(new_cols):
                r0 = r * 2
                r1 = min(r0 + 2, rows)
                c0 = c * 2
                c1 = min(c0 + 2, cols)
                out[r, c] = np.sum(arr[r0:r1, c0:c1])
        return out

    def _build_maps(self):
        traversable = self.known_grid != 1
        unexplored = traversable & (~self.explored)

        level_maps = []
        level_maps.append(self._compute_level0_potential(traversable, unexplored))

        trav_count = traversable.astype(int)
        unexp_count = unexplored.astype(int)

        while trav_count.shape[0] > 1 or trav_count.shape[1] > 1:
            trav_count = self._coarsen_sum(trav_count)
            unexp_count = self._coarsen_sum(unexp_count)
            level_maps.append(self._compute_level_potential(trav_count, unexp_count))

        self.level_maps = level_maps

    def _neighbors(self, node):
        x, y = node
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        result = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.rows and 0 <= ny < self.cols:
                if self.known_grid[nx, ny] != 1:
                    result.append((nx, ny))
        return result

    def _is_traversable(self, node):
        x, y = node
        if x < 0 or x >= self.rows or y < 0 or y >= self.cols:
            return False
        return self.known_grid[x, y] != 1

    def _has_vertical_pair(self, node, positive_only=False):
        x, y = node
        up = (x - 1, y)
        down = (x + 1, y)

        if not self._is_traversable(up) or not self._is_traversable(down):
            return False

        if not positive_only:
            return True

        return self.level_maps[0][up[0], up[1]] > 0 and self.level_maps[0][down[0], down[1]] > 0

    def _should_defer_mark_current(self):
        if self.etm_state != "CP0":
            return False
        if self.sweep_mode is not None:
            return False

        x, y = self.current_node
        up = (x - 1, y)
        down = (x + 1, y)

        if not self._is_traversable(up) or not self._is_traversable(down):
            return False

        # level-0 positive potential condition without level-map dependency:
        # traversable and unexplored
        return (not self.explored[up[0], up[1]]) and (not self.explored[down[0], down[1]])

    def _has_positive_neighbor(self):
        for n in self._neighbors(self.current_node):
            if self.level_maps[0][n[0], n[1]] > 0:
                return True
        return False

    def _select_next_level0(self):
        x, y = self.current_node
        neighbors = self._neighbors(self.current_node)
        candidates = [n for n in neighbors if self.level_maps[0][n[0], n[1]] > 0]
        if not candidates:
            return None

        # overlap rule: if both vertical neighbors are unexplored (+potential),
        # keep moving one vertical direction without marking, then return with marking.
        if self._has_vertical_pair(self.current_node, positive_only=True):
            up = (x - 1, y)
            down = (x + 1, y)
            up_val = self.level_maps[0][up[0], up[1]]
            down_val = self.level_maps[0][down[0], down[1]]
            first = up if up_val >= down_val else down
            self.sweep_mode = "forward"
            self.sweep_dir = -1 if first == up else 1
            self.sweep_start = self.current_node
            self.sweep_trace = []
            self.sweep_return_path = []
            return first

        return max(candidates, key=lambda n: self.level_maps[0][n[0], n[1]])

    def _reset_sweep(self):
        self.sweep_mode = None
        self.sweep_dir = 0
        self.sweep_start = None
        self.sweep_trace = []
        self.sweep_return_path = []

    def _step_vertical_sweep(self):
        if self.sweep_mode == "forward":
            x, y = self.current_node
            next_node = (x + self.sweep_dir, y)
            can_continue = self._has_vertical_pair(self.current_node, positive_only=False)

            if can_continue and self._is_traversable(next_node):
                self._move_to(next_node)
                self.sweep_trace.append(next_node)
                return True

            self.sweep_mode = "return"
            self.sweep_return_path = [self.sweep_start] + self.sweep_trace[:-1]
            return False

        if self.sweep_mode == "return":
            if not self.sweep_return_path:
                self._reset_sweep()
                return False

            backtrack_node = self.sweep_return_path[-1]

            alternatives = []
            for n in self._neighbors(self.current_node):
                if n == backtrack_node:
                    continue
                p = self.level_maps[0][n[0], n[1]]
                if p > 0:
                    alternatives.append((p, n))

            if alternatives:
                best_alt_p, best_alt = max(alternatives, key=lambda x: x[0])
                backtrack_p = self.level_maps[0][backtrack_node[0], backtrack_node[1]]
                if best_alt_p > backtrack_p:
                    self._reset_sweep()
                    self._move_to(best_alt)
                    return True

            next_node = self.sweep_return_path.pop()
            if not self._is_traversable(next_node):
                self._reset_sweep()
                return False

            self._move_to(next_node)

            if not self.sweep_return_path:
                self._reset_sweep()

            return True

        return False

    def _coarse_neighbors(self, cell, shape):
        r, c = cell
        rows, cols = shape
        result = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result.append((nr, nc))
        return result

    def _pick_fine_cell_in_coarse(self, coarse_cell, level):
        scale = 2 ** level
        r0 = coarse_cell[0] * scale
        r1 = min((coarse_cell[0] + 1) * scale, self.rows)
        c0 = coarse_cell[1] * scale
        c1 = min((coarse_cell[1] + 1) * scale, self.cols)

        best = None
        best_val = -float("inf")

        for r in range(r0, r1):
            for c in range(c0, c1):
                if self.known_grid[r, c] == 1:
                    continue
                pot = self.level_maps[0][r, c]
                if pot > best_val and pot > 0:
                    best = (r, c)
                    best_val = pot

        if best is not None:
            return best

        for r in range(r0, r1):
            for c in range(c0, c1):
                if self.known_grid[r, c] != 1:
                    return (r, c)

        return None

    def _find_escape_target(self):
        max_level = len(self.level_maps) - 1
        for level in range(1, max_level + 1):
            coarse = self.level_maps[level]
            cur_cell = (self.current_node[0] // (2 ** level), self.current_node[1] // (2 ** level))
            neighbors = self._coarse_neighbors(cur_cell, coarse.shape)
            candidates = [n for n in neighbors if coarse[n[0], n[1]] > 0]
            if not candidates:
                continue
            target_coarse = max(candidates, key=lambda n: coarse[n[0], n[1]])
            target = self._pick_fine_cell_in_coarse(target_coarse, level)
            if target is not None:
                return target, level
        return None, None

    def _a_star(self, start, goal):
        if self.known_grid[goal[0], goal[1]] == 1:
            return []

        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        came_from = {}
        g_score = {start: 0}

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        while open_set:
            _, g, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for neighbor in self._neighbors(current):
                if self.known_grid[neighbor[0], neighbor[1]] == 1:
                    continue
                tentative = g + 1
                if neighbor not in g_score or tentative < g_score[neighbor]:
                    g_score[neighbor] = tentative
                    f = tentative + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, tentative, neighbor))
                    came_from[neighbor] = current

        return []

    def _move_to(self, node):
        self.current_node = node
        self.path.append(node)

    def run(self, max_steps=None):
        if max_steps is None:
            max_steps = self.rows * self.cols * 10

        self.etm_state = "CP0"
        steps = 0

        while steps < max_steps:
            self._sense()
            defer_mark = self._should_defer_mark_current()
            if self.sweep_mode != "forward" and not defer_mark:
                self.explored[self.current_node[0], self.current_node[1]] = True
            self._build_maps()

            if self.sweep_mode in ("forward", "return"):
                moved = self._step_vertical_sweep()
                if moved:
                    steps += 1
                    continue
                if self.sweep_mode in ("forward", "return"):
                    continue

            if self.etm_state == "CP0":
                next_node = self._select_next_level0()
                if next_node is None:
                    self.etm_state = "CPH"
                    self.escape_path = []
                    self.waypoint = None
                else:
                    self._move_to(next_node)
                    if self.sweep_mode == "forward":
                        self.sweep_trace.append(next_node)
                    steps += 1
                    continue

            if self.etm_state == "CPH":
                target, level = self._find_escape_target()
                if target is None:
                    break
                self.waypoint = target
                self.current_level = level
                self.escape_path = self._a_star(self.current_node, self.waypoint)
                if not self.escape_path:
                    break
                self.etm_state = "WT"

            if self.etm_state == "WT":
                if self.escape_path and self.escape_path[0] == self.current_node:
                    self.escape_path.pop(0)
                if not self.escape_path:
                    self.etm_state = "CP0"
                    continue

                next_node = self.escape_path.pop(0)
                if self.known_grid[next_node[0], next_node[1]] == 1:
                    self.escape_path = []
                    self.etm_state = "CPH"
                    continue

                self._move_to(next_node)
                steps += 1

                if self._has_positive_neighbor():
                    self.etm_state = "CP0"
                    self.escape_path = []

        return self.path


def visualize_path(grid, path, title="Coverage Path"):
    plt.figure(figsize=(11, 10))
    plt.imshow(grid, cmap="binary", origin="upper")

    if path:
        rows = [node[0] for node in path]
        cols = [node[1] for node in path]

        points = np.array([cols, rows]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        norm = plt.Normalize(0, len(path))
        lc = LineCollection(segments, cmap="jet", norm=norm, alpha=0.8)
        lc.set_array(np.arange(len(path)))
        lc.set_linewidth(2)
        line = plt.gca().add_collection(lc)

        plt.colorbar(line, fraction=0.046, pad=0.04, label="Step Sequence")

        step_interval = max(1, len(path) // 40)

        arrow_cols = []
        arrow_rows = []
        arrow_dx = []
        arrow_dy = []

        for i in range(0, len(path) - 1, step_interval):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]

            if r1 != r2 or c1 != c2:
                arrow_cols.append(c1)
                arrow_rows.append(r1)
                arrow_dx.append(c2 - c1)
                arrow_dy.append(r2 - r1)

        plt.quiver(
            arrow_cols,
            arrow_rows,
            arrow_dx,
            arrow_dy,
            angles="xy",
            scale_units="xy",
            scale=1,
            color="black",
            alpha=0.5,
            width=0.003,
            headwidth=4,
            headlength=4,
            pivot="mid",
            label="Direction",
        )

        plt.scatter(cols[0], rows[0], c="lime", s=150, zorder=6, edgecolors="black", label="Start")
        plt.scatter(cols[-1], rows[-1], c="red", s=150, zorder=6, marker="X", edgecolors="black", label="End")

    plt.title(title)
    plt.xlabel("Column (X)")
    plt.ylabel("Row (Y)")

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], color="blue", lw=2, label="Path (Gradient Color)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lime", markeredgecolor="black", markersize=10, label="Start"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="red", markeredgecolor="black", markersize=10, label="End"),
        Patch(facecolor="black", edgecolor="black", label="Obstacle"),
        Patch(facecolor="white", edgecolor="black", label="Free Space"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")

    plt.grid(which="major", color="gray", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    height, width = 20, 20
    seed = 123

    print(f"Generating new map ({height}x{width}) with seed {seed}...")
    map_gen = MapGenerator(height=height, width=width, seed=seed)
    grid, meta = map_gen.generate_map(stage=3, return_metadata=True)

    start_node = (0, 0)
    if grid[start_node[0]][start_node[1]] == 1:
        print("Warning: Start node (0,0) is blocked. Clearing it.")
        grid[start_node[0]][start_node[1]] = 0

    print("Running Epsilon* Online Coverage Path Planning...")
    planner = EpsilonStarCPP(grid, start_node=start_node, epsilon=0.5, sensor_range=5)
    path = planner.run()

    print(f"Path generated! Total steps: {len(path)}")
    visualize_path(grid, path, title=f"E* Online Coverage Path (Map: {height}x{width}, Steps: {len(path)})")

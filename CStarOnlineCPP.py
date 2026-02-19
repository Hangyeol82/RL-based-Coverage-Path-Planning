from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from cstar.control_mixin import CStarControlMixin
from cstar.graph_mixin import CStarGraphMixin
from cstar.types import GridPos, RCGEdge, RCGNode
from cstar.visualization import show_path_plot


class CStarOnlineCPP(CStarGraphMixin, CStarControlMixin):
    """
    Cell-based C* style online coverage with RCG construction.

    Refactor note:
    - Graph/RCG construction and pruning logic lives in `cstar/graph_mixin.py`.
    - Coverage-control logic lives in `cstar/control_mixin.py`.
    - Plotting logic lives in `cstar/visualization.py`.
    """

    def __init__(
        self,
        grid_map: np.ndarray,
        start_node: GridPos,
        sensor_range: int = 5,
        lap_width: int = 2,
        frontier_sample_stride: int = 2,
        frontier_connectivity: int = 8,
        gate_min_cluster_size: int = 6,
        gate_max_width: int = 2,
        adjacency_cell_dist: int = 2,
        obstacle_inflation: int = 1,
        min_traversable_width: int = 1,
    ):
        self.true_grid = grid_map
        self.rows, self.cols = grid_map.shape
        self.current_node = start_node
        self.sensor_range = sensor_range
        self.lap_width = max(1, lap_width)
        self.frontier_sample_stride = max(1, frontier_sample_stride)
        self.frontier_connectivity = 8 if frontier_connectivity == 8 else 4
        self.gate_min_cluster_size = gate_min_cluster_size
        self.gate_max_width = max(1, gate_max_width)
        self.adjacency_cell_dist = max(1, adjacency_cell_dist)
        self.obstacle_inflation = max(0, obstacle_inflation)
        self.min_traversable_width = max(1, min_traversable_width)

        # known_grid: -1 unknown, 0 free, 1 obstacle
        self.known_grid = np.full_like(grid_map, fill_value=-1, dtype=int)
        self.explored = np.zeros_like(grid_map, dtype=bool)
        self.path: List[GridPos] = [start_node]
        self.iteration = 0

        # RCG containers
        self.nodes: Dict[int, RCGNode] = {}
        self.edges: Dict[Tuple[int, int], RCGEdge] = {}
        self.node_ids_by_lap: Dict[int, Set[int]] = defaultdict(set)
        self.node_key_to_id: Dict[Tuple[str, GridPos], int] = {}
        self.next_node_id = 0

        self.region_node_ids: Set[int] = set()
        self.gate_node_ids: Set[int] = set()
        self.frontier_node_ids: Set[int] = set()
        self.hole_candidate_node_ids: Set[int] = set()
        self.hole_node_ids: Set[int] = set()
        self.link_node_ids: Set[int] = set()
        self.active_frontier_positions: Set[GridPos] = set()
        self.active_gate_positions: Set[GridPos] = set()
        self.total_pruned_nodes = 0
        self.total_pruned_edges = 0

        self.current_plan: List[GridPos] = []
        self.current_waypoint: Optional[GridPos] = None
        self.current_graph_node_id: Optional[int] = None
        self.current_goal_node_id: Optional[int] = None
        self.closed_graph_node_ids: Set[int] = set()
        self.node_state: Dict[int, str] = {}
        self.hole_tcp_queue: List[GridPos] = []
        self.active_hole_cells: Set[GridPos] = set()

        self._sense()

    def _sense(self):
        cx, cy = self.current_node
        r0 = max(0, cx - self.sensor_range)
        r1 = min(self.rows - 1, cx + self.sensor_range)
        c0 = max(0, cy - self.sensor_range)
        c1 = min(self.cols - 1, cy + self.sensor_range)
        self.known_grid[r0 : r1 + 1, c0 : c1 + 1] = self.true_grid[r0 : r1 + 1, c0 : c1 + 1]

    def show_path_plot(self):
        show_path_plot(self)


if __name__ == "__main__":
    from MapGenerator import MapGenerator

    h, w, seed = 30, 30, 42
    gen = MapGenerator(height=h, width=w, seed=seed)
    grid, _ = gen.generate_map(stage=3, return_metadata=True)
    if grid[0, 0] == 1:
        grid[0, 0] = 0

    planner = CStarOnlineCPP(
        grid_map=grid,
        start_node=(0, 0),
        sensor_range=5,
        lap_width=2,
        frontier_sample_stride=2,
        frontier_connectivity=8,
        gate_min_cluster_size=5,
        gate_max_width=2,
        adjacency_cell_dist=2,
        obstacle_inflation=1,
        min_traversable_width=1,
    )
    path = planner.run(max_steps=4000)
    stats = planner.rcg_stats()

    print(f"Path length: {len(path)}")
    print(f"Explored: {int(planner.explored.sum())}/{grid.size}")
    print("RCG stats:", stats)
    planner.show_path_plot()

import argparse

import numpy as np

from CStarOnlineCPP import CStarOnlineCPP


CUSTOM_MAP_TEXT = """
0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
"""


def parse_custom_map(text: str) -> np.ndarray:
    rows = [list(map(int, line.split())) for line in text.strip().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Custom map is empty.")

    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"Custom map rows have inconsistent widths: {sorted(widths)}")

    grid = np.array(rows, dtype=int)
    if not np.isin(grid, [0, 1]).all():
        raise ValueError("Custom map must contain only 0 (free) and 1 (obstacle).")
    return grid


def main():
    parser = argparse.ArgumentParser(description="Run C* on a fixed hand-crafted map.")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--start-row", type=int, default=0)
    parser.add_argument("--start-col", type=int, default=0)
    parser.add_argument("--sensor-range", type=int, default=5)
    parser.add_argument("--lap-width", type=int, default=2)
    parser.add_argument("--no-plot", action="store_true", help="Do not open matplotlib window.")
    args = parser.parse_args()

    grid = parse_custom_map(CUSTOM_MAP_TEXT)
    start = (args.start_row, args.start_col)

    if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]):
        raise ValueError(f"Start {start} is out of map bounds {grid.shape}.")
    if grid[start[0], start[1]] == 1:
        raise ValueError(f"Start {start} is an obstacle cell (1).")

    planner = CStarOnlineCPP(
        grid_map=grid,
        start_node=start,
        sensor_range=args.sensor_range,
        lap_width=args.lap_width,
        frontier_sample_stride=2,
        frontier_connectivity=8,
        gate_min_cluster_size=5,
        gate_max_width=2,
        adjacency_cell_dist=2,
        obstacle_inflation=1,
        min_traversable_width=1,
    )
    path = planner.run(max_steps=args.max_steps)
    stats = planner.rcg_stats()

    print(f"Map size: {grid.shape[0]}x{grid.shape[1]}")
    print(f"Path length: {len(path)}")
    print(f"Explored: {int(planner.explored.sum())}/{grid.size}")
    print("RCG stats:", stats)

    if not args.no_plot:
        planner.show_path_plot()


if __name__ == "__main__":
    main()

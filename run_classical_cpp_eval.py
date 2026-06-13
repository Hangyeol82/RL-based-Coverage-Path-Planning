import argparse
import csv
import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
if "MKL_NUM_THREADS" not in os.environ:
    os.environ["MKL_NUM_THREADS"] = "1"
if "OPENBLAS_NUM_THREADS" not in os.environ:
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

from CStarOnlineCPP import CStarOnlineCPP
from EStarOlineCpp import EpsilonStarCPP
from evaluation.cpp_metrics import compute_cpp_path_metrics
from map_generators.shape_grid_presets import build_validated_shape_grid_map, get_paper_shape_grid_preset
from map_generators.validation import analyze_grid_map, parse_map_txt


GridPos = Tuple[int, int]


def _parse_int_list(raw: str) -> List[int]:
    out = []
    for tok in raw.split(","):
        s = tok.strip()
        if s:
            out.append(int(s))
    if not out:
        raise ValueError("No integers provided")
    return out


def _parse_map_files(raw: str) -> List[Path]:
    out = []
    for tok in raw.split(","):
        s = tok.strip()
        if s:
            out.append(Path(s).expanduser().resolve())
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate deterministic classical/heuristic CPP baselines on grid maps."
    )
    p.add_argument(
        "--algorithm",
        type=str,
        default="epsilon_star",
        choices=["epsilon_star", "cstar", "offline_sweep"],
    )
    p.add_argument("--map-files", type=str, default="", help="Comma-separated txt map files.")
    p.add_argument("--seeds", type=str, default="1001,1002,1003,1004,1005")
    p.add_argument("--map-size", type=int, default=128)
    p.add_argument("--shapegrid-level", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=30000)
    p.add_argument("--sensor-range", type=int, default=3)
    p.add_argument("--start-row", type=int, default=0)
    p.add_argument("--start-col", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="log_analysis/logs/classical_cpp_eval")
    p.add_argument("--save-paths", action="store_true")
    p.add_argument("--save-plots", action="store_true")
    p.add_argument("--lap-width", type=int, default=2, help="C* lap width.")
    return p.parse_args()


def _neighbors(pos: GridPos, grid: np.ndarray) -> Iterable[GridPos]:
    r, c = pos
    rows, cols = grid.shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            yield nr, nc


def _bfs_path(grid: np.ndarray, start: GridPos, goal: GridPos) -> List[GridPos]:
    if start == goal:
        return [start]
    if grid[start[0], start[1]] != 0 or grid[goal[0], goal[1]] != 0:
        return []
    q = deque([start])
    parent: Dict[GridPos, GridPos] = {}
    visited = {start}
    while q:
        cur = q.popleft()
        for nxt in _neighbors(cur, grid):
            if nxt in visited:
                continue
            visited.add(nxt)
            parent[nxt] = cur
            if nxt == goal:
                path = [nxt]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                path.reverse()
                return path
            q.append(nxt)
    return []


def _offline_sweep_path(grid: np.ndarray, start: GridPos, max_steps: int) -> List[GridPos]:
    targets: List[GridPos] = []
    rows, cols = grid.shape
    for r in range(rows):
        cols_iter = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        for c in cols_iter:
            if grid[r, c] == 0:
                targets.append((r, c))

    path = [start]
    visited = {start} if grid[start[0], start[1]] == 0 else set()
    cur = start
    for target in targets:
        if target in visited:
            continue
        route = _bfs_path(grid, cur, target)
        if len(route) <= 1:
            continue
        for pos in route[1:]:
            path.append(pos)
            cur = pos
            if grid[pos[0], pos[1]] == 0:
                visited.add(pos)
            if len(path) - 1 >= int(max_steps):
                return path
    return path


def _run_algorithm(args: argparse.Namespace, grid: np.ndarray, start: GridPos) -> Tuple[List[GridPos], Dict[str, object]]:
    if args.algorithm == "epsilon_star":
        planner = EpsilonStarCPP(
            grid,
            start_node=start,
            sensor_range=int(args.sensor_range),
        )
        path = planner.run(max_steps=int(args.max_steps))
        return [(int(r), int(c)) for r, c in path], {}

    if args.algorithm == "cstar":
        planner = CStarOnlineCPP(
            grid_map=grid,
            start_node=start,
            sensor_range=int(args.sensor_range),
            lap_width=int(args.lap_width),
            frontier_sample_stride=2,
            frontier_connectivity=8,
            gate_min_cluster_size=5,
            gate_max_width=2,
            adjacency_cell_dist=2,
            obstacle_inflation=1,
            min_traversable_width=1,
        )
        path = planner.run(max_steps=int(args.max_steps))
        return [(int(r), int(c)) for r, c in path], planner.rcg_stats()

    path = _offline_sweep_path(grid, start, int(args.max_steps))
    return path, {"known_map": "full"}


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "algorithm",
        "map_id",
        "seed",
        "map_file",
        "map_size",
        "obstacle_ratio",
        "start_component_ratio",
        "steps",
        "path_length",
        "coverage_cells",
        "free_cells",
        "coverage_ratio",
        "revisit_count",
        "revisit_ratio",
        "overlap_count",
        "overlap_ratio",
        "step_to_90",
        "step_to_95",
        "step_to_99",
        "success_90",
        "success_95",
        "success_99",
        "invalid_step_count",
        "obstacle_step_count",
        "out_of_bounds_step_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _save_plot(grid: np.ndarray, path: Sequence[GridPos], out_png: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(grid == 1, cmap="gray_r", origin="upper")
    arr = np.asarray(path, dtype=np.int32)
    if len(arr) > 1:
        pts = np.stack([arr[:, 1], arr[:, 0]], axis=1)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, cmap="turbo", norm=Normalize(0, max(1, len(segs) - 1)), linewidths=1.6)
        lc.set_array(np.arange(len(segs), dtype=np.float32))
        ax.add_collection(lc)
    if len(arr) > 0:
        ax.scatter(arr[0, 1], arr[0, 0], color="tab:blue", s=30, zorder=3)
        ax.scatter(arr[-1, 1], arr[-1, 0], color="tab:green", s=30, zorder=3)
    ax.set_title(title)
    ax.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax.set_ylim(grid.shape[0] - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _maps_from_args(args: argparse.Namespace) -> List[Tuple[str, Optional[int], np.ndarray, str]]:
    map_files = _parse_map_files(str(args.map_files))
    if map_files:
        out = []
        for path in map_files:
            grid = parse_map_txt(path)
            out.append((path.stem, None, grid, str(path)))
        return out

    preset = get_paper_shape_grid_preset(int(args.shapegrid_level))
    out = []
    for seed in _parse_int_list(str(args.seeds)):
        grid, _stats, used_seed, _attempt = build_validated_shape_grid_map(
            size=int(args.map_size),
            seed=int(seed),
            level=int(args.shapegrid_level),
        )
        map_id = f"shapegrid_L{int(args.shapegrid_level)}_{preset.name}_seed{seed}"
        out.append((map_id, int(used_seed), grid, "generated"))
    return out


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = (int(args.start_row), int(args.start_col))

    rows: List[Dict[str, object]] = []
    for map_id, seed, grid, map_file in _maps_from_args(args):
        if not (0 <= start[0] < grid.shape[0] and 0 <= start[1] < grid.shape[1]):
            raise ValueError(f"Start {start} is out of bounds for {map_id}: {grid.shape}")
        if grid[start[0], start[1]] != 0:
            grid = grid.copy()
            grid[start[0], start[1]] = 0

        path, extra = _run_algorithm(args, grid, start)
        metrics = compute_cpp_path_metrics(grid, path)
        map_stats = analyze_grid_map(grid, start=start)
        row = {
            "algorithm": str(args.algorithm),
            "map_id": map_id,
            "seed": "" if seed is None else int(seed),
            "map_file": map_file,
            "map_size": f"{grid.shape[0]}x{grid.shape[1]}",
            "obstacle_ratio": float(map_stats.obstacle_ratio),
            "start_component_ratio": float(map_stats.start_component_ratio),
            **metrics.as_dict(),
        }
        rows.append(row)

        if bool(args.save_paths):
            payload = {**row, "path": [[int(r), int(c)] for r, c in path], "algorithm_extra": extra}
            (out_dir / "paths").mkdir(parents=True, exist_ok=True)
            (out_dir / "paths" / f"{map_id}_{args.algorithm}.json").write_text(
                json.dumps(payload, indent=2),
                encoding="utf-8",
            )
        if bool(args.save_plots):
            _save_plot(
                grid,
                path,
                out_dir / "plots" / f"{map_id}_{args.algorithm}.png",
                title=f"{args.algorithm} | {map_id}",
            )

        print(
            f"[DONE] {args.algorithm} {map_id}: "
            f"cov={metrics.coverage_ratio:.4f} revisit={metrics.revisit_ratio:.4f} "
            f"steps={metrics.steps} t90={metrics.step_to_90}"
        )

    _write_csv(out_dir / "metrics.csv", rows)
    aggregate = {
        "algorithm": str(args.algorithm),
        "num_maps": int(len(rows)),
        "mean_coverage_ratio": float(np.mean([float(r["coverage_ratio"]) for r in rows])) if rows else float("nan"),
        "mean_revisit_ratio": float(np.mean([float(r["revisit_ratio"]) for r in rows])) if rows else float("nan"),
        "mean_overlap_ratio": float(np.mean([float(r["overlap_ratio"]) for r in rows])) if rows else float("nan"),
        "success_90_count": int(sum(1 for r in rows if bool(r["success_90"]))),
        "success_95_count": int(sum(1 for r in rows if bool(r["success_95"]))),
        "success_99_count": int(sum(1 for r in rows if bool(r["success_99"]))),
    }
    (out_dir / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"[DONE] metrics: {out_dir / 'metrics.csv'}")
    print(f"[DONE] aggregate: {out_dir / 'aggregate.json'}")


if __name__ == "__main__":
    main()

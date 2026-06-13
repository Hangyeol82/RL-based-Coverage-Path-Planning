import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_generators.io_utils import save_map_png, write_map_txt
from map_generators.large_obstacle_grid import build_large_obstacle_grid_map


def _parse_csv_ints(raw: str):
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate large-obstacle base-map samples.")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--seeds", type=str, default="101,202,303,404,505,606")
    p.add_argument(
        "--obstacle-count",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="0 samples one or two obstacles by seed; 1/2 forces the count.",
    )
    p.add_argument("--out-dir", type=str, default="map_generators/samples/large_obstacle_grid")
    p.add_argument("--prefix", type=str, default="largeobs_base")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_csv_ints(args.seeds)
    rows = []
    forced_count = None if int(args.obstacle_count) == 0 else int(args.obstacle_count)
    for seed in seeds:
        grid, meta = build_large_obstacle_grid_map(
            size=int(args.size),
            seed=int(seed),
            obstacle_count=forced_count,
            return_metadata=True,
        )
        count = int(meta["obstacle_count_placed"])
        stem = f"{args.prefix}{int(args.size)}_n{count}_seed{int(seed)}"
        txt_path = out_dir / f"{stem}.txt"
        png_path = out_dir / f"{stem}.png"
        write_map_txt(txt_path, grid)
        png_ok = save_map_png(png_path, grid)
        row = {
            **meta,
            "free_cells": int(np.count_nonzero(grid == 0)),
            "obstacle_cells": int(np.count_nonzero(grid == 1)),
            "txt": str(txt_path),
            "png": str(png_path) if png_ok else "",
        }
        rows.append(row)
        print(
            f"[OK] seed={seed} obs={float(meta['obstacle_ratio']):.3f} "
            f"count={meta['obstacle_count_placed']}/{meta['obstacle_count_target']}"
        )

    if rows:
        summary_path = out_dir / f"{args.prefix}{int(args.size)}_summary.csv"
        keys = sorted({k for row in rows for k in row.keys()})
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"[DONE] summary={summary_path}")


if __name__ == "__main__":
    main()

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_generators.io_utils import save_map_png, write_map_txt
from map_generators.trail_grid import build_trail_grid_map


def _parse_csv_ints(raw: str):
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate branching trail map samples.")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--levels", type=str, default="1,2,3,4")
    p.add_argument("--seeds", type=str, default="101,202,303")
    p.add_argument("--out-dir", type=str, default="map_generators/samples/trail_grid")
    p.add_argument("--prefix", type=str, default="trail")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for level in _parse_csv_ints(args.levels):
        for seed in _parse_csv_ints(args.seeds):
            grid, meta = build_trail_grid_map(
                size=int(args.size),
                seed=int(seed),
                level=int(level),
                return_metadata=True,
            )
            stem = f"{args.prefix}{int(args.size)}_L{int(level)}_seed{int(seed)}"
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
                f"[OK] L{level} seed={seed} "
                f"free={float(meta['free_ratio']):.3f} obs={float(meta['obstacle_ratio']):.3f} "
                f"branch={meta['branch_count_placed']}/{meta['branch_count_target']} "
                f"connected={meta['free_connected']}"
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

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_generators.io_utils import save_map_png, write_map_txt
from map_generators.macro_detail_grid import build_macro_detail_grid_map


def _parse_csv_ints(raw: str):
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Generate macro-detail map samples.")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--levels", type=str, default="1,2,3,4")
    p.add_argument("--seeds", type=str, default="101,202,303")
    p.add_argument("--out-dir", type=str, default="map_generators/samples/macro_detail_grid")
    p.add_argument("--prefix", type=str, default="macrodetail")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    levels = _parse_csv_ints(args.levels)
    seeds = _parse_csv_ints(args.seeds)
    rows = []
    for level in levels:
        for seed in seeds:
            grid, meta = build_macro_detail_grid_map(
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
                f"[OK] L{level} seed={seed} obs={float(meta['obstacle_ratio']):.3f} "
                f"detail/free={float(meta['detail_obstacle_ratio_of_base_free']):.3f} "
                f"target={float(meta['target_detail_ratio_of_base_free']):.3f} "
                f"base={meta['base_obstacle_count']} "
                f"block={meta['block_count_placed']} "
                f"trap={meta['trap_count_placed']}/{meta['trap_count_target']}"
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

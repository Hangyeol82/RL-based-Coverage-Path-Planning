import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_generators.io_utils import save_map_png, write_map_txt
from map_generators.shape_grid import build_shape_grid_map


def _entry(seed: int, txt_path: Path, png_path: Path, grid: np.ndarray, png_ok: bool) -> Dict[str, object]:
    try:
        txt_rel = txt_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        txt_rel = txt_path.as_posix()
    try:
        png_rel = png_path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        png_rel = png_path.as_posix()

    item: Dict[str, object] = {
        "seed": int(seed),
        "txt": txt_rel,
        "obstacle_cells": int(np.count_nonzero(grid == 1)),
        "free_cells": int(np.count_nonzero(grid == 0)),
    }
    if png_ok:
        item["png"] = png_rel
    return item


def main() -> None:
    p = argparse.ArgumentParser(description="Generate shape-grid map set and optional manifest group.")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seeds", type=str, default="101,202,303,404,505")
    p.add_argument("--out-dir", type=str, default="map")
    p.add_argument("--prefix", type=str, default="shapegrid")
    p.add_argument("--write-manifest", type=str, default="", help="Optional output json path for this group only.")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_list = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seed_list:
        raise ValueError("no seeds provided")

    entries: List[Dict[str, object]] = []
    for seed in seed_list:
        grid = build_shape_grid_map(size=int(args.size), seed=seed)
        txt_path = out_dir / f"{args.prefix}{int(args.size)}_seed{seed}.txt"
        png_path = out_dir / f"{args.prefix}{int(args.size)}_seed{seed}.png"
        write_map_txt(txt_path, grid)
        png_ok = save_map_png(png_path, grid)
        entries.append(_entry(seed, txt_path, png_path, grid, png_ok))
        print(f"[OK] seed={seed}: {txt_path}")

    if args.write_manifest.strip():
        manifest_path = Path(args.write_manifest).resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"shape_grid": entries}
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[DONE] manifest: {manifest_path}")


if __name__ == "__main__":
    main()

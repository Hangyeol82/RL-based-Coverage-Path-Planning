import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_generators.indoor import build_indoor_map
from map_generators.io_utils import save_map_png, write_map_txt
from map_generators.random_map import build_random_map


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


def generate_manifest_32(
    *,
    out_dir: Path,
    seeds: Sequence[int],
    size: int = 32,
    random_stage: int = 3,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, List[Dict[str, object]]] = {"indoor": [], "random": []}

    for seed in seeds:
        indoor = build_indoor_map(size=size, seed=int(seed))
        indoor_txt = out_dir / f"indoor32_seed{int(seed)}.txt"
        indoor_png = out_dir / f"indoor32_seed{int(seed)}.png"
        write_map_txt(indoor_txt, indoor)
        indoor_png_ok = save_map_png(indoor_png, indoor)
        manifest["indoor"].append(_entry(int(seed), indoor_txt, indoor_png, indoor, indoor_png_ok))

        random_map = build_random_map(size=size, seed=int(seed), stage=random_stage)
        random_txt = out_dir / f"random32_stage{int(random_stage)}_seed{int(seed)}.txt"
        random_png = out_dir / f"random32_stage{int(random_stage)}_seed{int(seed)}.png"
        write_map_txt(random_txt, random_map)
        random_png_ok = save_map_png(random_png, random_map)
        manifest["random"].append(_entry(int(seed), random_txt, random_png, random_map, random_png_ok))

    manifest_path = out_dir / "manifest_32.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    seeds = (101, 202, 303, 404, 505)
    out_dir = Path(__file__).resolve().parent
    manifest_path = generate_manifest_32(out_dir=out_dir, seeds=seeds, size=32, random_stage=3)
    print(f"[DONE] generated 32x32 map set: {manifest_path}")


if __name__ == "__main__":
    main()

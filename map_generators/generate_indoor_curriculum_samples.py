import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_generators.indoor import INDOOR_CURRICULUM_SPECS, build_indoor_curriculum_map
from map_generators.io_utils import save_map_png, write_map_txt


def _parse_stage_list(raw: str):
    out = []
    for tok in str(raw).split(","):
        s = tok.strip()
        if not s:
            continue
        out.append(int(s))
    return tuple(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate 3-stage indoor curriculum samples.")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--out-dir", type=str, default="map_generators/samples/indoor_curriculum_64")
    p.add_argument("--stages", type=str, default="1,2,3")
    p.add_argument("--room-inner", type=int, default=8)
    p.add_argument("--wall-thickness", type=int, default=1)
    p.add_argument("--door-width", type=int, default=2)
    p.add_argument("--extra-connection-prob", type=float, default=0.35)
    p.add_argument("--two-door-prob", type=float, default=0.2)
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = _parse_stage_list(args.stages)

    summary_rows = []
    png_ok = True
    for stage in stages:
        grid, meta = build_indoor_curriculum_map(
            size=args.size,
            seed=args.seed,
            stage=stage,
            room_inner=args.room_inner,
            wall_thickness=args.wall_thickness,
            door_width=args.door_width,
            extra_connection_prob=args.extra_connection_prob,
            two_door_prob=args.two_door_prob,
            return_metadata=True,
        )
        grid = np.asarray(grid, dtype=np.int32)
        txt_path = out_dir / f"indoor_stage{stage}_seed{args.seed}.txt"
        png_path = out_dir / f"indoor_stage{stage}_seed{args.seed}.png"
        write_map_txt(txt_path, grid)
        png_ok = save_map_png(png_path, grid) and png_ok

        obstacle_cells = int(np.count_nonzero(grid == 1))
        total_cells = int(grid.size)
        summary_rows.append(
            {
                "stage": int(stage),
                "seed": int(args.seed),
                "size": int(args.size),
                "merge_edge_count_target": int(INDOOR_CURRICULUM_SPECS[int(stage)]["merge_edge_count"]),
                "merge_grow_prob": float(INDOOR_CURRICULUM_SPECS[int(stage)]["merge_grow_prob"]),
                "merge_seed_edge_count": int(INDOOR_CURRICULUM_SPECS[int(stage)]["merge_seed_edge_count"]),
                "rows": int(meta["rows"]),
                "cols": int(meta["cols"]),
                "active_edge_count": int(meta["active_edge_count"]),
                "merge_edge_count": int(meta["merge_edge_count"]),
                "merge_cluster_sizes": str(meta["merge_cluster_sizes"]),
                "obstacle_cells": obstacle_cells,
                "obstacle_ratio": float(obstacle_cells) / float(max(1, total_cells)),
                "txt_path": str(txt_path),
                "png_path": str(png_path),
            }
        )

    csv_path = out_dir / f"indoor_curriculum_summary_seed{args.seed}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[DONE] indoor curriculum samples -> {out_dir}")
    print(f"[DONE] summary csv -> {csv_path}")
    for row in summary_rows:
        print(
            f"  stage={row['stage']} merge_edges={row['merge_edge_count']} "
            f"clusters={row['merge_cluster_sizes']} seeds={row['merge_seed_edge_count']} "
            f"obstacle_ratio={row['obstacle_ratio']:.3f}"
        )

    if not png_ok:
        raise RuntimeError("matplotlib not available: failed to write one or more PNG files.")


if __name__ == "__main__":
    main()

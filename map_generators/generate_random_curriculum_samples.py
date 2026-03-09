import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MapGenerator import MapGenerator
from map_generators.curriculum_profiles import (
    available_curriculum_profiles,
    default_stages_for_profile,
    stage_difficulty_label,
    stage_token,
)
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
    p = argparse.ArgumentParser(
        description="Generate random-map curriculum stage samples (stage 1~4)."
    )
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--out-dir", type=str, default="map_generators/samples/random_curriculum")
    p.add_argument(
        "--profile",
        type=str,
        default="legacy4",
        choices=available_curriculum_profiles(),
        help="Curriculum profile name.",
    )
    p.add_argument(
        "--stages",
        type=str,
        default="",
        help="Comma-separated stage ids. Defaults to the profile's default stages.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stages = _parse_stage_list(args.stages) if args.stages.strip() else default_stages_for_profile(args.profile)

    summary_rows = []
    png_ok = True

    for stage in stages:
        gen = MapGenerator(height=args.size, width=args.size, seed=args.seed, curriculum_profile=args.profile)
        grid, meta = gen.generate_map(stage=stage, return_metadata=True)
        grid = np.asarray(grid, dtype=np.int32)
        u_shape_count = int(sum(1 for m in meta if str(m.get("type", "")) == "u_shape"))
        difficulty_label = stage_difficulty_label(args.profile, stage)
        token = stage_token(args.profile, stage)

        txt_path = out_dir / f"random_stage{token}_seed{args.seed}.txt"
        png_path = out_dir / f"random_stage{token}_seed{args.seed}.png"
        write_map_txt(txt_path, grid)
        png_ok = save_map_png(png_path, grid) and png_ok

        free_cells = int(np.count_nonzero(grid == 0))
        obstacle_cells = int(np.count_nonzero(grid == 1))
        total = int(grid.size)
        summary_rows.append(
            {
                "stage": int(stage),
                "difficulty_label": difficulty_label,
                "profile": str(args.profile),
                "seed": int(args.seed),
                "size": int(args.size),
                "obstacle_instances": int(len(meta)),
                "u_shape_instances": u_shape_count,
                "obstacle_cells": obstacle_cells,
                "obstacle_ratio": float(obstacle_cells) / float(max(1, total)),
                "free_ratio": float(free_cells) / float(max(1, total)),
                "txt_path": str(txt_path),
                "png_path": str(png_path),
            }
        )

    csv_path = out_dir / f"random_curriculum_summary_seed{args.seed}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[DONE] random curriculum samples -> {out_dir}")
    print(f"[DONE] profile -> {args.profile}")
    print(f"[DONE] summary csv -> {csv_path}")
    for row in summary_rows:
        print(
            f"  stage={row['stage']}({row['difficulty_label']}) "
            f"obs_inst={row['obstacle_instances']} "
            f"u_shape={row['u_shape_instances']} "
            f"obs_ratio={row['obstacle_ratio']:.3f} "
            f"free_ratio={row['free_ratio']:.3f}"
        )

    if not png_ok:
        raise RuntimeError("matplotlib not available: failed to write one or more PNG files.")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_generators.io_utils import save_map_png, write_map_txt
from map_generators.shape_grid_presets import (
    build_validated_shape_grid_map,
    get_paper_shape_grid_preset,
)


def _parse_int_list(raw: str) -> List[int]:
    out: List[int] = []
    for tok in raw.split(","):
        s = tok.strip()
        if s:
            out.append(int(s))
    if not out:
        raise ValueError("List argument must not be empty")
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate validated shape-grid maps for paper experiments."
    )
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--levels", type=str, default="1,2,3,4")
    p.add_argument("--train-seeds", type=str, default="101,202,303,404,505")
    p.add_argument("--test-seeds", type=str, default="1001,1002,1003,1004,1005")
    p.add_argument("--out-dir", type=str, default="map/paper_shapegrid128")
    p.add_argument("--prefix", type=str, default="shapegrid")
    p.add_argument("--max-retries", type=int, default=64)
    p.add_argument("--min-start-component-ratio", type=float, default=0.995)
    p.add_argument("--min-free-ratio", type=float, default=0.50)
    p.add_argument("--max-obstacle-ratio", type=float, default=0.45)
    p.add_argument("--no-png", action="store_true")
    return p.parse_args()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "level",
        "level_name",
        "requested_seed",
        "used_seed",
        "retry_attempt",
        "size",
        "txt",
        "png",
        "free_cells",
        "obstacle_cells",
        "free_ratio",
        "obstacle_ratio",
        "free_component_count",
        "largest_free_component_ratio",
        "start_component_ratio",
        "unreachable_free_cells_from_start",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    args = _parse_args()
    levels = _parse_int_list(args.levels)
    split_to_seeds = {
        "train": _parse_int_list(args.train_seeds),
        "test": _parse_int_list(args.test_seeds),
    }
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    manifest: Dict[str, object] = {
        "generator": "shape_grid",
        "size": int(args.size),
        "levels": levels,
        "checks": {
            "min_start_component_ratio": float(args.min_start_component_ratio),
            "min_free_ratio": float(args.min_free_ratio),
            "max_obstacle_ratio": float(args.max_obstacle_ratio),
        },
        "splits": {"train": [], "test": []},
    }

    for split, seeds in split_to_seeds.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for level in levels:
            preset = get_paper_shape_grid_preset(level)
            for seed in seeds:
                grid, stats, used_seed, attempt = build_validated_shape_grid_map(
                    size=int(args.size),
                    seed=int(seed),
                    level=int(level),
                    max_retries=int(args.max_retries),
                    min_start_component_ratio=float(args.min_start_component_ratio),
                    min_free_ratio=float(args.min_free_ratio),
                    max_obstacle_ratio=float(args.max_obstacle_ratio),
                )
                stem = (
                    f"{args.prefix}{int(args.size)}_L{int(level)}_"
                    f"{preset.name}_seed{int(seed)}"
                )
                txt_path = split_dir / f"{stem}.txt"
                png_path = split_dir / f"{stem}.png"
                write_map_txt(txt_path, grid)
                png_rel = ""
                if not bool(args.no_png):
                    if save_map_png(png_path, grid):
                        png_rel = _rel(png_path)

                row: Dict[str, object] = {
                    "split": split,
                    "level": int(level),
                    "level_name": preset.name,
                    "requested_seed": int(seed),
                    "used_seed": int(used_seed),
                    "retry_attempt": int(attempt),
                    "size": int(args.size),
                    "txt": _rel(txt_path),
                    "png": png_rel,
                    **stats.as_dict(),
                }
                rows.append(row)
                manifest["splits"][split].append(row)  # type: ignore[index]
                print(
                    f"[OK] {split} L{level} seed={seed} used={used_seed} "
                    f"obs={stats.obstacle_ratio:.3f} start_conn={stats.start_component_ratio:.3f} "
                    f"{txt_path}"
                )

    _write_summary_csv(out_dir / "summary.csv", rows)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[DONE] summary: {out_dir / 'summary.csv'}")
    print(f"[DONE] manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

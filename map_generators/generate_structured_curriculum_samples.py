import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_generators.io_utils import save_map_png, write_map_txt
from map_generators.structured import FAMILIES, build_structured_map, compute_map_stats


def _parse_csv_ints(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty integer list")
    return vals


def _parse_csv_strings(raw: str) -> List[str]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("empty string list")
    return vals


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _save_overview(path: Path, entries: List[Dict[str, object]], *, seed: int, size: int) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    subset = [e for e in entries if int(e["seed"]) == int(seed)]
    if not subset:
        return False
    families = list(dict.fromkeys(str(e["family"]) for e in subset))
    levels = sorted({int(e["level"]) for e in subset})
    fig, axes = plt.subplots(
        len(families),
        len(levels),
        figsize=(3.0 * len(levels), 3.0 * len(families)),
        squeeze=False,
    )
    for ax in axes.flat:
        ax.axis("off")
    by_key = {(str(e["family"]), int(e["level"])): e for e in subset}
    for r, family in enumerate(families):
        for c, level in enumerate(levels):
            ax = axes[r][c]
            item = by_key.get((family, level))
            if item is None:
                continue
            grid = np.loadtxt(str(item["txt_path_abs"]), dtype=np.int32)
            ax.imshow(grid == 1, cmap="gray_r", origin="upper")
            ax.set_title(
                f"{family} L{level}\nobs={float(item['obstacle_ratio']):.2f}, narrow={float(item['narrow_ratio']):.2f}",
                fontsize=9,
            )
            ax.axis("off")
    fig.suptitle(f"Structured map families, size={size}, seed={seed}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description="Generate structured CPP map-family samples and stats.")
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--seeds", type=str, default="101,202,303")
    p.add_argument("--levels", type=str, default="1,2,3,4")
    p.add_argument(
        "--families",
        type=str,
        default="pocket_trap,bridge_maze,room_corridor,mixed",
        help=f"Comma-list from {','.join(FAMILIES)}",
    )
    p.add_argument("--out-dir", type=str, default="map_generators/samples/structured_curriculum")
    p.add_argument("--prefix", type=str, default="structured")
    p.add_argument("--overview-seed", type=int, default=101)
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_csv_ints(args.seeds)
    levels = _parse_csv_ints(args.levels)
    families = _parse_csv_strings(args.families)

    rows: List[Dict[str, object]] = []
    for family in families:
        for level in levels:
            for seed in seeds:
                grid, meta = build_structured_map(
                    family=family,
                    size=int(args.size),
                    seed=int(seed),
                    level=int(level),
                    return_metadata=True,
                )
                stats = compute_map_stats(grid)
                stem = f"{args.prefix}_{family}_L{int(level)}_seed{int(seed)}"
                txt_path = out_dir / f"{stem}.txt"
                png_path = out_dir / f"{stem}.png"
                write_map_txt(txt_path, grid)
                png_ok = save_map_png(png_path, grid)
                row: Dict[str, object] = {
                    "family": family,
                    "level": int(level),
                    "seed": int(seed),
                    "size": int(args.size),
                    "txt": _rel(txt_path),
                    "png": _rel(png_path) if png_ok else "",
                    "txt_path_abs": str(txt_path),
                    "png_path_abs": str(png_path),
                }
                row.update(stats)
                # Keep generator metadata compact and CSV-friendly.
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        row[f"meta_{k}"] = v
                rows.append(row)
                print(
                    f"[OK] {family} L{level} seed={seed} "
                    f"obs={stats['obstacle_ratio']:.3f} comp={stats['free_components']} "
                    f"reach={stats['reachable_free_ratio']:.3f}: {txt_path}",
                    flush=True,
                )

    csv_path = out_dir / f"{args.prefix}_summary.csv"
    keys = sorted({k for row in rows for k in row.keys() if not k.endswith("_path_abs")})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})

    overview_path = out_dir / f"{args.prefix}_overview_seed{int(args.overview_seed)}.png"
    ok = _save_overview(overview_path, rows, seed=int(args.overview_seed), size=int(args.size))
    print(f"[DONE] summary: {csv_path}")
    if ok:
        print(f"[DONE] overview: {overview_path}")


if __name__ == "__main__":
    main()

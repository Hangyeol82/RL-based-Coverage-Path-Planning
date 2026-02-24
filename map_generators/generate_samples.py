import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from map_generators.indoor import build_indoor_map
from map_generators.io_utils import save_map_png, write_map_txt
from map_generators.random_map import build_random_map
from map_generators.shape_grid import build_shape_grid_map


def main() -> None:
    p = argparse.ArgumentParser(description="Generate one sample map image per generator.")
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--out-dir", type=str, default="map_generators/samples")
    args = p.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    indoor = build_indoor_map(size=args.size, seed=args.seed, room_inner=8, wall_thickness=1, door_width=2)
    random_map = build_random_map(size=args.size, seed=args.seed, stage=3)
    shape_grid = build_shape_grid_map(size=args.size, seed=args.seed, grid_step=10)

    write_map_txt(out_dir / f"indoor_sample_seed{args.seed}.txt", indoor)
    write_map_txt(out_dir / f"random_sample_seed{args.seed}.txt", random_map)
    write_map_txt(out_dir / f"shape_grid_sample_seed{args.seed}.txt", shape_grid)

    ok1 = save_map_png(out_dir / f"indoor_sample_seed{args.seed}.png", indoor)
    ok2 = save_map_png(out_dir / f"random_sample_seed{args.seed}.png", random_map)
    ok3 = save_map_png(out_dir / f"shape_grid_sample_seed{args.seed}.png", shape_grid)

    if not (ok1 and ok2 and ok3):
        raise RuntimeError("matplotlib not available: failed to write one or more PNG files.")
    print(f"[DONE] samples written to: {out_dir}")


if __name__ == "__main__":
    main()

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MapGenerator import MapGenerator


GridPos = Tuple[int, int]


def _write_map_txt(path: Path, grid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(str(int(v)) for v in row) for row in grid.tolist()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_map_png(path: Path, grid: np.ndarray) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(grid == 1, cmap="gray_r", origin="upper")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return True


def _room_neighbors(r: int, c: int, rows: int, cols: int) -> List[GridPos]:
    out: List[GridPos] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _canon_edge(a: GridPos, b: GridPos) -> Tuple[GridPos, GridPos]:
    return (a, b) if a <= b else (b, a)


def _sample_non_overlapping_intervals(
    segment_len: int,
    width: int,
    n: int,
    rng: np.random.RandomState,
) -> List[Tuple[int, int]]:
    width = max(1, int(width))
    n = max(1, int(n))
    if segment_len <= width:
        return [(0, min(segment_len, width))]

    starts = list(range(0, segment_len - width + 1))
    rng.shuffle(starts)
    out: List[Tuple[int, int]] = []
    for s in starts:
        e = s + width
        bad = False
        for ss, ee in out:
            if not (e <= ss or s >= ee):
                bad = True
                break
        if bad:
            continue
        out.append((s, e))
        if len(out) >= n:
            break
    if not out:
        out = [(0, width)]
    return out


def _build_indoor_map(
    *,
    size: int,
    seed: int,
    room_inner: int = 4,
    wall_thickness: int = 1,
    door_width: int = 1,
    extra_connection_prob: float = 0.35,
    two_door_prob: float = 0.2,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    size = int(size)
    room_inner = max(3, int(room_inner))
    wall_thickness = max(1, int(wall_thickness))
    door_width = max(1, int(door_width))
    extra_connection_prob = float(np.clip(extra_connection_prob, 0.0, 1.0))
    two_door_prob = float(np.clip(two_door_prob, 0.0, 1.0))

    rows = max(2, (size - wall_thickness) // (room_inner + wall_thickness))
    cols = max(2, (size - wall_thickness) // (room_inner + wall_thickness))
    req_h = rows * room_inner + (rows + 1) * wall_thickness
    req_w = cols * room_inner + (cols + 1) * wall_thickness
    if req_h > size or req_w > size:
        raise ValueError("room layout does not fit requested map size")

    off_r = (size - req_h) // 2
    off_c = (size - req_w) // 2

    grid = np.zeros((size, size), dtype=np.int32)

    # Draw walls.
    for k in range(rows + 1):
        r0 = off_r + k * (room_inner + wall_thickness)
        r1 = r0 + wall_thickness
        grid[r0:r1, off_c : off_c + req_w] = 1
    for k in range(cols + 1):
        c0 = off_c + k * (room_inner + wall_thickness)
        c1 = c0 + wall_thickness
        grid[off_r : off_r + req_h, c0:c1] = 1

    # Build room adjacency edges.
    all_edges: List[Tuple[GridPos, GridPos]] = []
    for rr in range(rows):
        for cc in range(cols):
            if rr + 1 < rows:
                all_edges.append(((rr, cc), (rr + 1, cc)))
            if cc + 1 < cols:
                all_edges.append(((rr, cc), (rr, cc + 1)))

    # Random spanning tree for guaranteed room connectivity.
    visited = {(0, 0)}
    stack = [(0, 0)]
    tree_edges = set()
    while stack:
        cur = stack[-1]
        neigh = [n for n in _room_neighbors(cur[0], cur[1], rows, cols) if n not in visited]
        if not neigh:
            stack.pop()
            continue
        nxt = neigh[int(rng.randint(0, len(neigh)))]
        visited.add(nxt)
        tree_edges.add(_canon_edge(cur, nxt))
        stack.append(nxt)

    active_edges = set(tree_edges)
    for e in all_edges:
        ce = _canon_edge(e[0], e[1])
        if ce in active_edges:
            continue
        if rng.rand() < extra_connection_prob:
            active_edges.add(ce)

    # Carve doors for active room adjacencies.
    for a, b in sorted(active_edges):
        ar, ac = a
        br, bc = b
        door_count = 2 if (rng.rand() < two_door_prob) else 1

        if ar == br:
            # Vertical wall between left/right rooms.
            left_c = min(ac, bc)
            wall_c0 = off_c + (left_c + 1) * (room_inner + wall_thickness)
            seg_r0 = off_r + ar * (room_inner + wall_thickness) + wall_thickness
            seg_r1 = seg_r0 + room_inner
            intervals = _sample_non_overlapping_intervals(seg_r1 - seg_r0, door_width, door_count, rng)
            for s, e in intervals:
                grid[seg_r0 + s : seg_r0 + e, wall_c0 : wall_c0 + wall_thickness] = 0
        else:
            # Horizontal wall between top/bottom rooms.
            top_r = min(ar, br)
            wall_r0 = off_r + (top_r + 1) * (room_inner + wall_thickness)
            seg_c0 = off_c + ac * (room_inner + wall_thickness) + wall_thickness
            seg_c1 = seg_c0 + room_inner
            intervals = _sample_non_overlapping_intervals(seg_c1 - seg_c0, door_width, door_count, rng)
            for s, e in intervals:
                grid[wall_r0 : wall_r0 + wall_thickness, seg_c0 + s : seg_c0 + e] = 0

    # Ensure start-near area is free.
    grid[0, 0] = 0
    if size > 1:
        grid[0, 1] = 0
        grid[1, 0] = 0
    return grid


def _build_random_map(*, size: int, seed: int, stage: int = 3) -> np.ndarray:
    gen = MapGenerator(height=size, width=size, seed=seed)
    grid = np.asarray(gen.generate_map(stage=stage), dtype=np.int32)
    if grid.shape != (size, size):
        out = np.zeros((size, size), dtype=np.int32)
        h = min(size, grid.shape[0])
        w = min(size, grid.shape[1])
        out[:h, :w] = grid[:h, :w]
        grid = out
    grid[0, 0] = 0
    return grid


def _entry(seed: int, txt_path: Path, png_path: Path, grid: np.ndarray, png_ok: bool) -> Dict[str, object]:
    txt_rel = txt_path.relative_to(REPO_ROOT).as_posix()
    png_rel = png_path.relative_to(REPO_ROOT).as_posix()
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
        indoor = _build_indoor_map(size=size, seed=int(seed))
        indoor_txt = out_dir / f"indoor32_seed{int(seed)}.txt"
        indoor_png = out_dir / f"indoor32_seed{int(seed)}.png"
        _write_map_txt(indoor_txt, indoor)
        indoor_png_ok = _save_map_png(indoor_png, indoor)
        manifest["indoor"].append(_entry(int(seed), indoor_txt, indoor_png, indoor, indoor_png_ok))

        random_map = _build_random_map(size=size, seed=int(seed), stage=random_stage)
        random_txt = out_dir / f"random32_stage{int(random_stage)}_seed{int(seed)}.txt"
        random_png = out_dir / f"random32_stage{int(random_stage)}_seed{int(seed)}.png"
        _write_map_txt(random_txt, random_map)
        random_png_ok = _save_map_png(random_png, random_map)
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

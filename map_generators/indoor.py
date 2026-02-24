from typing import List, Tuple

import numpy as np


GridPos = Tuple[int, int]


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
        has_overlap = False
        for ss, ee in out:
            if not (e <= ss or s >= ee):
                has_overlap = True
                break
        if has_overlap:
            continue
        out.append((s, e))
        if len(out) >= n:
            break
    if not out:
        out = [(0, width)]
    return out


def build_indoor_map(
    *,
    size: int,
    seed: int,
    room_inner: int = 4,
    wall_thickness: int = 1,
    door_width: int = 1,
    extra_connection_prob: float = 0.35,
    two_door_prob: float = 0.2,
    ensure_start_clear: bool = True,
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

    # Room walls
    for k in range(rows + 1):
        r0 = off_r + k * (room_inner + wall_thickness)
        r1 = r0 + wall_thickness
        grid[r0:r1, off_c : off_c + req_w] = 1
    for k in range(cols + 1):
        c0 = off_c + k * (room_inner + wall_thickness)
        c1 = c0 + wall_thickness
        grid[off_r : off_r + req_h, c0:c1] = 1

    all_edges: List[Tuple[GridPos, GridPos]] = []
    for rr in range(rows):
        for cc in range(cols):
            if rr + 1 < rows:
                all_edges.append(((rr, cc), (rr + 1, cc)))
            if cc + 1 < cols:
                all_edges.append(((rr, cc), (rr, cc + 1)))

    # Spanning tree guarantees that all rooms are connected.
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

    for a, b in sorted(active_edges):
        ar, ac = a
        br, bc = b
        door_count = 2 if (rng.rand() < two_door_prob) else 1
        if ar == br:
            left_c = min(ac, bc)
            wall_c0 = off_c + (left_c + 1) * (room_inner + wall_thickness)
            seg_r0 = off_r + ar * (room_inner + wall_thickness) + wall_thickness
            seg_r1 = seg_r0 + room_inner
            intervals = _sample_non_overlapping_intervals(seg_r1 - seg_r0, door_width, door_count, rng)
            for s, e in intervals:
                grid[seg_r0 + s : seg_r0 + e, wall_c0 : wall_c0 + wall_thickness] = 0
        else:
            top_r = min(ar, br)
            wall_r0 = off_r + (top_r + 1) * (room_inner + wall_thickness)
            seg_c0 = off_c + ac * (room_inner + wall_thickness) + wall_thickness
            seg_c1 = seg_c0 + room_inner
            intervals = _sample_non_overlapping_intervals(seg_c1 - seg_c0, door_width, door_count, rng)
            for s, e in intervals:
                grid[wall_r0 : wall_r0 + wall_thickness, seg_c0 + s : seg_c0 + e] = 0

    if ensure_start_clear:
        grid[0, 0] = 0
        if size > 1:
            grid[0, 1] = 0
            grid[1, 0] = 0
    return grid

from typing import List, Tuple

import numpy as np


GridPos = Tuple[int, int]
Edge = Tuple[GridPos, GridPos]


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


def _sample_clustered_edges(
    edges: List[Edge],
    count: int,
    rng: np.random.RandomState,
    grow_prob: float,
    seed_edge_count: int,
) -> List[Edge]:
    count = max(0, int(count))
    if count <= 0 or not edges:
        return []

    grow_prob = float(np.clip(grow_prob, 0.0, 1.0))
    seed_edge_count = max(1, min(int(seed_edge_count), count))
    remaining = list(edges)
    rng.shuffle(remaining)
    selected: List[Edge] = []
    frontier_nodes = set()

    seed_candidates = list(remaining)
    rng.shuffle(seed_candidates)
    for edge in seed_candidates:
        if len(selected) >= seed_edge_count:
            break
        a, b = edge
        if a in frontier_nodes or b in frontier_nodes:
            continue
        selected.append(edge)
        frontier_nodes.add(a)
        frontier_nodes.add(b)
        remaining.remove(edge)

    while remaining and len(selected) < seed_edge_count:
        edge = remaining.pop(0)
        selected.append(edge)
        frontier_nodes.add(edge[0])
        frontier_nodes.add(edge[1])

    while remaining and len(selected) < count:
        candidate_pool = []
        if rng.rand() < grow_prob:
            candidate_pool = [edge for edge in remaining if edge[0] in frontier_nodes or edge[1] in frontier_nodes]
        if not candidate_pool:
            candidate_pool = remaining
        pick_idx = int(rng.randint(0, len(candidate_pool)))
        edge = candidate_pool[pick_idx]
        remaining.remove(edge)
        selected.append(edge)
        frontier_nodes.add(edge[0])
        frontier_nodes.add(edge[1])
    return selected


def _merge_cluster_sizes(rows: int, cols: int, merge_edges: List[Edge]) -> List[int]:
    if not merge_edges:
        return []

    parent = {}

    def find(x: GridPos) -> GridPos:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: GridPos, b: GridPos) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    for rr in range(rows):
        for cc in range(cols):
            parent[(rr, cc)] = (rr, cc)
    for a, b in merge_edges:
        union(a, b)

    sizes = {}
    for node in parent:
        root = find(node)
        sizes[root] = sizes.get(root, 0) + 1
    return sorted((size for size in sizes.values() if size > 1), reverse=True)


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
    merge_edge_count: int = 0,
    merge_room_ratio: float = 0.0,
    merge_grow_prob: float = 0.75,
    merge_seed_edge_count: int = 1,
    ensure_start_clear: bool = True,
    return_metadata: bool = False,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    size = int(size)
    room_inner = max(3, int(room_inner))
    wall_thickness = max(1, int(wall_thickness))
    door_width = max(1, int(door_width))
    extra_connection_prob = float(np.clip(extra_connection_prob, 0.0, 1.0))
    two_door_prob = float(np.clip(two_door_prob, 0.0, 1.0))
    merge_room_ratio = float(np.clip(merge_room_ratio, 0.0, 1.0))

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

    merge_edge_count = max(0, int(merge_edge_count))
    if merge_edge_count <= 0 and merge_room_ratio > 0.0:
        merge_edge_count = max(1, int(round(rows * cols * merge_room_ratio)))
    merge_edge_count = min(merge_edge_count, len(active_edges))
    merge_edges = set(
        _sample_clustered_edges(
            sorted(active_edges),
            merge_edge_count,
            rng,
            merge_grow_prob,
            merge_seed_edge_count,
        )
    )

    for a, b in sorted(active_edges):
        ar, ac = a
        br, bc = b
        if ar == br:
            left_c = min(ac, bc)
            wall_c0 = off_c + (left_c + 1) * (room_inner + wall_thickness)
            seg_r0 = off_r + ar * (room_inner + wall_thickness) + wall_thickness
            seg_r1 = seg_r0 + room_inner
            if _canon_edge(a, b) in merge_edges:
                grid[seg_r0:seg_r1, wall_c0 : wall_c0 + wall_thickness] = 0
            else:
                door_count = 2 if (rng.rand() < two_door_prob) else 1
                intervals = _sample_non_overlapping_intervals(seg_r1 - seg_r0, door_width, door_count, rng)
                for s, e in intervals:
                    grid[seg_r0 + s : seg_r0 + e, wall_c0 : wall_c0 + wall_thickness] = 0
        else:
            top_r = min(ar, br)
            wall_r0 = off_r + (top_r + 1) * (room_inner + wall_thickness)
            seg_c0 = off_c + ac * (room_inner + wall_thickness) + wall_thickness
            seg_c1 = seg_c0 + room_inner
            if _canon_edge(a, b) in merge_edges:
                grid[wall_r0 : wall_r0 + wall_thickness, seg_c0:seg_c1] = 0
            else:
                door_count = 2 if (rng.rand() < two_door_prob) else 1
                intervals = _sample_non_overlapping_intervals(seg_c1 - seg_c0, door_width, door_count, rng)
                for s, e in intervals:
                    grid[wall_r0 : wall_r0 + wall_thickness, seg_c0 + s : seg_c0 + e] = 0

    if ensure_start_clear:
        grid[0, 0] = 0
        if size > 1:
            grid[0, 1] = 0
            grid[1, 0] = 0
    if not return_metadata:
        return grid

    metadata = {
        "size": int(size),
        "seed": int(seed),
        "rows": int(rows),
        "cols": int(cols),
        "room_inner": int(room_inner),
        "wall_thickness": int(wall_thickness),
        "door_width": int(door_width),
        "active_edge_count": int(len(active_edges)),
        "merge_edge_count": int(len(merge_edges)),
        "merge_room_ratio": float(merge_room_ratio),
        "merge_grow_prob": float(merge_grow_prob),
        "merge_seed_edge_count": int(merge_seed_edge_count),
        "merge_cluster_sizes": _merge_cluster_sizes(rows, cols, sorted(merge_edges)),
    }
    return grid, metadata


INDOOR_CURRICULUM_SPECS = {
    1: {"merge_edge_count": 48, "merge_grow_prob": 0.78, "merge_seed_edge_count": 6},
    2: {"merge_edge_count": 30, "merge_grow_prob": 0.76, "merge_seed_edge_count": 5},
    3: {"merge_edge_count": 0, "merge_grow_prob": 0.0, "merge_seed_edge_count": 1},
}


def build_indoor_curriculum_map(
    *,
    size: int,
    seed: int,
    stage: int,
    room_inner: int = 8,
    wall_thickness: int = 1,
    door_width: int = 2,
    extra_connection_prob: float = 0.35,
    two_door_prob: float = 0.2,
    ensure_start_clear: bool = True,
    return_metadata: bool = False,
):
    if int(stage) not in INDOOR_CURRICULUM_SPECS:
        raise ValueError(f"unsupported indoor curriculum stage: {stage}")
    spec = dict(INDOOR_CURRICULUM_SPECS[int(stage)])
    return build_indoor_map(
        size=size,
        seed=seed,
        room_inner=room_inner,
        wall_thickness=wall_thickness,
        door_width=door_width,
        extra_connection_prob=extra_connection_prob,
        two_door_prob=two_door_prob,
        ensure_start_clear=ensure_start_clear,
        return_metadata=return_metadata,
        **spec,
    )

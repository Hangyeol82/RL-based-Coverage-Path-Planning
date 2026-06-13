from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .indoor import build_indoor_map


GridPos = Tuple[int, int]


FAMILIES = ("pocket_trap", "bridge_maze", "room_corridor", "mixed")


_POCKET_SPECS = {
    1: {"count": 7, "block_count": 4, "min_size": 11, "max_size": 22, "gap": 5, "thickness": (1, 1), "overlap": 0.10},
    2: {"count": 10, "block_count": 7, "min_size": 16, "max_size": 38, "gap": 7, "thickness": (1, 2), "overlap": 0.22},
    3: {"count": 13, "block_count": 11, "min_size": 20, "max_size": 50, "gap": 8, "thickness": (1, 3), "overlap": 0.32},
    4: {"count": 16, "block_count": 15, "min_size": 22, "max_size": 52, "gap": 8, "thickness": (1, 3), "overlap": 0.34},
}

_BRIDGE_SPECS = {
    1: {"walls": 6, "block_count": 2, "gap_count": 3, "thickness": 1, "min_spacing": 14, "span_ratio": 0.62},
    2: {"walls": 8, "block_count": 5, "gap_count": 3, "thickness": 1, "min_spacing": 12, "span_ratio": 0.72},
    3: {"walls": 10, "block_count": 8, "gap_count": 3, "thickness": 1, "min_spacing": 9, "span_ratio": 0.82},
    4: {"walls": 12, "block_count": 10, "gap_count": 2, "thickness": 1, "min_spacing": 7, "span_ratio": 0.90},
}

_ROOM_SPECS = {
    1: {"room_inner": 20, "door_width": 5, "extra_connection_prob": 0.65, "two_door_prob": 0.55, "merge_room_ratio": 0.32},
    2: {"room_inner": 17, "door_width": 4, "extra_connection_prob": 0.55, "two_door_prob": 0.40, "merge_room_ratio": 0.18},
    3: {"room_inner": 13, "door_width": 3, "extra_connection_prob": 0.40, "two_door_prob": 0.24, "merge_room_ratio": 0.06},
    4: {"room_inner": 8, "door_width": 2, "extra_connection_prob": 0.28, "two_door_prob": 0.10, "merge_room_ratio": 0.00},
}


def _norm_level(level: int) -> int:
    level = int(level)
    if level not in {1, 2, 3, 4}:
        raise ValueError(f"level must be one of 1,2,3,4; got {level}")
    return level


def _clear_start(grid: np.ndarray, radius: int = 2) -> None:
    rr = min(grid.shape[0], max(1, int(radius) + 1))
    cc = min(grid.shape[1], max(1, int(radius) + 1))
    grid[:rr, :cc] = 0


def _neighbors(r: int, c: int, rows: int, cols: int) -> Iterable[GridPos]:
    if r > 0:
        yield r - 1, c
    if r + 1 < rows:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < cols:
        yield r, c + 1


def _reachable_mask(grid: np.ndarray, start: GridPos = (0, 0)) -> np.ndarray:
    rows, cols = grid.shape
    sr, sc = start
    seen = np.zeros_like(grid, dtype=bool)
    if not (0 <= sr < rows and 0 <= sc < cols) or grid[sr, sc] != 0:
        return seen
    q: deque[GridPos] = deque([(int(sr), int(sc))])
    seen[sr, sc] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors(r, c, rows, cols):
            if seen[nr, nc] or grid[nr, nc] != 0:
                continue
            seen[nr, nc] = True
            q.append((nr, nc))
    return seen


def _keep_start_component(grid: np.ndarray, start: GridPos = (0, 0)) -> np.ndarray:
    out = np.asarray(grid, dtype=np.int32).copy()
    seen = _reachable_mask(out, start=start)
    out[(out == 0) & (~seen)] = 1
    return out


def _free_components(grid: np.ndarray) -> List[List[GridPos]]:
    rows, cols = grid.shape
    unseen = grid == 0
    comps: List[List[GridPos]] = []
    for sr, sc in np.argwhere(unseen):
        if not unseen[sr, sc]:
            continue
        comp: List[GridPos] = []
        q: deque[GridPos] = deque([(int(sr), int(sc))])
        unseen[sr, sc] = False
        while q:
            r, c = q.popleft()
            comp.append((int(r), int(c)))
            for nr, nc in _neighbors(r, c, rows, cols):
                if unseen[nr, nc]:
                    unseen[nr, nc] = False
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _carve_l_corridor(grid: np.ndarray, a: GridPos, b: GridPos, *, width: int = 1) -> None:
    ar, ac = a
    br, bc = b
    half = max(0, int(width) // 2)
    r0, r1 = sorted((int(ar), int(br)))
    c0, c1 = sorted((int(ac), int(bc)))
    _draw_rect(grid, r0, int(ac) - half, r1 - r0 + 1, int(width), 0)
    _draw_rect(grid, int(br) - half, c0, int(width), c1 - c0 + 1, 0)


def _connect_free_components(grid: np.ndarray, start: GridPos = (0, 0)) -> np.ndarray:
    out = np.asarray(grid, dtype=np.int32).copy()
    if out[start] != 0:
        out[start] = 0

    # Connect all disconnected free components back to the default start.
    # This preserves procedural structure without making coverage impossible.
    for _ in range(8):
        comps = _free_components(out)
        if len(comps) <= 1:
            return out
        for comp in comps:
            if start in comp:
                continue
            rep = comp[len(comp) // 2]
            _carve_l_corridor(out, start, rep, width=1)
    return out


def _draw_rect(grid: np.ndarray, r0: int, c0: int, h: int, w: int, value: int = 1) -> None:
    rows, cols = grid.shape
    r0 = int(np.clip(r0, 0, rows))
    c0 = int(np.clip(c0, 0, cols))
    r1 = int(np.clip(r0 + max(0, int(h)), 0, rows))
    c1 = int(np.clip(c0 + max(0, int(w)), 0, cols))
    if r1 > r0 and c1 > c0:
        grid[r0:r1, c0:c1] = int(value)


def _draw_u_pocket(
    grid: np.ndarray,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
    open_side: str,
    gap_width: int,
    thickness: int,
) -> None:
    bottom = top + height - 1
    right = left + width - 1
    t = max(1, int(thickness))
    if bottom <= top + 2 or right <= left + 2:
        return

    # Draw closed rectangle walls first.
    _draw_rect(grid, top, left, t, width, 1)
    _draw_rect(grid, bottom - t + 1, left, t, width, 1)
    _draw_rect(grid, top, left, height, t, 1)
    _draw_rect(grid, top, right - t + 1, height, t, 1)

    # Open one side around the middle. This creates C/U-shaped obstacle pockets.
    gap_width = max(1, int(gap_width))
    if open_side in {"left", "right"}:
        center = top + height // 2
        g0 = center - gap_width // 2
        if open_side == "left":
            _draw_rect(grid, g0, left, gap_width, t, 0)
        else:
            _draw_rect(grid, g0, right - t + 1, gap_width, t, 0)
    else:
        center = left + width // 2
        g0 = center - gap_width // 2
        if open_side == "up":
            _draw_rect(grid, top, g0, t, gap_width, 0)
        else:
            _draw_rect(grid, bottom - t + 1, g0, t, gap_width, 0)


def _u_pocket_candidate(
    shape: Tuple[int, int],
    *,
    top: int,
    left: int,
    height: int,
    width: int,
    open_side: str,
    gap_width: int,
    thickness: int,
) -> np.ndarray:
    candidate = np.zeros(shape, dtype=np.int32)
    _draw_u_pocket(
        candidate,
        top=top,
        left=left,
        height=height,
        width=width,
        open_side=open_side,
        gap_width=gap_width,
        thickness=thickness,
    )
    return candidate == 1


def _has_obstacle_overlap(grid: np.ndarray, candidate: np.ndarray, clearance: int) -> bool:
    rows, cols = np.where(candidate)
    if rows.size == 0:
        return True
    r0 = max(0, int(rows.min()) - int(clearance))
    r1 = min(grid.shape[0], int(rows.max()) + int(clearance) + 1)
    c0 = max(0, int(cols.min()) - int(clearance))
    c1 = min(grid.shape[1], int(cols.max()) + int(clearance) + 1)
    return bool(np.any(grid[r0:r1, c0:c1] == 1))


def _candidate_overlap(grid: np.ndarray, candidate: np.ndarray) -> Tuple[int, int, int]:
    area = int(np.count_nonzero(candidate))
    overlap = int(np.count_nonzero(candidate & (grid == 1)))
    return area, overlap, area - overlap


def _add_random_blocks(grid: np.ndarray, rng: np.random.RandomState, *, count: int, level: int) -> None:
    size = grid.shape[0]
    for _ in range(max(0, int(count))):
        h = int(rng.randint(3 + level, 7 + 2 * level))
        w = int(rng.randint(3 + level, 7 + 2 * level))
        r = int(rng.randint(4, max(5, size - h - 3)))
        c = int(rng.randint(4, max(5, size - w - 3)))
        if r < 6 and c < 6:
            continue
        _draw_rect(grid, r, c, h, w, 1)


def _add_nonoverlap_blocks(
    grid: np.ndarray,
    rng: np.random.RandomState,
    *,
    count: int,
    min_size: int,
    max_size: int,
    clearance: int,
) -> int:
    size = grid.shape[0]
    placed = 0
    for _ in range(max(0, int(count)) * 30):
        if placed >= int(count):
            break
        h = int(rng.randint(max(3, int(min_size)), max(4, int(max_size)) + 1))
        w = int(rng.randint(max(3, int(min_size)), max(4, int(max_size)) + 1))
        if h >= size - 8 or w >= size - 8:
            continue
        r = int(rng.randint(4, max(5, size - h - 3)))
        c = int(rng.randint(4, max(5, size - w - 3)))
        if r < 8 and c < 8:
            continue
        candidate = np.zeros_like(grid, dtype=bool)
        candidate[r : r + h, c : c + w] = True
        if _has_obstacle_overlap(grid, candidate, clearance=clearance):
            continue
        test = grid.copy()
        test[candidate] = 1
        if len(_free_components(test)) != 1:
            continue
        grid[candidate] = 1
        placed += 1
    return placed


def _add_overlap_blocks(
    grid: np.ndarray,
    rng: np.random.RandomState,
    *,
    count: int,
    min_size: int,
    max_size: int,
    max_overlap_ratio: float,
) -> int:
    size = grid.shape[0]
    placed = 0
    max_overlap_ratio = float(np.clip(max_overlap_ratio, 0.0, 0.95))
    for _ in range(max(0, int(count)) * 60):
        if placed >= int(count):
            break
        h = int(rng.randint(max(3, int(min_size)), max(4, int(max_size)) + 1))
        w = int(rng.randint(max(3, int(min_size)), max(4, int(max_size)) + 1))
        if h >= size - 8 or w >= size - 8:
            continue
        r = int(rng.randint(4, max(5, size - h - 3)))
        c = int(rng.randint(4, max(5, size - w - 3)))
        if r < 8 and c < 8:
            continue
        candidate = np.zeros_like(grid, dtype=bool)
        candidate[r : r + h, c : c + w] = True
        area, overlap, new_cells = _candidate_overlap(grid, candidate)
        if area <= 0 or new_cells < max(4, area // 4):
            continue
        if (float(overlap) / float(area)) > max_overlap_ratio:
            continue
        test = grid.copy()
        test[candidate] = 1
        if len(_free_components(test)) != 1:
            continue
        grid[candidate] = 1
        placed += 1
    return placed


def _finalize(grid: np.ndarray, *, ensure_start_clear: bool) -> np.ndarray:
    out = np.asarray(grid, dtype=np.int32).copy()
    if ensure_start_clear:
        _clear_start(out, radius=3)
    out = _connect_free_components(out, start=(0, 0))
    if ensure_start_clear:
        _clear_start(out, radius=2)
    return out


def build_pocket_trap_map(
    *,
    size: int,
    seed: int,
    level: int = 3,
    ensure_start_clear: bool = True,
    return_metadata: bool = False,
):
    """Build maps with U/C-shaped obstacle pockets and local traps."""
    level = _norm_level(level)
    size = max(24, int(size))
    rng = np.random.RandomState(seed)
    spec = dict(_POCKET_SPECS[level])
    scale = max(0.6, (size / 128.0) ** 2)
    pocket_count = max(1, int(round(spec["count"] * scale)))
    block_count = max(0, int(round(spec["block_count"] * scale)))

    grid = np.zeros((size, size), dtype=np.int32)
    margin = max(5, size // 18)
    placed = 0
    attempts = pocket_count * 50
    clearance = max(1, size // 96)
    max_overlap_ratio = float(spec.get("overlap", 0.0))
    for _ in range(attempts):
        if placed >= pocket_count:
            break
        h = int(rng.randint(spec["min_size"], spec["max_size"] + 1))
        w = int(rng.randint(spec["min_size"], spec["max_size"] + 1))
        h = min(h, size - 2 * margin)
        w = min(w, size - 2 * margin)
        if h < 7 or w < 7:
            continue
        top = int(rng.randint(margin, max(margin + 1, size - h - margin)))
        left = int(rng.randint(margin, max(margin + 1, size - w - margin)))
        if top < 10 and left < 10:
            continue
        side = ("left", "right", "up", "down")[int(rng.randint(0, 4))]
        thickness_spec = spec.get("thickness", (1, 1))
        if isinstance(thickness_spec, tuple):
            thickness = int(rng.randint(int(thickness_spec[0]), int(thickness_spec[1]) + 1))
        else:
            thickness = int(thickness_spec)
        candidate = _u_pocket_candidate(
            grid.shape,
            top=top,
            left=left,
            height=h,
            width=w,
            open_side=side,
            gap_width=int(spec["gap"]),
            thickness=thickness,
        )
        area, overlap, new_cells = _candidate_overlap(grid, candidate)
        if area <= 0:
            continue
        if level == 1 and _has_obstacle_overlap(grid, candidate, clearance=clearance):
            continue
        if level > 1:
            if (float(overlap) / float(area)) > max_overlap_ratio:
                continue
            if new_cells < max(12, area // 3):
                continue

        test = grid.copy()
        test[candidate] = 1
        if len(_free_components(test)) != 1:
            continue
        free_ratio = float(np.count_nonzero(test == 0)) / float(test.size)
        if free_ratio < 0.45:
            continue
        grid = test
        placed += 1

    placed_blocks = _add_overlap_blocks(
        grid,
        rng,
        count=block_count,
        min_size=max(3, int(spec["min_size"]) // 4),
        max_size=max(4, int(spec["max_size"]) // 3),
        max_overlap_ratio=max(0.15, max_overlap_ratio),
    )
    grid = _finalize(grid, ensure_start_clear=ensure_start_clear)
    meta = {
        "family": "pocket_trap",
        "size": int(size),
        "seed": int(seed),
        "level": int(level),
        "pocket_count_target": int(pocket_count),
        "pocket_count_placed": int(placed),
        "block_count": int(block_count),
        "block_count_placed": int(placed_blocks),
    }
    return (grid, meta) if return_metadata else grid


def _draw_wall_with_gaps(
    grid: np.ndarray,
    rng: np.random.RandomState,
    *,
    orientation: str,
    pos: int,
    span0: int,
    span1: int,
    thickness: int,
    gap_count: int,
    gap_width: int,
) -> None:
    size = grid.shape[0]
    span0 = int(np.clip(span0, 0, size - 1))
    span1 = int(np.clip(span1, span0 + 1, size))
    length = span1 - span0
    gap_count = max(1, int(gap_count))
    gap_width = max(2, int(gap_width))
    gaps: List[Tuple[int, int]] = []
    for _ in range(gap_count):
        if length <= gap_width + 2:
            break
        g0 = int(rng.randint(span0 + 1, max(span0 + 2, span1 - gap_width - 1)))
        gaps.append((g0, g0 + gap_width))

    if orientation == "vertical":
        _draw_rect(grid, span0, pos, length, thickness, 1)
        for g0, g1 in gaps:
            _draw_rect(grid, g0, pos, g1 - g0, thickness, 0)
    else:
        _draw_rect(grid, pos, span0, thickness, length, 1)
        for g0, g1 in gaps:
            _draw_rect(grid, pos, g0, thickness, g1 - g0, 0)


def build_bridge_maze_map(
    *,
    size: int,
    seed: int,
    level: int = 3,
    ensure_start_clear: bool = True,
    return_metadata: bool = False,
):
    """Build maps with long barriers, narrow passages, and bottleneck bridges."""
    level = _norm_level(level)
    size = max(24, int(size))
    rng = np.random.RandomState(seed)
    spec = dict(_BRIDGE_SPECS[level])
    scale = max(0.7, size / 128.0)
    wall_count = max(1, int(round(spec["walls"] * scale)))
    block_count = max(0, int(round(spec["block_count"] * scale)))

    grid = np.zeros((size, size), dtype=np.int32)
    margin = max(6, size // 12)
    used_positions: List[int] = []
    min_spacing = max(6, int(spec.get("min_spacing", 12) * scale))
    span_ratio = float(np.clip(float(spec.get("span_ratio", 0.65)), 0.25, 0.95))
    for _ in range(wall_count * 10):
        if len(used_positions) >= wall_count:
            break
        vertical = bool(rng.rand() < 0.5)
        pos = int(rng.randint(margin, size - margin))
        if any(abs(pos - p) < min_spacing for p in used_positions):
            continue
        used_positions.append(pos)
        span_len = int(np.clip(round(size * span_ratio), 12, size - 2 * margin))
        span0 = int(rng.randint(2, max(3, size - span_len - 2)))
        span1 = int(span0 + span_len)
        gap_width = int(rng.randint(4, 8 if level < 4 else 7))
        _draw_wall_with_gaps(
            grid,
            rng,
            orientation="vertical" if vertical else "horizontal",
            pos=pos,
            span0=span0,
            span1=span1,
            thickness=int(spec["thickness"]),
            gap_count=int(spec["gap_count"]),
            gap_width=gap_width,
        )

    grid = _connect_free_components(grid, start=(0, 0))
    placed_blocks = _add_overlap_blocks(
        grid,
        rng,
        count=block_count,
        min_size=4 + level,
        max_size=8 + 2 * level,
        max_overlap_ratio=0.45 if level >= 3 else 0.30,
    )
    grid = _finalize(grid, ensure_start_clear=ensure_start_clear)
    meta = {
        "family": "bridge_maze",
        "size": int(size),
        "seed": int(seed),
        "level": int(level),
        "wall_count_target": int(wall_count),
        "wall_count_placed": int(len(used_positions)),
        "block_count": int(block_count),
        "block_count_placed": int(placed_blocks),
    }
    return (grid, meta) if return_metadata else grid


def build_room_corridor_map(
    *,
    size: int,
    seed: int,
    level: int = 3,
    ensure_start_clear: bool = True,
    return_metadata: bool = False,
):
    """Build connected room/corridor maps with explicit access from the default start."""
    level = _norm_level(level)
    spec = dict(_ROOM_SPECS[level])
    grid, indoor_meta = build_indoor_map(
        size=int(size),
        seed=int(seed),
        wall_thickness=1,
        ensure_start_clear=False,
        return_metadata=True,
        **spec,
    )
    grid = np.asarray(grid, dtype=np.int32)

    room_inner = int(indoor_meta["room_inner"])
    wall_thickness = int(indoor_meta["wall_thickness"])
    rows = int(indoor_meta["rows"])
    cols = int(indoor_meta["cols"])
    req_h = rows * room_inner + (rows + 1) * wall_thickness
    req_w = cols * room_inner + (cols + 1) * wall_thickness
    off_r = (int(size) - req_h) // 2
    off_c = (int(size) - req_w) // 2

    # Open a small doorway in the first room's outer wall so the default start
    # at (0, 0) is connected without carving a long artificial side corridor.
    door_width = max(1, min(int(indoor_meta["door_width"]), room_inner))
    door_c0 = off_c + wall_thickness
    door_c1 = min(door_c0 + door_width, off_c + wall_thickness + room_inner)
    grid[off_r : off_r + wall_thickness, door_c0:door_c1] = 0
    door_center = (door_c0 + door_c1 - 1) // 2
    grid[0, 0 : door_center + 1] = 0
    grid[0 : off_r + 1, door_center] = 0

    if ensure_start_clear:
        _clear_start(grid, radius=2)
    grid = _keep_start_component(grid, start=(0, 0))
    if ensure_start_clear:
        _clear_start(grid, radius=2)
    meta = {"family": "room_corridor", "level": int(level), **indoor_meta}
    return (grid, meta) if return_metadata else grid


def build_mixed_structured_map(
    *,
    size: int,
    seed: int,
    level: int = 3,
    ensure_start_clear: bool = True,
    return_metadata: bool = False,
):
    """Sample one structured family. Useful as a mixed curriculum source."""
    rng = np.random.RandomState(seed)
    families = ("pocket_trap", "bridge_maze", "room_corridor")
    family = families[int(rng.randint(0, len(families)))]
    grid, meta = build_structured_map(
        family=family,
        size=size,
        seed=seed + 7919,
        level=level,
        ensure_start_clear=ensure_start_clear,
        return_metadata=True,
    )
    meta = {"family": "mixed", "sampled_family": family, **meta}
    return (grid, meta) if return_metadata else grid


def build_structured_map(
    *,
    family: str,
    size: int,
    seed: int,
    level: int = 3,
    ensure_start_clear: bool = True,
    return_metadata: bool = False,
):
    family = str(family).strip().lower()
    if family == "pocket":
        family = "pocket_trap"
    if family == "bridge":
        family = "bridge_maze"
    if family == "rooms":
        family = "room_corridor"
    if family == "pocket_trap":
        return build_pocket_trap_map(
            size=size,
            seed=seed,
            level=level,
            ensure_start_clear=ensure_start_clear,
            return_metadata=return_metadata,
        )
    if family == "bridge_maze":
        return build_bridge_maze_map(
            size=size,
            seed=seed,
            level=level,
            ensure_start_clear=ensure_start_clear,
            return_metadata=return_metadata,
        )
    if family == "room_corridor":
        return build_room_corridor_map(
            size=size,
            seed=seed,
            level=level,
            ensure_start_clear=ensure_start_clear,
            return_metadata=return_metadata,
        )
    if family == "mixed":
        return build_mixed_structured_map(
            size=size,
            seed=seed,
            level=level,
            ensure_start_clear=ensure_start_clear,
            return_metadata=return_metadata,
        )
    raise ValueError(f"unknown structured map family: {family}; expected one of {FAMILIES}")


def compute_map_stats(grid: np.ndarray, *, start: GridPos = (0, 0)) -> Dict[str, float]:
    g = np.asarray(grid, dtype=np.int32)
    rows, cols = g.shape
    free = g == 0
    obs = g == 1
    reachable = _reachable_mask(g, start=start)

    dead_ends = 0
    narrow = 0
    perimeter = 0
    for r in range(rows):
        for c in range(cols):
            if free[r, c]:
                deg = sum(1 for nr, nc in _neighbors(r, c, rows, cols) if free[nr, nc])
                if deg <= 1:
                    dead_ends += 1
                if deg <= 2:
                    narrow += 1
            else:
                for nr, nc in _neighbors(r, c, rows, cols):
                    if free[nr, nc]:
                        perimeter += 1

    # Count free components for diagnostics.
    unseen = free.copy()
    components = 0
    for sr, sc in np.argwhere(unseen):
        if not unseen[sr, sc]:
            continue
        components += 1
        q: deque[GridPos] = deque([(int(sr), int(sc))])
        unseen[sr, sc] = False
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors(r, c, rows, cols):
                if unseen[nr, nc]:
                    unseen[nr, nc] = False
                    q.append((nr, nc))

    total = float(g.size)
    free_cells = int(np.count_nonzero(free))
    return {
        "size": int(rows),
        "free_cells": int(free_cells),
        "obstacle_cells": int(np.count_nonzero(obs)),
        "free_ratio": float(free_cells) / total,
        "obstacle_ratio": float(np.count_nonzero(obs)) / total,
        "reachable_free_cells": int(np.count_nonzero(reachable)),
        "reachable_free_ratio": float(np.count_nonzero(reachable)) / float(max(1, free_cells)),
        "free_components": int(components),
        "dead_end_cells": int(dead_ends),
        "dead_end_ratio": float(dead_ends) / float(max(1, free_cells)),
        "narrow_cells": int(narrow),
        "narrow_ratio": float(narrow) / float(max(1, free_cells)),
        "obstacle_perimeter": int(perimeter),
    }

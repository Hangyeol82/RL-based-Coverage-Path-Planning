from __future__ import annotations

import argparse
import json
import os
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

THIS_DIR = Path(__file__).resolve().parent
MPL_CACHE_DIR = THIS_DIR / ".mplconfig"
if "MPLCONFIGDIR" not in os.environ:
    MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

from learning.observation import MultiScaleCPPObservationConfig
from learning.observation.cpp.directional_traversability import compute_directional_traversability
from learning.observation.cpp.grid_features import (
    BLOCKED_STATE,
    FREE_STATE,
    UNKNOWN_STATE,
    block_reduce_max,
    block_reduce_mean,
    block_reduce_state,
    compute_frontier_map,
    extract_known_masks,
    global_reduce_max,
    global_reduce_mean,
    global_reduce_state,
)
from learning.reinforcement.cpp_env import ACTION_TO_DELTA, CPPDiscreteEnv, CPPDiscreteEnvConfig

GridPos = Tuple[int, int]



def parse_map_text(text: str) -> np.ndarray:
    rows: List[List[int]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        rows.append([int(x) for x in line.split()])
    if not rows:
        raise ValueError('empty map text')
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        raise ValueError('ragged map rows')
    arr = np.asarray(rows, dtype=np.int32)
    if not np.isin(arr, [0, 1]).all():
        raise ValueError('map must contain only 0/1')
    return arr


def neighbors4(pos: GridPos, shape: Tuple[int, int]) -> Iterable[GridPos]:
    r, c = pos
    h, w = shape
    for dr, dc in ACTION_TO_DELTA.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def bfs_distances(grid: np.ndarray, start: GridPos) -> Dict[GridPos, int]:
    q: deque[GridPos] = deque([start])
    dist: Dict[GridPos, int] = {start: 0}
    while q:
        cur = q.popleft()
        for nxt in neighbors4(cur, grid.shape):
            if grid[nxt] != 0 or nxt in dist:
                continue
            dist[nxt] = dist[cur] + 1
            q.append(nxt)
    return dist


def shortest_path(grid: np.ndarray, start: GridPos, goal: GridPos) -> List[GridPos]:
    q: deque[GridPos] = deque([start])
    parent: Dict[GridPos, Optional[GridPos]] = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in neighbors4(cur, grid.shape):
            if grid[nxt] != 0 or nxt in parent:
                continue
            parent[nxt] = cur
            q.append(nxt)
    if goal not in parent:
        raise ValueError(f'goal {goal} unreachable from {start}')
    path: List[GridPos] = []
    cur: Optional[GridPos] = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _local_complexity(grid: np.ndarray, pos: GridPos, radius: int = 4) -> float:
    r, c = pos
    h, w = grid.shape
    r0 = max(0, r - radius)
    r1 = min(h, r + radius + 1)
    c0 = max(0, c - radius)
    c1 = min(w, c + radius + 1)
    patch = grid[r0:r1, c0:c1]
    obstacle = int(np.count_nonzero(patch == 1))
    horiz = int(np.count_nonzero(patch[:, 1:] != patch[:, :-1])) if patch.shape[1] > 1 else 0
    vert = int(np.count_nonzero(patch[1:, :] != patch[:-1, :])) if patch.shape[0] > 1 else 0
    nfree = sum(1 for nb in neighbors4(pos, grid.shape) if grid[nb] == 0)
    return float(8 * nfree + obstacle + horiz + vert)


def choose_interesting_target(
    grid: np.ndarray,
    start: GridPos,
    *,
    pass_score_map: Optional[np.ndarray] = None,
) -> GridPos:
    dist = bfs_distances(grid, start)
    if not dist:
        raise ValueError('no reachable free cells')
    center = np.asarray([grid.shape[0] / 2.0, grid.shape[1] / 2.0], dtype=np.float64)
    best: Optional[GridPos] = None
    best_score = -1e18
    min_dist = max(8, min(grid.shape) // 4)
    for pos, d in dist.items():
        if d < min_dist:
            continue
        nfree = sum(1 for nb in neighbors4(pos, grid.shape) if grid[nb] == 0)
        if nfree < 2:
            continue
        complexity = _local_complexity(grid, pos, radius=4)
        center_penalty = float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - center))
        pass_score = float(pass_score_map[pos]) if pass_score_map is not None else 0.0
        score = 20.0 * pass_score + complexity + 0.15 * d - 0.1 * center_penalty
        if score > best_score:
            best_score = score
            best = pos
    if best is None:
        best = max(dist, key=lambda p: dist[p])
    return best


def action_from_step(a: GridPos, b: GridPos) -> int:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    for act, delta in ACTION_TO_DELTA.items():
        if delta == (dr, dc):
            return int(act)
    raise ValueError(f'non-adjacent move {a}->{b}')


def run_snapshot(
    grid: np.ndarray,
    *,
    sensor_range: int,
    dtm_output_mode: str,
    dtm_connectivity: int,
) -> Dict[str, object]:
    obs_cfg = MultiScaleCPPObservationConfig(
        local_blocks=(1, 2, 4, 8, 16),
        local_window_size=7,
        global_window_size=4,
        dtm_patch_size=7,
        dtm_connectivity=int(dtm_connectivity),
        dtm_require_fully_known_patch=False,
        dtm_min_known_ratio=0.6,
        dtm_patch_min_known_ratio=0.6,
        dtm_coarse_mode='bfs',
        dtm_output_mode=str(dtm_output_mode),
        include_cell_phase_channels=True,
    )
    env = CPPDiscreteEnv(
        grid,
        config=CPPDiscreteEnvConfig(
            sensor_range=int(sensor_range),
            max_steps=2000,
            include_dtm=True,
            use_action_mask=False,
            observation=obs_cfg,
        ),
    )
    env.reset()
    start = env.current_pos
    full_state = np.where(grid == 0, 1, 0).astype(np.int8)
    raw_full_dtm = compute_directional_traversability(
        full_state,
        known_ratio_map=np.ones_like(grid, dtype=np.float32),
        patch_size=obs_cfg.dtm_patch_size,
        connectivity=obs_cfg.dtm_connectivity,
        require_fully_known_patch=False,
        min_center_known_ratio=obs_cfg.dtm_min_known_ratio,
        min_patch_known_ratio=obs_cfg.dtm_patch_min_known_ratio,
        uncertain_fill=obs_cfg.dtm_uncertain_fill,
        unknown_fill=obs_cfg.dtm_unknown_fill,
        output_mode=env.maps_builder._native_dtm_mode(),
    )
    projected_full_dtm = env.maps_builder._project_dtm_output(raw_full_dtm)
    if str(dtm_output_mode) == 'axis2km':
        pass_score_map = projected_full_dtm[0] + projected_full_dtm[1]
    else:
        pass_score_map = np.maximum(projected_full_dtm, 0.0).sum(axis=0)
    target = choose_interesting_target(grid, start, pass_score_map=pass_score_map)
    path = shortest_path(grid, start, target)
    for prev, cur in zip(path[:-1], path[1:]):
        act = action_from_step(prev, cur)
        env.step(act)

    covered = env.explored & (env.known_map == 0)
    _, known_obstacle, _ = extract_known_masks(env.known_map)
    frontier = compute_frontier_map(covered, known_obstacle)

    online_levels = env.maps_builder.build_levels(env.known_map, robot_pos=env.current_pos, explored=env.explored)
    full_levels = env.maps_builder.build_levels(env.true_map.copy(), robot_pos=env.current_pos, explored=env.explored)

    return {
        'start': start,
        'target': target,
        'path': path,
        'current_pos': env.current_pos,
        'known_map': env.known_map.copy(),
        'true_map_known': env.true_map.copy(),
        'explored': env.explored.copy(),
        'frontier': frontier.astype(np.int32),
        'online_levels': online_levels,
        'full_levels': full_levels,
        'channel_names': env.maps_builder.channel_names,
        'config': obs_cfg,
        'builder': env.maps_builder,
    }


def plot_context(grid: np.ndarray, snapshot: Dict[str, object], out_path: Path, title: str) -> None:
    known_map = snapshot['known_map']
    explored = snapshot['explored']
    frontier = snapshot['frontier']
    path = snapshot['path']
    current_pos = snapshot['current_pos']
    target = snapshot['target']

    known_vis = np.full_like(known_map, 0, dtype=np.int32)
    known_vis[known_map == -1] = 0
    known_vis[known_map == 0] = 1
    known_vis[known_map == 1] = 2

    covered_vis = np.full_like(known_map, 0, dtype=np.int32)
    covered_vis[known_map == -1] = 0
    covered_vis[known_map == 1] = 1
    covered_vis[(known_map == 0) & (~explored)] = 2
    covered_vis[(known_map == 0) & explored] = 3

    cmap_gt = colors.ListedColormap(['white', 'black'])
    cmap_known = colors.ListedColormap(['#bdbdbd', 'white', 'black'])
    cmap_cov = colors.ListedColormap(['#bdbdbd', 'black', '#d9d9d9', '#2ca25f'])
    cmap_front = colors.ListedColormap(['#f7f7f7', '#ff8c00'])

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    axes[0].imshow(grid, cmap=cmap_gt, origin='upper', vmin=0, vmax=1)
    axes[0].set_title('GT map')

    axes[1].imshow(known_vis, cmap=cmap_known, origin='upper', vmin=0, vmax=2)
    axes[1].set_title('Known map\n(gray=unknown)')

    axes[2].imshow(covered_vis, cmap=cmap_cov, origin='upper', vmin=0, vmax=3)
    axes[2].set_title('Covered / known state')

    axes[3].imshow(frontier, cmap=cmap_front, origin='upper', vmin=0, vmax=1)
    axes[3].set_title('Frontier map')

    path_r = [p[0] for p in path]
    path_c = [p[1] for p in path]
    for ax in axes:
        ax.plot(path_c, path_r, color='#377eb8', linewidth=1.5, alpha=0.9)
        ax.scatter([path[0][1]], [path[0][0]], c='tab:blue', s=30, label='start')
        ax.scatter([target[1]], [target[0]], c='tab:orange', s=36, label='target')
        ax.scatter([current_pos[1]], [current_pos[0]], c='tab:red', s=30, label='robot')
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc='lower right', fontsize=8)
    fig.suptitle(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _format_val(v: float) -> str:
    if abs(v - round(v)) < 1e-6:
        return str(int(round(v)))
    return f'{v:.2f}'


def plot_dtm_levels(
    levels: Dict[int, np.ndarray],
    channel_names: Tuple[str, ...],
    config: MultiScaleCPPObservationConfig,
    robot_pos: GridPos,
    map_shape: Tuple[int, int],
    out_path: Path,
    title: str,
) -> None:
    dtm_names = [n for n in channel_names if n.startswith('dtm_')]
    dtm_indices = [i for i, n in enumerate(channel_names) if n.startswith('dtm_')]
    n_levels = len(levels)
    n_cols = len(dtm_indices)

    cmap = colors.ListedColormap(['#9e9e9e', '#2166ac', '#1a9850'])
    norm = colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, axes = plt.subplots(n_levels, n_cols, figsize=(3.2 * n_cols, 2.9 * n_levels), constrained_layout=True)
    if n_levels == 1:
        axes = np.asarray([axes])
    if n_cols == 1:
        axes = axes[:, None]

    local_blocks = list(config.local_blocks)
    for level_id in sorted(levels.keys()):
        level = levels[level_id]
        if level_id < len(local_blocks):
            block = local_blocks[level_id]
            row_label = f'L{level_id} local\nblock={block}\nrf={config.local_window_size * block}'
            marker_r = config.local_window_size // 2
            marker_c = config.local_window_size // 2
        else:
            row_label = f'L{level_id} global\n4x4 summary'
            marker_r = min(
                config.global_window_size - 1,
                max(0, int((robot_pos[0] * config.global_window_size) / max(1, map_shape[0]))),
            )
            marker_c = min(
                config.global_window_size - 1,
                max(0, int((robot_pos[1] * config.global_window_size) / max(1, map_shape[1]))),
            )
        for col, (ch_idx, ch_name) in enumerate(zip(dtm_indices, dtm_names)):
            ax = axes[level_id, col]
            arr = level[ch_idx]
            ax.imshow(arr, cmap=cmap, norm=norm, origin='upper')
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    txt = _format_val(float(arr[r, c]))
                    ax.text(c, r, txt, ha='center', va='center', fontsize=7, color='black')
            ax.add_patch(plt.Rectangle((marker_c - 0.5, marker_r - 0.5), 1.0, 1.0, fill=False, edgecolor='red', linewidth=2.0))
            if col == 0:
                ax.set_ylabel(row_label, rotation=0, labelpad=42, fontsize=9, va='center')
            ax.set_title(ch_name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_level_dump(levels: Dict[int, np.ndarray], channel_names: Tuple[str, ...], out_path: Path) -> None:
    payload: Dict[str, object] = {}
    for level_id, arr in levels.items():
        payload[str(level_id)] = {
            'shape': list(arr.shape),
            'channels': {name: arr[idx].tolist() for idx, name in enumerate(channel_names) if name.startswith('dtm_')},
        }
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def compute_fullscale_levels(
    known_map: np.ndarray,
    explored: np.ndarray,
    builder,
) -> Dict[int, Dict[str, np.ndarray]]:
    known_free, known_obstacle, unknown = extract_known_masks(
        known_map,
        unknown_value=builder.config.unknown_value,
        obstacle_value=builder.config.obstacle_value,
    )
    covered = explored.astype(bool) & known_free
    frontier = compute_frontier_map(covered, known_obstacle)

    covered_f = covered.astype(np.float32)
    obstacle_f = known_obstacle.astype(np.float32)
    frontier_f = frontier.astype(np.float32)
    known_f = (~unknown).astype(np.float32)

    levels: Dict[int, Dict[str, np.ndarray]] = {}
    native_mode = builder._native_dtm_mode()
    state_fine = np.full(known_map.shape, BLOCKED_STATE, dtype=np.int8)
    state_fine[unknown] = UNKNOWN_STATE
    state_fine[known_free] = FREE_STATE
    native0 = builder._compute_level0_native_dtm(state_fine)

    for level_id, block in enumerate(builder.config.local_blocks):
        state = block_reduce_state(
            known_free,
            known_obstacle,
            unknown,
            block,
            min_known_ratio=builder.config.dtm_min_known_ratio,
        )
        known_ratio = block_reduce_mean(known_f, block)
        if level_id == 0:
            native = native0
        elif native0 is not None:
            native = builder._compose_dtm_block_from_children(
                dtm_child=native0,
                state_child=state_fine,
                known_ratio_parent=known_ratio,
                block_r=block,
            )
        else:
            native = compute_directional_traversability(
                state,
                known_ratio_map=known_ratio,
                patch_size=builder.config.dtm_patch_size,
                connectivity=builder.config.dtm_connectivity,
                require_fully_known_patch=builder.config.dtm_require_fully_known_patch,
                min_center_known_ratio=builder.config.dtm_min_known_ratio,
                min_patch_known_ratio=builder.config.dtm_patch_min_known_ratio,
                uncertain_fill=builder.config.dtm_uncertain_fill,
                unknown_fill=builder.config.dtm_unknown_fill,
                output_mode=native_mode,
            )
        levels[level_id] = {
            'coverage': block_reduce_mean(covered_f, block),
            'obstacle': block_reduce_mean(obstacle_f, block),
            'frontier': block_reduce_max(frontier_f, block),
            'known_ratio': known_ratio,
            'state': state,
            'dtm': builder._project_dtm_output(native),
        }
    global_id = len(builder.config.local_blocks)
    g = builder.config.global_window_size
    g_state = global_reduce_state(
        known_free,
        known_obstacle,
        unknown,
        g,
        g,
        min_known_ratio=builder.config.dtm_min_known_ratio,
    )
    g_known_ratio = global_reduce_mean(known_f, g, g)
    if native0 is not None:
        row_edges = builder._global_edges(known_map.shape[0], g)
        col_edges = builder._global_edges(known_map.shape[1], g)
        g_native = builder._compose_dtm_partition_from_children(
            dtm_child=native0,
            state_child=state_fine,
            known_ratio_parent=g_known_ratio,
            row_edges=row_edges,
            col_edges=col_edges,
        )
    else:
        g_native = compute_directional_traversability(
            g_state,
            known_ratio_map=g_known_ratio,
            patch_size=builder.config.dtm_patch_size,
            connectivity=builder.config.dtm_connectivity,
            require_fully_known_patch=builder.config.dtm_require_fully_known_patch,
            min_center_known_ratio=builder.config.dtm_min_known_ratio,
            min_patch_known_ratio=builder.config.dtm_patch_min_known_ratio,
            uncertain_fill=builder.config.dtm_uncertain_fill,
            unknown_fill=builder.config.dtm_unknown_fill,
            output_mode=native_mode,
        )
    levels[global_id] = {
        'coverage': global_reduce_mean(covered_f, g, g),
        'obstacle': global_reduce_mean(obstacle_f, g, g),
        'frontier': global_reduce_max(frontier_f, g, g),
        'known_ratio': g_known_ratio,
        'state': g_state,
        'dtm': builder._project_dtm_output(g_native),
    }
    return levels


def plot_fullscale_dtm_levels(
    fullscale_levels: Dict[int, Dict[str, np.ndarray]],
    channel_names: Tuple[str, ...],
    config: MultiScaleCPPObservationConfig,
    out_path: Path,
    title: str,
) -> None:
    dtm_names = [n for n in channel_names if n.startswith('dtm_')]
    n_levels = len(fullscale_levels)
    n_cols = len(dtm_names)

    cmap = colors.ListedColormap(['#9e9e9e', '#2166ac', '#1a9850'])
    norm = colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, axes = plt.subplots(n_levels, n_cols, figsize=(3.2 * n_cols, 2.9 * n_levels), constrained_layout=True)
    if n_levels == 1:
        axes = np.asarray([axes])
    if n_cols == 1:
        axes = axes[:, None]

    local_blocks = list(config.local_blocks)
    for row, level_id in enumerate(sorted(fullscale_levels.keys())):
        data = fullscale_levels[level_id]
        dtm = data['dtm']
        if level_id < len(local_blocks):
            block = local_blocks[level_id]
            row_label = f'L{level_id} full\\nblock={block}\\nshape={dtm.shape[1]}x{dtm.shape[2]}'
        else:
            row_label = f'L{level_id} global\\nshape={dtm.shape[1]}x{dtm.shape[2]}'

        for col, ch_name in enumerate(dtm_names):
            ax = axes[row, col]
            arr = dtm[col]
            ax.imshow(arr, cmap=cmap, norm=norm, origin='upper')
            for r in range(arr.shape[0]):
                for c in range(arr.shape[1]):
                    ax.text(c, r, _format_val(float(arr[r, c])), ha='center', va='center', fontsize=7, color='black')
            if col == 0:
                ax.set_ylabel(row_label, rotation=0, labelpad=46, fontsize=9, va='center')
            ax.set_title(ch_name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(title)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def save_fullscale_dump(
    fullscale_levels: Dict[int, Dict[str, np.ndarray]],
    channel_names: Tuple[str, ...],
    out_path: Path,
) -> None:
    dtm_names = [name for name in channel_names if name.startswith('dtm_')]
    payload: Dict[str, object] = {}
    for level_id, data in fullscale_levels.items():
        dtm = data['dtm']
        payload[str(level_id)] = {
            'dtm_shape': list(dtm.shape),
            'channels': {
                name: dtm[idx].tolist()
                for idx, name in enumerate(dtm_names)
            },
            'known_ratio': data['known_ratio'].tolist(),
            'state': data['state'].tolist(),
        }
    out_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')


def main() -> None:
    p = argparse.ArgumentParser(description='Visualize multiscale DTM snapshot on a complex map.')
    p.add_argument('--map-file', type=str, default='map/indoor_seed101.txt')
    p.add_argument('--sensor-range', type=int, default=2)
    p.add_argument('--dtm-output-mode', type=str, default='axis2km', choices=['axis2km', 'axis2', 'six', 'extent6', 'port12'])
    p.add_argument('--dtm-connectivity', type=int, default=4, choices=[4, 8])
    p.add_argument('--out-dir', type=str, default='log_analysis/logs/dtm_snapshot_indoor_20260311')
    args = p.parse_args()

    map_path = Path(args.map_file)
    grid = parse_map_text(map_path.read_text(encoding='utf-8'))
    snapshot = run_snapshot(
        grid,
        sensor_range=int(args.sensor_range),
        dtm_output_mode=str(args.dtm_output_mode),
        dtm_connectivity=int(args.dtm_connectivity),
    )
    fullscale_online = compute_fullscale_levels(snapshot['known_map'], snapshot['explored'], snapshot['builder'])
    fullscale_full = compute_fullscale_levels(snapshot['true_map_known'], snapshot['explored'], snapshot['builder'])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    context_png = out_dir / 'context_online.png'
    dtm_online_png = out_dir / 'dtm_levels_online.png'
    dtm_full_png = out_dir / 'dtm_levels_full_known.png'
    dtm_fullmap_online_png = out_dir / 'dtm_fullmap_levels_online.png'
    dtm_fullmap_full_png = out_dir / 'dtm_fullmap_levels_full_known.png'
    online_json = out_dir / 'dtm_levels_online.json'
    full_json = out_dir / 'dtm_levels_full_known.json'
    fullmap_online_json = out_dir / 'dtm_fullmap_levels_online.json'
    fullmap_full_json = out_dir / 'dtm_fullmap_levels_full_known.json'
    meta_json = out_dir / 'meta.json'

    title_prefix = f"map={map_path.name} sensor={args.sensor_range} mode={args.dtm_output_mode} connectivity={args.dtm_connectivity}"
    plot_context(grid, snapshot, context_png, title=f'{title_prefix} | online snapshot')
    plot_dtm_levels(
        snapshot['online_levels'],
        snapshot['channel_names'],
        snapshot['config'],
        snapshot['current_pos'],
        grid.shape,
        dtm_online_png,
        title=f'{title_prefix} | DTM online levels',
    )
    plot_dtm_levels(
        snapshot['full_levels'],
        snapshot['channel_names'],
        snapshot['config'],
        snapshot['current_pos'],
        grid.shape,
        dtm_full_png,
        title=f'{title_prefix} | DTM full-known levels',
    )
    plot_fullscale_dtm_levels(
        fullscale_online,
        snapshot['channel_names'],
        snapshot['config'],
        dtm_fullmap_online_png,
        title=f'{title_prefix} | FULL MAP DTM levels (online known map)',
    )
    plot_fullscale_dtm_levels(
        fullscale_full,
        snapshot['channel_names'],
        snapshot['config'],
        dtm_fullmap_full_png,
        title=f'{title_prefix} | FULL MAP DTM levels (full-known map)',
    )
    save_level_dump(snapshot['online_levels'], snapshot['channel_names'], online_json)
    save_level_dump(snapshot['full_levels'], snapshot['channel_names'], full_json)
    save_fullscale_dump(fullscale_online, snapshot['channel_names'], fullmap_online_json)
    save_fullscale_dump(fullscale_full, snapshot['channel_names'], fullmap_full_json)

    meta = {
        'map_file': str(map_path.resolve()),
        'start': list(snapshot['start']),
        'target': list(snapshot['target']),
        'current_pos': list(snapshot['current_pos']),
        'path_length': len(snapshot['path']) - 1,
        'known_cells': int(np.count_nonzero(snapshot['known_map'] != -1)),
        'covered_cells': int(np.count_nonzero(snapshot['explored'])),
        'channel_names': list(snapshot['channel_names']),
        'dtm_output_mode': args.dtm_output_mode,
        'dtm_connectivity': args.dtm_connectivity,
    }
    meta_json.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print(f'[DONE] {context_png.resolve()}')
    print(f'[DONE] {dtm_online_png.resolve()}')
    print(f'[DONE] {dtm_full_png.resolve()}')
    print(f'[DONE] {dtm_fullmap_online_png.resolve()}')
    print(f'[DONE] {dtm_fullmap_full_png.resolve()}')
    print(f'[DONE] {online_json.resolve()}')
    print(f'[DONE] {full_json.resolve()}')
    print(f'[DONE] {fullmap_online_json.resolve()}')
    print(f'[DONE] {fullmap_full_json.resolve()}')
    print(f'[DONE] {meta_json.resolve()}')
    print(json.dumps(meta, indent=2))


if __name__ == '__main__':
    main()

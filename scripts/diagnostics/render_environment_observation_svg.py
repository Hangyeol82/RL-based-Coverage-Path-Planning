from __future__ import annotations

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle

from learning.observation import MultiScaleCPPObservationConfig
from learning.reinforcement.cpp_env import ACTION_TO_DELTA, CPPDiscreteEnv, CPPDiscreteEnvConfig
from map_generators.structured import build_structured_map


GridPos = Tuple[int, int]


def _neighbors(pos: GridPos, grid: np.ndarray) -> Iterable[Tuple[int, GridPos]]:
    r, c = pos
    rows, cols = grid.shape
    for action, (dr, dc) in ACTION_TO_DELTA.items():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == 0:
            yield int(action), (int(nr), int(nc))


def _bfs_to_nearest_unvisited(grid: np.ndarray, start: GridPos, visited: np.ndarray) -> List[int]:
    q: deque[GridPos] = deque([start])
    parent: Dict[GridPos, Tuple[Optional[GridPos], Optional[int]]] = {start: (None, None)}
    target: Optional[GridPos] = None

    while q:
        pos = q.popleft()
        if not visited[pos] and grid[pos] == 0:
            target = pos
            break
        for action, nxt in _neighbors(pos, grid):
            if nxt in parent:
                continue
            parent[nxt] = (pos, action)
            q.append(nxt)

    if target is None:
        return []

    actions: List[int] = []
    cur = target
    while cur != start:
        prev, action = parent[cur]
        if prev is None or action is None:
            break
        actions.append(int(action))
        cur = prev
    actions.reverse()
    return actions


def _oracle_coverage_rollout(env: CPPDiscreteEnv, steps: int) -> None:
    """Create a clean illustrative rollout. This is not a learned policy."""
    visited = np.zeros_like(env.true_map, dtype=bool)
    visited[env.current_pos] = True
    queue: List[int] = []
    for _ in range(int(steps)):
        if env.done:
            break
        if not queue:
            queue = _bfs_to_nearest_unvisited(env.true_map, env.current_pos, visited)
        if not queue:
            break
        action = int(queue.pop(0))
        _, _, done, _ = env.step(action)
        visited[env.current_pos] = True
        if done:
            break


def _nearest_free(grid: np.ndarray, target: GridPos) -> GridPos:
    rows, cols = grid.shape
    tr, tc = target
    q: deque[GridPos] = deque([(int(np.clip(tr, 0, rows - 1)), int(np.clip(tc, 0, cols - 1)))])
    seen = {q[0]}
    while q:
        pos = q.popleft()
        if grid[pos] == 0:
            return pos
        for _, nxt in _neighbors(pos, np.zeros_like(grid)):
            if nxt in seen:
                continue
            seen.add(nxt)
            q.append(nxt)
    raise ValueError("No free cell found")


def _state_image(env: CPPDiscreteEnv) -> np.ndarray:
    # 0 unknown, 1 known free, 2 covered, 3 known obstacle.
    img = np.zeros(env.known_map.shape, dtype=np.int8)
    img[env.known_map == 0] = 1
    img[env.explored & env.free_mask] = 2
    img[env.known_map == 1] = 3
    return img


def _draw_grid(ax, data: np.ndarray, *, grid_lw: float = 0.15) -> None:
    rows, cols = data.shape
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#5a5a5a", linewidth=grid_lw, alpha=0.45)
    ax.tick_params(which="both", left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#222222")


def render_environment_state(env: CPPDiscreteEnv, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    known_cmap = ListedColormap(["#bcc6d4", "#ffffff", "#73c58d", "#222222"])
    known_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], known_cmap.N)

    fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.6), dpi=180)
    ax.imshow(_state_image(env), cmap=known_cmap, norm=known_norm, interpolation="nearest")
    _draw_grid(ax, env.known_map)
    path = np.asarray(env.path, dtype=float)
    if path.shape[0] > 1:
        ax.plot(path[:, 1], path[:, 0], color="#1f77b4", linewidth=1.5, alpha=0.9)
    rr, cc = env.current_pos
    ax.scatter([cc], [rr], s=72, color="#d90429", edgecolors="white", linewidths=1.0, zorder=5)

    rng = int(env.config.sensor_range)
    r0, r1 = max(0, rr - rng), min(env.rows - 1, rr + rng)
    c0, c1 = max(0, cc - rng), min(env.cols - 1, cc + rng)
    ax.add_patch(
        Rectangle(
            (c0 - 0.5, r0 - 0.5),
            c1 - c0 + 1,
            r1 - r0 + 1,
            fill=False,
            edgecolor="#ff9f1c",
            linewidth=2.0,
            linestyle="--",
            zorder=4,
        )
    )

    fig.tight_layout(pad=0.05)
    save_kwargs = {"bbox_inches": "tight", "pad_inches": 0.02}
    fig.savefig(output_dir / "environment_known_map_path.svg", format="svg", **save_kwargs)
    fig.savefig(output_dir / "environment_known_map_path.png", format="png", **save_kwargs)
    plt.close(fig)


def _plot_channel(ax, arr: np.ndarray, title: str, *, ternary: bool = False) -> None:
    if ternary:
        cmap = ListedColormap(["#bcc6d4", "#f7f7f7", "#1f77b4"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
        ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
    else:
        ax.imshow(arr, cmap="Greens", vmin=0.0, vmax=1.0, interpolation="nearest")
    _draw_grid(ax, arr, grid_lw=0.3)
    ax.set_title(title, fontsize=8, pad=3)


def render_multiscale_observation(env: CPPDiscreteEnv, output_dir: Path) -> None:
    # A figure-friendly subset of the paper observation: base map channels + two-way DTM.
    cfg = MultiScaleCPPObservationConfig(
        local_blocks=(1, 2, 4, 8, 16),
        local_window_size=7,
        global_window_size=4,
        dtm_output_mode="axis2",
        dtm_coarse_mode="bfs",
        dtm_connectivity=4,
        include_cell_phase_channels=False,
    )
    obs = env.maps_builder.__class__(cfg, include_dtm=True).build_levels(
        occupancy=env.known_map,
        robot_pos=env.current_pos,
        explored=env.explored,
    )

    channel_names = ["Coverage", "Obstacle", "Frontier", "DTM-LR", "DTM-UD"]
    row_labels = [
        "Level 0\nblock=1",
        "Level 1\nblock=2",
        "Level 2\nblock=4",
        "Level 3\nblock=8",
        "Level 4\nblock=16",
        "Global\n4x4",
    ]
    fig, axes = plt.subplots(len(row_labels), len(channel_names), figsize=(11.5, 12.8), dpi=180)
    for r, level_id in enumerate(sorted(obs.keys())):
        tensor = np.asarray(obs[level_id], dtype=float)
        for c, name in enumerate(channel_names):
            ax = axes[r, c]
            _plot_channel(ax, tensor[c], name if r == 0 else "", ternary=(c >= 3))
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=9, rotation=0, labelpad=32, va="center")

    fig.suptitle(
        "Multi-scale Observation Example: coverage / obstacle / frontier + two-way DTM",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.015,
        "DTM values: blue=traversable(1), white=blocked(0), gray=uncertain/unknown(-1). "
        "Cell-phase channels are omitted here for readability.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.04, 0.04, 1.0, 0.965))
    fig.savefig(output_dir / "observation_multiscale_axis2.svg", format="svg")
    fig.savefig(output_dir / "observation_multiscale_axis2.png", format="png")
    plt.close(fig)


def build_demo_env(size: int, seed: int, level: int, family: str, steps: int) -> CPPDiscreteEnv:
    grid = build_structured_map(
        family=family,
        size=size,
        seed=seed,
        level=level,
        ensure_start_clear=True,
    )
    cfg = CPPDiscreteEnvConfig(
        sensor_range=3,
        max_steps=max(steps + 1, 1_000),
        collision_ends_episode=False,
        stop_on_full_coverage=False,
        include_dtm=True,
        use_action_mask=True,
        observation=MultiScaleCPPObservationConfig(
            local_blocks=(1, 2, 4, 8, 16),
            local_window_size=7,
            global_window_size=4,
            dtm_output_mode="axis2",
            dtm_coarse_mode="bfs",
            dtm_connectivity=4,
            include_cell_phase_channels=True,
        ),
    )
    start = _nearest_free(grid, (size // 2, size // 2))
    env = CPPDiscreteEnv(grid, start_pos=start, config=cfg)
    _oracle_coverage_rollout(env, steps)
    return env


def main() -> None:
    p = argparse.ArgumentParser(description="Render paper-friendly environment/observation SVG examples.")
    p.add_argument("--out-dir", type=str, default="docs/figures/environment_observation")
    p.add_argument("--size", type=int, default=48)
    p.add_argument("--seed", type=int, default=113)
    p.add_argument("--level", type=int, default=2)
    p.add_argument("--family", type=str, default="room_corridor")
    p.add_argument("--steps", type=int, default=180)
    args = p.parse_args()

    out = Path(args.out_dir)
    env = build_demo_env(
        size=int(args.size),
        seed=int(args.seed),
        level=int(args.level),
        family=str(args.family),
        steps=int(args.steps),
    )
    render_environment_state(env, out)
    render_multiscale_observation(env, out)
    print(f"saved: {out.resolve()}")
    print(f"steps={env.steps}, coverage={env._coverage_ratio():.4f}, robot={env.current_pos}")


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from MapGenerator import MapGenerator
from learning.observation import MultiScaleCPPObservationConfig
from learning.reinforcement.cpp_env import CPPDiscreteEnv, CPPDiscreteEnvConfig
from learning.reinforcement.reward import CPPRewardConfig
from run_cstar_custom_map import CUSTOM_MAP_TEXT, parse_custom_map


GridPos = Tuple[int, int]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile CPP observation/env step timings.")
    p.add_argument("--map-source", type=str, default="random", choices=["random", "custom", "file"])
    p.add_argument("--map-file", type=str, default="")
    p.add_argument("--map-size", type=int, default=64)
    p.add_argument("--map-stage", type=int, default=3)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--sensor-range", type=int, default=2)
    p.add_argument(
        "--local-blocks",
        type=str,
        default="",
        help="Optional comma list overriding MAPS local block sizes, e.g. 1,2,4,8",
    )
    p.add_argument("--max-episode-steps", type=int, default=6000)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--profile-interval", type=int, default=100)
    p.add_argument("--include-dtm", action="store_true")
    p.add_argument(
        "--dtm-coarse-mode",
        type=str,
        default="bfs",
        choices=["bfs", "aggregate", "aggregate_transfer"],
    )
    p.add_argument(
        "--dtm-output-mode",
        type=str,
        default="axis2km",
        choices=["six", "extent6", "axis2", "axis2km", "four", "port12"],
    )
    p.add_argument(
        "--obs-unknown-policy",
        type=str,
        default="keep",
        choices=["keep", "as_free", "as_obstacle"],
    )
    p.add_argument("--dtm-connectivity", type=int, default=4, choices=[4, 8])
    return p.parse_args()


def _parse_local_blocks(raw: str):
    s = str(raw).strip()
    if not s:
        return None
    vals = []
    for tok in s.split(","):
        t = tok.strip()
        if not t:
            continue
        vals.append(int(t))
    if not vals:
        return None
    if any(v <= 0 for v in vals):
        raise ValueError("--local-blocks values must be positive")
    if sorted(vals) != vals:
        raise ValueError("--local-blocks must be in increasing order")
    if len(set(vals)) != len(vals):
        raise ValueError("--local-blocks must not contain duplicates")
    return tuple(vals)


def _prepare_square_map(base_map: np.ndarray, size: int) -> np.ndarray:
    out = np.zeros((size, size), dtype=np.int32)
    h, w = base_map.shape
    out[: min(h, size), : min(w, size)] = base_map[: min(h, size), : min(w, size)]
    return out


def _parse_map_text(text: str) -> np.ndarray:
    rows = [list(map(int, line.split())) for line in text.strip().splitlines() if line.strip()]
    grid = np.asarray(rows, dtype=np.int32)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("Invalid map text")
    return grid


def _build_map(args: argparse.Namespace) -> np.ndarray:
    if args.map_source == "custom":
        base = parse_custom_map(CUSTOM_MAP_TEXT).astype(np.int32)
        return _prepare_square_map(base, args.map_size)
    if args.map_source == "file":
        if not args.map_file:
            raise ValueError("--map-file is required with --map-source file")
        base = _parse_map_text(Path(args.map_file).read_text(encoding="utf-8")).astype(np.int32)
        return _prepare_square_map(base, args.map_size)
    gen = MapGenerator(height=args.map_size, width=args.map_size, seed=args.seed)
    return np.asarray(gen.generate_map(stage=int(args.map_stage)), dtype=np.int32)


def _pick_start(grid: np.ndarray) -> GridPos:
    free = np.argwhere(grid == 0)
    if free.size == 0:
        raise RuntimeError("No free cell exists in map")
    r, c = free[0]
    return int(r), int(c)


def _sample_action(env: CPPDiscreteEnv, rng: np.random.Generator) -> int:
    mask = env.get_action_mask()
    valid = np.flatnonzero(mask)
    if valid.size <= 0:
        return 0
    return int(valid[int(rng.integers(0, valid.size))])


def main():
    args = _parse_args()
    rng = np.random.default_rng(int(args.seed))
    grid = _build_map(args)
    local_blocks = _parse_local_blocks(args.local_blocks)
    if int(grid[0, 0]) == 1:
        grid[0, 0] = 0
    start = _pick_start(grid)
    cfg = CPPDiscreteEnvConfig(
        sensor_range=int(args.sensor_range),
        max_steps=int(args.max_episode_steps),
        include_dtm=bool(args.include_dtm),
        use_action_mask=True,
        profile_observation=True,
        profile_interval_steps=int(args.profile_interval),
        profile_name=f"{'dtm' if args.include_dtm else 'base'}_{args.map_size}",
        observation=MultiScaleCPPObservationConfig(
            local_blocks=local_blocks or MultiScaleCPPObservationConfig().local_blocks,
            unknown_policy=str(args.obs_unknown_policy),
            dtm_coarse_mode=str(args.dtm_coarse_mode),
            dtm_output_mode=str(args.dtm_output_mode),
            dtm_connectivity=int(args.dtm_connectivity),
        ),
        reward=CPPRewardConfig(),
    )
    env = CPPDiscreteEnv(grid_map=grid, start_pos=start, config=cfg)
    env.reset()

    total_steps = int(args.warmup_steps) + int(args.steps)
    done_count = 0
    for i in range(total_steps):
        action = _sample_action(env, rng)
        _, _, done, _ = env.step(action)
        if done:
            done_count += 1
            env.reset()
        if i + 1 == int(args.warmup_steps):
            # Reset profiler after warmup so the printed windows are stable.
            env._obs_profile_totals = {}
            env._obs_profile_calls = 0
            env.maps_builder.reset_profile()
            if env._boundary_maps_builder is not None:
                env._boundary_maps_builder.reset_profile()

    print(
        f"[OBS-PROFILE] completed steps={int(args.steps)} warmup={int(args.warmup_steps)} "
        f"episodes_reset={done_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()

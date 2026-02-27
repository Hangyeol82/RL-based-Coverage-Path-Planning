import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.observation import MultiScaleCPPObservationConfig  # noqa: E402
from learning.reinforcement.cpp_env import CPPDiscreteEnv, CPPDiscreteEnvConfig  # noqa: E402


GridPos = Tuple[int, int]


def _make_map(rng: np.random.RandomState, size: int, obstacle_prob: float) -> np.ndarray:
    grid = (rng.rand(size, size) < obstacle_prob).astype(np.int32)
    grid[0, 0] = 0
    return grid


def _dtm_channels(mode: str) -> int:
    m = str(mode).strip().lower()
    if m in {"six", "extent6"}:
        return 6
    if m == "four":
        return 4
    if m == "port12":
        return 12
    raise ValueError(f"Unsupported mode: {mode}")


def _exit_scores(mode: str, dtm_values: np.ndarray) -> Tuple[float, float, float, float]:
    m = str(mode).strip().lower()
    vals = np.maximum(np.asarray(dtm_values, dtype=np.float32), 0.0)
    if m in {"six", "extent6"}:
        lr, ud, nw_se, se_nw, ne_sw, sw_ne = [float(v) for v in vals[:6]]
        up = max(ud, se_nw, sw_ne)
        right = max(lr, nw_se, sw_ne)
        down = max(ud, nw_se, ne_sw)
        left = max(lr, se_nw, ne_sw)
        return up, right, down, left
    if m == "four":
        lr, ud, d1, d2 = [float(v) for v in vals[:4]]
        up = max(ud, d1, d2)
        right = max(lr, d1, d2)
        down = max(ud, d1, d2)
        left = max(lr, d1, d2)
        return up, right, down, left
    if m == "port12":
        (
            u_r,
            u_d,
            u_l,
            r_u,
            r_d,
            r_l,
            d_u,
            d_r,
            d_l,
            l_u,
            l_r,
            l_d,
        ) = [float(v) for v in vals[:12]]
        up = max(r_u, d_u, l_u)
        right = max(u_r, d_r, l_r)
        down = max(u_d, r_d, l_d)
        left = max(u_l, r_l, d_l)
        return up, right, down, left
    raise ValueError(f"Unsupported mode: {mode}")


def _level_cell_index(
    *,
    level_id: int,
    local_level_count: int,
    local_window_size: int,
    global_window_size: int,
    robot_pos: GridPos,
    rows: int,
    cols: int,
) -> GridPos:
    if level_id < local_level_count:
        c = int(local_window_size) // 2
        return int(c), int(c)
    rr, cc = robot_pos
    gr = min(global_window_size - 1, max(0, int((float(rr) * float(global_window_size)) / float(max(1, rows)))))
    gc = min(global_window_size - 1, max(0, int((float(cc) * float(global_window_size)) / float(max(1, cols)))))
    return int(gr), int(gc)


def _expected_boundary_features(
    *,
    levels: Dict[int, np.ndarray],
    mode: str,
    threshold: float,
    include_valid: bool,
    robot_pos: GridPos,
    rows: int,
    cols: int,
    local_level_count: int,
    local_window_size: int,
    global_window_size: int,
) -> np.ndarray:
    thr = float(np.clip(threshold, 0.0, 1.0))
    dtm_ch = _dtm_channels(mode)
    out: List[float] = []
    for lv in sorted(levels.keys()):
        arr = np.asarray(levels[lv], dtype=np.float32)
        ri, ci = _level_cell_index(
            level_id=int(lv),
            local_level_count=int(local_level_count),
            local_window_size=int(local_window_size),
            global_window_size=int(global_window_size),
            robot_pos=robot_pos,
            rows=int(rows),
            cols=int(cols),
        )
        dtm = arr[3 : 3 + dtm_ch, ri, ci]
        valid = float(np.all(dtm >= 0.0))
        up_s, right_s, down_s, left_s = _exit_scores(mode, dtm)
        if valid > 0.5:
            out.extend(
                [
                    1.0 if up_s > thr else 0.0,
                    1.0 if right_s > thr else 0.0,
                    1.0 if down_s > thr else 0.0,
                    1.0 if left_s > thr else 0.0,
                ]
            )
        else:
            out.extend([0.0, 0.0, 0.0, 0.0])
        if include_valid:
            out.append(valid)
    return np.asarray(out, dtype=np.float32)


def verify(
    *,
    trials: int,
    steps: int,
    map_size: int,
    obstacle_prob: float,
    sensor_range: int,
    dtm_output_mode: str,
    threshold: float,
) -> Tuple[int, int, int]:
    rng = np.random.RandomState(1234)
    mismatch_decode = 0
    mismatch_parity = 0
    checks = 0

    for _ in range(int(trials)):
        grid = _make_map(rng, int(map_size), float(obstacle_prob))

        obs_cfg = MultiScaleCPPObservationConfig(
            dtm_output_mode=str(dtm_output_mode),
            dtm_coarse_mode="bfs",
        )

        # A: include_dtm=True path (CNN has DTM), plus boundary features in robot_state.
        cfg_a = CPPDiscreteEnvConfig(
            sensor_range=int(sensor_range),
            include_dtm=True,
            use_boundary_exit_features=True,
            boundary_exit_threshold=float(threshold),
            observation=obs_cfg,
        )
        env_a = CPPDiscreteEnv(grid, config=cfg_a)

        # B: include_dtm=False path (CNN without DTM), boundary features still enabled.
        cfg_b = CPPDiscreteEnvConfig(
            sensor_range=int(sensor_range),
            include_dtm=False,
            use_boundary_exit_features=True,
            boundary_exit_threshold=float(threshold),
            observation=obs_cfg,
        )
        env_b = CPPDiscreteEnv(grid, config=cfg_b)

        obs_a = env_a.reset()
        obs_b = env_b.reset()
        base_dim = 9

        for _step in range(int(steps)):
            extra_a = np.asarray(obs_a["robot_state"], dtype=np.float32)[base_dim:]
            extra_b = np.asarray(obs_b["robot_state"], dtype=np.float32)[base_dim:]

            expected = _expected_boundary_features(
                levels=obs_a["levels"],
                mode=str(dtm_output_mode),
                threshold=float(threshold),
                include_valid=bool(cfg_a.boundary_exit_include_valid),
                robot_pos=env_a.current_pos,
                rows=int(env_a.rows),
                cols=int(env_a.cols),
                local_level_count=len(obs_cfg.local_blocks),
                local_window_size=int(obs_cfg.local_window_size),
                global_window_size=int(obs_cfg.global_window_size),
            )

            checks += 1
            if not np.array_equal(extra_a, expected):
                mismatch_decode += 1
            if not np.array_equal(extra_a, extra_b):
                mismatch_parity += 1

            action = int(rng.randint(0, 4))
            obs_a, _, done_a, _ = env_a.step(action)
            obs_b, _, done_b, _ = env_b.step(action)
            if bool(done_a) != bool(done_b):
                mismatch_parity += 1
            if done_a or done_b:
                obs_a = env_a.reset()
                obs_b = env_b.reset()

    return checks, mismatch_decode, mismatch_parity


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify boundary-exit robot_state features against level DTM channels.")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--map-size", type=int, default=32)
    ap.add_argument("--obstacle-prob", type=float, default=0.22)
    ap.add_argument("--sensor-range", type=int, default=2)
    ap.add_argument("--dtm-output-mode", type=str, default="six", choices=["six", "extent6", "four", "port12"])
    ap.add_argument("--boundary-exit-threshold", type=float, default=0.0)
    args = ap.parse_args()

    checks, mismatch_decode, mismatch_parity = verify(
        trials=int(args.trials),
        steps=int(args.steps),
        map_size=int(args.map_size),
        obstacle_prob=float(args.obstacle_prob),
        sensor_range=int(args.sensor_range),
        dtm_output_mode=str(args.dtm_output_mode),
        threshold=float(args.boundary_exit_threshold),
    )
    print(
        f"[VERIFY-BOUNDARY] checks={checks} "
        f"mismatch_decode={mismatch_decode} mismatch_parity={mismatch_parity}"
    )
    if mismatch_decode > 0 or mismatch_parity > 0:
        raise SystemExit(1)
    print("[VERIFY-BOUNDARY] PASS")


if __name__ == "__main__":
    main()

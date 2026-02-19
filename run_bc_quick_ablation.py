import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from learning.common import FusedMAPSStateEncoderConfig, MultiLevelMAPSEncoderConfig
from learning.imitation import BCPolicy, BCPolicyConfig
from learning.imitation.epsilon_rollout import (
    BCTensorDataset,
    build_bc_tensors_from_rollout,
    collect_epsilon_rollout,
)
from learning.observation import MultiScaleCPPObservationBuilder, MultiScaleCPPObservationConfig
from MapGenerator import MapGenerator
from run_cstar_custom_map import CUSTOM_MAP_TEXT, parse_custom_map


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick BC ablation: baseline(coverage/obstacle/frontier) vs +DTM.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--sensor-range", type=int, default=3, help="range=3 -> 7x7 local sensing")
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--map-size", type=int, default=32, help="Pad/crop custom map to square size.")
    parser.add_argument(
        "--map-source",
        type=str,
        default="random",
        choices=["random", "custom", "file", "indoor"],
        help="random: MapGenerator, custom: built-in fixed map, file: user map text file, indoor: room-grid map",
    )
    parser.add_argument("--map-stage", type=int, default=3, choices=[1, 2, 3, 4])
    parser.add_argument("--num-obstacles", type=int, default=None)
    parser.add_argument("--map-file", type=str, default=None, help="Path to whitespace-separated 0/1 map file.")
    parser.add_argument("--indoor-room-inner-size", type=int, default=8)
    parser.add_argument("--indoor-rooms-rows", type=int, default=0, help="0 means auto from map-size/room size.")
    parser.add_argument("--indoor-rooms-cols", type=int, default=0, help="0 means auto from map-size/room size.")
    parser.add_argument("--indoor-extra-connection-prob", type=float, default=0.35)
    parser.add_argument("--indoor-two-door-prob", type=float, default=0.2, help="2-door ratio (target 0.2 => 2:8)")
    parser.add_argument("--indoor-door-width", type=int, default=2)
    parser.add_argument("--indoor-wall-thickness", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-log-json", type=str, default=None)
    return parser.parse_args()


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_slice_levels(levels: Dict[int, torch.Tensor], idx: torch.Tensor) -> Dict[int, torch.Tensor]:
    return {lv: x[idx] for lv, x in levels.items()}


def _evaluate(policy: BCPolicy, dataset: BCTensorDataset, device: torch.device) -> Tuple[float, float]:
    policy.eval()
    with torch.no_grad():
        levels = {lv: x.to(device) for lv, x in dataset.levels.items()}
        state = dataset.robot_state.to(device)
        actions = dataset.actions.to(device)
        logits = policy(levels, state)
        loss = torch.nn.functional.cross_entropy(logits, actions).item()
        pred = torch.argmax(logits, dim=1)
        acc = (pred == actions).float().mean().item()
    return float(loss), float(acc)


def _prepare_square_map(base_map: np.ndarray, size: int) -> np.ndarray:
    if size <= 0:
        raise ValueError("map-size must be positive")
    out = np.zeros((size, size), dtype=np.int32)
    h, w = base_map.shape
    rh = min(h, size)
    rw = min(w, size)
    out[:rh, :rw] = base_map[:rh, :rw]
    return out


def _parse_map_text(text: str) -> np.ndarray:
    rows = [list(map(int, line.split())) for line in text.strip().splitlines() if line.strip()]
    if not rows:
        raise ValueError("Map text is empty")
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"Map rows have inconsistent widths: {sorted(widths)}")
    grid = np.asarray(rows, dtype=np.int32)
    if not np.isin(grid, [0, 1]).all():
        raise ValueError("Map must contain only 0 and 1")
    return grid


def _carve_door(
    grid: np.ndarray,
    *,
    is_vertical_wall: bool,
    wall_coord: int,
    seg_start: int,
    seg_end: int,
    door_width: int,
    wall_thickness: int,
    rng: np.random.RandomState,
):
    if seg_end - seg_start <= 2:
        return
    door_width = max(1, int(door_width))
    wall_thickness = max(1, int(wall_thickness))
    # Keep doors away from wall intersections to avoid leaking through corners.
    low = seg_start + wall_thickness
    high = seg_end - wall_thickness
    if high - low <= 1:
        return
    center = int(rng.randint(low, high))
    d0 = center - door_width // 2
    d1 = d0 + door_width
    d0 = max(low, d0)
    d1 = min(high, d1)
    if d1 <= d0:
        return

    half = wall_thickness // 2
    if is_vertical_wall:
        c0 = max(0, wall_coord - half)
        c1 = min(grid.shape[1], c0 + wall_thickness)
        grid[d0:d1, c0:c1] = 0
    else:
        r0 = max(0, wall_coord - half)
        r1 = min(grid.shape[0], r0 + wall_thickness)
        grid[r0:r1, d0:d1] = 0


def _room_neighbors(r: int, c: int, rows: int, cols: int):
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < rows:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < cols:
        out.append((r, c + 1))
    return out


def _canonical_edge(a, b):
    return (a, b) if a <= b else (b, a)


def _sample_door_intervals(segment_len: int, door_width: int, n_doors: int, rng: np.random.RandomState):
    door_width = max(1, int(door_width))
    n_doors = max(1, int(n_doors))
    if segment_len <= door_width:
        return [(0, min(segment_len, door_width))]

    starts = list(range(0, segment_len - door_width + 1))
    rng.shuffle(starts)
    intervals = []
    for s in starts:
        e = s + door_width
        overlap = False
        for ss, ee in intervals:
            if not (e <= ss or s >= ee):
                overlap = True
                break
        if overlap:
            continue
        intervals.append((s, e))
        if len(intervals) >= n_doors:
            break
    if not intervals:
        intervals = [(0, door_width)]
    return intervals


def _build_indoor_map(args: argparse.Namespace) -> np.ndarray:
    size = int(args.map_size)
    room_inner = max(3, int(args.indoor_room_inner_size))
    door_width = max(1, int(args.indoor_door_width))
    wall_thickness = max(1, int(args.indoor_wall_thickness))
    rows = int(args.indoor_rooms_rows)
    cols = int(args.indoor_rooms_cols)
    if rows <= 0:
        rows = max(2, (size - wall_thickness) // (room_inner + wall_thickness))
    if cols <= 0:
        cols = max(2, (size - wall_thickness) // (room_inner + wall_thickness))
    if rows < 2 or cols < 2:
        raise ValueError("indoor map must have at least 2x2 rooms")
    req_h = rows * room_inner + (rows + 1) * wall_thickness
    req_w = cols * room_inner + (cols + 1) * wall_thickness
    if req_h > size or req_w > size:
        raise ValueError(
            f"room layout does not fit map-size={size}. "
            f"required=({req_h},{req_w}), room_inner={room_inner}, wall={wall_thickness}, rooms=({rows},{cols})"
        )

    extra_prob = float(np.clip(args.indoor_extra_connection_prob, 0.0, 1.0))
    two_door_prob = float(np.clip(args.indoor_two_door_prob, 0.0, 1.0))
    rng = np.random.RandomState(args.seed)

    grid = np.zeros((size, size), dtype=np.int32)
    grid[:, :] = 0

    # Place layout at top-left; leftover area stays free.
    layout_h = req_h
    layout_w = req_w

    # Draw all walls first.
    for k in range(rows + 1):
        r0 = k * (room_inner + wall_thickness)
        r1 = min(layout_h, r0 + wall_thickness)
        grid[r0:r1, :layout_w] = 1
    for k in range(cols + 1):
        c0 = k * (room_inner + wall_thickness)
        c1 = min(layout_w, c0 + wall_thickness)
        grid[:layout_h, c0:c1] = 1

    # Build adjacency edges between rooms.
    all_edges = []
    for r in range(rows):
        for c in range(cols):
            if r + 1 < rows:
                all_edges.append(((r, c), (r + 1, c)))
            if c + 1 < cols:
                all_edges.append(((r, c), (r, c + 1)))

    # Random spanning tree to guarantee full connectivity.
    visited = set()
    stack = [(0, 0)]
    visited.add((0, 0))
    tree_edges = set()
    while stack:
        cur = stack[-1]
        neigh = [n for n in _room_neighbors(cur[0], cur[1], rows, cols) if n not in visited]
        if not neigh:
            stack.pop()
            continue
        nxt = neigh[int(rng.randint(0, len(neigh)))]
        visited.add(nxt)
        stack.append(nxt)
        tree_edges.add(_canonical_edge(cur, nxt))

    # Open additional random edges (direction can have door or not).
    open_edges = set(tree_edges)
    for e in all_edges:
        ce = _canonical_edge(e[0], e[1])
        if ce in open_edges:
            continue
        if rng.rand() < extra_prob:
            open_edges.add(ce)

    # Carve doors for each open edge, with 2-door ratio around 2:8.
    for a, b in sorted(open_edges):
        n_doors = 2 if rng.rand() < two_door_prob else 1
        if a[0] == b[0]:
            # left-right neighbors => vertical wall
            r = a[0]
            c_left = min(a[1], b[1])
            r0 = wall_thickness + r * (room_inner + wall_thickness)
            wall_c = wall_thickness + c_left * (room_inner + wall_thickness) + room_inner
            intervals = _sample_door_intervals(room_inner, door_width, n_doors, rng)
            for s, e in intervals:
                rr0 = r0 + s
                rr1 = r0 + e
                grid[rr0:rr1, wall_c : wall_c + wall_thickness] = 0
        else:
            # up-down neighbors => horizontal wall
            c = a[1]
            r_up = min(a[0], b[0])
            c0 = wall_thickness + c * (room_inner + wall_thickness)
            wall_r = wall_thickness + r_up * (room_inner + wall_thickness) + room_inner
            intervals = _sample_door_intervals(room_inner, door_width, n_doors, rng)
            for s, e in intervals:
                cc0 = c0 + s
                cc1 = c0 + e
                grid[wall_r : wall_r + wall_thickness, cc0:cc1] = 0

    # Ensure top-left spawn and local move are free.
    grid[0, 0] = 0
    if size > 1:
        grid[1, 0] = 0
        grid[0, 1] = 0
    return grid


def _build_map(args: argparse.Namespace) -> np.ndarray:
    if args.map_source == "custom":
        base_map = parse_custom_map(CUSTOM_MAP_TEXT).astype(np.int32)
        if base_map[0, 0] == 1:
            base_map[0, 0] = 0
        return _prepare_square_map(base_map, args.map_size)
    if args.map_source == "file":
        if not args.map_file:
            raise ValueError("--map-file is required when --map-source file")
        p = Path(args.map_file)
        if not p.exists():
            raise FileNotFoundError(f"Map file not found: {p}")
        base_map = _parse_map_text(p.read_text(encoding="utf-8")).astype(np.int32)
        if base_map[0, 0] == 1:
            base_map[0, 0] = 0
        return _prepare_square_map(base_map, args.map_size)
    if args.map_source == "indoor":
        return _build_indoor_map(args)

    gen = MapGenerator(height=args.map_size, width=args.map_size, seed=args.seed)
    rand_map = gen.generate_map(stage=args.map_stage, num_obstacles=args.num_obstacles)
    rand_map = np.asarray(rand_map, dtype=np.int32)
    if rand_map.shape != (args.map_size, args.map_size):
        rand_map = _prepare_square_map(rand_map, args.map_size)
    if rand_map[0, 0] == 1:
        rand_map[0, 0] = 0
    return rand_map


def _pick_start_node(grid: np.ndarray) -> Tuple[int, int]:
    free = np.argwhere(grid == 0)
    if free.size == 0:
        raise RuntimeError("Map has no free cell for start")
    r, c = free[0]
    return int(r), int(c)


def _train_one_mode(
    mode_name: str,
    dataset: BCTensorDataset,
    in_channels_per_level: int,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, float, Dict[str, list]]:
    enc = FusedMAPSStateEncoderConfig(
        maps=MultiLevelMAPSEncoderConfig(
            num_levels=len(dataset.levels),
            in_channels_per_level=in_channels_per_level,
            conv_channels=(16, 32),
            level_embed_dim=64,
        )
    )
    policy = BCPolicy(BCPolicyConfig(action_dim=args.action_dim, encoder=enc)).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=args.lr)

    n = dataset.size
    history: Dict[str, list] = {
        "epoch": [],
        "train_loss": [],
        "eval_loss": [],
        "eval_acc": [],
    }
    for epoch in range(1, args.epochs + 1):
        policy.train()
        perm = torch.randperm(n)
        epoch_loss_sum = 0.0
        epoch_count = 0
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size]
            levels_b = _batch_slice_levels(dataset.levels, idx)
            levels_b = {lv: x.to(device) for lv, x in levels_b.items()}
            state_b = dataset.robot_state[idx].to(device)
            action_b = dataset.actions[idx].to(device)

            optim.zero_grad(set_to_none=True)
            loss = policy.loss(levels_b, state_b, action_b)
            loss.backward()
            optim.step()
            batch_n = int(action_b.shape[0])
            epoch_loss_sum += float(loss.item()) * batch_n
            epoch_count += batch_n

        train_loss = epoch_loss_sum / float(max(1, epoch_count))
        eval_loss, eval_acc = _evaluate(policy, dataset, device)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["eval_loss"].append(eval_loss)
        history["eval_acc"].append(eval_acc)

        if epoch == 1 or epoch == args.epochs or (epoch % max(1, args.log_every) == 0):
            print(
                f"  [{mode_name}][ep {epoch:03d}] "
                f"train_loss={train_loss:.4f} eval_loss={eval_loss:.4f} acc={eval_acc * 100.0:6.2f}%",
                flush=True,
            )

    loss = float(history["eval_loss"][-1])
    acc = float(history["eval_acc"][-1])
    print(f"{mode_name:>10s} | samples={n:4d} | loss={loss:.4f} | acc={acc * 100.0:6.2f}%")
    return loss, acc, history


def _history_diagnostics(history: Dict[str, list]) -> Dict[str, float]:
    eval_loss = np.asarray(history["eval_loss"], dtype=np.float64)
    eval_acc = np.asarray(history["eval_acc"], dtype=np.float64)
    if eval_loss.size == 0:
        return {
            "start_eval_loss": float("nan"),
            "end_eval_loss": float("nan"),
            "max_eval_loss_jump": float("nan"),
            "start_acc": float("nan"),
            "end_acc": float("nan"),
            "max_acc_drop": float("nan"),
        }

    loss_deltas = np.diff(eval_loss) if eval_loss.size >= 2 else np.array([0.0], dtype=np.float64)
    acc_deltas = np.diff(eval_acc) if eval_acc.size >= 2 else np.array([0.0], dtype=np.float64)
    return {
        "start_eval_loss": float(eval_loss[0]),
        "end_eval_loss": float(eval_loss[-1]),
        "max_eval_loss_jump": float(np.max(loss_deltas)),
        "start_acc": float(eval_acc[0]),
        "end_acc": float(eval_acc[-1]),
        "max_acc_drop": float(np.min(acc_deltas)),
    }


def main():
    args = _parse_args()
    device = _select_device(args.device)
    _set_seed(args.seed)

    grid = _build_map(args)
    obs_count = int(np.count_nonzero(grid == 1))
    free_count = int(np.count_nonzero(grid == 0))
    total = int(grid.size)
    print(
        f"Map source: {args.map_source} "
        f"(size={args.map_size}, stage={args.map_stage if args.map_source == 'random' else 'n/a'})",
        flush=True,
    )
    print(f"Obstacle cells: {obs_count}/{total} ({100.0 * obs_count / total:.2f}%)", flush=True)
    print(f"Free cells: {free_count}/{total} ({100.0 * free_count / total:.2f}%)", flush=True)
    print("Collecting E* rollout...", flush=True)

    start_node = _pick_start_node(grid)
    print(f"Start node: {start_node}", flush=True)

    transitions, teacher_path = collect_epsilon_rollout(
        grid_map=grid,
        start_node=start_node,
        epsilon=args.epsilon,
        sensor_range=args.sensor_range,
        max_steps=args.max_steps,
    )
    if len(transitions) == 0:
        raise RuntimeError("Collected zero transitions from teacher rollout")
    print(f"Collected transitions: {len(transitions)}", flush=True)

    obs_cfg = MultiScaleCPPObservationConfig(
        local_blocks=(1, 2, 4, 8, 16),
        local_window_size=7,
        global_window_size=4,
        dtm_patch_size=7,
        dtm_connectivity=8,
        dtm_require_fully_known_patch=False,
        dtm_min_known_ratio=0.6,
    )
    baseline_builder = MultiScaleCPPObservationBuilder(obs_cfg, include_dtm=False)
    dtm_builder = MultiScaleCPPObservationBuilder(obs_cfg, include_dtm=True)

    print("Building baseline dataset...", flush=True)
    baseline_ds = build_bc_tensors_from_rollout(transitions, maps_builder=baseline_builder)
    print("Building DTM dataset...", flush=True)
    dtm_ds = build_bc_tensors_from_rollout(transitions, maps_builder=dtm_builder)

    print(f"Device: {device}")
    print(f"Map shape: {grid.shape}")
    print(f"Teacher path length: {len(teacher_path)}")
    print(f"Transitions: {len(transitions)}")
    print(f"Epochs: {args.epochs}, batch_size: {args.batch_size}")
    print("Mode results:")

    _set_seed(args.seed)
    print("Training baseline...", flush=True)
    baseline_loss, baseline_acc, baseline_hist = _train_one_mode(
        "baseline",
        baseline_ds,
        in_channels_per_level=baseline_builder.channels_per_level,
        args=args,
        device=device,
    )
    _set_seed(args.seed)
    print("Training DTM...", flush=True)
    dtm_loss, dtm_acc, dtm_hist = _train_one_mode(
        "dtm",
        dtm_ds,
        in_channels_per_level=dtm_builder.channels_per_level,
        args=args,
        device=device,
    )

    print("\nDelta (dtm - baseline):")
    print(f"  loss: {dtm_loss - baseline_loss:+.4f}")
    print(f"  acc : {(dtm_acc - baseline_acc) * 100.0:+.2f}%")

    bdiag = _history_diagnostics(baseline_hist)
    ddiag = _history_diagnostics(dtm_hist)
    print("\nConvergence diagnostics:")
    print(
        "  baseline: "
        f"eval_loss {bdiag['start_eval_loss']:.4f}->{bdiag['end_eval_loss']:.4f}, "
        f"max_loss_jump {bdiag['max_eval_loss_jump']:+.4f}, "
        f"acc {bdiag['start_acc']*100.0:.2f}%->{bdiag['end_acc']*100.0:.2f}%, "
        f"worst_acc_drop {bdiag['max_acc_drop']*100.0:+.2f}%p"
    )
    print(
        "  dtm:      "
        f"eval_loss {ddiag['start_eval_loss']:.4f}->{ddiag['end_eval_loss']:.4f}, "
        f"max_loss_jump {ddiag['max_eval_loss_jump']:+.4f}, "
        f"acc {ddiag['start_acc']*100.0:.2f}%->{ddiag['end_acc']*100.0:.2f}%, "
        f"worst_acc_drop {ddiag['max_acc_drop']*100.0:+.2f}%p"
    )

    if args.save_log_json:
        payload = {
            "config": vars(args),
            "baseline": {
                "final_loss": baseline_loss,
                "final_acc": baseline_acc,
                "history": baseline_hist,
                "diagnostics": bdiag,
            },
            "dtm": {
                "final_loss": dtm_loss,
                "final_acc": dtm_acc,
                "history": dtm_hist,
                "diagnostics": ddiag,
            },
        }
        with open(args.save_log_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved log json: {args.save_log_json}")


if __name__ == "__main__":
    main()

import argparse
from collections import deque
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.observation.cpp.grid_features import FREE_STATE
from learning.observation.cpp.multiscale_observation import (
    MultiScaleCPPObservationBuilder,
    MultiScaleCPPObservationConfig,
)


def _build_ref_graph(
    dtm_block: np.ndarray,
    state_block: np.ndarray,
    mode: str,
) -> List[List[int]]:
    bh, bw = state_block.shape
    adj: List[List[int]] = [[] for _ in range(bh * bw * 4)]

    def nid(r: int, c: int, p: int) -> int:
        return ((r * bw + c) * 4) + p

    def add_edge(u: int, v: int) -> None:
        adj[u].append(v)

    def is_free(r: int, c: int) -> bool:
        return int(state_block[r, c]) == int(FREE_STATE)

    # Intra-cell transfer.
    for r in range(bh):
        for c in range(bw):
            if not is_free(r, c):
                continue
            ch = dtm_block[:, r, c]
            n = nid(r, c, 0)
            e = nid(r, c, 1)
            s = nid(r, c, 2)
            w = nid(r, c, 3)

            if mode == "six":
                if float(ch[0]) > 0.5:
                    add_edge(w, e)
                    add_edge(e, w)
                if float(ch[1]) > 0.5:
                    add_edge(n, s)
                    add_edge(s, n)
                if float(ch[2]) > 0.5:
                    add_edge(n, e)
                    add_edge(w, s)
                if float(ch[3]) > 0.5:
                    add_edge(e, n)
                    add_edge(s, w)
                if float(ch[4]) > 0.5:
                    add_edge(n, w)
                    add_edge(e, s)
                if float(ch[5]) > 0.5:
                    add_edge(w, n)
                    add_edge(s, e)
            else:
                # port12:
                # 0 U->R, 1 U->D, 2 U->L,
                # 3 R->U, 4 R->D, 5 R->L,
                # 6 D->U, 7 D->R, 8 D->L,
                # 9 L->U,10 L->R,11 L->D
                if float(ch[0]) > 0.5:
                    add_edge(n, e)
                if float(ch[1]) > 0.5:
                    add_edge(n, s)
                if float(ch[2]) > 0.5:
                    add_edge(n, w)
                if float(ch[3]) > 0.5:
                    add_edge(e, n)
                if float(ch[4]) > 0.5:
                    add_edge(e, s)
                if float(ch[5]) > 0.5:
                    add_edge(e, w)
                if float(ch[6]) > 0.5:
                    add_edge(s, n)
                if float(ch[7]) > 0.5:
                    add_edge(s, e)
                if float(ch[8]) > 0.5:
                    add_edge(s, w)
                if float(ch[9]) > 0.5:
                    add_edge(w, n)
                if float(ch[10]) > 0.5:
                    add_edge(w, e)
                if float(ch[11]) > 0.5:
                    add_edge(w, s)

    # Inter-cell boundary crossing (undirected).
    for r in range(bh):
        for c in range(bw):
            if not is_free(r, c):
                continue
            if c + 1 < bw and is_free(r, c + 1):
                u = nid(r, c, 1)
                v = nid(r, c + 1, 3)
                add_edge(u, v)
                add_edge(v, u)
            if r + 1 < bh and is_free(r + 1, c):
                u = nid(r, c, 2)
                v = nid(r + 1, c, 0)
                add_edge(u, v)
                add_edge(v, u)
    return adj


def _reachable(adj: Sequence[Sequence[int]], src_nodes: Sequence[int], tgt_nodes: Sequence[int]) -> bool:
    src = [x for x in src_nodes if x is not None]
    tgt = {x for x in tgt_nodes if x is not None}
    if not src or not tgt:
        return False
    if any(x in tgt for x in src):
        return True

    q = deque(src)
    vis = set(src)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in vis:
                continue
            if v in tgt:
                return True
            vis.add(v)
            q.append(v)
    return False


def _ref_flags(
    dtm_block: np.ndarray,
    state_block: np.ndarray,
    mode: str,
) -> np.ndarray:
    bh, bw = state_block.shape

    def is_free(r: int, c: int) -> bool:
        return 0 <= r < bh and 0 <= c < bw and int(state_block[r, c]) == int(FREE_STATE)

    def nid(r: int, c: int, p: int) -> int:
        return ((r * bw + c) * 4) + p

    def port_node(r: int, c: int, p: int):
        if not is_free(r, c):
            return None
        return nid(r, c, p)

    adj = _build_ref_graph(dtm_block, state_block, mode)
    side_nodes = {
        "up": [port_node(0, c, 0) for c in range(bw)],
        "right": [port_node(r, bw - 1, 1) for r in range(bh)],
        "down": [port_node(bh - 1, c, 2) for c in range(bw)],
        "left": [port_node(r, 0, 3) for r in range(bh)],
    }

    if mode == "six":
        def cell_nodes(r: int, c: int):
            return [port_node(r, c, p) for p in range(4)]

        nw_nodes = cell_nodes(0, 0)
        se_nodes = cell_nodes(bh - 1, bw - 1)
        ne_nodes = cell_nodes(0, bw - 1)
        sw_nodes = cell_nodes(bh - 1, 0)
        out = np.zeros(6, dtype=np.float32)
        out[0] = 1.0 if (_reachable(adj, side_nodes["left"], side_nodes["right"]) or _reachable(adj, side_nodes["right"], side_nodes["left"])) else 0.0
        out[1] = 1.0 if (_reachable(adj, side_nodes["up"], side_nodes["down"]) or _reachable(adj, side_nodes["down"], side_nodes["up"])) else 0.0
        out[2] = 1.0 if _reachable(adj, nw_nodes, se_nodes) else 0.0
        out[3] = 1.0 if _reachable(adj, se_nodes, nw_nodes) else 0.0
        out[4] = 1.0 if _reachable(adj, ne_nodes, sw_nodes) else 0.0
        out[5] = 1.0 if _reachable(adj, sw_nodes, ne_nodes) else 0.0
        return out

    pairs = (
        ("up", "right"),
        ("up", "down"),
        ("up", "left"),
        ("right", "up"),
        ("right", "down"),
        ("right", "left"),
        ("down", "up"),
        ("down", "right"),
        ("down", "left"),
        ("left", "up"),
        ("left", "right"),
        ("left", "down"),
    )
    out = np.zeros(12, dtype=np.float32)
    for k, (a, b) in enumerate(pairs):
        out[k] = 1.0 if _reachable(adj, side_nodes[a], side_nodes[b]) else 0.0
    return out


def _run_handcrafted_cases(builder: MultiScaleCPPObservationBuilder, mode: str) -> None:
    state = np.full((2, 2), FREE_STATE, dtype=np.int8)

    if mode == "port12":
        dtm = np.zeros((12, 2, 2), dtype=np.float32)

        # Loop-like case discovered by user:
        # (0,0) -> (1,0) -> (1,1) -> (0,1), entering from left boundary and exiting to right boundary.
        # Needed local transitions:
        # (0,0): L->D, (1,0): U->R, (1,1): L->U, (0,1): D->R
        dtm[11, 0, 0] = 1.0  # L->D
        dtm[0, 1, 0] = 1.0   # U->R
        dtm[9, 1, 1] = 1.0   # L->U
        dtm[7, 0, 1] = 1.0   # D->R

        got = builder._aggregate_transfer_flags_block(dtm, state, dtm_mode=mode)
        ref = _ref_flags(dtm, state, mode)
        if not np.allclose(got, ref, atol=1e-6):
            raise RuntimeError("handcrafted case mismatch against reference BFS")
        # index 10 is L->R
        if float(got[10]) != 1.0:
            raise RuntimeError("handcrafted loop path did not produce expected L->R=1")
        return

    if mode == "six":
        dtm = np.zeros((6, 2, 2), dtype=np.float32)
        # Diagonal case through all four cells:
        # NW(S->E) -> NE(W->S) -> SE(N->E)
        # This should set NW->SE due to quadrant-cell connectivity.
        dtm[5, 0, 0] = 1.0  # SW->NE gives S->E
        dtm[2, 0, 1] = 1.0  # NW->SE gives W->S
        dtm[2, 1, 1] = 1.0  # NW->SE gives N->E
        got = builder._aggregate_transfer_flags_block(dtm, state, dtm_mode=mode)
        ref = _ref_flags(dtm, state, mode)
        if not np.allclose(got, ref, atol=1e-6):
            raise RuntimeError("handcrafted six-mode diagonal case mismatch")
        # index 2 is NW->SE
        if float(got[2]) != 1.0:
            raise RuntimeError("handcrafted six diagonal path did not produce expected NW->SE=1")


def main() -> None:
    ap = argparse.ArgumentParser(description="Strict exact-check: aggregate_transfer vs independent BFS reference.")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--height", type=int, default=4)
    ap.add_argument("--width", type=int, default=4)
    ap.add_argument("--mode", type=str, default="port12", choices=["six", "port12"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(int(args.seed))
    mode = str(args.mode)
    ch = 6 if mode == "six" else 12

    cfg = MultiScaleCPPObservationConfig(dtm_output_mode=mode, dtm_coarse_mode="aggregate_transfer")
    builder = MultiScaleCPPObservationBuilder(cfg, include_dtm=True)
    _run_handcrafted_cases(builder, mode)

    checks = 0
    mismatches = 0
    for _ in range(int(args.trials)):
        state = np.where(rng.rand(int(args.height), int(args.width)) < 0.72, FREE_STATE, 1).astype(np.int8)
        raw = rng.rand(ch, int(args.height), int(args.width))
        dtm_block = np.full((ch, int(args.height), int(args.width)), -1.0, dtype=np.float32)
        dtm_block[raw < 0.33] = 0.0
        dtm_block[raw > 0.66] = 1.0

        got = builder._aggregate_transfer_flags_block(dtm_block, state, dtm_mode=mode)
        ref = _ref_flags(dtm_block, state, mode)
        checks += 1
        if not np.allclose(got, ref, atol=1e-6):
            mismatches += 1

    print(f"[VERIFY-TRANSFER] mode={mode} checks={checks} mismatches={mismatches}")
    if mismatches > 0:
        raise SystemExit(1)
    print("[VERIFY-TRANSFER] PASS: exact match with independent BFS reference")


if __name__ == "__main__":
    main()

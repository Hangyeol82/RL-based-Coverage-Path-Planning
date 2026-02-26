import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.observation.cpp.multiscale_observation import (
    MultiScaleCPPObservationBuilder,
    MultiScaleCPPObservationConfig,
)


GridPos = Tuple[int, int]


def _random_free_start(true_map: np.ndarray, rng: np.random.RandomState) -> GridPos:
    free = np.argwhere(true_map == 0)
    if free.size == 0:
        true_map[0, 0] = 0
        return (0, 0)
    idx = int(rng.randint(0, free.shape[0]))
    r, c = free[idx]
    return (int(r), int(c))


def _simulate_sequence(
    *,
    map_size: int,
    steps: int,
    sensor_range: int,
    obstacle_prob: float,
    seed: int,
    fully_known: bool,
) -> List[Tuple[np.ndarray, np.ndarray, GridPos]]:
    rng = np.random.RandomState(seed)
    h = int(map_size)
    w = int(map_size)
    true_map = (rng.rand(h, w) < float(obstacle_prob)).astype(np.int32)
    pos = _random_free_start(true_map, rng)
    known = true_map.copy() if fully_known else np.full((h, w), -1, dtype=np.int32)
    explored = np.zeros((h, w), dtype=bool)

    seq: List[Tuple[np.ndarray, np.ndarray, GridPos]] = []
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    sr = max(0, int(sensor_range))

    for _ in range(int(steps)):
        dr, dc = deltas[int(rng.randint(0, 4))]
        nr = min(h - 1, max(0, pos[0] + dr))
        nc = min(w - 1, max(0, pos[1] + dc))
        pos = (nr, nc)

        if not fully_known:
            r0 = max(0, nr - sr)
            r1 = min(h - 1, nr + sr)
            c0 = max(0, nc - sr)
            c1 = min(w - 1, nc + sr)
            known[r0 : r1 + 1, c0 : c1 + 1] = true_map[r0 : r1 + 1, c0 : c1 + 1]
        if true_map[nr, nc] == 0:
            explored[nr, nc] = True

        seq.append((known.copy(), explored.copy(), pos))
    return seq


def _value_index(arr: np.ndarray) -> np.ndarray:
    idx = np.full(arr.shape, 3, dtype=np.int32)
    idx[arr == -1.0] = 0
    idx[arr == 0.0] = 1
    idx[arr == 1.0] = 2
    return idx


def _compare_sequence(
    seq: List[Tuple[np.ndarray, np.ndarray, GridPos]],
    *,
    output_mode: str,
) -> Dict[str, object]:
    cfg_bfs = MultiScaleCPPObservationConfig(dtm_coarse_mode="bfs", dtm_output_mode=output_mode)
    cfg_transfer = MultiScaleCPPObservationConfig(dtm_coarse_mode="aggregate_transfer", dtm_output_mode=output_mode)
    bfs_builder = MultiScaleCPPObservationBuilder(cfg_bfs, include_dtm=True)
    transfer_builder = MultiScaleCPPObservationBuilder(cfg_transfer, include_dtm=True)

    # Infer structure from first sample.
    first_occ, first_exp, first_pos = seq[0]
    first_ref = bfs_builder.build_levels(first_occ, robot_pos=first_pos, explored=first_exp)
    level_ids = sorted(first_ref.keys())
    dtm_ch = int(first_ref[level_ids[0]][3:].shape[0])

    total = 0
    mismatch = 0
    total_ref_unknown = 0
    mismatch_ref_unknown = 0
    total_ref_certain = 0
    mismatch_ref_certain = 0
    fp1 = 0
    fn1 = 0

    level_total = {lv: 0 for lv in level_ids}
    level_mismatch = {lv: 0 for lv in level_ids}
    channel_total = {ch: 0 for ch in range(dtm_ch)}
    channel_mismatch = {ch: 0 for ch in range(dtm_ch)}
    confusion = np.zeros((4, 4), dtype=np.int64)  # row=transfer, col=bfs

    # Re-run first sample from fresh builders for alignment with loop.
    bfs_builder = MultiScaleCPPObservationBuilder(cfg_bfs, include_dtm=True)
    transfer_builder = MultiScaleCPPObservationBuilder(cfg_transfer, include_dtm=True)

    for occ, exp, pos in seq:
        ref = bfs_builder.build_levels(occ, robot_pos=pos, explored=exp)
        got = transfer_builder.build_levels(occ, robot_pos=pos, explored=exp)
        for lv in level_ids:
            ref_d = ref[lv][3:]
            got_d = got[lv][3:]
            eq = got_d == ref_d
            neq = ~eq

            cnt = int(ref_d.size)
            mis = int(np.count_nonzero(neq))
            total += cnt
            mismatch += mis
            level_total[lv] += cnt
            level_mismatch[lv] += mis

            ref_unknown = ref_d == -1.0
            ref_certain = ~ref_unknown
            total_ref_unknown += int(np.count_nonzero(ref_unknown))
            total_ref_certain += int(np.count_nonzero(ref_certain))
            mismatch_ref_unknown += int(np.count_nonzero(neq & ref_unknown))
            mismatch_ref_certain += int(np.count_nonzero(neq & ref_certain))

            fp1 += int(np.count_nonzero((got_d == 1.0) & (ref_d != 1.0)))
            fn1 += int(np.count_nonzero((got_d != 1.0) & (ref_d == 1.0)))

            for ch in range(dtm_ch):
                rc = ref_d[ch]
                gc = got_d[ch]
                channel_total[ch] += int(rc.size)
                channel_mismatch[ch] += int(np.count_nonzero(gc != rc))

            gi = _value_index(got_d)
            ri = _value_index(ref_d)
            for a in range(4):
                for b in range(4):
                    confusion[a, b] += int(np.count_nonzero((gi == a) & (ri == b)))

    out = {
        "total": total,
        "mismatch": mismatch,
        "mismatch_ratio": float(mismatch) / float(max(1, total)),
        "ref_unknown_total": total_ref_unknown,
        "ref_unknown_mismatch": mismatch_ref_unknown,
        "ref_unknown_mismatch_ratio": float(mismatch_ref_unknown) / float(max(1, total_ref_unknown)),
        "ref_certain_total": total_ref_certain,
        "ref_certain_mismatch": mismatch_ref_certain,
        "ref_certain_mismatch_ratio": float(mismatch_ref_certain) / float(max(1, total_ref_certain)),
        "fp1_count": fp1,
        "fn1_count": fn1,
        "levels": [
            {
                "level": int(lv),
                "mismatch_ratio": float(level_mismatch[lv]) / float(max(1, level_total[lv])),
                "total": int(level_total[lv]),
                "mismatch": int(level_mismatch[lv]),
            }
            for lv in level_ids
        ],
        "channels": [
            {
                "channel": int(ch),
                "mismatch_ratio": float(channel_mismatch[ch]) / float(max(1, channel_total[ch])),
                "total": int(channel_total[ch]),
                "mismatch": int(channel_mismatch[ch]),
            }
            for ch in range(dtm_ch)
        ],
        "confusion_transfer_vs_bfs": confusion.tolist(),
    }
    return out


def _print_report(header: str, report: Dict[str, object]) -> None:
    print(header)
    print(
        "  total={total} mismatch={mismatch} mismatch_ratio={mismatch_ratio:.8f}".format(
            **report
        )
    )
    print(
        "  ref_unknown: total={ref_unknown_total} mismatch={ref_unknown_mismatch} ratio={ref_unknown_mismatch_ratio:.8f}".format(
            **report
        )
    )
    print(
        "  ref_certain: total={ref_certain_total} mismatch={ref_certain_mismatch} ratio={ref_certain_mismatch_ratio:.8f}".format(
            **report
        )
    )
    print("  fp1_count={fp1_count} fn1_count={fn1_count}".format(**report))
    print("  per-level mismatch:")
    for x in report["levels"]:
        print(
            "    L{level}: ratio={mismatch_ratio:.8f} ({mismatch}/{total})".format(
                **x
            )
        )
    print("  per-channel mismatch:")
    for x in report["channels"]:
        print(
            "    ch{channel:02d}: ratio={mismatch_ratio:.8f} ({mismatch}/{total})".format(
                **x
            )
        )
    print("  confusion rows=transfer(-1,0,1,other), cols=bfs(-1,0,1,other):")
    for row in report["confusion_transfer_vs_bfs"]:
        print("   ", row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Detailed comparison: aggregate_transfer vs raw BFS.")
    ap.add_argument("--map-size", type=int, default=32)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--sensor-range", type=int, default=2)
    ap.add_argument("--obstacle-prob", type=float, default=0.28)
    ap.add_argument("--seeds", type=str, default="101,202,303")
    ap.add_argument("--output-mode", type=str, default="port12", choices=["six", "four", "port12"])
    ap.add_argument("--scenario", type=str, default="both", choices=["online", "fully_known", "both"])
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    if not seeds:
        raise ValueError("No seeds parsed from --seeds")

    reports_online = []
    reports_known = []

    for seed in seeds:
        if args.scenario in {"online", "both"}:
            seq = _simulate_sequence(
                map_size=int(args.map_size),
                steps=int(args.steps),
                sensor_range=int(args.sensor_range),
                obstacle_prob=float(args.obstacle_prob),
                seed=int(seed),
                fully_known=False,
            )
            reports_online.append(_compare_sequence(seq, output_mode=str(args.output_mode)))
        if args.scenario in {"fully_known", "both"}:
            seq = _simulate_sequence(
                map_size=int(args.map_size),
                steps=int(args.steps),
                sensor_range=int(args.sensor_range),
                obstacle_prob=float(args.obstacle_prob),
                seed=int(seed),
                fully_known=True,
            )
            reports_known.append(_compare_sequence(seq, output_mode=str(args.output_mode)))

    def _merge(reports: List[Dict[str, object]]) -> Dict[str, object]:
        if not reports:
            return {}
        # Weighted merge by counts.
        merged = {
            "total": sum(int(r["total"]) for r in reports),
            "mismatch": sum(int(r["mismatch"]) for r in reports),
            "ref_unknown_total": sum(int(r["ref_unknown_total"]) for r in reports),
            "ref_unknown_mismatch": sum(int(r["ref_unknown_mismatch"]) for r in reports),
            "ref_certain_total": sum(int(r["ref_certain_total"]) for r in reports),
            "ref_certain_mismatch": sum(int(r["ref_certain_mismatch"]) for r in reports),
            "fp1_count": sum(int(r["fp1_count"]) for r in reports),
            "fn1_count": sum(int(r["fn1_count"]) for r in reports),
        }
        merged["mismatch_ratio"] = float(merged["mismatch"]) / float(max(1, merged["total"]))
        merged["ref_unknown_mismatch_ratio"] = float(merged["ref_unknown_mismatch"]) / float(max(1, merged["ref_unknown_total"]))
        merged["ref_certain_mismatch_ratio"] = float(merged["ref_certain_mismatch"]) / float(max(1, merged["ref_certain_total"]))

        # Levels/channels: sum counts.
        level_ids = [int(x["level"]) for x in reports[0]["levels"]]
        ch_ids = [int(x["channel"]) for x in reports[0]["channels"]]
        levels = []
        for lv in level_ids:
            t = sum(int(next(y for y in r["levels"] if int(y["level"]) == lv)["total"]) for r in reports)
            m = sum(int(next(y for y in r["levels"] if int(y["level"]) == lv)["mismatch"]) for r in reports)
            levels.append({"level": lv, "total": t, "mismatch": m, "mismatch_ratio": float(m) / float(max(1, t))})
        channels = []
        for ch in ch_ids:
            t = sum(int(next(y for y in r["channels"] if int(y["channel"]) == ch)["total"]) for r in reports)
            m = sum(int(next(y for y in r["channels"] if int(y["channel"]) == ch)["mismatch"]) for r in reports)
            channels.append({"channel": ch, "total": t, "mismatch": m, "mismatch_ratio": float(m) / float(max(1, t))})
        merged["levels"] = levels
        merged["channels"] = channels

        conf = np.zeros((4, 4), dtype=np.int64)
        for r in reports:
            conf += np.asarray(r["confusion_transfer_vs_bfs"], dtype=np.int64)
        merged["confusion_transfer_vs_bfs"] = conf.tolist()
        return merged

    print(
        f"[DETAIL] output_mode={args.output_mode} map={args.map_size} steps={args.steps} "
        f"sensor={args.sensor_range} seeds={seeds}"
    )
    if reports_online:
        _print_report("[SCENARIO] online(partial-known)", _merge(reports_online))
    if reports_known:
        _print_report("[SCENARIO] fully-known", _merge(reports_known))


if __name__ == "__main__":
    main()


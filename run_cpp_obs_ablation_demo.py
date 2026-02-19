import numpy as np

from learning.observation import (
    MultiScaleCPPObservationBuilder,
    MultiScaleCPPObservationConfig,
)
from run_cstar_custom_map import CUSTOM_MAP_TEXT, parse_custom_map


def _reveal_local(gt_map: np.ndarray, robot_pos, radius: int = 3) -> np.ndarray:
    """
    Build an online known map from GT map by revealing only a local 7x7 window.
    unknown = -1, free = 0, obstacle = 1.
    """
    known = np.full_like(gt_map, -1, dtype=np.int32)
    rr, cc = robot_pos
    h, w = gt_map.shape
    for r in range(max(0, rr - radius), min(h, rr + radius + 1)):
        for c in range(max(0, cc - radius), min(w, cc + radius + 1)):
            known[r, c] = int(gt_map[r, c])
    return known


def _print_stats(name: str, levels, channel_names):
    print(f"\n[{name}] channels={channel_names}")
    for lv in sorted(levels.keys()):
        x = levels[lv]
        c, h, w = x.shape
        print(
            f"  level {lv}: shape=({c},{h},{w}) "
            f"min={x.min():.3f} max={x.max():.3f}"
        )


def main():
    gt_map = parse_custom_map(CUSTOM_MAP_TEXT).astype(np.int32)
    robot_pos = (0, 0)
    known_map = _reveal_local(gt_map, robot_pos, radius=3)
    explored = np.zeros_like(gt_map, dtype=bool)
    explored[robot_pos[0], robot_pos[1]] = True

    cfg = MultiScaleCPPObservationConfig(
        local_blocks=(1, 2, 4, 8, 16),
        local_window_size=7,
        global_window_size=4,
        dtm_patch_size=7,
        dtm_connectivity=8,
        dtm_require_fully_known_patch=False,
        dtm_min_known_ratio=0.6,
    )

    baseline_builder = MultiScaleCPPObservationBuilder(cfg, include_dtm=False)
    dtm_builder = MultiScaleCPPObservationBuilder(cfg, include_dtm=True)

    baseline_levels = baseline_builder.build_levels(
        known_map,
        robot_pos=robot_pos,
        explored=explored,
    )
    dtm_levels = dtm_builder.build_levels(
        known_map,
        robot_pos=robot_pos,
        explored=explored,
    )

    print("Map shape:", gt_map.shape)
    print("Known cells:", int(np.count_nonzero(known_map != -1)))
    _print_stats("baseline", baseline_levels, baseline_builder.channel_names)
    _print_stats("dtm", dtm_levels, dtm_builder.channel_names)


if __name__ == "__main__":
    main()

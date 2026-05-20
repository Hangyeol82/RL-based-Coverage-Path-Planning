from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from learning.observation.cpp.directional_traversability import compute_directional_traversability
from learning.observation.cpp.grid_features import BLOCKED_STATE, FREE_STATE
from learning.observation.cpp.multiscale_observation import (
    MultiScaleCPPObservationBuilder,
    MultiScaleCPPObservationConfig,
)


OUT_DIR = REPO_ROOT / "docs" / "figures" / "map_generator"
CHANNELS = ("U-R", "U-D", "U-L", "R-D", "R-L", "D-L")
GridPos = Tuple[int, int]

plt.rcParams["font.family"] = ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _state_from_cells(n: int, free_cells: Iterable[GridPos]) -> np.ndarray:
    state = np.full((n, n), BLOCKED_STATE, dtype=np.int8)
    for r, c in free_cells:
        if 0 <= r < n and 0 <= c < n:
            state[int(r), int(c)] = FREE_STATE
    return state


def _path_cells_8(kind: str) -> Sequence[GridPos]:
    mid = 3
    if kind == "U-R":
        return [(r, mid) for r in range(0, mid + 1)] + [(mid, c) for c in range(mid, 8)]
    if kind == "U-D":
        return [(r, mid) for r in range(8)]
    if kind == "U-L":
        return [(r, mid) for r in range(0, mid + 1)] + [(mid, c) for c in range(0, mid + 1)]
    if kind == "R-D":
        return [(mid, c) for c in range(mid, 8)] + [(r, mid) for r in range(mid, 8)]
    if kind == "R-L":
        return [(mid, c) for c in range(8)]
    if kind == "D-L":
        return [(r, mid) for r in range(mid, 8)] + [(mid, c) for c in range(0, mid + 1)]
    raise ValueError(f"unknown kind: {kind}")


def _hierarchical_flags8(state: np.ndarray) -> np.ndarray:
    cfg = MultiScaleCPPObservationConfig(
        local_blocks=(1, 2, 4, 8),
        dtm_output_mode="six",
        dtm_coarse_mode="bfs",
        include_cell_phase_channels=False,
    )
    builder = MultiScaleCPPObservationBuilder(cfg, include_dtm=True)
    changed = np.ones_like(state, dtype=bool)
    builder._update_boundary_summary_pyramid(state, changed_mask=changed, max_power=3)
    dtm = builder._dtm_from_boundary_summary_level(
        level_id=999,
        power=3,
        dirty_mask=np.ones((1, 1), dtype=bool),
    )
    return dtm[:, 0, 0].astype(np.float32)


def _patch_bfs_flags7(kind: str) -> np.ndarray:
    # Same visual path shrunk to a 7x7 patch for the direct BFS DTM function.
    cells8 = np.asarray(_path_cells_8(kind), dtype=np.int32)
    cells7 = np.clip(np.rint(cells8 * (6.0 / 7.0)).astype(np.int32), 0, 6)
    state = _state_from_cells(7, [tuple(x) for x in cells7.tolist()])
    known_ratio = np.ones_like(state, dtype=np.float32)
    dtm = compute_directional_traversability(
        state,
        known_ratio_map=known_ratio,
        patch_size=7,
        connectivity=4,
        require_fully_known_patch=False,
        min_center_known_ratio=0.0,
        min_patch_known_ratio=0.0,
        output_mode="six",
    )
    return dtm[:, 3, 3].astype(np.float32)


def _draw_state(ax, state: np.ndarray, title: str, flags: np.ndarray) -> None:
    n = int(state.shape[0])
    ax.set_xlim(0, n)
    ax.set_ylim(n, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    for r in range(n):
        for c in range(n):
            fc = "#f7f7f7" if int(state[r, c]) == FREE_STATE else "#242424"
            ax.add_patch(Rectangle((c, r), 1, 1, fc=fc, ec="#666666", lw=0.65))
    values = ",".join(str(int(v)) for v in flags.tolist())
    ax.text(
        n / 2,
        n + 0.55,
        f"[UR,UD,UL,RD,RL,DL]=[{values}]",
        ha="center",
        va="top",
        fontsize=8.2,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    states: Dict[str, np.ndarray] = {}
    for kind in CHANNELS:
        state = _state_from_cells(8, _path_cells_8(kind))
        states[kind] = state
        hierarchical = _hierarchical_flags8(state)
        patch_bfs = _patch_bfs_flags7(kind)
        expected = np.zeros(6, dtype=np.float32)
        expected[CHANNELS.index(kind)] = 1.0
        rows.append(
            {
                "case": kind,
                "expected": " ".join(str(int(v)) for v in expected.tolist()),
                "hierarchical_8x8": " ".join(str(int(v)) for v in hierarchical.tolist()),
                "patch_bfs_7x7": " ".join(str(int(v)) for v in patch_bfs.tolist()),
                "hierarchical_ok": bool(np.allclose(hierarchical, expected)),
                "patch_bfs_ok": bool(np.allclose(patch_bfs, expected)),
            }
        )

    csv_path = OUT_DIR / "dtm_six_sidepair_value_check.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "expected",
                "hierarchical_8x8",
                "patch_bfs_7x7",
                "hierarchical_ok",
                "patch_bfs_ok",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(10.0, 6.6), dpi=220)
    for ax, kind in zip(axes.flat, CHANNELS):
        flags = np.fromstring(next(r for r in rows if r["case"] == kind)["hierarchical_8x8"], sep=" ")
        _draw_state(ax, states[kind], kind, flags)
    fig.suptitle("DTM six side-pair channel value check", fontsize=14, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.035,
        "Each 8x8 coarse cell connects exactly one side pair. Value order: U-R, U-D, U-L, R-D, R-L, D-L.",
        ha="center",
        fontsize=9.2,
    )
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.935])
    png_path = OUT_DIR / "dtm_six_sidepair_value_check.png"
    svg_path = OUT_DIR / "dtm_six_sidepair_value_check.svg"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    print(csv_path)
    print(png_path)
    print(svg_path)
    for row in rows:
        print(
            f"{row['case']}: expected={row['expected']} "
            f"hierarchical_8x8={row['hierarchical_8x8']} "
            f"patch_bfs_7x7={row['patch_bfs_7x7']} "
            f"ok={row['hierarchical_ok'] and row['patch_bfs_ok']}"
        )


if __name__ == "__main__":
    main()

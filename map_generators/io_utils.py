from pathlib import Path

import numpy as np


def write_map_txt(path: Path, grid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(str(int(v)) for v in row) for row in grid.tolist()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_map_png(path: Path, grid: np.ndarray) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(grid == 1, cmap="gray_r", origin="upper")
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(path, dpi=140, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return True

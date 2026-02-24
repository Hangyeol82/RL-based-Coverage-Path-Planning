import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MapGenerator import MapGenerator


def build_random_map(
    *,
    size: int,
    seed: int,
    stage: int = 3,
    ensure_start_clear: bool = True,
) -> np.ndarray:
    gen = MapGenerator(height=size, width=size, seed=seed)
    grid = np.asarray(gen.generate_map(stage=stage), dtype=np.int32)
    if grid.shape != (size, size):
        out = np.zeros((size, size), dtype=np.int32)
        h = min(size, grid.shape[0])
        w = min(size, grid.shape[1])
        out[:h, :w] = grid[:h, :w]
        grid = out
    if ensure_start_clear:
        grid[0, 0] = 0
    return grid

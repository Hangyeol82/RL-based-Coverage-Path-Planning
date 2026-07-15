import numpy as np

from learning.observation.cpp.grid_features import global_reduce_max, global_reduce_mean


def _loop_reduce_mean(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = arr.shape
    row_edges = np.linspace(0, h, out_h + 1, dtype=np.int32)
    col_edges = np.linspace(0, w, out_w + 1, dtype=np.int32)
    row_edges[-1] = h
    col_edges[-1] = w
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for r in range(out_h):
        rs = int(np.clip(row_edges[r], 0, h - 1))
        re = int(np.clip(row_edges[r + 1], rs + 1, h))
        for c in range(out_w):
            cs = int(np.clip(col_edges[c], 0, w - 1))
            ce = int(np.clip(col_edges[c + 1], cs + 1, w))
            out[r, c] = float(np.mean(arr[rs:re, cs:ce]))
    return out


def _loop_reduce_max(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    h, w = arr.shape
    row_edges = np.linspace(0, h, out_h + 1, dtype=np.int32)
    col_edges = np.linspace(0, w, out_w + 1, dtype=np.int32)
    row_edges[-1] = h
    col_edges[-1] = w
    out = np.zeros((out_h, out_w), dtype=np.float32)
    for r in range(out_h):
        rs = int(np.clip(row_edges[r], 0, h - 1))
        re = int(np.clip(row_edges[r + 1], rs + 1, h))
        for c in range(out_w):
            cs = int(np.clip(col_edges[c], 0, w - 1))
            ce = int(np.clip(col_edges[c + 1], cs + 1, w))
            out[r, c] = float(np.max(arr[rs:re, cs:ce]))
    return out


def test_global_reduce_fast_path_matches_loop_for_divisible_blocks() -> None:
    rng = np.random.default_rng(123)
    arr = rng.normal(size=(128, 128)).astype(np.float32)

    for out_h, out_w in ((64, 64), (32, 32), (16, 16)):
        np.testing.assert_allclose(
            global_reduce_mean(arr, out_h, out_w),
            _loop_reduce_mean(arr, out_h, out_w),
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            global_reduce_max(arr, out_h, out_w),
            _loop_reduce_max(arr, out_h, out_w),
            rtol=0.0,
            atol=0.0,
        )


def test_global_reduce_fallback_matches_loop_for_nondivisible_blocks() -> None:
    rng = np.random.default_rng(456)
    arr = rng.normal(size=(127, 131)).astype(np.float32)

    np.testing.assert_allclose(
        global_reduce_mean(arr, 16, 17),
        _loop_reduce_mean(arr, 16, 17),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        global_reduce_max(arr, 16, 17),
        _loop_reduce_max(arr, 16, 17),
        rtol=0.0,
        atol=0.0,
    )

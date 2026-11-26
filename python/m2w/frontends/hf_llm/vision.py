"""Host-side math of the Qwen3.5 image tower that does not depend on the pixels, plus the pixel layout the graphs expect (docs/models/qwen35-vision.md).

Everything that depends on the GRID only is computed here at compile time and stored in the package, so the vision graphs have no gather and no data-dependent shape:
  * `interpolated_pos_embed`: the learned (side x side) position table bilinearly resampled to the grid (transformers' `fast_pos_embed_interpolate`), rows in merge-block order;
  * `rotary_tables`: the 2-D rotary cos / sin [N, head_dim] (half of the dims rotate with the patch row, half with the column), rows in merge-block order.
The patches of an image are in MERGE-BLOCK order (the 2x2 patches that the merger turns into one token are consecutive), each patch is a vector in the (channel, temporal, kh, kw) order of the
Conv3d kernel: `patchify` (transformers' Qwen2VLImageProcessor `_preprocess`).
"""
from __future__ import annotations

import math

import numpy as np

DEFAULT_GRIDS = ((16, 16), (24, 24), (32, 32), (16, 24), (24, 16), (24, 32), (32, 24))   # (grid_h, grid_w) in patches: 256 / 576 / 1024 patches, the 3:2 and 2:3 rectangles
ROTARY_THETA = 10000.0   # Qwen3_5VisionRotaryEmbedding


def grid_name(grid) -> str:
    return f"{grid[0]}x{grid[1]}"


def parse_grid(text: str) -> tuple:
    h, w = (int(x) for x in text.lower().split("x"))
    return h, w


def check_grid(grid, merge: int) -> None:
    if grid[0] % merge or grid[1] % merge or min(grid) < merge:
        raise ValueError(f"the grid {grid_name(grid)} must be a multiple of the merge size {merge} in both directions")


def merge_order(grid_h: int, grid_w: int, merge: int):
    """(rows, cols) int arrays [N]: the patch row / column of every position of the merge-block order (block row, block column, row in block, column in block)."""
    rows = np.arange(grid_h).reshape(grid_h // merge, merge, 1, 1) * np.ones((1, 1, grid_w // merge, merge), np.int64)
    cols = np.ones((grid_h // merge, merge, 1, 1), np.int64) * np.arange(grid_w).reshape(1, 1, grid_w // merge, merge)
    return rows.transpose(0, 2, 1, 3).reshape(-1), cols.transpose(0, 2, 1, 3).reshape(-1)


def interpolated_pos_embed(table: np.ndarray, grid_h: int, grid_w: int, merge: int) -> np.ndarray:
    """table [side*side, C] (row-major (side x side) learned position embedding) -> [N, C] float32: bilinear interpolation to the grid with align-corners spacing
    (transformers' fast_pos_embed_interpolate), then the merge-block order."""
    side = int(round(table.shape[0] ** 0.5))
    table = np.asarray(table, np.float32)
    h_idx, w_idx = np.linspace(0, side - 1, grid_h, dtype=np.float32), np.linspace(0, side - 1, grid_w, dtype=np.float32)
    h0, w0 = h_idx.astype(np.int64), w_idx.astype(np.int64)
    h1, w1 = np.minimum(h0 + 1, side - 1), np.minimum(w0 + 1, side - 1)
    dh, dw = (h_idx - h0)[:, None], (w_idx - w0)[None, :]
    corner = lambda hi, wi: table[(hi[:, None] * side + wi[None, :]).reshape(-1)]   # noqa: E731
    pos = (corner(h0, w0) * ((1 - dh) * (1 - dw)).reshape(-1, 1) + corner(h0, w1) * ((1 - dh) * dw).reshape(-1, 1)
           + corner(h1, w0) * (dh * (1 - dw)).reshape(-1, 1) + corner(h1, w1) * (dh * dw).reshape(-1, 1)).astype(np.float32)
    rows, cols = merge_order(grid_h, grid_w, merge)
    return pos[rows * grid_w + cols]


def rotary_tables(head_dim: int, grid_h: int, grid_w: int, merge: int):
    """(cos, sin) float32 [N, head_dim]: per patch the angles of its row on the first head_dim / 4 frequencies and of its column on the next head_dim / 4, repeated for the second half of the
    head dim (rotate_half layout): [row | col | row | col]."""
    quarter = head_dim // 4
    inv = 1.0 / (ROTARY_THETA ** (np.arange(0, head_dim // 2, 2, dtype=np.float32) / (head_dim // 2)))   # [quarter]
    rows, cols = merge_order(grid_h, grid_w, merge)
    half = np.concatenate([np.outer(rows.astype(np.float32), inv), np.outer(cols.astype(np.float32), inv)], axis=1)
    assert half.shape[1] == 2 * quarter
    angles = np.concatenate([half, half], axis=1)
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def patchify(pixels: np.ndarray, grid_h: int, grid_w: int, patch: int = 16, temporal: int = 2, merge: int = 2) -> np.ndarray:
    """pixels [C, grid_h * patch, grid_w * patch] (already rescaled and normalised) -> [N, C * temporal * patch * patch] in merge-block order; a single image is repeated over the
    temporal patch (what the Qwen2-VL image processor does for an image)."""
    c = pixels.shape[0]
    if pixels.shape[1:] != (grid_h * patch, grid_w * patch):
        raise ValueError(f"the image is {pixels.shape[2]}x{pixels.shape[1]}, the grid {grid_name((grid_h, grid_w))} wants {grid_w * patch}x{grid_h * patch}")
    frames = np.broadcast_to(pixels[None], (temporal, c, grid_h * patch, grid_w * patch))                     # [T, C, H, W]
    x = frames.reshape(temporal, c, grid_h // merge, merge, patch, grid_w // merge, merge, patch)               # T, C, bh, mh, ph, bw, mw, pw
    x = x.transpose(2, 5, 3, 6, 1, 0, 4, 7)                                                                     # bh, bw, mh, mw, C, T, ph, pw
    return np.ascontiguousarray(x.reshape(grid_h * grid_w, c * temporal * patch * patch)).astype(np.float32)


DETAIL_PATCHES = {"low": 256, "auto": 576, "high": 1024}   # the patch budget of a detail level: 16x16 (64 image tokens), 24x24 (144), 32x32 (256)
UPSCALE_AREA = 1.5                                          # a preset may have at most this many times the (budget limited) patches of the picture, unless it is the smallest preset


def choose_grid(grids, width: int, height: int, patch: int = 16, detail: str = "auto") -> tuple:
    """The preset for an image of width x height pixels among `grids`, automatically (docs/models/qwen35-vision.md). `detail` low | auto | high sets the patch budget (256 / 576 / 1024 patches).
      1. size: the picture is never upscaled far: target = min(its patches, the budget); presets with more than UPSCALE_AREA x target patches are out (the smallest preset always stays);
      2. aspect ratio: the closest log aspect ratio among them (presets within 0.1 of the best count as equal);
      3. of those the one whose patch count is closest to the target (log distance, ties -> larger).
    Mirrors protocol.ts chooseGrid (tests/golden/protocol_vectors.json)."""
    if detail not in DETAIL_PATCHES:
        raise ValueError(f"unknown detail level '{detail}' ({' | '.join(DETAIL_PATCHES)})")
    patches = lambda g: g[0] * g[1]   # noqa: E731
    target = min(width * height / (patch * patch), DETAIL_PATCHES[detail])
    limit = max(UPSCALE_AREA * target, min(patches(g) for g in grids))
    fit = [g for g in grids if patches(g) <= limit]
    aspect = lambda g: abs(math.log((g[1] / g[0]) / (width / height)))   # noqa: E731
    best = min(aspect(g) for g in fit)
    near = [g for g in fit if aspect(g) <= best + 0.1]
    return min(near, key=lambda g: (abs(math.log(patches(g) / target)), -patches(g), g))


def describe_choice(grid, width: int, height: int, patch: int = 16, merge: int = 2, detail: str = "auto") -> str:
    """what choose_grid decided, for a caption: 'landscape 3:2 image -> grid 16x24 (384 x 256 px, 96 image tokens, detail auto)'"""
    ratio = width / height
    shape = "square" if abs(math.log(ratio)) < 0.1 else ("landscape" if ratio > 1 else "portrait")
    return (f"{shape} {max(width, height) / min(width, height):.2f}:1 image ({width} x {height} px) -> grid {grid_name(grid)} ({grid[1] * patch} x {grid[0] * patch} px, "
            f"{grid[0] * grid[1] // merge ** 2} image tokens, detail {detail})")

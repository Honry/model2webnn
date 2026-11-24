"""Weight quantization into the layouts the NPUW-style graphs consume.

int4, symmetric, block size 128 along the input channels, stored GROUP-MAJOR:  q [groups, out, 128] (int4, nibble-packed on disk, low nibble first),
scale [groups, out, 1] (float16).  dequantized W[n, g*128 + k] = q[g, n, k] * scale[g, n].
Convention (the one of OpenVINO / NNCF and MLAS, checked against the Qwen3 weights): the SIGNED scale is  max_element / -8,  where max_element is the
element of the block with the largest magnitude, so that element lands exactly on -8 and every block uses the whole range [-8, 7].
"""
from __future__ import annotations

import numpy as np

# A scale of exactly 0 must never reach the device: an lm_head whose dequantizeLinear had a few zero-scale rows (the padded vocabulary rows of a model) ran dramatically slower
# on the NPU. Rows / blocks whose scale would be 0 are all-zero (q - zp = 0 or q = 0), so any non-zero scale gives the same result.
ZERO_SCALE_REPLACEMENT = 1e-3

DEFAULT_GRID = (1.0, 0.985, 0.97, 0.955, 0.94, 0.92, 0.90, 0.88)   # scale multipliers tried by the error search (1.0 = the plain absmax rule)


def pack_int4(q: np.ndarray) -> np.ndarray:
    """int8 values in [-8, 7] (any shape) -> uint8 bytes, two elements per byte in row-major order, low nibble first."""
    flat = q.reshape(-1)
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, flat.dtype)])
    nibbles = (flat.astype(np.int16) & 0xF).astype(np.uint8)
    return (nibbles[0::2] | (nibbles[1::2] << 4)).astype(np.uint8)


def unpack_int4(data: np.ndarray, shape, signed: bool = True) -> np.ndarray:
    """Inverse of pack_int4: bytes -> int8 (signed) / uint8 values of the logical ``shape``."""
    b = np.asarray(data, dtype=np.uint8).reshape(-1)
    out = np.empty(b.size * 2, np.int8)
    out[0::2] = b & 0xF
    out[1::2] = b >> 4
    n = int(np.prod(shape))
    out = out[:n]
    if signed:
        out = np.where(out >= 8, out - 16, out).astype(np.int8)
    return out.reshape(shape)


def _scales_absmax(blocks: np.ndarray) -> np.ndarray:
    """blocks [..., 128] float32 -> signed scale [..., 1] = (element with the largest magnitude) / -8; 0 for all-zero blocks."""
    index = np.abs(blocks).argmax(axis=-1)[..., None]
    peak = np.take_along_axis(blocks, index, axis=-1)
    return peak / -8.0


def _quantize_with(blocks: np.ndarray, scale: np.ndarray) -> np.ndarray:
    safe = np.where(scale == 0, 1.0, scale)
    return np.clip(np.rint(blocks / safe), -8, 7)


def quantize_int4_symmetric(weight: np.ndarray, group_size: int = 128, grid=None, rows_per_chunk: int = 1024):
    """weight [N, K] (out, in; the HF ``nn.Linear.weight`` layout) -> (q [G, N, group] int8 in [-8, 7], scale [G, N, 1] float16).

    ``grid``: scale multipliers to try per block (default: only the plain rule). With several candidates, the one with the smallest squared
    reconstruction error of each block is used; the multiplier shrinks the scale, which clips the largest elements but resolves the others finer."""
    n, k = weight.shape
    if k % group_size:
        raise ValueError(f"input channels {k} is not a multiple of the group size {group_size}")
    groups = k // group_size
    grid = (1.0,) if grid is None else tuple(grid)
    q_out = np.empty((groups, n, group_size), np.int8)
    s_out = np.empty((groups, n, 1), np.float16)
    for start in range(0, n, rows_per_chunk):
        w = np.ascontiguousarray(weight[start:start + rows_per_chunk], dtype=np.float32).reshape(-1, groups, group_size)   # [r, G, 128]
        base = _scales_absmax(w)                                                                                           # [r, G, 1]
        best_error = None
        best_scale = base
        for multiplier in grid:
            # the scale is stored as float16: search with the value the device will see
            scale = (base * multiplier).astype(np.float16).astype(np.float32)
            error = np.square(w - _quantize_with(w, scale) * scale).sum(axis=-1, keepdims=True)
            if best_error is None:
                best_error, best_scale = error, scale
            else:
                better = error < best_error
                best_error = np.where(better, error, best_error)
                best_scale = np.where(better, scale, best_scale)
        scale16 = best_scale.astype(np.float16)
        scale16 = np.where(scale16 == 0, np.float16(ZERO_SCALE_REPLACEMENT), scale16)   # all-zero block: q is 0 whatever the scale
        q = _quantize_with(w, scale16.astype(np.float32)).astype(np.int8)
        q_out[:, start:start + w.shape[0]] = q.transpose(1, 0, 2)
        s_out[:, start:start + w.shape[0]] = scale16.transpose(1, 0, 2)
    return q_out, s_out


def dequantize_int4(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(q [G, N, group], scale [G, N, 1]) -> W [N, G*group] float32, with the scale rounded as float16 like the device does."""
    w = (q.astype(np.float32) * scale.astype(np.float32)).astype(np.float16).astype(np.float32)
    return np.ascontiguousarray(w.transpose(1, 0, 2).reshape(w.shape[1], -1))


def quantize_u8_rowwise(weight: np.ndarray, rows_per_chunk: int = 8192):
    """weight [V, E] -> (q uint8 [V, E], zero_point uint8 [V, 1], scale float16 [V, 1]): asymmetric per row,  W ~ (q - zp) * scale
    (the form of the tied embedding / lm_head table)."""
    v, e = weight.shape
    q = np.empty((v, e), np.uint8)
    zp = np.empty((v, 1), np.uint8)
    scale = np.empty((v, 1), np.float16)
    for start in range(0, v, rows_per_chunk):
        w = np.ascontiguousarray(weight[start:start + rows_per_chunk], dtype=np.float32)
        low, high = np.minimum(w.min(axis=1, keepdims=True), 0), np.maximum(w.max(axis=1, keepdims=True), 0)
        s = ((high - low) / 255.0).astype(np.float16)
        s = np.where(s == 0, np.float16(ZERO_SCALE_REPLACEMENT), s)   # all-zero row: q = zero point = 0 whatever the scale
        s32 = s.astype(np.float32)
        z = np.clip(np.rint(-low / s32), 0, 255)
        q[start:start + w.shape[0]] = np.clip(np.rint(w / s32) + z, 0, 255).astype(np.uint8)
        zp[start:start + w.shape[0]] = z.astype(np.uint8)
        scale[start:start + w.shape[0]] = s
    return q, zp, scale


def quantize_i8_symmetric_rowwise(weight: np.ndarray, rows_per_chunk: int = 8192):
    """weight [V, E] -> (q int8 [V, E] in [-127, 127], scale float16 [V, 1]): symmetric per row, scale = max|w| / 127, W ~ q * scale (no zero point).
    The alternative lm_head format: as accurate as the asymmetric uint8 one (tools/lm_head_formats.py), no zero point to carry."""
    v, e = weight.shape
    q = np.empty((v, e), np.int8)
    scale = np.empty((v, 1), np.float16)
    for start in range(0, v, rows_per_chunk):
        w = np.ascontiguousarray(weight[start:start + rows_per_chunk], dtype=np.float32)
        s = (np.abs(w).max(axis=1, keepdims=True) / 127.0).astype(np.float16)
        s = np.where(s == 0, np.float16(ZERO_SCALE_REPLACEMENT), s)   # all-zero row: q = 0 whatever the scale (a scale of 0 makes the NPU slow, see above)
        q[start:start + w.shape[0]] = np.clip(np.rint(w / s.astype(np.float32)), -127, 127).astype(np.int8)
        scale[start:start + w.shape[0]] = s
    return q, scale


def dequantize_i8_symmetric_rowwise(q, scale) -> np.ndarray:
    return (q.astype(np.float32) * scale.astype(np.float32)).astype(np.float32)


def dequantize_u8_rowwise(q, zp, scale) -> np.ndarray:
    return ((q.astype(np.float32) - zp.astype(np.float32)) * scale.astype(np.float32)).astype(np.float32)

"""RoPE tables on the host. The graphs take cos / sin as inputs ([1,1,S,head_dim], HF half-split layout: both halves hold the same angles), so everything model
specific about positions (theta, YaRN / linear scaling, the YaRN attention factor) lives here and in the runtime's table builder, never in a graph.

Mirrors transformers' ``modeling_rope_utils`` (checked against transformers 5.x in tests/test_rope.py)."""
from __future__ import annotations

import math

import numpy as np


def default_inv_freq(theta: float, dim: int) -> np.ndarray:
    return 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim))


def linear_inv_freq(theta: float, dim: int, factor: float) -> np.ndarray:
    return default_inv_freq(theta, dim) / factor


def yarn_inv_freq(theta: float, dim: int, factor: float, original_max_position_embeddings: int, beta_fast: float = 32.0, beta_slow: float = 1.0,
                  truncate: bool = True, attention_factor=None, mscale=None, mscale_all_dim=None):
    """-> (inv_freq [dim/2], attention_factor). The attention factor multiplies cos and sin (so q and k, and the logits by its square)."""
    def get_mscale(scale, m=1.0):
        return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0

    if attention_factor is None:
        attention_factor = float(get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)) if (mscale and mscale_all_dim) else get_mscale(factor)

    def correction_dim(rotations):
        return (dim * math.log(original_max_position_embeddings / (rotations * 2 * math.pi))) / (2 * math.log(theta))

    low, high = correction_dim(beta_fast), correction_dim(beta_slow)
    if truncate:
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001   # prevent a singularity
    ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / (high - low), 0, 1)
    extrapolation_factor = 1 - ramp
    pos_freqs = theta ** (np.arange(0, dim, 2, dtype=np.float32) / dim)
    inv_freq = (1.0 / (factor * pos_freqs)) * (1 - extrapolation_factor) + (1.0 / pos_freqs) * extrapolation_factor
    return inv_freq.astype(np.float32), attention_factor


def proportional_inv_freq(theta: float, dim: int, partial_rotary_factor: float) -> np.ndarray:
    """Gemma 4 full attention: only the first `partial_rotary_factor` of the frequencies rotate (the exponent still divides by the FULL head dim); the others are 0 (cos 1, sin 0)."""
    angles = int(partial_rotary_factor * dim // 2)
    rotated = 1.0 / (theta ** (np.arange(0, 2 * angles, 2, dtype=np.float32) / dim))
    return np.concatenate([rotated, np.zeros(dim // 2 - angles, np.float32)]).astype(np.float32)


def inv_freq_for(rope: dict, head_dim: int):
    """rope: a resolved entry of model.json ("type", "theta", ...) -> (inv_freq, attention_factor)."""
    kind, theta = rope["type"], float(rope["theta"])
    if kind == "default":
        return default_inv_freq(theta, head_dim), 1.0
    if kind == "linear":
        return linear_inv_freq(theta, head_dim, float(rope["factor"])), 1.0
    if kind == "proportional":
        return proportional_inv_freq(theta, head_dim, float(rope.get("partial_rotary_factor", 1.0))), 1.0
    if kind == "yarn":
        return yarn_inv_freq(theta, head_dim, float(rope["factor"]), int(rope["original_max_position_embeddings"]), float(rope.get("beta_fast", 32.0)),
                             float(rope.get("beta_slow", 1.0)), bool(rope.get("truncate", True)), rope.get("attention_factor"), rope.get("mscale"),
                             rope.get("mscale_all_dim"))
    raise NotImplementedError(f"rope type '{kind}' is not supported yet")


def cos_sin(positions, inv_freq: np.ndarray, attention_factor: float = 1.0):
    """positions [S] -> (cos, sin) float32 [S, head_dim]; the two halves repeat the same angles (rotate-half layout)."""
    angles = np.outer(np.asarray(positions, dtype=np.float32), inv_freq)
    emb = np.concatenate([angles, angles], axis=-1)
    return (np.cos(emb) * attention_factor).astype(np.float32), (np.sin(emb) * attention_factor).astype(np.float32)

"""The host side of the runtime protocol as pure functions: everything a runtime has to write into the graph inputs of one dispatch chain.
The TS runtime (src/runtime/protocol.ts) implements the same functions; tests/golden/protocol_vectors.json (generated from here) keeps the two identical.

Conventions:
  * the KV cache has P = max_length - 1 rows, cache row j holds the key / value of position j; the staging buffer has R rows;
  * masks are additive, float16: 0 = visible, -65504 = hidden; one mask and one rope table per LAYER TYPE (sliding vs full attention);
  * prefill chunk of C rows, the last chunk is LEFT-padded: [pad rows | real rows]; pad rows see only themselves, their RoPE position is the first real position;
    prefill mask columns = [cache rows 0..P-1 | chunk rows 0..C-1];
  * decode (staged KV): mask columns = [main cache rows 0..P-1 | staging rows 0..R-1]; the new token goes into staging row `filled`, which holds position base+filled.
"""
from __future__ import annotations

import numpy as np

from ..frontends.hf_llm import rope as rope_lib
from ..frontends.hf_llm.arch import ModelSpec

NEG = np.float16(-65504)


def visible(spec: ModelSpec, layer_type: str, q_pos: int, k_pos: int) -> bool:
    if k_pos > q_pos:
        return False
    return layer_type != "sliding_attention" or q_pos - k_pos < spec.sliding_window


def rope_tables(spec: ModelSpec, positions: list, rows: int, pad: int) -> dict:
    """layer type -> (cos, sin) float16 [1,1,rows,head_dim]; the first `pad` rows reuse the first real position."""
    out = {}
    for ltype, entry in spec.rope.items():
        inv, factor = rope_lib.inv_freq_for(entry, spec.head_dim)
        cos, sin = rope_lib.cos_sin(positions, inv, factor)
        index = np.maximum(0, np.arange(rows) - pad)
        out[ltype] = (cos[index][None, None].astype(np.float16), sin[index][None, None].astype(np.float16))
    return out


def prefill_masks(spec: ModelSpec, P: int, C: int, start: int, valid: int) -> dict:
    """layer type -> mask float16 [1,1,C,P+C] of the chunk that holds `valid` real tokens (positions start..start+valid-1)."""
    pad = C - valid
    masks = {}
    for ltype in sorted(set(spec.layer_types)):
        mask = np.full((C, P + C), NEG, np.float16)
        for row in range(C):
            if row < pad:
                mask[row, P + row] = 0           # pad row sees only itself
                continue
            q_pos = start + (row - pad)
            for j in range(start):                # cache rows of earlier chunks (cache row j holds position j)
                if visible(spec, ltype, q_pos, j):
                    mask[row, j] = 0
            for c in range(pad, row + 1):         # real rows of this chunk up to the row itself
                if visible(spec, ltype, q_pos, start + (c - pad)):
                    mask[row, P + c] = 0
        masks[ltype] = mask[None, None]
    return masks


def prefill_selectors(P: int, C: int, start: int, valid: int):
    """(rows_to_keep [1,1,P,1], row_selector [1,1,P,C]) float16: chunk row pad+i writes cache row start+i."""
    pad = C - valid
    keep = np.ones((1, 1, P, 1), np.float16)
    selector = np.zeros((1, 1, P, C), np.float16)
    for i in range(valid):
        keep[0, 0, start + i, 0] = 0
        selector[0, 0, start + i, pad + i] = 1
    return keep, selector


def decode_masks(spec: ModelSpec, P: int, R: int, pos: int, base: int, filled: int) -> dict:
    """layer type -> mask float16 [1,1,1,P+R] for the token at position `pos` = base + filled."""
    masks = {}
    for ltype in sorted(set(spec.layer_types)):
        mask = np.full((P + R,), NEG, np.float16)
        for j in range(base):
            if visible(spec, ltype, pos, j):
                mask[j] = 0
        for i in range(filled + 1):
            if visible(spec, ltype, pos, base + i):
                mask[P + i] = 0
        masks[ltype] = mask.reshape(1, 1, 1, -1)
    return masks


def decode_selectors(R: int, filled: int):
    """(staging_rows_to_keep [1,1,R,1], staging_row_selector [1,1,R,1]): the new token goes into staging row `filled`."""
    keep = np.ones((1, 1, R, 1), np.float16)
    selector = np.zeros((1, 1, R, 1), np.float16)
    keep[0, 0, filled, 0], selector[0, 0, filled, 0] = 0, 1
    return keep, selector


def flush_selectors(P: int, R: int, base: int):
    """(rows_to_keep [1,1,P,1], row_selector [1,1,P,R]): staging row i -> main cache row base+i (rows beyond the cache are dropped)."""
    keep = np.ones((1, 1, P, 1), np.float16)
    selector = np.zeros((1, 1, P, R), np.float16)
    for i in range(R):
        if base + i < P:
            keep[0, 0, base + i, 0] = 0
            selector[0, 0, base + i, i] = 1
    return keep, selector

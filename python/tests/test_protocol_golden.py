"""Golden vectors of the host-side runtime protocol (tests/golden/protocol_vectors.json), read by the TS tests (src/runtime/protocol.test.ts) too."""
import json
import os
from pathlib import Path

import numpy as np

from m2w.frontends.hf_llm import rope as rope_lib, vision as V
from m2w.frontends.hf_llm.arch import spec_from_config
from m2w.reference import protocol

from tiny_model import CONFIG
from tiny_qwen35 import CONFIG as QWEN35_CONFIG

GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "golden" / "protocol_vectors.json"
P, C, R = 23, 8, 4
PS = 40   # cache rows of the tensorScatter vectors (the padded prompt needs rows beyond the window)


def visible_columns(mask):   # [rows, cols] f16 -> per row the indices that are 0
    return [np.flatnonzero(row == 0).tolist() for row in mask.reshape(-1, mask.shape[-1])]


def f(arr, digits=6):
    return np.round(arr.astype(np.float32).reshape(-1), digits).tolist()


def build():
    spec = spec_from_config(CONFIG)
    out = {"spec": {"layer_types": spec.layer_types, "sliding_window": spec.sliding_window, "head_dim": spec.head_dim, "rope": spec.rope}, "P": P, "C": C, "R": R, "PS": PS}
    rope = protocol.rope_tables(spec, [5, 6, 7], C, 5)      # a left-padded chunk: 5 pad rows, 3 real rows at positions 5..7
    out["rope_padded_chunk"] = {"positions": [5, 6, 7], "rows": C, "pad": 5, **{t: {"cos": f(c), "sin": f(s)} for t, (c, s) in rope.items()}}
    out["prefill_full_chunk"] = {"start": 0, "valid": 8, "masks": {t: visible_columns(m) for t, m in protocol.prefill_masks(spec, P, C, 0, 8).items()}}
    masks = protocol.prefill_masks(spec, P, C, 16, 5)        # start 16, 5 real rows -> 3 pad rows; sliding window 16 hides cache rows < qpos - 15
    out["prefill_padded_chunk"] = {"start": 16, "valid": 5, "masks": {t: visible_columns(m) for t, m in masks.items()}}
    keep, selector = protocol.prefill_selectors(P, C, 16, 5)
    out["prefill_padded_selectors"] = {"rows_to_keep": f(keep), "row_selector_ones": np.argwhere(selector.reshape(P, C) == 1).tolist()}
    out["decode"] = {"pos": 23, "base": 21, "filled": 2,
                     "masks": {t: visible_columns(m) for t, m in protocol.decode_masks(spec, P, R, 23, 21, 2).items()}}
    keep, selector = protocol.decode_selectors(R, 2)
    out["decode_selectors"] = {"rows_to_keep": f(keep), "row_selector": f(selector)}
    for base in (20, 22):
        keep, selector = protocol.flush_selectors(P, R, base)
        out[f"flush_{base}"] = {"base": base, "rows_to_keep": f(keep), "row_selector_ones": np.argwhere(selector.reshape(P, R) == 1).tolist()}
    # onehot KV: decode writes cache row `pos`
    out["onehot_decode"] = {"pos": 22, "masks": {t: visible_columns(m) for t, m in protocol.decode_masks_onehot(spec, P, 22).items()}}
    keep, selector = protocol.decode_selectors_onehot(P, 22)
    out["onehot_decode_selectors"] = {"rows_to_keep": f(keep), "row_selector_ones": np.argwhere(selector.reshape(P, 1) == 1).tolist()}
    # tensorScatter KV: the prompt padding is in front (3 pad rows), every chunk is full
    out["scatter_prefill_pad_chunk"] = {"start": 0, "pad": 3, "masks": {t: visible_columns(m) for t, m in protocol.scatter_prefill_masks(spec, PS, C, 0, 3).items()}}
    out["scatter_prefill_later_chunk"] = {"start": 16, "pad": 3, "masks": {t: visible_columns(m) for t, m in protocol.scatter_prefill_masks(spec, PS, C, 16, 3).items()}}
    for name, (start, pad) in {"pad": (0, 3), "later": (16, 3)}.items():
        tables = protocol.scatter_prefill_rope(spec, C, start, pad)
        out[f"scatter_prefill_rope_{name}"] = {"start": start, "pad": pad, **{t: {"cos": f(c), "sin": f(s_)} for t, (c, s_) in tables.items()}}
    out["scatter_decode"] = {"pos": 20, "pad": 3, "masks": {t: visible_columns(m) for t, m in protocol.scatter_decode_masks(spec, PS, 20, 3).items()}}
    # scatterND KV: real rows go to their position, the pad rows of the last chunk are parked behind them (clamped to P-1)
    out["scatternd_prefill"] = {"start": 16, "valid": 5, "kv_heads": 2, "masks": {t: visible_columns(m) for t, m in protocol.scatternd_prefill_masks(spec, P, C, 16, 5).items()},
                                "indices": protocol.scatternd_prefill_indices(2, P, C, 16, 5).reshape(-1).tolist()}
    out["scatternd_decode"] = {"pos": 20, "kv_heads": 2, "masks": {t: visible_columns(m) for t, m in protocol.scatternd_decode_masks(spec, P, 20).items()},
                               "indices": protocol.scatternd_decode_indices(2, 20).reshape(-1).tolist()}
    # attention over another layer's updated cache (KV sharing): pad rows see row 0
    out["incache_prefill"] = {"start": 16, "valid": 5, "masks": {t: visible_columns(m) for t, m in protocol.incache_prefill_masks(spec, P, C, 16, 5).items()}}
    # prefill planning
    out["plans"] = [{"sizes": sizes, "tokens": n, "fixed": fixed, "plan": protocol.plan_prefill(sizes, n, fixed)}
                    for sizes, n, fixed in (([64], 20, 200), ([64], 130, 200), ([64, 256], 20, 200), ([64, 256], 300, 200), ([64, 256], 200, 200), ([32, 64, 128], 95, 50), ([8, 16], 21, 2))]
    # RoPE: default and YaRN tables of the real Aion entries
    aion = {"sliding_attention": {"type": "default", "theta": 32000.0},
            "full_attention": {"type": "yarn", "theta": 32000.0, "factor": 16.0, "original_max_position_embeddings": 2048, "beta_fast": 32.0, "beta_slow": 1.0}}
    out["rope_aion"] = {}
    for name, entry in aion.items():
        inv, factor = rope_lib.inv_freq_for(entry, 128)
        cos, sin = rope_lib.cos_sin([0, 1, 100, 2047], inv, factor)
        out["rope_aion"][name] = {"entry": entry, "head_dim": 128, "positions": [0, 1, 100, 2047], "inv_freq": f(inv, 9), "attention_factor": float(factor),
                                  "cos": f(cos, 5), "sin": f(sin, 5)}
    out["vision"] = vision_vectors()
    return out


def vision_vectors() -> dict:
    """image prompts of Qwen3.5: grid choice, pad expansion, (t, h, w) positions, interleaved MRoPE tables (theta 10 so that every axis moves the angles), patchify"""
    import dataclasses

    spec = spec_from_config(QWEN35_CONFIG)
    spec = dataclasses.replace(spec, rope={"full_attention": {**spec.rope["full_attention"], "theta": 10.0}})
    grids = [(4, 4), (4, 6), (6, 4)]
    short = [4, 5, 288, 290, 289, 6, 288, 290, 289, 7, 8]
    tokens = protocol.expand_image_tokens(spec, short, grids[1:])
    positions, delta = protocol.mrope_positions(spec, tokens, grids[1:])
    tables = protocol.rope_tables_mrope(spec, positions[:, 3:11], 12, 4)["full_attention"]   # a left-padded chunk: 4 pad rows + 8 real rows (positions of tokens 3..10)
    rng = np.random.default_rng(0)
    pixels = rng.integers(-8, 9, (3, 8, 12)).astype(np.float32)
    return {"spec": {"head_dim": spec.head_dim, "rope": spec.rope, "mrope": spec.mrope, "image_token_id": spec.vision["image_token_id"], "merge_size": spec.vision["merge_size"]},
            "choose": [{"grids": [list(g) for g in V.DEFAULT_GRIDS], "width": w, "height": h, "patch": 16, "grid": list(V.choose_grid(V.DEFAULT_GRIDS, w, h))}
                       for w, h in ((1000, 1000), (400, 400), (200, 200), (1824, 2358), (1500, 1000), (1000, 1500), (1280, 960), (640, 480), (4000, 3000))],
            "prompt": {"short": short, "grids": [list(g) for g in grids[1:]], "expanded": tokens, "positions": positions.tolist(), "rope_delta": delta},
            "rope_chunk": {"first": 3, "valid": 8, "rows": 12, "pad": 4, "cos": f(tables[0]), "sin": f(tables[1])},
            "axes": {"section": [11, 11, 10], "count": 32, "axes": protocol.mrope_axes([11, 11, 10], 32).tolist()},
            "patchify": {"grid": [4, 6], "patch": 2, "temporal": 2, "merge": 2, "pixels": pixels.reshape(-1).tolist(), "patches": V.patchify(pixels, 4, 6, 2, 2, 2).reshape(-1).tolist()}}


def test_protocol_vectors_are_current():
    text = json.dumps(build(), separators=(",", ":"))
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.write_text(text + "\n", encoding="utf-8", newline="\n")
    assert GOLDEN.read_text(encoding="utf-8").strip() == text, "stale: run `UPDATE_GOLDEN=1 pytest tests/test_protocol_golden.py`"

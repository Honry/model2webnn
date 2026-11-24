import json
import os
from pathlib import Path

import numpy as np
import pytest

from m2w.quantize import (DEFAULT_GRID, dequantize_i8_symmetric_rowwise, dequantize_int4, dequantize_u8_rowwise, pack_int4, quantize_i8_symmetric_rowwise, quantize_int4_symmetric,
                          quantize_u8_rowwise, unpack_int4)

# the weights extracted from the OpenVINO Qwen3-4B IR (manifest.json + lN.bin): set QWEN3_WEIGHTS to run the format regression below (skipped otherwise)
QWEN3_WEIGHTS = Path(os.environ.get("QWEN3_WEIGHTS", "qwen3_4b_weights_not_set"))


def test_pack_unpack_roundtrip_and_nibble_order():
    q = np.array([-8, 7, 0, -1, 3, 5], np.int8)
    packed = pack_int4(q)
    assert packed.tolist() == [0x78, 0xF0, 0x53]    # low nibble = first element
    np.testing.assert_array_equal(unpack_int4(packed, (6,)), q)
    rng = np.random.default_rng(0)
    big = rng.integers(-8, 8, (3, 5, 128)).astype(np.int8)
    np.testing.assert_array_equal(unpack_int4(pack_int4(big), big.shape), big)


def test_symmetric_int4_uses_full_range_with_signed_scale():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((16, 256)).astype(np.float32)
    q, s = quantize_int4_symmetric(w)
    assert q.shape == (2, 16, 128) and s.shape == (2, 16, 1) and q.dtype == np.int8 and s.dtype == np.float16
    assert q.min() == -8 and q.max() <= 7
    assert (np.abs(q).max(-1) == 8).all()             # the largest element of every block sits on -8
    rec = dequantize_int4(q, s)
    # half a step everywhere, except the largest element of a block with a positive peak: it maps to +8 and is clipped to +7 (error <= one step)
    assert np.abs(rec - w).max() <= np.abs(s.astype(np.float32)).max() * 1.0 + 1e-3
    inner = np.abs(rec - w).reshape(16, 2, 128)
    assert np.median(inner) <= np.abs(s.astype(np.float32)).mean() * 0.3


def test_error_search_never_increases_the_error():
    rng = np.random.default_rng(2)
    w = (rng.standard_normal((64, 512)) * rng.uniform(0.5, 2, (64, 1))).astype(np.float32)
    plain = np.square(dequantize_int4(*quantize_int4_symmetric(w)) - w).sum()
    searched = np.square(dequantize_int4(*quantize_int4_symmetric(w, grid=DEFAULT_GRID)) - w).sum()
    assert searched <= plain * (1 + 1e-6)
    assert searched < plain * 0.97   # gaussian weights clip well: the search must find something


def test_u8_rowwise_asymmetric():
    rng = np.random.default_rng(3)
    w = (rng.standard_normal((50, 96)) * 0.1 + 0.02).astype(np.float32)
    q, zp, s = quantize_u8_rowwise(w)
    rec = dequantize_u8_rowwise(q, zp, s)
    assert q.dtype == np.uint8 and zp.shape == (50, 1) and s.dtype == np.float16
    assert np.abs(rec - w).max() <= s.astype(np.float32).max() * 0.6


@pytest.mark.skipif(not (QWEN3_WEIGHTS / "manifest.json").exists(), reason="Qwen3 OpenVINO-extracted weights not available")
def test_requantizing_the_openvino_weights_reproduces_them():
    """Format regression against the weights the runtime already uses: layout, nibble order, signed-scale convention."""
    manifest = json.loads((QWEN3_WEIGHTS / "manifest.json").read_text())
    layer = manifest["layers"][0]
    raw = (QWEN3_WEIGHTS / layer["file"]).read_bytes()
    tensors = {t["name"]: t for t in layer["tensors"]}
    for prefix in ("key", "down_projection"):
        wt = tensors[f"{prefix}_weight" if prefix != "down_projection" else "down_projection_weight"]
        st = tensors[wt["name"].replace("_weight", "_scale")]
        q_ref = unpack_int4(np.frombuffer(raw, np.uint8, wt["bytes"], wt["offset"]), tuple(wt["shape"]))
        s_ref = np.frombuffer(raw, np.float16, int(np.prod(st["shape"])), st["offset"]).reshape(st["shape"])
        w = dequantize_int4(q_ref, s_ref)                       # [N, K] the weight the NPU sees
        q, s = quantize_int4_symmetric(w)                       # our quantizer, plain rule
        same_q = (q == q_ref).mean()
        assert same_q > 0.999, f"{wt['name']}: only {same_q:.4%} of the int4 values are reproduced"
        np.testing.assert_allclose(s.astype(np.float32), s_ref.astype(np.float32), rtol=2e-3)
        assert (pack_int4(q_ref) == np.frombuffer(raw, np.uint8, wt["bytes"], wt["offset"])).all()   # packing is byte-exact


def test_zero_scales_never_reach_the_device():
    """A zero scale can make the NPU lm_head dramatically slower (padded vocabulary rows). All-zero rows / blocks get a small non-zero scale and still dequantize to 0."""
    w = np.random.default_rng(5).standard_normal((8, 256)).astype(np.float32)
    w[3] = 0                                        # an all-zero row
    q, zp, s = quantize_u8_rowwise(w)
    assert (s > 0).all() and dequantize_u8_rowwise(q, zp, s)[3].max() == 0 and dequantize_u8_rowwise(q, zp, s)[3].min() == 0
    wi = np.random.default_rng(6).standard_normal((4, 256)).astype(np.float32)
    wi[:, :128] = 0                                 # all-zero blocks
    qi, si = quantize_int4_symmetric(wi)
    assert (si.astype(np.float32) != 0).all() and not dequantize_int4(qi, si)[:, :128].any()


def test_i8_symmetric_rowwise_is_as_accurate_as_u8_asymmetric_and_never_has_a_zero_scale():
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((200, 128)) * 0.05 + 0.01).astype(np.float32)
    w[5] = 0                                                       # a dead row
    q, s = quantize_i8_symmetric_rowwise(w)
    assert q.dtype == np.int8 and q.min() >= -127 and s.dtype == np.float16 and (s > 0).all()
    rec = dequantize_i8_symmetric_rowwise(q, s)
    assert not rec[5].any()
    asym = dequantize_u8_rowwise(*quantize_u8_rowwise(w))
    err = lambda r: np.linalg.norm(r - w) / np.linalg.norm(w)   # noqa: E731
    assert err(rec) < 0.01 and err(rec) < 1.3 * err(asym)

import numpy as np
import pytest

from m2w.ir import GraphBuilder, GraphError
from m2w.ir import serialize
from m2w.ir.interpreter import run


def test_shape_rules_reject_bad_graphs():
    b = GraphBuilder("t")
    x = b.input("x", "float32", [2, 3])
    y = b.input("y", "float16", [2, 3])
    with pytest.raises(GraphError):
        b.add(x, y)  # dtype mismatch
    with pytest.raises(GraphError):
        b.reshape(x, [4, 2])  # element count changes
    with pytest.raises(GraphError):
        b.matmul(x, x)  # inner dims differ
    with pytest.raises(GraphError):
        b.slice(x, [0, 1], [2, 3])  # out of range
    with pytest.raises(GraphError):
        b.output("o", x)  # a graph input cannot be an output


def test_interpreter_matches_numpy_and_roundtrips_json():
    rng = np.random.default_rng(0)
    b = GraphBuilder("t")
    x = b.input("x", "float32", [1, 4, 8])
    w = b.input("w", "int4", [2, 6, 4], "weight")   # groups of 4 along the last axis
    s = b.input("s", "float16", [2, 6, 1], "weight")
    zp = b.constant("int4", [2, 6, 1], fill=0)
    dq = b.dequantize_linear(w, s, zp)                        # f16 [2,6,4]
    mat = b.cast(b.reshape(b.transpose(dq, [1, 0, 2]), [6, 8]), "float32")
    y = b.gemm(b.reshape(x, [4, 8]), mat, b_transpose=True)   # [4,6]
    out = b.softmax(b.add(y, b.scalar(0.5, "float32", 2)), 1)
    b.output("out", out)
    g = b.build()

    xv = rng.standard_normal((1, 4, 8)).astype(np.float32)
    wv = rng.integers(-8, 8, (2, 6, 4)).astype(np.int8)
    sv = rng.uniform(0.05, 0.2, (2, 6, 1)).astype(np.float16)
    got = run(g, {"x": xv, "w": wv, "s": sv})["out"]

    deq = (wv.astype(np.float32) * sv.astype(np.float32)).astype(np.float16).astype(np.float32)
    mat_ref = deq.transpose(1, 0, 2).reshape(6, 8)
    logits = xv.reshape(4, 8) @ mat_ref.T + 0.5
    e = np.exp(logits - logits.max(1, keepdims=True))
    np.testing.assert_allclose(got, e / e.sum(1, keepdims=True), rtol=1e-5, atol=1e-6)

    g2 = serialize.loads(serialize.dumps(g))   # JSON roundtrip re-derives every shape through the same rules
    np.testing.assert_array_equal(run(g2, {"x": xv, "w": wv, "s": sv})["out"], got)


def test_tensor_scatter_writes_rows_at_index():
    b = GraphBuilder("t")
    cache = b.input("cache", "float16", [1, 2, 6, 4], "kv")
    rows = b.input("rows", "float16", [1, 2, 2, 4])
    index = b.input("index", "int32", [1], "control")
    b.output("out", b.tensor_scatter(cache, rows, index, axis=2))
    g = b.build()
    cv = np.zeros((1, 2, 6, 4), np.float16)
    rv = np.ones((1, 2, 2, 4), np.float16)
    out = run(g, {"cache": cv, "rows": rv, "index": np.array([3], np.int32)})["out"]
    assert out[0, :, 3:5].min() == 1 and out[0, :, :3].max() == 0 and out[0, :, 5:].max() == 0


def test_scatter_nd_shape_rule_and_interpreter():
    import numpy as np
    import pytest

    from m2w.ir.builder import GraphBuilder, GraphError
    from m2w.ir.interpreter import run

    b = GraphBuilder("scatter")
    cache = b.input("cache", "float16", [1, 2, 6, 4])
    indices = b.input("indices", "int32", [1, 2, 3, 3])
    rows = b.input("rows", "float16", [1, 2, 3, 4])
    b.output("out", b.scatter_nd(cache, indices, rows))
    g = b.build()
    new = np.arange(24, dtype=np.float16).reshape(1, 2, 3, 4) + 1
    idx = np.array([[[[0, 0, 4], [0, 0, 1], [0, 0, 2]], [[0, 1, 0], [0, 1, 5], [0, 1, 3]]]], np.int32)   # (batch, head, row) of every new row, in any order
    out = run(g, {"cache": np.zeros((1, 2, 6, 4), np.float16), "indices": idx, "rows": new})["out"]
    assert (out[0, 0, 4] == new[0, 0, 0]).all() and (out[0, 0, 1] == new[0, 0, 1]).all() and (out[0, 1, 5] == new[0, 1, 1]).all()
    assert (out[0, 0, 0] == 0).all() and (out[0, 1, 4] == 0).all()   # rows that are not indexed stay
    b = GraphBuilder("bad")
    with pytest.raises(GraphError):
        b.scatter_nd(b.input("c", "float16", [1, 2, 6, 4]), b.input("i", "int32", [1, 2, 3, 3]), b.input("u", "float16", [1, 2, 3, 5]))   # updates must end with the head dim


def test_tensor_size_limit_of_webnn():
    """WebNN rejects tensors above 2^31 - 1 bytes: the dequantized lm_head of a 262144-row vocabulary in float32 (2.7 GB) is caught when the graph is written."""
    import pytest

    from m2w.frontends.hf_llm.builders import decoder as D
    from m2w.ir import GraphError, check_tensor_limit

    dims = D.DecoderDims(hidden=2560, num_heads=8, num_kv_heads=2, head_dim=256, ffn=10240, vocab=262144)
    with pytest.raises(GraphError, match=r"above the WebNN limit.*2\.5\d GiB"):
        check_tensor_limit(D.build_lm_head_graph(dims, D.GraphOptions(compute_type="float32")))
    check_tensor_limit(D.build_lm_head_graph(dims, D.GraphOptions(compute_type="float16")))   # 1.25 GiB: fine
    check_tensor_limit(D.build_lm_head_graph(D.DecoderDims(hidden=2560, num_heads=8, num_kv_heads=2, head_dim=256, ffn=10240, vocab=151936), D.GraphOptions()))   # Qwen3: 1.45 GiB


def test_clamp_and_the_float16_safe_gelu():
    """clamp bounds the polynomial of the vision GELU (x^3 would overflow float16): equal to the exact tanh GELU for any x, finite in float16"""
    from m2w.frontends.hf_llm.builders import qwen35_vision as QV
    from m2w.frontends.hf_llm.builders.decoder import Ctx, DecoderDims, GraphOptions

    for compute in ("float32", "float16"):
        b = GraphBuilder("g")
        c = Ctx(b, DecoderDims(hidden=128, num_heads=2, num_kv_heads=2, head_dim=64, ffn=128, vocab=0), GraphOptions(compute_type=compute))
        x = b.input("x", compute, [1, 1, 9])
        b.output("y", QV.gelu(c, x))
        xs = np.array([-700, -40, -3, -0.5, 0, 0.5, 3, 40, 700], np.float32).reshape(1, 1, 9)
        y = run(b.build(), {"x": xs.astype(np.float16 if compute == "float16" else np.float32)})["y"].astype(np.float32)
        want = 0.5 * xs * (1 + np.tanh(np.sqrt(2 / np.pi) * (xs + 0.044715 * xs ** 3)))
        assert np.isfinite(y).all() and np.abs(y - want).max() < (0.5 if compute == "float16" else 1e-3), (compute, y, want)

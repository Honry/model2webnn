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

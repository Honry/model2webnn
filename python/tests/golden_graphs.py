"""A tiny graph that uses every operator of the IR once. Its JSON is committed under tests/golden/ and read by BOTH the Python and the TS tests."""
from m2w.ir import GraphBuilder


def tiny_all_ops():
    b = GraphBuilder("tiny_all_ops")
    x = b.input("x", "float32", [1, 4, 8])
    q = b.input("q", "int4", [2, 6, 4], "weight")
    s = b.input("s", "float16", [2, 6, 1], "weight")
    cache = b.input("cache", "float16", [1, 2, 6, 4], "kv")
    rows = b.input("rows", "float16", [1, 2, 2, 4])
    index = b.input("index", "int32", [1], "control")

    zp = b.constant("int4", [2, 6, 1], fill=0)
    dq = b.dequantize_linear(q, s, zp)
    dq_no_zp = b.dequantize_linear(q, s, None)
    matrix = b.cast(b.reshape(b.transpose(b.add(dq, dq_no_zp), [1, 0, 2]), [6, 8]), "float32")
    y = b.gemm(b.reshape(x, [4, 8]), matrix, b_transpose=True)                        # [4,6]
    m = b.matmul(b.reshape(y, [1, 4, 6]), b.reshape(matrix, [1, 6, 8]))             # [1,4,8]
    n = b.mul(m, b.div(b.scalar(1, "float32", 3), b.sqrt(b.add(b.reduce_mean(b.pow(m, b.scalar(2, "float32", 3)), [2], True), b.scalar(1e-6, "float32", 3)))))
    p = b.softmax(b.sub(n, b.reduce_max(n, [2], True)), 2)
    r = b.reduce_sum(b.mul(p, b.sigmoid(n)), [1], False)                              # [1,8]
    parts = b.concat([b.slice(r, [0, 0], [1, 4]), b.slice(r, [0, 4], [1, 4])], 1)     # [1,8]
    e = b.reshape(b.expand(b.reshape(parts, [1, 1, 1, 8]), [1, 2, 3, 8]), [1, 6, 8])
    updated = b.tensor_scatter(cache, rows, index, axis=2)
    b.output("e", e)
    b.output("updated", updated)
    b.output("half_const", b.mul(b.cast(parts, "float16"), b.scalar(1e-6, "float16", 2)))
    return b.build()

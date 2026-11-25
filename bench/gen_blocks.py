#!/usr/bin/env python
"""Generate synthetic Qwen3-4B-shaped decoder blocks (static shape, int4 block-128, random weights)
for the NPU graph-splitting benchmark.

  python gen_blocks.py --layers 0:12 --query 1 --weights dq_const --out models
  python gen_blocks.py --kind tiny --out models

Per-block interface (all fp16, batch 1):
  inputs : hidden [1,Q,E]  cos [1,1,Q,D]  sin [1,1,Q,D]  mask [1,1,Q,T+Q]
           past_k.<i> / past_v.<i> [1,KVH,T,D]            (one pair per layer in the block)
  outputs: hidden_out [1,Q,E]
           present_k.<i> / present_v.<i> [1,KVH,Q,D]       (new slice only, NPUW RedirectNewKvToOutput style)
Weights forms (--weights):
  matmulnbits : com.microsoft MatMulNBits, B/scales are initializers (what the ORT exporters emit)
  dq_const    : DequantizeLinear(int4 init, block_size=128) + MatMul, weights are initializers
  dq_input    : same graph, but every int4 weight and scale is a graph *input* (weights-as-inputs test)
Random weights give the same compile/run cost as real ones; numerics are only kept finite.
"""
import argparse
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto as TP
from onnx import helper as H

E, NH, KVH, D, FF = 2560, 32, 8, 128, 9728  # Qwen3-4B
BLOCK = 128
EPS = 1e-6
F16 = TP.FLOAT16
rng = np.random.default_rng(1234)


class G:
    """Tiny graph builder."""

    def __init__(self, weights):
        self.nodes, self.inits, self.inputs, self.outputs = [], [], [], []
        self.weights = weights
        self.n = 0

    def name(self, p):
        self.n += 1
        return f"{p}_{self.n}"

    def node(self, op, ins, n_out=1, domain="", **attrs):
        outs = [self.name(op.lower()) for _ in range(n_out)]
        self.nodes.append(H.make_node(op, ins, outs, name=self.name("n_" + op), domain=domain, **attrs))
        return outs[0] if n_out == 1 else outs

    def init_f16(self, name, arr):
        self.inits.append(H.make_tensor(name, F16, arr.shape, np.asarray(arr, np.float16).tobytes(), raw=True))
        return name

    def init_i64(self, vals):
        name = self.name("i64c")
        self.inits.append(H.make_tensor(name, TP.INT64, [len(vals)], np.asarray(vals, np.int64).tobytes(), raw=True))
        return name

    def init_scalar_f16(self, v):
        return self.init_f16(self.name("sc"), np.array(v, np.float16).reshape(()))

    def add_input(self, name, dtype, shape):
        self.inputs.append(H.make_tensor_value_info(name, dtype, shape))
        return name

    def add_output(self, name, dtype, shape):
        self.outputs.append(H.make_tensor_value_info(name, dtype, shape))


def rand_int4_bytes(numel):
    assert numel % 2 == 0
    return rng.integers(0, 256, size=numel // 2, dtype=np.uint8).tobytes()


def rand_scales(shape):
    return (0.0006 + 0.0006 * rng.random(shape)).astype(np.float16)


def linear(g, x, k, n, label):
    """x [.., k] -> [.., n] with int4 block-128 weights in the configured form."""
    kb = k // BLOCK
    if g.weights == "matmulnbits":
        b = g.name(label + "_B")
        # B: uint8 [N, K/BLOCK, BLOCK/2] (two uint4 per byte); scales f16 [N*K/BLOCK]
        g.inits.append(H.make_tensor(b, TP.UINT8, [n, kb, BLOCK // 2], rng.integers(0, 256, size=(n, kb, BLOCK // 2), dtype=np.uint8).tobytes(), raw=True))
        s = g.init_f16(g.name(label + "_s"), rand_scales((n * kb,)))
        return g.node("MatMulNBits", [x, b, s], domain="com.microsoft", K=k, N=n, bits=4, block_size=BLOCK)
    # DequantizeLinear: W int4 [K, N], scale [K/BLOCK, N], axis 0, block_size BLOCK
    wn, sn = g.name(label + "_w"), g.name(label + "_s")
    if g.weights == "dq_input":
        g.add_input(wn, TP.INT4, [k, n])
        g.add_input(sn, F16, [kb, n])
    else:
        g.inits.append(H.make_tensor(wn, TP.INT4, [k, n], rand_int4_bytes(k * n), raw=True))
        g.init_f16(sn, rand_scales((kb, n)))
    w = g.node("DequantizeLinear", [wn, sn], axis=0, block_size=BLOCK)
    return g.node("MatMul", [x, w])


def rms_norm(g, x, dim, label, shape_rank_last=True):
    w = g.init_f16(g.name(label + "_g"), (0.8 + 0.4 * rng.random(dim)).astype(np.float16))
    two = g.init_scalar_f16(2.0)
    eps = g.init_scalar_f16(EPS)
    sq = g.node("Pow", [x, two])
    ms = g.node("ReduceMean", [sq, g.init_i64([-1])], keepdims=1)
    r = g.node("Sqrt", [g.node("Add", [ms, eps])])
    return g.node("Mul", [g.node("Div", [x, r]), w])


def rotate_half(g, x, q):
    half = D // 2
    ax = g.init_i64([3])
    s0, e0, s1, e1 = g.init_i64([0]), g.init_i64([half]), g.init_i64([half]), g.init_i64([D])
    x1 = g.node("Slice", [x, s0, e0, ax])
    x2 = g.node("Slice", [x, s1, e1, ax])
    return g.node("Concat", [g.node("Neg", [x2]), x1], axis=3)


def apply_rope(g, x, cos, sin, q):
    return g.node("Add", [g.node("Mul", [x, cos]), g.node("Mul", [rotate_half(g, x, q), sin])])


def decoder_layer(g, h, i, q, t, cos, sin, mask):
    s = t + q
    # --- attention
    xn = rms_norm(g, h, E, f"l{i}_ln1")
    qp = linear(g, xn, E, NH * D, f"l{i}_q")
    kp = linear(g, xn, E, KVH * D, f"l{i}_k")
    vp = linear(g, xn, E, KVH * D, f"l{i}_v")

    def heads(x, nh):
        return g.node("Reshape", [x, g.init_i64([1, q, nh, D])])

    qh, kh, vh = heads(qp, NH), heads(kp, KVH), heads(vp, KVH)
    qh = rms_norm(g, qh, D, f"l{i}_qn")  # qk-norm (Qwen3)
    kh = rms_norm(g, kh, D, f"l{i}_kn")
    tr = lambda x: g.node("Transpose", [x], perm=[0, 2, 1, 3])
    qh, kh, vh = tr(qh), tr(kh), tr(vh)  # [1,nh,Q,D]
    qh = apply_rope(g, qh, cos, sin, q)
    kh = apply_rope(g, kh, cos, sin, q)

    for nm, t_ in ((f"present_k.{i}", kh), (f"present_v.{i}", vh)):
        g.nodes.append(H.make_node("Identity", [t_], [nm], name=g.name("n_present")))
        g.add_output(nm, F16, [1, KVH, q, D])

    pk = g.add_input(f"past_k.{i}", F16, [1, KVH, t, D])
    pv = g.add_input(f"past_v.{i}", F16, [1, KVH, t, D])
    k_all = g.node("Concat", [pk, kh], axis=2)  # [1,KVH,S,D]
    v_all = g.node("Concat", [pv, vh], axis=2)

    rep = NH // KVH

    def expand_kv(x):
        u = g.node("Unsqueeze", [x, g.init_i64([2])])  # [1,KVH,1,S,D]
        e = g.node("Expand", [u, g.init_i64([1, KVH, rep, s, D])])
        return g.node("Reshape", [e, g.init_i64([1, NH, s, D])])

    k_e, v_e = expand_kv(k_all), expand_kv(v_all)
    kt = g.node("Transpose", [k_e], perm=[0, 1, 3, 2])  # [1,NH,D,S]
    sc = g.node("MatMul", [qh, kt])
    sc = g.node("Mul", [sc, g.init_scalar_f16(1.0 / np.sqrt(D))])
    sc = g.node("Add", [sc, mask])
    p = g.node("Softmax", [sc], axis=-1)
    o = g.node("MatMul", [p, v_e])  # [1,NH,Q,D]
    o = g.node("Transpose", [o], perm=[0, 2, 1, 3])
    o = g.node("Reshape", [o, g.init_i64([1, q, NH * D])])
    o = linear(g, o, NH * D, E, f"l{i}_o")
    h = g.node("Add", [h, o])
    # --- MLP
    xn = rms_norm(g, h, E, f"l{i}_ln2")
    gate = linear(g, xn, E, FF, f"l{i}_gate")
    up = linear(g, xn, E, FF, f"l{i}_up")
    act = g.node("Mul", [g.node("Mul", [gate, g.node("Sigmoid", [gate])]), up])
    down = linear(g, act, FF, E, f"l{i}_down")
    return g.node("Add", [h, down])


def build_block(a, b, q, t, weights):
    g = G(weights)
    h = g.add_input("hidden", F16, [1, q, E])
    cos = g.add_input("cos", F16, [1, 1, q, D])
    sin = g.add_input("sin", F16, [1, 1, q, D])
    mask = g.add_input("mask", F16, [1, 1, q, t + q])
    for i in range(a, b):
        h = decoder_layer(g, h, i, q, t, cos, sin, mask)
    out = g.node("Identity", [h])
    g.outputs.insert(0, H.make_tensor_value_info(out, F16, [1, q, E]))
    # rename the graph output to a stable name
    g.nodes[-1].output[0] = "hidden_out"
    g.outputs[0].name = "hidden_out"
    return g


def build_tiny():
    g = G("dq_const")
    h = g.add_input("hidden", F16, [1, 1, E])
    one = g.init_scalar_f16(1.0)
    out = g.node("Add", [h, one])
    g.nodes[-1].output[0] = "hidden_out"
    g.add_output("hidden_out", F16, [1, 1, E])
    return g


def save(g, path, opsets):
    graph = H.make_graph(g.nodes, os.path.basename(path), g.inputs, g.outputs, g.inits)
    model = H.make_model(graph, opset_imports=[H.make_opsetid(d, v) for d, v in opsets])
    model.ir_version = 10
    data_name = os.path.basename(path) + ".data"
    if os.path.exists(path):
        os.remove(path)
    if os.path.exists(os.path.join(os.path.dirname(path), data_name)):
        os.remove(os.path.join(os.path.dirname(path), data_name))
    onnx.save(model, path, save_as_external_data=True, all_tensors_to_one_file=True, location=data_name, size_threshold=1024)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="block", choices=["block", "tiny"])
    ap.add_argument("--layers", default="0:1", help="a:b (global layer indices, b exclusive)")
    ap.add_argument("--query", type=int, default=1)
    ap.add_argument("--kv", type=int, default=512, help="past KV capacity T")
    ap.add_argument("--weights", default="dq_const", choices=["matmulnbits", "dq_const", "dq_input"])
    ap.add_argument("--out", default="models")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    global rng
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    if args.kind == "tiny":
        g = build_tiny()
        name = "tiny"
        opsets = [("", 21)]
    else:
        a, b = (int(x) for x in args.layers.split(":"))
        g = build_block(a, b, args.query, args.kv, args.weights)
        name = f"blk_{args.weights}_L{a}-{b}_q{args.query}_t{args.kv}"
        opsets = [("", 21)] + ([("com.microsoft", 1)] if args.weights == "matmulnbits" else [])
    path = os.path.join(args.out, name + ".onnx")
    save(g, path, opsets)
    meta = {
        "name": name,
        "file": name + ".onnx",
        "data": name + ".onnx.data",
        "inputs": [(v.name, TP.DataType.Name(v.type.tensor_type.elem_type), [d.dim_value for d in v.type.tensor_type.shape.dim]) for v in g.inputs],
        "outputs": [(v.name, TP.DataType.Name(v.type.tensor_type.elem_type), [d.dim_value for d in v.type.tensor_type.shape.dim]) for v in g.outputs],
    }
    with open(os.path.join(args.out, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(name, "inputs:", len(meta["inputs"]), "outputs:", len(meta["outputs"]))


if __name__ == "__main__":
    sys.exit(main())

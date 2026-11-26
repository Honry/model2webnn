"""Reference interpreter: runs a Graph with numpy. It is the numerical oracle for the compiler (no WebNN needed).

Semantics notes
  * float16 tensors stay float16 between operators (every operator rounds to float16 like the device does); reductions, softmax,
    sigmoid and matmul accumulate in float32 and round once at the end.
  * int4 / uint4 tensors are held UNPACKED (one value per int8 / uint8 element) with their logical shape.
  * After every node the result is checked against the dtype / shape the builder inferred, which cross-checks the shape rules.
"""
from __future__ import annotations

import numpy as np

from .graph import Graph

NUMPY_DTYPES = {
    "float32": np.float32, "float16": np.float16, "int32": np.int32, "int64": np.int64,
    "int8": np.int8, "uint8": np.uint8, "int4": np.int8, "uint4": np.uint8,
}


def np_dtype(dtype: str):
    return NUMPY_DTYPES[dtype]


def _f32(x):
    return x.astype(np.float32) if x.dtype == np.float16 else x


def _expand_blockwise(arr: np.ndarray, shape) -> np.ndarray:
    for axis, (have, want) in enumerate(zip(arr.shape, shape)):
        if have != want:
            arr = np.repeat(arr, want // have, axis=axis)
    return arr


def _eval(op: str, ins: list, opt: dict, out_dtype: str):
    x = ins[0] if ins else None
    if op in ("add", "sub", "mul", "div", "pow"):
        a, b = ins
        fn = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide, "pow": np.power}[op]
        with np.errstate(all="ignore"):
            return fn(a, b).astype(np_dtype(out_dtype))
    if op == "sqrt":
        return np.sqrt(_f32(x)).astype(np_dtype(out_dtype))
    if op == "sigmoid":
        return (1.0 / (1.0 + np.exp(-_f32(x)))).astype(np_dtype(out_dtype))
    if op == "tanh":
        return np.tanh(_f32(x)).astype(np_dtype(out_dtype))
    if op == "clamp":
        return np.clip(_f32(x), opt["minValue"], opt["maxValue"]).astype(np_dtype(out_dtype))
    if op == "softplus":
        return np.logaddexp(0.0, _f32(x)).astype(np_dtype(out_dtype))
    if op in ("exp", "log"):
        with np.errstate(all="ignore"):
            return (np.exp if op == "exp" else np.log)(_f32(x)).astype(np_dtype(out_dtype))
    if op == "cast":
        return x.astype(np_dtype(out_dtype))
    if op == "reshape":
        return x.reshape(opt["newShape"])
    if op == "transpose":
        return np.ascontiguousarray(np.transpose(x, opt["permutation"]))
    if op == "slice":
        index = tuple(slice(s, s + z) for s, z in zip(opt["starts"], opt["sizes"]))
        return np.ascontiguousarray(x[index])
    if op == "concat":
        return np.concatenate(ins, axis=opt["axis"])
    if op == "expand":
        return np.ascontiguousarray(np.broadcast_to(x, opt["newShape"]))
    if op in ("reduceMean", "reduceMax", "reduceSum"):
        fn = {"reduceMean": np.mean, "reduceMax": np.max, "reduceSum": np.sum}[op]
        return fn(_f32(x), axis=tuple(opt["axes"]), keepdims=opt.get("keepDimensions", False)).astype(np_dtype(out_dtype))
    if op == "softmax":
        v = _f32(x)
        e = np.exp(v - v.max(axis=opt["axis"], keepdims=True))
        return (e / e.sum(axis=opt["axis"], keepdims=True)).astype(np_dtype(out_dtype))
    if op == "matmul":
        a, b = ins
        return np.matmul(_f32(a), _f32(b)).astype(np_dtype(out_dtype))
    if op == "gemm":
        a, b = _f32(ins[0]), _f32(ins[1])
        if opt.get("aTranspose"): a = a.T
        if opt.get("bTranspose"): b = b.T
        return (a @ b).astype(np_dtype(out_dtype))
    if op == "conv2d":
        a, w = _f32(ins[0]), _f32(ins[1])
        groups, (sh, sw), (dh, dw), pad = opt.get("groups", 1), opt.get("strides", [1, 1]), opt.get("dilations", [1, 1]), opt.get("padding", [0, 0, 0, 0])
        a = np.pad(a, ((0, 0), (0, 0), (pad[0], pad[1]), (pad[2], pad[3])))
        n, c, _, _ = a.shape
        o, ci, kh, kw = w.shape
        oh, ow = (a.shape[2] - dh * (kh - 1) - 1) // sh + 1, (a.shape[3] - dw * (kw - 1) - 1) // sw + 1
        out = np.zeros((n, o, oh, ow), np.float32)
        og = o // groups
        for g in range(groups):
            for i in range(kh):
                for j in range(kw):
                    window = a[:, g * ci:(g + 1) * ci, i * dh: i * dh + sh * (oh - 1) + 1: sh, j * dw: j * dw + sw * (ow - 1) + 1: sw]   # [n, ci, oh, ow]
                    out[:, g * og:(g + 1) * og] += np.einsum("nchw,oc->nohw", window, w[g * og:(g + 1) * og, :, i, j])
        return out.astype(np_dtype(out_dtype))
    if op == "dequantizeLinear":
        q, scale, zp = ins
        qf = q.astype(np.float32)
        if zp is not None:
            qf = qf - _expand_blockwise(zp.astype(np.float32), q.shape)
        return (qf * _expand_blockwise(_f32(scale), q.shape)).astype(np_dtype(out_dtype))
    if op == "tensorScatter":
        inp, update, index = ins
        out = inp.copy()
        start = int(index[0])
        axis = opt["axis"]
        sel = [slice(None)] * inp.ndim
        sel[axis] = slice(start, start + update.shape[axis])
        out[tuple(sel)] = update
        return out
    if op == "scatterND":
        inp, indices, updates = ins
        out = inp.copy()
        k = indices.shape[-1]
        flat = indices.reshape(-1, k).astype(np.int64)
        out[tuple(flat.T)] = updates.reshape((flat.shape[0],) + tuple(inp.shape[k:]))   # duplicate indices: the last one wins
        return out
    raise NotImplementedError(op)


def run(graph: Graph, feeds: dict, externals: dict | None = None) -> dict:
    """Execute ``graph``. ``feeds``: input name -> array. ``externals``: name -> array for constants that reference weights."""
    env: dict = {}
    for name, spec in graph.inputs.items():
        if name not in feeds:
            raise KeyError(f"missing input '{name}'")
        arr = np.asarray(feeds[name])
        if arr.shape != tuple(spec.shape):
            raise ValueError(f"input '{name}': shape {arr.shape}, expected {tuple(spec.shape)}")
        if arr.dtype != np_dtype(spec.dtype):
            raise ValueError(f"input '{name}': dtype {arr.dtype}, expected {np_dtype(spec.dtype)} ({spec.dtype})")
        env[name] = arr
    for name, c in graph.constants.items():
        dt = np_dtype(c.dtype)
        if c.fill is not None:
            env[name] = np.full(c.shape, c.fill, dtype=dt)
        elif c.data is not None:
            env[name] = np.array(c.data, dtype=dt).reshape(c.shape)
        else:
            if not externals or c.external not in externals:
                raise KeyError(f"missing external constant '{c.external}'")
            env[name] = np.asarray(externals[c.external]).astype(dt, copy=False).reshape(c.shape)
    for node in graph.nodes:
        ins = [env[i] if i is not None else None for i in node.inputs]
        out_name = node.outputs[0]
        expected = graph.values[out_name]
        result = _eval(node.op, ins, node.options, expected.dtype)
        if result.shape != tuple(expected.shape) or result.dtype != np_dtype(expected.dtype):
            raise AssertionError(f"{node.op} -> {out_name}: got {result.dtype}{result.shape}, builder inferred {expected.dtype}{expected.shape}")
        env[out_name] = result
    return {name: env[value] for name, value in graph.outputs.items()}

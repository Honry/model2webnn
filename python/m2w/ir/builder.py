"""GraphBuilder: records WebNN operators and infers static shapes / data types as it goes.

The method names mirror ``MLGraphBuilder`` (snake_case instead of camelCase), so graph code written against the JS builder
(``builder.reduceMean(x, {axes, keepDimensions})``) ports almost line by line (``b.reduce_mean(x, axes=..., keep_dimensions=...)``).
``infer_node`` is the single place that knows the shape / dtype rule of each operator; ``Graph.from_dict`` uses it too.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .graph import DATA_TYPES, Constant, Graph, InputSpec, Node, Value

FLOATS = ("float32", "float16")
UNARY_FLOAT_OPS = ("sigmoid", "sqrt")
ELEMENTWISE_BINARY_OPS = ("add", "sub", "mul", "div", "pow")
REDUCE_OPS = ("reduceMean", "reduceMax", "reduceSum")


class GraphError(ValueError):
    pass


def broadcast_shapes(a: Sequence[int], b: Sequence[int]) -> tuple:
    out = []
    for i in range(1, max(len(a), len(b)) + 1):
        da = a[-i] if i <= len(a) else 1
        db = b[-i] if i <= len(b) else 1
        if da != db and da != 1 and db != 1:
            raise GraphError(f"cannot broadcast {tuple(a)} with {tuple(b)}")
        out.append(db if da == 1 else da)
    return tuple(reversed(out))


def _norm_axes(axes: Sequence[int], rank: int) -> list:
    out = []
    for a in axes:
        if not -rank <= a < rank:
            raise GraphError(f"axis {a} out of range for rank {rank}")
        out.append(a % rank)
    return out


def infer_node(graph: Graph, node: Node):
    """Return [(dtype, shape)] for the outputs of ``node`` (its inputs must already have values in ``graph``)."""
    op, opt = node.op, node.options
    ins = [graph.values[i] if i is not None else None for i in node.inputs]
    x = ins[0] if ins else None

    if op in ELEMENTWISE_BINARY_OPS:
        a, b = ins
        if a.dtype != b.dtype:
            raise GraphError(f"{op}: operand types differ ({a.dtype} vs {b.dtype}) for {node.outputs}")
        return [(a.dtype, broadcast_shapes(a.shape, b.shape))]
    if op in UNARY_FLOAT_OPS:
        if x.dtype not in FLOATS:
            raise GraphError(f"{op}: needs a float input, got {x.dtype}")
        return [(x.dtype, x.shape)]
    if op == "cast":
        to = opt["to"]
        if to not in DATA_TYPES:
            raise GraphError(f"cast: unknown type {to}")
        return [(to, x.shape)]
    if op == "reshape":
        new_shape = tuple(opt["newShape"])
        if any((not isinstance(d, int)) or d <= 0 for d in new_shape):
            raise GraphError(f"reshape: newShape must be explicit positive integers, got {new_shape}")
        if _count(new_shape) != _count(x.shape):
            raise GraphError(f"reshape: {x.shape} -> {new_shape} changes the element count")
        return [(x.dtype, new_shape)]
    if op == "transpose":
        perm = list(opt["permutation"])
        if sorted(perm) != list(range(x.rank)):
            raise GraphError(f"transpose: bad permutation {perm} for rank {x.rank}")
        return [(x.dtype, tuple(x.shape[p] for p in perm))]
    if op == "slice":
        starts, sizes = list(opt["starts"]), list(opt["sizes"])
        if len(starts) != x.rank or len(sizes) != x.rank:
            raise GraphError(f"slice: starts/sizes must have rank {x.rank}")
        for s, z, d in zip(starts, sizes, x.shape):
            if s < 0 or z <= 0 or s + z > d:
                raise GraphError(f"slice: [{s}, {s + z}) is outside dimension {d}")
        return [(x.dtype, tuple(sizes))]
    if op == "concat":
        axis = opt["axis"]
        first = ins[0]
        total = 0
        for v in ins:
            if v.dtype != first.dtype or v.rank != first.rank:
                raise GraphError(f"concat: operand mismatch {v.dtype}{v.shape} vs {first.dtype}{first.shape}")
            for d in range(first.rank):
                if d != axis and v.shape[d] != first.shape[d]:
                    raise GraphError(f"concat: shapes differ off the axis: {v.shape} vs {first.shape}")
            total += v.shape[axis]
        shape = list(first.shape)
        shape[axis] = total
        return [(first.dtype, tuple(shape))]
    if op == "expand":
        new_shape = tuple(opt["newShape"])
        broadcast_shapes(x.shape, new_shape)
        if broadcast_shapes(x.shape, new_shape) != new_shape:
            raise GraphError(f"expand: {x.shape} does not expand to {new_shape}")
        return [(x.dtype, new_shape)]
    if op in REDUCE_OPS:
        axes = _norm_axes(opt["axes"], x.rank)
        keep = opt.get("keepDimensions", False)
        shape = [1 if i in axes else d for i, d in enumerate(x.shape)]
        if not keep:
            shape = [d for i, d in enumerate(x.shape) if i not in axes]
        return [(x.dtype, tuple(shape))]
    if op == "softmax":
        _norm_axes([opt["axis"]], x.rank)
        return [(x.dtype, x.shape)]
    if op == "matmul":
        a, b = ins
        if a.dtype != b.dtype:
            raise GraphError(f"matmul: operand types differ ({a.dtype} vs {b.dtype})")
        if a.rank < 2 or b.rank < 2:
            raise GraphError("matmul: operands must have rank >= 2")
        if a.shape[-1] != b.shape[-2]:
            raise GraphError(f"matmul: inner dimensions differ {a.shape} @ {b.shape}")
        batch = broadcast_shapes(a.shape[:-2], b.shape[:-2])
        return [(a.dtype, batch + (a.shape[-2], b.shape[-1]))]
    if op == "gemm":
        a, b = ins[0], ins[1]
        if a.dtype != b.dtype:
            raise GraphError(f"gemm: operand types differ ({a.dtype} vs {b.dtype})")
        if a.rank != 2 or b.rank != 2:
            raise GraphError("gemm: operands must be 2-D")
        m, k = (a.shape[1], a.shape[0]) if opt.get("aTranspose") else a.shape
        k2, n = (b.shape[1], b.shape[0]) if opt.get("bTranspose") else b.shape
        if k != k2:
            raise GraphError(f"gemm: inner dimensions differ {a.shape} x {b.shape} (aT={opt.get('aTranspose')}, bT={opt.get('bTranspose')})")
        return [(a.dtype, (m, n))]
    if op == "dequantizeLinear":
        q, scale, zp = ins
        if q.dtype not in ("int4", "uint4", "int8", "uint8", "int32"):
            raise GraphError(f"dequantizeLinear: cannot dequantize {q.dtype}")
        if scale.rank != q.rank or any(q.shape[i] % scale.shape[i] for i in range(q.rank)):
            raise GraphError(f"dequantizeLinear: scale {scale.shape} must divide input {q.shape} per dimension")
        if zp is not None and (zp.shape != scale.shape or zp.dtype != q.dtype):
            raise GraphError(f"dequantizeLinear: zero point must have the input type and the scale shape ({zp.dtype}{zp.shape})")
        return [(scale.dtype, q.shape)]
    if op == "tensorScatter":
        inp, update, index = ins
        axis = opt["axis"]
        if inp.dtype != update.dtype:
            raise GraphError("tensorScatter: input/update types differ")
        if index.dtype not in ("int32", "int64") or index.shape != (1,):
            raise GraphError("tensorScatter: write index must be int32/int64 of shape [1]")
        for d in range(inp.rank):
            if d != axis and inp.shape[d] != update.shape[d]:
                raise GraphError(f"tensorScatter: update {update.shape} does not match input {inp.shape} off axis {axis}")
        if update.shape[axis] > inp.shape[axis]:
            raise GraphError("tensorScatter: update is longer than the input along the axis")
        return [(inp.dtype, inp.shape)]
    if op == "scatterND":
        inp, indices, updates = ins
        if indices.dtype not in ("int32", "int64") or indices.rank < 1:
            raise GraphError("scatterND: indices must be int32/int64 with rank >= 1")
        k = indices.shape[-1]
        if k > inp.rank:
            raise GraphError(f"scatterND: index tuples of {k} values for an input of rank {inp.rank}")
        expected = tuple(indices.shape[:-1]) + tuple(inp.shape[k:])
        if tuple(updates.shape) != expected or updates.dtype != inp.dtype:
            raise GraphError(f"scatterND: updates {updates.dtype}{tuple(updates.shape)} must be {inp.dtype}{expected} (indices shape without the last axis + the input shape after it)")
        return [(inp.dtype, inp.shape)]
    raise GraphError(f"unsupported operator '{op}'")


def _count(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


class GraphBuilder:
    def __init__(self, name: str):
        self.graph = Graph(name=name)
        self._counter = 0
        self._constant_cache: dict = {}

    # ------------------------------------------------------------------ declarations
    def input(self, name: str, dtype: str, shape: Sequence[int], role: str = "activation") -> Value:
        if name in self.graph.values:
            raise GraphError(f"duplicate name '{name}'")
        shape = tuple(int(d) for d in shape)
        self.graph.inputs[name] = InputSpec(name, dtype, shape, role)
        v = Value(name, dtype, shape)
        self.graph.values[name] = v
        return v

    def constant(self, dtype: str, shape: Sequence[int], *, fill: Optional[float] = None, data: Optional[list] = None,
                 external: Optional[str] = None, cache_key=None) -> Value:
        if cache_key is not None and cache_key in self._constant_cache:
            return self._constant_cache[cache_key]
        shape = tuple(int(d) for d in shape)
        name = f"const{self._counter}"
        self._counter += 1
        self.graph.constants[name] = Constant(name, dtype, shape, fill, data, external)
        v = Value(name, dtype, shape)
        self.graph.values[name] = v
        if cache_key is not None:
            self._constant_cache[cache_key] = v
        return v

    def scalar(self, value: float, dtype: str, rank: int) -> Value:
        """A shape [1, ..., 1] constant (the exponent of pow, an epsilon ...), shared within the graph."""
        return self.constant(dtype, [1] * rank, fill=value, cache_key=("scalar", value, dtype, rank))

    def output(self, name: str, value: Value) -> None:
        if name in self.graph.outputs:
            raise GraphError(f"duplicate output '{name}'")
        if value.name in self.graph.inputs:
            raise GraphError(f"output '{name}' is directly a graph input; copy it first (WebNN cannot output an input)")
        self.graph.outputs[name] = value.name

    # ------------------------------------------------------------------ recording
    def _emit(self, op: str, inputs: list, options: Optional[dict] = None) -> Value:
        out = f"v{self._counter}"
        self._counter += 1
        node = Node(op, [v.name if v is not None else None for v in inputs], [out], options or {})
        ((dtype, shape),) = infer_node(self.graph, node)
        self.graph.nodes.append(node)
        v = Value(out, dtype, tuple(shape))
        self.graph.values[out] = v
        return v

    # ------------------------------------------------------------------ operators (WebNN names, snake_case)
    def add(self, a, b): return self._emit("add", [a, b])
    def sub(self, a, b): return self._emit("sub", [a, b])
    def mul(self, a, b): return self._emit("mul", [a, b])
    def div(self, a, b): return self._emit("div", [a, b])
    def pow(self, a, b): return self._emit("pow", [a, b])
    def sqrt(self, x): return self._emit("sqrt", [x])
    def sigmoid(self, x): return self._emit("sigmoid", [x])
    def cast(self, x, to: str): return self._emit("cast", [x], {"to": to})
    def reshape(self, x, new_shape): return self._emit("reshape", [x], {"newShape": [int(d) for d in new_shape]})
    def transpose(self, x, permutation): return self._emit("transpose", [x], {"permutation": [int(p) for p in permutation]})
    def slice(self, x, starts, sizes): return self._emit("slice", [x], {"starts": [int(s) for s in starts], "sizes": [int(s) for s in sizes]})
    def concat(self, xs, axis: int): return self._emit("concat", list(xs), {"axis": int(axis)})
    def expand(self, x, new_shape): return self._emit("expand", [x], {"newShape": [int(d) for d in new_shape]})
    def reduce_mean(self, x, axes, keep_dimensions=False): return self._emit("reduceMean", [x], {"axes": list(axes), "keepDimensions": keep_dimensions})
    def reduce_max(self, x, axes, keep_dimensions=False): return self._emit("reduceMax", [x], {"axes": list(axes), "keepDimensions": keep_dimensions})
    def reduce_sum(self, x, axes, keep_dimensions=False): return self._emit("reduceSum", [x], {"axes": list(axes), "keepDimensions": keep_dimensions})
    def softmax(self, x, axis: int): return self._emit("softmax", [x], {"axis": int(axis)})
    def matmul(self, a, b): return self._emit("matmul", [a, b])
    def gemm(self, a, b, b_transpose=False, a_transpose=False):
        options = {}
        if b_transpose: options["bTranspose"] = True
        if a_transpose: options["aTranspose"] = True
        return self._emit("gemm", [a, b], options)
    def dequantize_linear(self, x, scale, zero_point=None): return self._emit("dequantizeLinear", [x, scale, zero_point])
    def scatter_nd(self, x, indices, updates): return self._emit("scatterND", [x, indices, updates], {})
    def tensor_scatter(self, x, update, write_index, axis: int): return self._emit("tensorScatter", [x, update, write_index], {"axis": int(axis)})

    def build(self) -> Graph:
        if not self.graph.outputs:
            raise GraphError("graph has no outputs")
        return self.graph

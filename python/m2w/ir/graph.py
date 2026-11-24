"""WebNN-level IR: a flat graph of WebNN operators over statically shaped tensors.

Operator names and option names follow the W3C WebNN spec (``dequantizeLinear``, ``reduceMean``, ``bTranspose`` ...), so a
node maps 1:1 onto a ``MLGraphBuilder`` call. The IR carries no weight data: graph inputs are either runtime tensors
(activations, KV cache, weights that are bound per dispatch) or small constants (scalars, all-zero zero points).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# WebNN data types that appear in the IR. int4 / uint4 are logical types: on disk and on the device they are packed two per byte
# (low nibble first); in numpy (interpreter) they are held UNPACKED in int8 / uint8 arrays.
DATA_TYPES = ("float32", "float16", "int32", "int64", "int8", "uint8", "int4", "uint4")
BITS_PER_ELEMENT = {"float32": 32, "float16": 16, "int32": 32, "int64": 64, "int8": 8, "uint8": 8, "int4": 4, "uint4": 4}
FORMAT = "m2w-graph"
VERSION = 0


def element_count(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def byte_length(dtype: str, shape) -> int:
    return (element_count(shape) * BITS_PER_ELEMENT[dtype] + 7) // 8


@dataclass(frozen=True)
class Value:
    """A tensor in the graph (the result of an input, a constant or a node)."""

    name: str
    dtype: str
    shape: tuple

    @property
    def rank(self) -> int:
        return len(self.shape)


@dataclass
class Node:
    op: str
    inputs: list  # list[Optional[str]]  (None = an omitted optional operand, e.g. the zero point of dequantizeLinear)
    outputs: list  # list[str]
    options: dict = field(default_factory=dict)


@dataclass
class Constant:
    """A constant operand. Exactly one of ``fill`` / ``data`` / ``external`` is set."""

    name: str
    dtype: str
    shape: tuple
    fill: Optional[float] = None  # every element has this value (zero points, scalars)
    data: Optional[list] = None  # small literal tensors, row-major
    external: Optional[str] = None  # name of a tensor in the weights manifest


@dataclass
class InputSpec:
    name: str
    dtype: str
    shape: tuple
    role: str = "activation"  # "activation" | "weight" | "kv" | "control": how the runtime plan binds it


@dataclass
class Graph:
    name: str
    inputs: dict = field(default_factory=dict)  # name -> InputSpec
    constants: dict = field(default_factory=dict)  # name -> Constant
    nodes: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)  # output name -> value name
    values: dict = field(default_factory=dict)  # value name -> Value
    meta: dict = field(default_factory=dict)

    def value(self, name: str) -> Value:
        return self.values[name]

    def to_dict(self) -> dict:
        def spec(v: Value) -> dict:
            return {"dataType": v.dtype, "shape": list(v.shape)}

        constants = []
        for c in self.constants.values():
            entry: dict[str, Any] = {"name": c.name, "dataType": c.dtype, "shape": list(c.shape)}
            if c.fill is not None:
                entry["fill"] = c.fill
            elif c.data is not None:
                entry["data"] = c.data
            else:
                entry["external"] = c.external
            constants.append(entry)
        return {
            "format": FORMAT,
            "version": VERSION,
            "name": self.name,
            "inputs": [
                {"name": i.name, "dataType": i.dtype, "shape": list(i.shape), "role": i.role} for i in self.inputs.values()
            ],
            "constants": constants,
            "nodes": [
                {"op": n.op, "inputs": n.inputs, "outputs": n.outputs, **({"options": n.options} if n.options else {})}
                for n in self.nodes
            ],
            "outputs": [{"name": k, **spec(self.values[v]), "value": v} for k, v in self.outputs.items()],
            "meta": self.meta,
        }

    @staticmethod
    def from_dict(d: dict) -> "Graph":
        if d.get("format") != FORMAT:
            raise ValueError(f"not an {FORMAT} document")
        if d.get("version") != VERSION:
            raise ValueError(f"unsupported {FORMAT} version {d.get('version')} (this build reads version {VERSION})")
        g = Graph(name=d["name"], meta=d.get("meta", {}))
        for i in d["inputs"]:
            shape = tuple(i["shape"])
            g.inputs[i["name"]] = InputSpec(i["name"], i["dataType"], shape, i.get("role", "activation"))
            g.values[i["name"]] = Value(i["name"], i["dataType"], shape)
        for c in d["constants"]:
            shape = tuple(c["shape"])
            g.constants[c["name"]] = Constant(c["name"], c["dataType"], shape, c.get("fill"), c.get("data"), c.get("external"))
            g.values[c["name"]] = Value(c["name"], c["dataType"], shape)
        for n in d["nodes"]:
            g.nodes.append(Node(n["op"], list(n["inputs"]), list(n["outputs"]), dict(n.get("options", {}))))
        for o in d["outputs"]:
            g.outputs[o["name"]] = o["value"]
            g.values[o["value"]] = Value(o["value"], o["dataType"], tuple(o["shape"]))
        # node results: re-derive types from the interpreter-independent shape rules
        from .builder import infer_node

        for n in g.nodes:
            infos = infer_node(g, n)
            for name, (dtype, shape) in zip(n.outputs, infos):
                g.values[name] = Value(name, dtype, tuple(shape))
        return g

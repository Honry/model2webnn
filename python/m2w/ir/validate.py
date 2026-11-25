"""Limits of the WebNN implementations a graph has to respect."""
from __future__ import annotations

from .builder import GraphError
from .graph import Graph, byte_length

# WebNN (Chromium) rejects a tensor of more than 2^31 - 1 bytes: an operand of a graph (input, constant, the result of any operator: e.g. the cast of a dequantized weight matrix) as well as an MLTensor.
WEBNN_MAX_TENSOR_BYTES = 2 ** 31 - 1


def check_tensor_limit(graph: Graph, limit: int = WEBNN_MAX_TENSOR_BYTES) -> None:
    """Raise GraphError naming every tensor of the graph that is bigger than `limit` bytes (inputs, constants and the results of the operators)."""
    too_big = [(byte_length(v.dtype, v.shape), name, v) for name, v in graph.values.items() if byte_length(v.dtype, v.shape) > limit]
    if too_big:
        listing = "; ".join(f"{name} {v.dtype}{list(v.shape)} = {size / 2 ** 30:.2f} GiB" for size, name, v in sorted(too_big, reverse=True)[:5])
        raise GraphError(f"graph '{graph.name}' has tensors above the WebNN limit of {limit} bytes ({listing}): split the operation into blocks (see lm_head_parts in builders/gemma4.py)")

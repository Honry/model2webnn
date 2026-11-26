"""Static-shape subgraphs of the Qwen3.5 image tower (a 24-block ViT + a patch merger) in the NPUW-style split of qwen35.py: HEAD / BLOCK / MERGER, one set per GRID preset (docs/models/qwen35-vision.md).

    HEAD    patches [N, 3*2*16*16] -> patch embedding (the Conv3d is one matmul over the flattened patch) + bias + the position embedding of the grid -> hidden_states [1, N, 1024]
    BLOCK   ONE graph for all 24 layers, the weights are graph inputs (the runtime binds the weights of layer i, like the BLOCK graphs of the text decoder):
            x + proj(attention(rotary(qkv(LayerNorm(x)))));  x + fc2(gelu_tanh(fc1(LayerNorm(x))))      bidirectional attention over all N patches, 16 heads, no mask
    MERGER  LayerNorm -> the 2x2 patches of a merge block are consecutive rows -> reshape [N / 4, 4096] -> fc1 -> GELU -> fc2 -> image_embeds [1, N / 4, 2560]

What is different from the text blocks (qwen35.py): LayerNorm (mean subtracted, with a bias) instead of RMSNorm, biases everywhere, a 2-D rotary on the patch row / column, no KV cache and no mask.
The tables that depend on the grid only (the interpolated position embedding of HEAD, the rotary cos / sin of BLOCK) are EXTERNAL constants (frontends/hf_llm/vision.py computes them, the package
stores them per grid). Linear weights: int8 per row symmetric (`dequantizeLinear`, the path the lm_head runs on the NPU; scale [out, 1]), int4 group 128 or float16 (compile option
--vision-weights); biases, LayerNorm and the patch-embed bias are float16.

NPU notes (carried over from the text path; the numerics have to be checked on the device, bench/raw/qwen35vision.html): float32 graphs run as float16 there, and the activations of this ViT leave
the float16 comfort zone (measured on a real image, float32 reference): a few "register" tokens carry values of 140 .. 230 in two channels from block 10 on, the last block raises every token (rms 67,
max 5300), the fc1 output of block 23 reaches 680. So
  * LayerNorm works on the INPUT SCALED by a power of two (VISION_NORM_SCALE = 1/16 in the blocks: centered values up to 231, row rms 0.16 .. 8.2, squares from 1e-4 to 208; the merger, whose input
    reaches 5300, uses 1/64): mean, centered values and squares all stay inside the float16 range, the result does not depend on the scale; eps (below ~6e-5 a float16 subnormal, flushed to 0)
    only matters for a row without variance;
  * the cube of the GELU overflows float16 for |x| > 40: the polynomial is formed on x clamped to +-8 (tanh is 1 there), the factor x stays unclamped;
  * an activation x activation matmul has all dims >= 16 here (head dim 64, N >= 64); the residual feeds the LayerNorm and the add (both elementwise), never a matmul and an add.
The merger's GELU is the tanh form although transformers uses the exact erf there: the two differ by 4.7e-4 at most (this IR has no erf).
"""
from __future__ import annotations

import math

from ....ir import GraphBuilder, Value
from .decoder import Ctx, DecoderDims, GraphOptions, apply_rotary

VISION_NORM_SCALE = 1.0 / 16.0          # LayerNorm of the blocks: the input is scaled by this first (and eps by its square); the result does not depend on it, the float16 range of the mean / squares does
VISION_MERGER_NORM_SCALE = 1.0 / 64.0   # the merger's LayerNorm sees the output of the last block (|x| up to 5300, row rms 2 .. 166)
GELU_CLAMP = 8.0                        # the polynomial of the tanh GELU is formed on x clamped to +-GELU_CLAMP (x^3 overflows float16 above 40; tanh is 1 for |inner| > 5)
WEIGHT_FORMATS = ("int8", "int4", "float16")
TABLES = ("pos_embed", "rope_cos", "rope_sin")   # the per-grid tensors of the package (weights/vision_grid_<grid>.safetensors), the external constants of HEAD (pos_embed) and BLOCK (rope_*)


def vision_dims(spec) -> dict:
    v = spec.vision
    merge_unit = v["merge_size"] ** 2
    return {"hidden": v["hidden"], "heads": v["heads"], "head_dim": v["hidden"] // v["heads"], "ffn": v["ffn"], "out": v["out_hidden"], "merge_unit": merge_unit,
            "patch_vector": v["channels"] * v["temporal_patch_size"] * v["patch_size"] ** 2, "merged_hidden": v["hidden"] * merge_unit, "eps": v["eps"]}


def _ctx(b: GraphBuilder, spec, compute_type: str) -> Ctx:
    d = vision_dims(spec)
    dims = DecoderDims(hidden=d["hidden"], num_heads=d["heads"], num_kv_heads=d["heads"], head_dim=d["head_dim"], ffn=d["ffn"], vocab=0, eps=d["eps"])
    return Ctx(b, dims, GraphOptions(compute_type=compute_type))


# ---------------------------------------------------------------------------------------------------------------------
# weight specs: (name, dtype, shape); the names are graph input names AND manifest keys
# ---------------------------------------------------------------------------------------------------------------------
def linear_specs(name: str, in_ch: int, out_ch: int, weights: str, group: int = 128) -> list:
    if weights == "int8":
        return [(f"{name}_weight", "int8", (out_ch, in_ch)), (f"{name}_scale", "float16", (out_ch, 1))]
    if weights == "int4":
        return [(f"{name}_weight", "int4", (in_ch // group, out_ch, group)), (f"{name}_scale", "float16", (in_ch // group, out_ch, 1))]
    if weights == "float16":
        return [(f"{name}_weight", "float16", (out_ch, in_ch))]
    raise ValueError(f"unknown vision weight format '{weights}' (int8 | int4 | float16)")


def _norm_specs(name: str, width: int) -> list:
    return [(f"{name}_gamma", "float16", (1, 1, width)), (f"{name}_beta", "float16", (1, 1, width))]


def _bias(name: str, width: int) -> list:
    return [(f"{name}_bias", "float16", (1, 1, width))]


def head_weight_specs(spec, weights: str) -> list:
    d = vision_dims(spec)
    return linear_specs("patch_embed", d["patch_vector"], d["hidden"], weights) + _bias("patch_embed", d["hidden"])


def block_weight_specs(spec, weights: str) -> list:
    h, f = spec.vision["hidden"], spec.vision["ffn"]
    return (_norm_specs("norm1", h) + linear_specs("qkv", h, 3 * h, weights) + _bias("qkv", 3 * h) + linear_specs("proj", h, h, weights) + _bias("proj", h)
            + _norm_specs("norm2", h) + linear_specs("fc1", h, f, weights) + _bias("fc1", f) + linear_specs("fc2", f, h, weights) + _bias("fc2", h))


def merger_weight_specs(spec, weights: str) -> list:
    d = vision_dims(spec)
    return (_norm_specs("merger_norm", d["hidden"]) + linear_specs("merger_fc1", d["merged_hidden"], d["merged_hidden"], weights) + _bias("merger_fc1", d["merged_hidden"])
            + linear_specs("merger_fc2", d["merged_hidden"], d["out"], weights) + _bias("merger_fc2", d["out"]))


def table_specs(spec, grid) -> list:
    n, d = grid[0] * grid[1], vision_dims(spec)
    return [("pos_embed", "float16", (n, d["hidden"])), ("rope_cos", "float16", (n, d["head_dim"])), ("rope_sin", "float16", (n, d["head_dim"]))]


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def linear(c: Ctx, x: Value, w: dict, name: str, in_ch: int, out_ch: int, rows: int, weights: str) -> Value:
    """y = x @ W^T + bias on x [1, rows, in]: the weight is dequantized (int8: weight * scale per output row; int4: [groups, out, 128] blocks) and cast to the compute type for a gemm."""
    b = c.b
    if weights == "float16":
        matrix = c.to_compute(w[f"{name}_weight"])
    elif weights == "int8":
        zero_point = b.constant("int8", (out_ch, 1), fill=0, cache_key=("zp", "int8", out_ch))
        matrix = c.to_compute(b.dequantize_linear(w[f"{name}_weight"], w[f"{name}_scale"], zero_point))
    else:
        groups = in_ch // c.dims.group_size
        dequantized = b.dequantize_linear(w[f"{name}_weight"], w[f"{name}_scale"], c.zero_point((groups, out_ch, 1)))
        matrix = c.to_compute(b.reshape(b.transpose(dequantized, [1, 0, 2]), [out_ch, in_ch]))
    y = b.reshape(b.gemm(b.reshape(x, [rows, in_ch]), matrix, b_transpose=True), [1, rows, out_ch])
    return b.add(y, c.to_compute(w[f"{name}_bias"]))


def layer_norm(c: Ctx, x: Value, gamma: Value, beta: Value, eps: float, scale: float = VISION_NORM_SCALE) -> Value:
    """gamma * (x - mean) / sqrt(var + eps) + beta over the last axis of [1, N, width], computed on y = x * scale (a power of two: exact) so that the mean, the centered values and their squares stay
    inside the float16 range: (y - mean(y)) / sqrt(mean((y - mean(y))^2) + eps scale^2) = (x - mean) / sqrt(var + eps) whatever the scale."""
    b = c.b
    y = b.mul(x, c.scalar(scale, 3))
    centered = b.sub(y, b.reduce_mean(y, axes=[2], keep_dimensions=True))
    mean_square = b.reduce_mean(b.pow(centered, c.scalar(2, 3)), axes=[2], keep_dimensions=True)
    inverse = b.div(c.scalar(1, 3), b.sqrt(b.add(mean_square, c.scalar(eps * scale * scale, 3))))
    return b.add(b.mul(b.mul(centered, inverse), c.to_compute(gamma)), c.to_compute(beta))


def gelu(c: Ctx, x: Value) -> Value:
    """0.5 x (1 + tanh(sqrt(2/pi) (u + 0.044715 u^3))) with u = x clamped to +-GELU_CLAMP (the tanh form of gemma4.gelu_tanh without the float16 overflow of the cube)"""
    b = c.b
    u = b.clamp(x, -GELU_CLAMP, GELU_CLAMP)
    inner = b.mul(b.mul(u, b.add(c.scalar(1.0, 3), b.mul(b.mul(u, u), c.scalar(0.044715, 3)))), c.scalar(math.sqrt(2.0 / math.pi), 3))
    return b.mul(b.mul(x, c.scalar(0.5, 3)), b.add(b.tanh(inner), c.scalar(1.0, 3)))


def attention(c: Ctx, spec, x: Value, w: dict, cos: Value, sin: Value, n: int, weights: str) -> Value:
    """bidirectional multi-head attention with the 2-D rotary on q and k, output projection included: [1, N, hidden] -> [1, N, hidden]"""
    b, d = c.b, vision_dims(spec)
    h, heads, hd = d["hidden"], d["heads"], d["head_dim"]
    c.seq = n
    qkv = linear(c, x, w, "qkv", h, 3 * h, n, weights)                                                 # channels: [q | k | v], each [heads, head_dim]
    heads_of = lambda start: b.transpose(b.reshape(b.slice(qkv, [0, 0, start], [1, n, h]), [1, n, heads, hd]), [0, 2, 1, 3])   # noqa: E731
    query, key, value = apply_rotary(c, heads_of(0), cos, sin, heads), apply_rotary(c, heads_of(h), cos, sin, heads), heads_of(2 * h)
    scores = b.matmul(b.mul(query, c.scalar(1.0 / math.sqrt(hd), 4)), b.transpose(key, [0, 1, 3, 2]))   # [1, heads, N, N]
    context = b.matmul(b.softmax(scores, 3), value)
    return linear(c, b.reshape(b.transpose(context, [0, 2, 1, 3]), [1, n, h]), w, "proj", h, h, n, weights)


# ---------------------------------------------------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------------------------------------------------
def build_head_graph(spec, grid, weights: str, compute_type: str = "float32", name: str = "vision_head"):
    b = GraphBuilder(name)
    c = _ctx(b, spec, compute_type)
    d, n = vision_dims(spec), grid[0] * grid[1]
    patches = c.to_compute(c.declare("patches", "float16", [n, d["patch_vector"]]))
    w = c.weights(head_weight_specs(spec, weights))
    x = linear(c, b.reshape(patches, [1, n, d["patch_vector"]]), w, "patch_embed", d["patch_vector"], d["hidden"], n, weights)
    position = b.constant("float16", (1, n, d["hidden"]), external="pos_embed")
    b.output("hidden_states", c.to_f16(b.add(x, c.to_compute(position))))
    return b.build()


def build_block_graph(spec, grid, weights: str, compute_type: str = "float32", name: str = "vision_block"):
    """one ViT block with the weights of ANY layer bound as inputs: hidden_states -> next_hidden_states"""
    b = GraphBuilder(name)
    c = _ctx(b, spec, compute_type)
    d, n = vision_dims(spec), grid[0] * grid[1]
    x = c.to_compute(c.declare("hidden_states", "float16", [1, n, d["hidden"]]))
    w = c.weights(block_weight_specs(spec, weights))
    cos, sin = b.constant("float16", (1, 1, n, d["head_dim"]), external="rope_cos"), b.constant("float16", (1, 1, n, d["head_dim"]), external="rope_sin")
    x1 = b.add(x, attention(c, spec, layer_norm(c, x, w["norm1_gamma"], w["norm1_beta"], d["eps"]), w, cos, sin, n, weights))
    hidden = linear(c, layer_norm(c, x1, w["norm2_gamma"], w["norm2_beta"], d["eps"]), w, "fc1", d["hidden"], d["ffn"], n, weights)
    out = b.add(x1, linear(c, gelu(c, hidden), w, "fc2", d["ffn"], d["hidden"], n, weights))
    b.output("next_hidden_states", c.to_f16(out))
    return b.build()


def build_merger_graph(spec, grid, weights: str, compute_type: str = "float32", name: str = "vision_merger"):
    b = GraphBuilder(name)
    c = _ctx(b, spec, compute_type)
    d, n = vision_dims(spec), grid[0] * grid[1]
    m = n // d["merge_unit"]
    x = c.to_compute(c.declare("hidden_states", "float16", [1, n, d["hidden"]]))
    w = c.weights(merger_weight_specs(spec, weights))
    merged = b.reshape(layer_norm(c, x, w["merger_norm_gamma"], w["merger_norm_beta"], d["eps"], VISION_MERGER_NORM_SCALE), [1, m, d["merged_hidden"]])   # the rows of a 2x2 merge block are consecutive
    hidden = gelu(c, linear(c, merged, w, "merger_fc1", d["merged_hidden"], d["merged_hidden"], m, weights))
    b.output("image_embeds", c.to_f16(linear(c, hidden, w, "merger_fc2", d["merged_hidden"], d["out"], m, weights)))
    return b.build()

"""Static-shape decoder subgraphs in the NPUW split: HEAD / BLOCK / TAIL / lm_head / KV update.

Ported from hand-written JavaScript graphs (production options only). Keep it a faithful port: the operator pattern decides what the NPU compiler recognises.

A decoder layer has two halves:
    attention half   : input_layernorm -> q/k/v projections -> (q/k RMSNorm) -> rotary -> attention over the KV cache
    feed-forward half: output projection + residual -> post_attention_layernorm -> gate/up -> SiLU(gate) * up -> down + residual
    HEAD  = attention half of layer 0
    BLOCK = feed-forward half of layer i + attention half of layer i+1     (one graph, dispatched N-1 times with that layer's weights bound)
    TAIL  = feed-forward half of the last layer + final RMSNorm (+ last row only when S > 1)
    LM HEAD, KV UPDATE (flush of the staging buffer / write of a prefill chunk)
What is copied 1:1 from the NPUW dump: f32 block interior with f16 boundaries; weights / scales / gammas are graph INPUTS; int4 weights laid out
[groups, outputChannels, 128] with scales [groups, outputChannels, 1] (symmetric, all-zero int4 zero point); RMSNorm = gamma * (x * (1 / sqrt(mean(x^2) + eps)));
the value cache has the natural layout [1, kvHeads, T, headDim]; GQA expands K/V by reshape -> expand -> reshape.
Everything that differs between layers of different *types* (sliding vs full attention, per-type RoPE) is a graph INPUT (attention_mask, rope_cos, rope_sin).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

from ....ir import GraphBuilder, Value


@dataclass(frozen=True)
class DecoderDims:
    hidden: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    ffn: int
    vocab: int  # the vocabulary size AS THE GRAPH SEES IT (already padded)
    eps: float = 1e-6
    group_size: int = 128
    qk_norm: bool = True

    @property
    def attention_width(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def kv_width(self) -> int:
        return self.num_kv_heads * self.head_dim

    def check(self) -> None:
        for name, value in (("hidden", self.hidden), ("attention_width", self.attention_width), ("ffn", self.ffn)):
            if value % self.group_size:
                raise ValueError(f"{name}={value} is not a multiple of the quantization group size {self.group_size}")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be a multiple of num_kv_heads")
        if self.head_dim % 2:
            raise ValueError("head_dim must be even (rotate-half RoPE)")


@dataclass(frozen=True)
class GraphOptions:
    compute_type: str = "float32"  # "float32" (the NPUW form) | "float16"
    zero_point: str = "const"  # "const": all-zero int4 constant (an optional-zero-point Chromium drops it) | "none"
    merge_projections: bool = True  # q/k/v -> one projection, gate/up -> one projection
    kv_style: str = "staged"  # decode: "staged" | "onehot" | "tensorScatter" | "scatterND";  prefill (S > 1) always one-hot (or tensorScatter / scatterND)
    staging_rows: int = 64
    scatter_index_type: str = "int32"
    value_layout: str = "natural"  # "natural": value cache [1,kvHeads,P,128] like the key cache | "transposed": [1,kvHeads,128,P] (onehot only; faster than natural on the NPU for the one-hot update,
    #                                the compiler needs the attention_guard output of the prefill graphs with it)
    lm_head_format: str = "int8-sym"  # "int8-sym": int8 per-row symmetric (scale only) | "uint8-asym": uint8 per-row asymmetric (scale + zero point, the OpenVINO export)


# ---------------------------------------------------------------------------------------------------------------------
# weight tensors of one layer: (name, dtype, shape). The names are graph input names AND manifest keys.
# ---------------------------------------------------------------------------------------------------------------------
def quantized_projection(dims: DecoderDims, prefix: str, in_ch: int, out_ch: int) -> list:
    groups = in_ch // dims.group_size
    return [(f"{prefix}_weight", "int4", (groups, out_ch, dims.group_size)), (f"{prefix}_scale", "float16", (groups, out_ch, 1))]


def attention_weight_specs(dims: DecoderDims, opt: GraphOptions) -> list:
    h, aw, kw, d = dims.hidden, dims.attention_width, dims.kv_width, dims.head_dim
    q_norm = [("query_norm_gamma", "float16", (1, 1, 1, d))] if dims.qk_norm else []
    k_norm = [("key_norm_gamma", "float16", (1, 1, 1, d))] if dims.qk_norm else []
    specs = [("input_layernorm_gamma", "float16", (1, 1, h))]
    if opt.merge_projections:
        return specs + quantized_projection(dims, "qkv_projection", h, aw + 2 * kw) + q_norm + k_norm
    return (specs + quantized_projection(dims, "query", h, aw) + q_norm + quantized_projection(dims, "key", h, kw) + k_norm
            + quantized_projection(dims, "value", h, kw))


def feed_forward_weight_specs(dims: DecoderDims, opt: GraphOptions) -> list:
    h, f = dims.hidden, dims.ffn
    specs = quantized_projection(dims, "output_projection", dims.attention_width, h) + [("post_attention_layernorm_gamma", "float16", (1, 1, h))]
    if opt.merge_projections:
        specs += quantized_projection(dims, "gate_up_projection", h, 2 * f)
    else:
        specs += quantized_projection(dims, "gate_projection", h, f) + quantized_projection(dims, "up_projection", h, f)
    return specs + quantized_projection(dims, "down_projection", f, h)


def tail_weight_specs(dims: DecoderDims, opt: GraphOptions) -> list:
    return feed_forward_weight_specs(dims, opt) + [("final_norm_gamma", "float16", (1, 1, dims.hidden))]


def lm_head_weight_specs(dims: DecoderDims, opt: "GraphOptions | None" = None) -> list:
    """The tied embedding / lm_head table, one scale (and one zero point) per vocabulary row.
    int8-sym (default): weight int8 + scale; uint8-asym: weight uint8 + zero point + scale."""
    if (opt or GraphOptions()).lm_head_format == "uint8-asym":
        return [("lm_head_weight", "uint8", (dims.vocab, dims.hidden)), ("lm_head_zero_point", "uint8", (dims.vocab, 1)), ("lm_head_scale", "float16", (dims.vocab, 1))]
    return [("lm_head_weight", "int8", (dims.vocab, dims.hidden)), ("lm_head_scale", "float16", (dims.vocab, 1))]


# ---------------------------------------------------------------------------------------------------------------------
# building context + small helpers
# ---------------------------------------------------------------------------------------------------------------------
class Ctx:
    def __init__(self, b: GraphBuilder, dims: DecoderDims, opt: GraphOptions):
        dims.check()
        self.b, self.dims, self.opt = b, dims, opt
        self.compute = opt.compute_type
        self.seq = 0
        self.rope = None
        self.per_block_controls = False   # block groups of hybrid models: every block declares its own rope_cos / rope_sin / attention_mask (b<j>_...)

    def declare(self, name: str, dtype: str, shape, role: str = "activation") -> Value:
        return self.b.input(name, dtype, shape, role)

    def shared(self, name: str, dtype: str, shape, role: str = "activation") -> Value:
        """An input that all layers of a multi-layer graph share: declared once."""
        if name in self.b.graph.inputs:
            return self.b.graph.values[name]
        return self.declare(name, dtype, shape, role)

    def weights(self, specs, prefix: str = "") -> dict:
        return {name: self.declare(prefix + name, dtype, shape, "weight") for name, dtype, shape in specs}

    def scalar(self, value: float, rank: int) -> Value:
        return self.b.scalar(value, self.compute, rank)

    def zero_point(self, shape) -> Value:
        return self.b.constant("int4", shape, fill=0, cache_key=("zp", "int4", tuple(shape)))

    def to_compute(self, x: Value) -> Value:
        return x if x.dtype == self.compute else self.b.cast(x, self.compute)

    def to_f16(self, x: Value) -> Value:
        if x.dtype != "float16":
            return self.b.cast(x, "float16")
        # an operand that is directly a graph input cannot be a graph output: with computeType=float16 the cast is a no-op, so copy it
        return self.b.reshape(x, x.shape) if x.name in self.b.graph.inputs else x


def quantized_linear(c: Ctx, x: Value, weight: Value, scale: Value, in_ch: int, out_ch: int, seq: int) -> Value:
    """y = x @ W^T, W symmetric int4 [groups, out, 128] + scale [groups, out, 1]:
    dequantizeLinear -> transpose [1,0,2] -> reshape [out, in] -> cast -> gemm(transposeB)."""
    b, groups = c.b, in_ch // c.dims.group_size
    zp = None if c.opt.zero_point == "none" else c.zero_point((groups, out_ch, 1))
    dequantized = b.dequantize_linear(weight, scale, zp)
    matrix = c.to_compute(b.reshape(b.transpose(dequantized, [1, 0, 2]), [out_ch, in_ch]))
    rows = b.reshape(x, [seq, in_ch])
    return b.reshape(b.gemm(rows, matrix, b_transpose=True), [1, seq, out_ch])


def rms_norm(c: Ctx, x: Value, gamma: Value, rank: int) -> Value:
    """gamma * (x * (1 / sqrt(mean(x^2) + eps))) over the last axis -- the NPUW form the NPU compiler pattern-matches."""
    b = c.b
    mean_square = b.reduce_mean(b.pow(x, c.scalar(2, rank)), axes=[rank - 1], keep_dimensions=True)
    rms = b.sqrt(b.add(mean_square, c.scalar(c.dims.eps, rank)))
    return b.mul(c.to_compute(gamma), b.mul(x, b.div(c.scalar(1, rank), rms)))


def apply_rotary(c: Ctx, x: Value, cos: Value, sin: Value, heads: int) -> Value:
    """x * cos + rotate_half(x) * sin on x [1, heads, S, head_dim] (HF half-split layout)."""
    b, s, half = c.b, c.seq, c.dims.head_dim // 2
    first = b.slice(x, [0, 0, 0, 0], [1, heads, s, half])
    second = b.slice(x, [0, 0, 0, half], [1, heads, s, half])
    rotated = b.concat([b.mul(second, c.scalar(-1, 4)), first], 3)
    return b.add(b.mul(x, c.to_compute(cos)), b.mul(rotated, c.to_compute(sin)))


def repeat_kv(c: Ctx, x: Value, trailing) -> Value:
    """GQA: [1, kvHeads, ...] -> reshape [1,kvHeads,1,...] -> expand [1,kvHeads,group,...] -> reshape [1,heads,...]."""
    b, d = c.b, c.dims
    group = d.num_heads // d.num_kv_heads
    expanded = b.expand(b.reshape(x, [1, d.num_kv_heads, 1, *trailing]), [1, d.num_kv_heads, group, *trailing])
    return b.reshape(expanded, [1, d.num_heads, *trailing])


def update_kv_cache(c: Ctx, past_key, past_value, key, value, past_len: int, seq: int, selector_prefix: str = "", style: str = "onehot", transposed_value: bool = False):
    """Write the new key / value rows into a fixed-size cache inside the graph. Returns (updated_key, updated_value).
    onehot       : past * rows_to_keep + row_selector @ new     (rows_to_keep [1,1,P,1], row_selector [1,1,P,S], both host inputs)
    tensorScatter: tensorScatter(past, new, kv_write_index, axis=2)   (one start row; the S rows go in contiguously)
    scatterND    : scatterND(past, kv_indices, new)                    (kv_indices int32 [1,kvHeads,S,3] = (0, head, cache row) of every new row)"""
    b, d = c.b, c.dims
    present_key, present_value = c.to_f16(key), c.to_f16(value)
    if style == "scatterND":
        indices = c.shared("kv_indices", "int32", [1, d.num_kv_heads, seq, 3], "control")
        return b.scatter_nd(past_key, indices, present_key), b.scatter_nd(past_value, indices, present_value)
    if style == "tensorScatter":
        index = c.shared("kv_write_index", c.opt.scatter_index_type, [1], "control")
        return b.tensor_scatter(past_key, present_key, index, axis=2), b.tensor_scatter(past_value, present_value, index, axis=2)
    row_selector = c.shared(selector_prefix + "row_selector", "float16", [1, 1, past_len, seq], "control")
    rows_to_keep = c.shared(selector_prefix + "rows_to_keep", "float16", [1, 1, past_len, 1], "control")
    if transposed_value:   # value cache [1,kvHeads,128,P], new rows [1,kvHeads,128,S]: the selectors are transposed inside the graph
        flip = lambda x: b.transpose(x, [0, 1, 3, 2])   # noqa: E731
        return (b.add(b.mul(past_key, rows_to_keep), b.matmul(row_selector, present_key)),
                b.add(b.mul(past_value, flip(rows_to_keep)), b.matmul(present_value, flip(row_selector))))
    return (b.add(b.mul(past_key, rows_to_keep), b.matmul(row_selector, present_key)),
            b.add(b.mul(past_value, rows_to_keep), b.matmul(row_selector, present_value)))


def attention_half(c: Ctx, residual: Value, w: dict, seq: int, past_len: int, mask_dtype: str, prefix: str = ""):
    """One layer's attention. Returns (attention_output f16 [1,S,attention_width], kv_outputs {name: Value})."""
    b, d, opt = c.b, c.dims, c.opt
    c.seq = seq
    use_scatter = opt.kv_style in ("tensorScatter", "scatterND") and past_len > 0
    staged = past_len > 0 and seq == 1 and opt.kv_style == "staged"
    staging_rows = opt.staging_rows
    # tensorScatter / scatterND: attention reads the scattered cache itself (length P); staged: concat(main, staging); otherwise concat(past, new rows)
    total = past_len if use_scatter else past_len + staging_rows if staged else past_len + seq

    control = prefix if c.per_block_controls else ""   # the rope tables / mask are shared by all blocks of a group, unless every block gets its own (layers of different types)
    cos = c.shared(control + "rope_cos", "float16", [1, 1, seq, d.head_dim], "control")
    sin = c.shared(control + "rope_sin", "float16", [1, 1, seq, d.head_dim], "control")
    mask = c.shared(control + "attention_mask", mask_dtype, [1, 1, seq, total], "control")
    past_key = c.declare(prefix + "past_key", "float16", [1, d.num_kv_heads, past_len, d.head_dim], "kv") if past_len > 0 else None
    transposed = opt.value_layout == "transposed"
    if transposed and opt.kv_style != "onehot":
        raise ValueError("the transposed value layout is only available with the onehot KV protocol")
    value_shape = [1, d.num_kv_heads, d.head_dim, past_len] if transposed else [1, d.num_kv_heads, past_len, d.head_dim]
    past_value = c.declare(prefix + "past_value", "float16", value_shape, "kv") if past_len > 0 else None

    # 1. normalise, project to q / k / v
    normalized = rms_norm(c, residual, w["input_layernorm_gamma"], 3)

    def split_heads(projected, heads, norm_gamma):  # [1,S,heads*hd] -> (per-head RMSNorm) -> [1,heads,S,hd]
        x = b.reshape(projected, [1, seq, heads, d.head_dim])
        if d.qk_norm:
            x = rms_norm(c, x, norm_gamma, 4)
        return b.transpose(x, [0, 2, 1, 3])

    if "qkv_projection_weight" in w:
        qkv = quantized_linear(c, normalized, w["qkv_projection_weight"], w["qkv_projection_scale"], d.hidden, d.attention_width + 2 * d.kv_width, seq)
        part = lambda start, width: b.slice(qkv, [0, 0, start], [1, seq, width])  # noqa: E731
        q_proj, k_proj, v_proj = part(0, d.attention_width), part(d.attention_width, d.kv_width), part(d.attention_width + d.kv_width, d.kv_width)
    else:
        q_proj = quantized_linear(c, normalized, w["query_weight"], w["query_scale"], d.hidden, d.attention_width, seq)
        k_proj = quantized_linear(c, normalized, w["key_weight"], w["key_scale"], d.hidden, d.kv_width, seq)
        v_proj = quantized_linear(c, normalized, w["value_weight"], w["value_scale"], d.hidden, d.kv_width, seq)
    query = apply_rotary(c, split_heads(q_proj, d.num_heads, w.get("query_norm_gamma")), cos, sin, d.num_heads)
    key = apply_rotary(c, split_heads(k_proj, d.num_kv_heads, w.get("key_norm_gamma")), cos, sin, d.num_kv_heads)
    value = b.transpose(b.reshape(v_proj, [1, seq, d.num_kv_heads, d.head_dim]), [0, 2, 3, 1] if transposed else [0, 2, 1, 3])   # natural layout [1,kvHeads,S,hd] | transposed [1,kvHeads,hd,S]
    scaled_query = b.mul(query, c.scalar(1.0 / math.sqrt(d.head_dim), 4))

    # 2. attention over [cache + new rows]; the cache update is part of the same graph
    to_c = c.to_compute
    kv_out: dict = {}
    if use_scatter:
        updated_key, updated_value = update_kv_cache(c, past_key, past_value, key, value, past_len, seq, style=opt.kv_style)
        all_keys, all_values = to_c(updated_key), to_c(updated_value)
        kv_out = {"updated_key": updated_key, "updated_value": updated_value}
    elif staged:
        staging_key = c.declare(prefix + "staging_key", "float16", [1, d.num_kv_heads, staging_rows, d.head_dim], "kv")
        staging_value = c.declare(prefix + "staging_value", "float16", [1, d.num_kv_heads, staging_rows, d.head_dim], "kv")
        updated_key, updated_value = update_kv_cache(c, staging_key, staging_value, key, value, staging_rows, seq, selector_prefix="staging_")
        all_keys = b.concat([to_c(past_key), to_c(updated_key)], 2)
        all_values = b.concat([to_c(past_value), to_c(updated_value)], 2)
        kv_out = {"updated_key": updated_key, "updated_value": updated_value}
    elif past_len > 0:
        updated_key, updated_value = update_kv_cache(c, past_key, past_value, key, value, past_len, seq, transposed_value=transposed)
        all_keys = b.concat([to_c(past_key), to_c(key)], 2)
        all_values = b.concat([to_c(past_value), to_c(value)], 3 if transposed else 2)
        kv_out = {"updated_key": updated_key, "updated_value": updated_value}
    else:
        all_keys, all_values = to_c(key), to_c(value)
        kv_out = {"present_key": c.to_f16(key), "present_value": c.to_f16(value)}

    keys_t = b.transpose(repeat_kv(c, all_keys, [total, d.head_dim]), [0, 1, 3, 2])           # [1,heads,hd,T]
    scores = b.add(b.matmul(to_c(scaled_query), keys_t), to_c(mask))                           # [1,heads,S,T]
    probabilities = b.softmax(scores, 3)
    if transposed:   # [1,kvHeads,hd,T] -> expand heads -> [1,heads,hd,T] -> [1,heads,T,hd]
        values_expanded = b.transpose(repeat_kv(c, all_values, [d.head_dim, total]), [0, 1, 3, 2])
        if seq > 1 and past_len > 0:
            # the NPU compiler returned WRONG attention results for S > 1 with a past cache and a transposed value cache unless the expanded values have a second (tiny) consumer
            kv_out["attention_guard"] = c.to_f16(b.reduce_max(values_expanded, [3], True))
    else:
        values_expanded = repeat_kv(c, all_values, [total, d.head_dim])                        # [1,heads,T,hd]
    context = b.matmul(probabilities, values_expanded)                                         # [1,heads,S,hd]
    attention_output = b.reshape(b.transpose(context, [0, 2, 1, 3]), [1, seq, d.attention_width])
    return c.to_f16(attention_output), kv_out


def feed_forward_half(c: Ctx, residual: Value, attention_output: Value, w: dict, seq: int) -> Value:
    """h = residual + o_proj(attention_output);  out = h + down(silu(gate(norm(h))) * up(norm(h)))."""
    b, d = c.b, c.dims
    after_attention = b.add(residual, quantized_linear(c, attention_output, w["output_projection_weight"], w["output_projection_scale"], d.attention_width, d.hidden, seq))
    normalized = rms_norm(c, after_attention, w["post_attention_layernorm_gamma"], 3)
    if "gate_up_projection_weight" in w:
        gate_up = quantized_linear(c, normalized, w["gate_up_projection_weight"], w["gate_up_projection_scale"], d.hidden, 2 * d.ffn, seq)
        gate = b.slice(gate_up, [0, 0, 0], [1, seq, d.ffn])
        up = b.slice(gate_up, [0, 0, d.ffn], [1, seq, d.ffn])
    else:
        gate = quantized_linear(c, normalized, w["gate_projection_weight"], w["gate_projection_scale"], d.hidden, d.ffn, seq)
        up = quantized_linear(c, normalized, w["up_projection_weight"], w["up_projection_scale"], d.hidden, d.ffn, seq)
    silu_gate = b.mul(gate, b.sigmoid(gate))
    down = quantized_linear(c, b.mul(silu_gate, up), w["down_projection_weight"], w["down_projection_scale"], d.ffn, d.hidden, seq)
    return b.add(after_attention, down)


# ---------------------------------------------------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------------------------------------------------
def build_head_graph(dims: DecoderDims, opt: GraphOptions, *, seq: int, past_len: int, mask_dtype: str = "float16", name: str = "head"):
    b = GraphBuilder(name)
    c = Ctx(b, dims, opt)
    hidden_states = c.declare("hidden_states", "float16", [1, seq, dims.hidden])
    weights = c.weights(attention_weight_specs(dims, opt))
    residual = c.to_compute(hidden_states)
    attention_output, kv_out = attention_half(c, residual, weights, seq, past_len, mask_dtype)
    b.output("residual", c.to_f16(residual))
    b.output("attention_output", attention_output)
    for k, v in kv_out.items():
        b.output(k, v)
    return b.build()


def build_block_graph(dims: DecoderDims, opt: GraphOptions, *, seq: int, past_len: int, mask_dtype: str = "float16", name: str = "block"):
    b = GraphBuilder(name)
    c = Ctx(b, dims, opt)
    residual = c.declare("residual", "float16", [1, seq, dims.hidden])
    attention_output = c.declare("attention_output", "float16", [1, seq, dims.attention_width])
    layer_output = feed_forward_half(c, c.to_compute(residual), c.to_compute(attention_output), c.weights(feed_forward_weight_specs(dims, opt)), seq)
    next_attention, kv_out = attention_half(c, layer_output, c.weights(attention_weight_specs(dims, opt)), seq, past_len, mask_dtype)
    b.output("next_residual", c.to_f16(layer_output))
    b.output("next_attention_output", next_attention)
    for k, v in kv_out.items():
        b.output(k, v)
    return b.build()


def build_block_group_graph(dims: DecoderDims, opt: GraphOptions, *, block_count: int, seq: int, past_len: int, mask_dtype: str = "float16", name: str | None = None,
                            per_block_controls: bool = False):
    """``block_count`` consecutive BLOCKs in one graph (fewer dispatches). Block j declares its inputs / outputs with the prefix ``b<j>_``;
    selectors are shared; rope / mask are shared too, unless ``per_block_controls``: then every block declares ``b<j>_rope_cos`` / ``b<j>_rope_sin`` /
    ``b<j>_attention_mask`` (a model whose layers have different types, e.g. sliding and full attention, groups any consecutive blocks that way)."""
    b = GraphBuilder(name or f"block_x{block_count}")
    c = Ctx(b, dims, opt)
    c.per_block_controls = per_block_controls
    residual = c.to_compute(c.declare("residual", "float16", [1, seq, dims.hidden]))
    attention_output = c.to_compute(c.declare("attention_output", "float16", [1, seq, dims.attention_width]))
    last_attention = None
    for j in range(block_count):
        prefix = f"b{j}_"
        layer_output = feed_forward_half(c, residual, attention_output, c.weights(feed_forward_weight_specs(dims, opt), prefix), seq)
        attention, kv_out = attention_half(c, layer_output, c.weights(attention_weight_specs(dims, opt), prefix), seq, past_len, mask_dtype, prefix=prefix)
        for k, v in kv_out.items():
            b.output(prefix + k, v)
        residual, last_attention = layer_output, attention
        if j < block_count - 1:
            attention_output = c.to_compute(attention)
        else:
            b.output("next_residual", c.to_f16(layer_output))
    b.output("next_attention_output", last_attention)
    return b.build()


def build_tail_graph(dims: DecoderDims, opt: GraphOptions, *, seq: int, name: str = "tail"):
    """FFN of the last layer + final RMSNorm; for S > 1 only the LAST row is kept (only the last prompt token yields the first output)."""
    b = GraphBuilder(name)
    c = Ctx(b, dims, opt)
    residual = c.declare("residual", "float16", [1, seq, dims.hidden])
    attention_output = c.declare("attention_output", "float16", [1, seq, dims.attention_width])
    weights = c.weights(tail_weight_specs(dims, opt))
    layer_output = feed_forward_half(c, c.to_compute(residual), c.to_compute(attention_output), weights, seq)
    final = rms_norm(c, layer_output, weights["final_norm_gamma"], 3)
    if seq > 1:
        final = b.slice(final, [0, seq - 1, 0], [1, 1, dims.hidden])
    b.output("final_hidden_states", c.to_f16(final))
    return b.build()


def build_lm_head_graph(dims: DecoderDims, opt: GraphOptions, *, weight_mode: str = "constant", name: str = "lm_head"):
    """final_hidden_states f16 [1,1,E] -> logits f16 [1,1,V]; weight int8 [V,E] per-row symmetric (default) or uint8 asymmetric: dequantizeLinear -> cast -> gemm(transposeB).
    weight_mode "constant": the weights are external constants of the graph (as in the NPUW dump) | "input": graph inputs."""
    b = GraphBuilder(name)
    c = Ctx(b, dims, opt)
    hidden = c.to_compute(c.declare("final_hidden_states", "float16", [1, 1, dims.hidden]))
    w = {}
    for wname, dtype, shape in lm_head_weight_specs(dims, opt):
        w[wname] = c.declare(wname, dtype, shape, "weight") if weight_mode == "input" else b.constant(dtype, shape, external=wname)
    # int8-sym: an all-zero int8 zero point (the optional zero point is dropped by some Chromium builds; this is what the hand-written graph does)
    zero_point = w.get("lm_head_zero_point") or b.constant("int8", (dims.vocab, 1), fill=0, cache_key=("zp", "int8", dims.vocab))
    dequantized = c.to_compute(b.dequantize_linear(w["lm_head_weight"], w["lm_head_scale"], zero_point))   # [V, E]
    logits = b.gemm(b.reshape(hidden, [1, dims.hidden]), dequantized, b_transpose=True)
    b.output("logits", c.to_f16(b.reshape(logits, [1, 1, dims.vocab])))
    return b.build()


def build_kv_update_graph(dims: DecoderDims, *, seq: int, past_len: int, name: str = "kv_update"):
    """Merge ``seq`` new key / value rows into one layer's cache on the device:  updated = past * rows_to_keep + row_selector @ new
    (also the flush graph of the staged KV cache, with seq = staging rows)."""
    b = GraphBuilder(name)
    d = dims
    past_key = b.input("past_key", "float16", [1, d.num_kv_heads, past_len, d.head_dim], "kv")
    new_key = b.input("new_key", "float16", [1, d.num_kv_heads, seq, d.head_dim])
    past_value = b.input("past_value", "float16", [1, d.num_kv_heads, past_len, d.head_dim], "kv")
    new_value = b.input("new_value", "float16", [1, d.num_kv_heads, seq, d.head_dim])
    rows_to_keep = b.input("rows_to_keep", "float16", [1, 1, past_len, 1], "control")
    row_selector = b.input("row_selector", "float16", [1, 1, past_len, seq], "control")
    b.output("updated_key", b.add(b.mul(past_key, rows_to_keep), b.matmul(row_selector, new_key)))
    b.output("updated_value", b.add(b.mul(past_value, rows_to_keep), b.matmul(row_selector, new_value)))
    return b.build()


def with_options(opt: GraphOptions, **changes) -> GraphOptions:
    return replace(opt, **changes)

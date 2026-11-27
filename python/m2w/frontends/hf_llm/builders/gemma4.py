"""Static-shape subgraphs of the Gemma 4 text decoder in the same NPUW-style split as decoder.py (HEAD / BLOCK / TAIL / lm_head / KV flush).

What differs from decoder.py (see docs/models/gemma4-e4b.md):
  * the attention half ends with the output projection and post_attention_layernorm, so the interface between the halves is `attention_output` [1,S,hidden], and the feed-forward half is the
    same for every layer: x1 = residual + attention_output; MLP with pre / post norms; the per-layer-embedding (PLE) branch; the layer scalar;
  * four kinds of attention half (layer type x own / shared K / V), because head dims differ per layer type and the last layers read the K / V of another layer: one BLOCK graph per kind
    (BLOCK i = feed-forward half of layer i + attention half of layer i + 1, of the kind of layer i + 1); HEAD = attention half of layer 0 + the PLE prologue; the K / V caches of the
    producer layers are read by the shared layers (decode: main cache + the staging buffer AFTER this step's update; prefill: the updated cache itself, mask columns = cache rows);
  * attention scale 1.0, a weightless v norm, RMSNorm weights used as they are, gelu-tanh, a logit softcap in the lm_head graph.
Only the staged KV protocol (decode) with the one-hot chunk update (prefill) is built.
Sliding window caches (opt.sliding_cache = "window"): `past_len` is the length of the FULL cache; the layers of a sliding kind hold a ring of cache_rows() rows (a graph of such a kind is
built with that length: cache inputs, mask columns, selectors). The kinds already separate sliding from full layers, so no graph is shared between the two.
"""
from __future__ import annotations

import math

from ....ir import GraphBuilder, Value
from .decoder import Ctx, DecoderDims, GraphOptions, apply_rotary, build_kv_update_graph, cache_rows, quantized_linear, repeat_kv, rms_norm, update_kv_cache

KINDS = ("sliding_attention_own", "full_attention_own", "sliding_attention_shared", "full_attention_shared")


def kind_type(kind: str) -> str:
    return kind.rsplit("_", 1)[0]


def kind_shared(kind: str) -> bool:
    return kind.endswith("_shared")


# ---------------------------------------------------------------------------------------------------------------------
# weight specs: (name, dtype, shape); the names are graph input names AND manifest keys
# ---------------------------------------------------------------------------------------------------------------------
def _linear(name: str, in_ch: int, out_ch: int, group: int) -> list:
    return [(f"{name}_weight", "int4", (in_ch // group, out_ch, group)), (f"{name}_scale", "float16", (in_ch // group, out_ch, 1))]


def attention_weight_specs(spec, kind: str) -> list:
    d = spec.decoder_dims(kind_type(kind))
    h, g = spec.hidden, spec.group_size
    out = [("input_layernorm_gamma", "float16", (1, 1, h))]
    if kind_shared(kind):
        out += _linear("query_projection", h, d.attention_width, g) + [("query_norm_gamma", "float16", (1, 1, 1, d.head_dim))]
    else:
        out += _linear("qkv_projection", h, d.attention_width + 2 * d.kv_width, g) + [("query_norm_gamma", "float16", (1, 1, 1, d.head_dim)), ("key_norm_gamma", "float16", (1, 1, 1, d.head_dim))]
    return out + _linear("output_projection", d.attention_width, h, g) + [("post_attention_layernorm_gamma", "float16", (1, 1, h))]


def feed_forward_weight_specs(spec) -> list:
    h, f, p, g = spec.hidden, spec.ffn, spec.ple_dim, spec.group_size
    return ([("pre_feedforward_layernorm_gamma", "float16", (1, 1, h))] + _linear("gate_up_projection", h, 2 * f, g) + _linear("down_projection", f, h, g)
            + [("post_feedforward_layernorm_gamma", "float16", (1, 1, h))] + _linear("per_layer_input_gate", h, p, g) + _linear("per_layer_projection", p, h, g)
            + [("post_per_layer_input_norm_gamma", "float16", (1, 1, h)), ("layer_scalar", "float16", (1, 1, 1))])


def head_extra_weight_specs(spec) -> list:
    """The PLE prologue of HEAD: the projection of the (scaled) embedding to num_layers x ple_dim and its norm."""
    return _linear("ple_model_projection", spec.hidden, spec.num_layers * spec.ple_dim, spec.group_size) + [("ple_projection_norm_gamma", "float16", (1, 1, 1, spec.ple_dim))]


def final_norm_specs(spec) -> list:
    return [("final_norm_gamma", "float16", (1, 1, spec.hidden))]


def lm_head_weight_specs(spec) -> list:
    return [("lm_head_weight", "int8", (spec.vocab_padded, spec.hidden)), ("lm_head_scale", "float16", (spec.vocab_padded, 1))]


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def rms_norm_no_scale(c: Ctx, x: Value, rank: int) -> Value:
    b = c.b
    mean_square = b.reduce_mean(b.pow(x, c.scalar(2, rank)), axes=[rank - 1], keep_dimensions=True)
    rms = b.sqrt(b.add(mean_square, c.scalar(c.dims.eps, rank)))
    return b.mul(x, b.div(c.scalar(1, rank), rms))


def gelu_tanh(c: Ctx, x: Value, rank: int) -> Value:
    """0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))"""
    b = c.b
    cube = b.mul(b.mul(x, x), x)
    inner = b.mul(b.add(x, b.mul(cube, c.scalar(0.044715, rank))), c.scalar(math.sqrt(2.0 / math.pi), rank))
    return b.mul(b.mul(x, c.scalar(0.5, rank)), b.add(b.tanh(inner), c.scalar(1.0, rank)))


def _ctx(b: GraphBuilder, spec, opt: GraphOptions, layer_type: str) -> Ctx:
    return Ctx(b, spec.decoder_dims(layer_type), opt)


# ---------------------------------------------------------------------------------------------------------------------
# the halves
# ---------------------------------------------------------------------------------------------------------------------
def attention_half(c: Ctx, spec, kind: str, residual: Value, w: dict, seq: int, past_len: int, opt: GraphOptions, mask_dtype: str = "float16", bind: "dict | None" = None):
    """-> (attention_output f16 [1,S,hidden] = post_attention_layernorm(o_proj(attention)), kv_out {name: Value})
    `bind` (block groups): the already declared inputs {cos, sin, mask, past_key, past_value, staging_key, staging_value} instead of declaring them here."""
    b, d = c.b, c.dims
    c.seq = seq
    shared, decode = kind_shared(kind), seq == 1
    past_len = cache_rows(opt, kind_type(kind), past_len)
    R = opt.staging_rows
    to_c = c.to_compute
    # which K / V does the attention read, and how long is it
    if shared:
        total = past_len + R if decode else past_len
    else:
        total = past_len + R if decode else past_len + seq
    if bind:
        cos, sin, mask, past_key, past_value = bind["cos"], bind["sin"], bind["mask"], bind["past_key"], bind["past_value"]
    else:
        cos = c.declare("rope_cos", "float16", [1, 1, seq, d.head_dim], "control")
        sin = c.declare("rope_sin", "float16", [1, 1, seq, d.head_dim], "control")
        mask = c.declare("attention_mask", mask_dtype, [1, 1, seq, total], "control")
        past_key = c.declare("past_key", "float16", [1, d.num_kv_heads, past_len, d.head_dim], "kv")
        past_value = c.declare("past_value", "float16", [1, d.num_kv_heads, past_len, d.head_dim], "kv")

    def staging():
        if bind:
            return bind["staging_key"], bind["staging_value"]
        return (c.declare("staging_key", "float16", [1, d.num_kv_heads, R, d.head_dim], "kv"), c.declare("staging_value", "float16", [1, d.num_kv_heads, R, d.head_dim], "kv"))

    normalized = rms_norm(c, residual, w["input_layernorm_gamma"], 3)

    def heads_of(projected, heads, gamma):   # [1,S,heads*d] -> per-head RMSNorm -> [1,heads,S,d]
        x = b.reshape(projected, [1, seq, heads, d.head_dim])
        x = rms_norm(c, x, gamma, 4) if gamma is not None else rms_norm_no_scale(c, x, 4)
        return b.transpose(x, [0, 2, 1, 3])

    kv_out: dict = {}
    if shared:
        q_proj = quantized_linear(c, normalized, w["query_projection_weight"], w["query_projection_scale"], d.hidden, d.attention_width, seq)
        query = apply_rotary(c, heads_of(q_proj, d.num_heads, w["query_norm_gamma"]), cos, sin, d.num_heads)
        if decode:   # the producer's main cache + its staging buffer after this step's update
            staging_key, staging_value = staging()
            all_keys, all_values = b.concat([to_c(past_key), to_c(staging_key)], 2), b.concat([to_c(past_value), to_c(staging_value)], 2)
        else:        # the producer's cache after the update of this chunk
            all_keys, all_values = to_c(past_key), to_c(past_value)
    else:
        qkv = quantized_linear(c, normalized, w["qkv_projection_weight"], w["qkv_projection_scale"], d.hidden, d.attention_width + 2 * d.kv_width, seq)
        part = lambda start, width: b.slice(qkv, [0, 0, start], [1, seq, width])  # noqa: E731
        query = apply_rotary(c, heads_of(part(0, d.attention_width), d.num_heads, w["query_norm_gamma"]), cos, sin, d.num_heads)
        key = apply_rotary(c, heads_of(part(d.attention_width, d.kv_width), d.num_kv_heads, w["key_norm_gamma"]), cos, sin, d.num_kv_heads)
        value = heads_of(part(d.attention_width + d.kv_width, d.kv_width), d.num_kv_heads, None)
        if decode:
            staging_key, staging_value = staging()
            updated_key, updated_value = update_kv_cache(c, staging_key, staging_value, key, value, R, seq, selector_prefix="staging_")
            all_keys, all_values = b.concat([to_c(past_key), to_c(updated_key)], 2), b.concat([to_c(past_value), to_c(updated_value)], 2)
        else:
            updated_key, updated_value = update_kv_cache(c, past_key, past_value, key, value, past_len, seq)
            all_keys, all_values = b.concat([to_c(past_key), to_c(key)], 2), b.concat([to_c(past_value), to_c(value)], 2)
        kv_out = {"updated_key": updated_key, "updated_value": updated_value}

    scaled = query if spec.attention_scale == 1.0 else b.mul(query, c.scalar(spec.attention_scale or 1.0 / math.sqrt(d.head_dim), 4))
    keys_t = b.transpose(repeat_kv(c, all_keys, [total, d.head_dim]), [0, 1, 3, 2])
    scores = b.add(b.matmul(to_c(scaled), keys_t), to_c(mask))
    context = b.matmul(b.softmax(scores, 3), repeat_kv(c, all_values, [total, d.head_dim]))
    context = b.reshape(b.transpose(context, [0, 2, 1, 3]), [1, seq, d.attention_width])
    projected = quantized_linear(c, context, w["output_projection_weight"], w["output_projection_scale"], d.attention_width, d.hidden, seq)
    return c.to_f16(rms_norm(c, projected, w["post_attention_layernorm_gamma"], 3)), kv_out


def feed_forward_half(c: Ctx, spec, residual: Value, attention_output: Value, per_layer_input: Value, w: dict, seq: int) -> Value:
    """x1 = residual + attention_output; x2 = x1 + post_ffn_norm(mlp(pre_ffn_norm(x1))); x3 = x2 + post_ple_norm(ple_proj(gelu(ple_gate(x2)) * per_layer_input)); out = x3 * layer_scalar"""
    b, h, f, p = c.b, spec.hidden, spec.ffn, spec.ple_dim
    x1 = b.add(residual, attention_output)
    normalized = rms_norm(c, x1, w["pre_feedforward_layernorm_gamma"], 3)
    gate_up = quantized_linear(c, normalized, w["gate_up_projection_weight"], w["gate_up_projection_scale"], h, 2 * f, seq)
    gate, up = b.slice(gate_up, [0, 0, 0], [1, seq, f]), b.slice(gate_up, [0, 0, f], [1, seq, f])
    mlp = quantized_linear(c, b.mul(gelu_tanh(c, gate, 3), up), w["down_projection_weight"], w["down_projection_scale"], f, h, seq)
    x2 = b.add(x1, rms_norm(c, mlp, w["post_feedforward_layernorm_gamma"], 3))
    gated = b.mul(gelu_tanh(c, quantized_linear(c, x2, w["per_layer_input_gate_weight"], w["per_layer_input_gate_scale"], h, p, seq), 3), c.to_compute(per_layer_input))
    ple = quantized_linear(c, gated, w["per_layer_projection_weight"], w["per_layer_projection_scale"], p, h, seq)
    x3 = b.add(x2, rms_norm(c, ple, w["post_per_layer_input_norm_gamma"], 3))
    return b.mul(x3, c.to_compute(w["layer_scalar"]))


# ---------------------------------------------------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------------------------------------------------
def build_head_graph(spec, opt: GraphOptions, *, seq: int, past_len: int, name: str = "head"):
    """hidden_states (the embedding times sqrt(hidden)) + ple_token (the PLE table rows of the tokens, [1,S,layers*ple]) -> the per-layer inputs ple_0 .. ple_{N-1} and the attention half of layer 0."""
    b = GraphBuilder(name)
    kind = spec.layer_kind(0)
    c = _ctx(b, spec, opt, spec.layer_types[0])
    hidden_states = c.declare("hidden_states", "float16", [1, seq, spec.hidden])
    ple_token = c.declare("ple_token", "float16", [1, seq, spec.num_layers * spec.ple_dim])
    ple_w = c.weights(head_extra_weight_specs(spec))
    attn_w = c.weights(attention_weight_specs(spec, kind))
    x = c.to_compute(hidden_states)
    p, n = spec.ple_dim, spec.num_layers
    projected = quantized_linear(c, x, ple_w["ple_model_projection_weight"], ple_w["ple_model_projection_scale"], spec.hidden, n * p, seq)
    projected = b.mul(projected, c.scalar(spec.hidden ** -0.5, 3))
    projected = rms_norm(c, b.reshape(projected, [1, seq, n, p]), ple_w["ple_projection_norm_gamma"], 4)
    token = b.mul(b.reshape(c.to_compute(ple_token), [1, seq, n, p]), c.scalar(p ** 0.5, 4))
    combined = b.mul(b.add(projected, token), c.scalar(2.0 ** -0.5, 4))
    attention_output, kv_out = attention_half(c, spec, kind, x, attn_w, seq, past_len, opt)
    b.output("residual", c.to_f16(x))
    b.output("attention_output", attention_output)
    for k, v in kv_out.items():
        b.output(k, v)
    for i in range(n):
        b.output(f"ple_{i}", c.to_f16(b.reshape(b.slice(combined, [0, 0, i, 0], [1, seq, 1, p]), [1, seq, p])))
    return b.build()


def build_block_graph(spec, opt: GraphOptions, kind: str, *, seq: int, past_len: int, name: str = "block"):
    """feed-forward half of layer i (uniform) + attention half of layer i + 1 of `kind`."""
    b = GraphBuilder(name)
    c = _ctx(b, spec, opt, kind_type(kind))
    residual = c.declare("residual", "float16", [1, seq, spec.hidden])
    attention_output = c.declare("attention_output", "float16", [1, seq, spec.hidden])
    per_layer_input = c.declare("per_layer_input", "float16", [1, seq, spec.ple_dim])
    layer_output = feed_forward_half(c, spec, c.to_compute(residual), c.to_compute(attention_output), per_layer_input, c.weights(feed_forward_weight_specs(spec)), seq)
    next_attention, kv_out = attention_half(c, spec, kind, layer_output, c.weights(attention_weight_specs(spec, kind)), seq, past_len, opt)
    b.output("next_residual", c.to_f16(layer_output))
    b.output("next_attention_output", next_attention)
    for k, v in kv_out.items():
        b.output(k, v)
    return b.build()


# ---------------------------------------------------------------------------------------------------------------------
# block groups (decode only): several consecutive BLOCKs in one graph
# ---------------------------------------------------------------------------------------------------------------------
# the key of a group graph spells its kinds (lower case only: graph file names must not differ in case alone): s / f = sliding / full attention, o / s = K / V of its own / shared
KIND_CODES = {"sliding_attention_own": "so", "full_attention_own": "fo", "sliding_attention_shared": "ss", "full_attention_shared": "fs"}


def block_group_schedule(spec, count: int) -> list:
    """The N - 1 BLOCKs cut into runs of `count`: [(first block, blocks)]; the last run may be shorter."""
    total = spec.num_layers - 1
    return [(first, min(count, total - first)) for first in range(0, total, count)]


def block_group_layout(spec, first: int, blocks: int) -> list:
    """What a group of BLOCKs first .. first + blocks - 1 needs to be built: per block (its attention half is that of layer first + j + 1) the kind and the OWNER = the block of the group
    that declares the cache tensors it reads: itself when it has K / V of its own or is the first block of the group that reads its producer's cache; for a block that reads the cache of
    a producer inside the group, the producer's block (it attends over the producer's main cache and its staging buffer AFTER the producer's update, which happens inside the graph)."""
    producer = [p if p >= 0 else i for i, p in enumerate(spec.kv_producer or [-1] * spec.num_layers)]
    layout, first_reader = [], {}
    for j in range(blocks):
        layer = first + j + 1
        p = producer[layer]
        if p == layer:
            owner = j
        elif first + 1 <= p <= first + blocks:
            owner = p - first - 1   # the block of the producer layer
        else:
            owner = first_reader.setdefault(p, j)
        layout.append({"kind": spec.layer_kind(layer), "owner": owner})
    return layout


def block_group_key(layout: list) -> str:
    return "-".join(KIND_CODES[item["kind"]] for item in layout)


def build_block_group_graph(spec, opt: GraphOptions, layout: list, *, seq: int = 1, past_len: int, mask_dtype: str = "float16", name: str | None = None):
    """len(layout) consecutive BLOCKs in one graph (fewer dispatches); decode only. `layout` = block_group_layout(): the kind and the cache OWNER of every block.
    Inputs: residual, attention_output (of the first block), the controls once per layer TYPE (<type>_rope_cos / <type>_rope_sin / <type>_attention_mask: own and shared layers
    of a type use the same mask in decode), and per block j with the prefix b<j>_: per_layer_input, the weights of the feed-forward half of layer i and of the attention half of
    layer i + 1 and, for the OWNER of a cache, past_key / past_value (the main cache of the layer whose K / V the block attends over) and staging_key / staging_value (its staging
    buffer). A block that reads a cache declared by an earlier block has no such inputs. Outputs: next_residual, next_attention_output, b<j>_updated_key / _value of the blocks with
    K / V of their own."""
    if seq != 1:
        raise ValueError("block groups of Gemma 4 are built for decode only (seq = 1)")
    b = GraphBuilder(name or f"block_x{len(layout)}")
    R = opt.staging_rows
    c0 = _ctx(b, spec, opt, kind_type(layout[0]["kind"]))
    residual = c0.to_compute(c0.declare("residual", "float16", [1, seq, spec.hidden]))
    attention_output = c0.to_compute(c0.declare("attention_output", "float16", [1, seq, spec.hidden]))
    controls: dict = {}   # layer type -> {cos, sin, mask}
    caches: dict = {}     # owner block -> {past_key, past_value, staging_key, staging_value}; the staging entries become the updated buffers once the owner's attention half has been built
    last_attention = None
    for j, item in enumerate(layout):
        kind, owner, prefix = item["kind"], item["owner"], f"b{j}_"
        t = kind_type(kind)
        c = _ctx(b, spec, opt, t)
        d = c.dims
        per_layer_input = c.declare(prefix + "per_layer_input", "float16", [1, seq, spec.ple_dim])
        layer_output = feed_forward_half(c, spec, residual, attention_output, per_layer_input, c.weights(feed_forward_weight_specs(spec), prefix), seq)
        attention_weights = c.weights(attention_weight_specs(spec, kind), prefix)
        rows = cache_rows(opt, t, past_len)   # the cache of this layer type (a ring for sliding layers of a window package)
        if t not in controls:
            controls[t] = {"cos": c.declare(f"{t}_rope_cos", "float16", [1, 1, seq, d.head_dim], "control"), "sin": c.declare(f"{t}_rope_sin", "float16", [1, 1, seq, d.head_dim], "control"),
                           "mask": c.declare(f"{t}_attention_mask", mask_dtype, [1, 1, seq, rows + R], "control")}
        if owner == j:
            shape, staging = [1, d.num_kv_heads, rows, d.head_dim], [1, d.num_kv_heads, R, d.head_dim]
            caches[j] = {"past_key": c.declare(prefix + "past_key", "float16", shape, "kv"), "past_value": c.declare(prefix + "past_value", "float16", shape, "kv"),
                         "staging_key": c.declare(prefix + "staging_key", "float16", staging, "kv"), "staging_value": c.declare(prefix + "staging_value", "float16", staging, "kv")}
        elif owner not in caches:
            raise ValueError(f"block {j} of the group reads the cache of block {owner}, which does not precede it")
        attention, kv_out = attention_half(c, spec, kind, layer_output, attention_weights, seq, past_len, opt, mask_dtype, bind={**controls[t], **caches[owner]})
        if kv_out:   # a block with K / V of its own: its updated staging buffer is an output and what the later blocks that share its K / V read
            caches[j]["staging_key"], caches[j]["staging_value"] = kv_out["updated_key"], kv_out["updated_value"]
            b.output(prefix + "updated_key", kv_out["updated_key"])
            b.output(prefix + "updated_value", kv_out["updated_value"])
        if j < len(layout) - 1:
            # the next block gets the boundary values of the ungrouped chain (float16 rounding): the same numbers as with one dispatch per block, and the add of the residual does not
            # read the result of the elementwise op that also feeds the next attention half
            residual, attention_output = c.to_compute(c.to_f16(layer_output)), c.to_compute(attention)
        else:
            b.output("next_residual", c.to_f16(layer_output))
            last_attention = attention
    b.output("next_attention_output", last_attention)
    return b.build()


def build_tail_graph(spec, opt: GraphOptions, *, seq: int, name: str = "tail"):
    """feed-forward half of the last layer + the final norm (the last row only when S > 1)."""
    b = GraphBuilder(name)
    c = _ctx(b, spec, opt, spec.layer_types[-1])
    residual = c.declare("residual", "float16", [1, seq, spec.hidden])
    attention_output = c.declare("attention_output", "float16", [1, seq, spec.hidden])
    per_layer_input = c.declare("per_layer_input", "float16", [1, seq, spec.ple_dim])
    weights = c.weights(feed_forward_weight_specs(spec) + final_norm_specs(spec))
    layer_output = feed_forward_half(c, spec, c.to_compute(residual), c.to_compute(attention_output), per_layer_input, weights, seq)
    final = rms_norm(c, layer_output, weights["final_norm_gamma"], 3)
    if seq > 1:
        final = b.slice(final, [0, seq - 1, 0], [1, 1, spec.hidden])
    b.output("final_hidden_states", c.to_f16(final))
    return b.build()


WEBNN_TENSOR_LIMIT = 1 << 31   # bytes: a WebNN tensor (the dequantized lm_head matrix of this vocabulary in float32 is 2.7 GB) may not be bigger than this


def lm_head_parts(spec, opt: GraphOptions) -> int:
    """How many row blocks the lm_head matrix is split into so that the dequantized (compute type) block stays well below the tensor size limit of WebNN."""
    k = max(1, -(-(spec.vocab_padded * spec.hidden * (4 if opt.compute_type == "float32" else 2)) // (WEBNN_TENSOR_LIMIT // 2)))
    while k < 256 and (spec.vocab_padded % k or (spec.vocab_padded // k) % 128):
        k += 1
    return k


def build_lm_head_graph(spec, opt: GraphOptions, *, name: str = "lm_head"):
    """final_hidden_states -> logits = softcap * tanh(logits / softcap); the int8 per-row table is a constant of the graph, in `lm_head_parts` row blocks
    (externals lm_head_weight_<i> / lm_head_scale_<i>, contiguous rows of the table) when the whole matrix would be too big for one WebNN tensor."""
    b = GraphBuilder(name)
    c = _ctx(b, spec, opt, spec.layer_types[0])
    hidden = c.to_compute(c.declare("final_hidden_states", "float16", [1, 1, spec.hidden]))
    parts = lm_head_parts(spec, opt)
    rows = spec.vocab_padded // parts
    zero_point = b.constant("int8", (rows, 1), fill=0, cache_key=("zp", "int8", rows))
    pieces = []
    for i in range(parts):
        suffix = f"_{i}" if parts > 1 else ""
        weight = b.constant("int8", (rows, spec.hidden), external=f"lm_head_weight{suffix}")
        scale = b.constant("float16", (rows, 1), external=f"lm_head_scale{suffix}")
        dequantized = c.to_compute(b.dequantize_linear(weight, scale, zero_point))
        pieces.append(b.gemm(b.reshape(hidden, [1, spec.hidden]), dequantized, b_transpose=True))
    logits = pieces[0] if parts == 1 else b.concat(pieces, 1)
    if spec.logit_softcap:
        cap = spec.logit_softcap
        logits = b.mul(b.tanh(b.mul(logits, c.scalar(1.0 / cap, 2))), c.scalar(cap, 2))
    b.output("logits", c.to_f16(b.reshape(logits, [1, 1, spec.vocab_padded])))
    return b.build()


def build_flush_graph(spec, layer_type: str, *, staging_rows: int, past_len: int, name: str = "kv_flush", opt: "GraphOptions | None" = None):
    """the main cache <- the staging rows, for the caches of one layer type (the head dims differ per type; `opt`: a sliding window package has the ring rows for a sliding type)."""
    return build_kv_update_graph(spec.decoder_dims(layer_type), seq=staging_rows, past_len=cache_rows(opt, layer_type, past_len) if opt else past_len, name=name)

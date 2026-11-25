"""Static-shape subgraphs of the Gemma 4 text decoder in the same NPUW-style split as decoder.py (HEAD / BLOCK / TAIL / lm_head / KV flush).

What differs from decoder.py (see docs/models/gemma4-e4b.md):
  * the attention half ends with the output projection and post_attention_layernorm, so the interface between the halves is `attention_output` [1,S,hidden], and the feed-forward half is the
    same for every layer: x1 = residual + attention_output; MLP with pre / post norms; the per-layer-embedding (PLE) branch; the layer scalar;
  * four kinds of attention half (layer type x own / shared K / V), because head dims differ per layer type and the last layers read the K / V of another layer: one BLOCK graph per kind
    (BLOCK i = feed-forward half of layer i + attention half of layer i + 1, of the kind of layer i + 1); HEAD = attention half of layer 0 + the PLE prologue; the K / V caches of the
    producer layers are read by the shared layers (decode: main cache + the staging buffer AFTER this step's update; prefill: the updated cache itself, mask columns = cache rows);
  * attention scale 1.0, a weightless v norm, RMSNorm weights used as they are, gelu-tanh, a logit softcap in the lm_head graph.
Only the staged KV protocol (decode) with the one-hot chunk update (prefill) is built.
"""
from __future__ import annotations

import math

from ....ir import GraphBuilder, Value
from .decoder import Ctx, DecoderDims, GraphOptions, apply_rotary, build_kv_update_graph, quantized_linear, repeat_kv, rms_norm, update_kv_cache

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
def attention_half(c: Ctx, spec, kind: str, residual: Value, w: dict, seq: int, past_len: int, opt: GraphOptions, mask_dtype: str = "float16"):
    """-> (attention_output f16 [1,S,hidden] = post_attention_layernorm(o_proj(attention)), kv_out {name: Value})"""
    b, d = c.b, c.dims
    c.seq = seq
    shared, decode = kind_shared(kind), seq == 1
    R = opt.staging_rows
    to_c = c.to_compute
    # which K / V does the attention read, and how long is it
    if shared:
        total = past_len + R if decode else past_len
    else:
        total = past_len + R if decode else past_len + seq
    cos = c.declare("rope_cos", "float16", [1, 1, seq, d.head_dim], "control")
    sin = c.declare("rope_sin", "float16", [1, 1, seq, d.head_dim], "control")
    mask = c.declare("attention_mask", mask_dtype, [1, 1, seq, total], "control")
    past_key = c.declare("past_key", "float16", [1, d.num_kv_heads, past_len, d.head_dim], "kv")
    past_value = c.declare("past_value", "float16", [1, d.num_kv_heads, past_len, d.head_dim], "kv")

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
            staging_key = c.declare("staging_key", "float16", [1, d.num_kv_heads, R, d.head_dim], "kv")
            staging_value = c.declare("staging_value", "float16", [1, d.num_kv_heads, R, d.head_dim], "kv")
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
            staging_key = c.declare("staging_key", "float16", [1, d.num_kv_heads, R, d.head_dim], "kv")
            staging_value = c.declare("staging_value", "float16", [1, d.num_kv_heads, R, d.head_dim], "kv")
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


def build_lm_head_graph(spec, opt: GraphOptions, *, name: str = "lm_head"):
    """final_hidden_states -> logits = softcap * tanh(logits / softcap); the int8 per-row table is a constant of the graph."""
    b = GraphBuilder(name)
    c = _ctx(b, spec, opt, spec.layer_types[0])
    hidden = c.to_compute(c.declare("final_hidden_states", "float16", [1, 1, spec.hidden]))
    w = {n: b.constant(dtype, shape, external=n) for n, dtype, shape in lm_head_weight_specs(spec)}
    zero_point = b.constant("int8", (spec.vocab_padded, 1), fill=0, cache_key=("zp", "int8", spec.vocab_padded))
    dequantized = c.to_compute(b.dequantize_linear(w["lm_head_weight"], w["lm_head_scale"], zero_point))
    logits = b.gemm(b.reshape(hidden, [1, spec.hidden]), dequantized, b_transpose=True)
    if spec.logit_softcap:
        cap = spec.logit_softcap
        logits = b.mul(b.tanh(b.mul(logits, c.scalar(1.0 / cap, 2))), c.scalar(cap, 2))
    b.output("logits", c.to_f16(b.reshape(logits, [1, 1, spec.vocab_padded])))
    return b.build()


def build_flush_graph(spec, layer_type: str, *, staging_rows: int, past_len: int, name: str = "kv_flush"):
    """the main cache <- the staging rows, for the caches of one layer type (the head dims differ per type)."""
    return build_kv_update_graph(spec.decoder_dims(layer_type), seq=staging_rows, past_len=past_len, name=name)

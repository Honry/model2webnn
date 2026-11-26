"""Static-shape subgraphs of the Qwen3.5 text decoder (hybrid: Gated DeltaNet "linear attention" layers and full attention layers) in the NPUW-style split of decoder.py
(HEAD / BLOCK / TAIL / lm_head / KV flush); see docs/models/qwen35.md.

No loop operator is needed for the Gated DeltaNet layers (OpenVINO's IR uses a `Loop` per layer; WebNN has none): see the "WHY THERE IS NO LOOP" note in gated_delta.py and docs/models/qwen35.md.

What differs from decoder.py:
  * the attention half ends with the output projection, the interface between the halves is `attention_output` [1,S,hidden] (like Gemma 4): x1 = residual + attention_output, then
    post_attention_layernorm -> gate / up -> silu -> down -> + x1;
  * two kinds of attention half = two BLOCK graphs: `full_attention` (own K / V cache, staged KV protocol, q_proj also yields an output gate, partial rotary on a permuted head layout)
    and `linear_attention` (Gated DeltaNet: causal conv over 8192 channels with a 3-row state, a [32,128,128] recurrent state; see gated_delta.py);
  * every RMSNorm weight is packed as (1 + w) (Qwen3.5 uses `x * (1 + w)`), except the gated norm inside the linear attention (plain w);
  * a linear attention layer keeps its state in two tensors that the graph reads and writes (`conv_state` -> `new_conv_state`, `recurrent_state` -> `new_recurrent_state`): no staging, no flush.
    Prefill chunks left-pad the last chunk; the pad rows are no-ops of the recurrence (valid_mask zeroes beta and g) and the conv state skips them (conv_perm moves the pad rows in front of the state rows).
NPU compiler facts that shaped the graphs (bench/raw/gdn.html): activation x activation matmuls need every dim >= 16; a multiplication result must not feed a matmul AND an add (the decode step
scales the prediction instead of the state); the triangular solve is a row-by-row substitution (the Neumann product explodes for nearly parallel keys).
"""
from __future__ import annotations

import math

from ....ir import GraphBuilder, Value
from . import gated_delta
from .decoder import Ctx, DecoderDims, GraphOptions, apply_rotary, build_kv_update_graph, quantized_linear, repeat_kv, rms_norm, update_kv_cache
from .gemma4 import build_lm_head_graph, lm_head_parts, lm_head_weight_specs  # noqa: F401  (the same row-block lm_head: the vocabulary is big)

KINDS = ("linear_attention", "full_attention")
CONV_TAPS = 4
SUB_CHUNK = 16        # the Gated DeltaNet sub-chunk of a prefill chunk (chunk sizes must be multiples of it)
PREFILL_SOLVER = "blockdiag"   # the triangular solve of the prefill core: "substitution" (16 dependent row steps: 3.2 ms of the 4.6 ms core on the NPU, accurate to 1e-3) | "blockdiag" (no dependent steps: the
#                               core takes 2.5 ms, accurate to 1.5e-2 on the NPU; see gated_delta.solve_block_diagonal and bench/raw/gdn.html dir=../data/gdn_seg/)
L2_EPS = 1e-6   # transformers: l2norm(x) = x * rsqrt(sum(x^2) + 1e-6). Faithful: an eps that is 100x bigger changed the hidden states by ~13% (it matters for the many q / k heads with sum(x^2) of 0.01 .. 0.1).
#                 1e-6 is a float16 subnormal: on the NPU it is flushed to 0, which is harmless for a real row but turns an all-zero row (the pad rows of a left-padded chunk) into 0 * inf = NaN;
#                 the pad rows therefore get a non-zero guard in prefill (attention_half_linear), they are no-ops of the recurrence anyway (beta = 0, g = 0).


# ---------------------------------------------------------------------------------------------------------------------
# weight specs: (name, dtype, shape); the names are graph input names AND manifest keys
# ---------------------------------------------------------------------------------------------------------------------
def _linear(name: str, in_ch: int, out_ch: int, group: int) -> list:
    return [(f"{name}_weight", "int4", (in_ch // group, out_ch, group)), (f"{name}_scale", "float16", (in_ch // group, out_ch, 1))]


def linear_dims(spec) -> dict:
    L = spec.linear
    key, value = L["num_k_heads"] * L["k_dim"], L["num_v_heads"] * L["v_dim"]
    return {"k_heads": L["num_k_heads"], "v_heads": L["num_v_heads"], "k_dim": L["k_dim"], "v_dim": L["v_dim"], "key_width": key, "value_width": value, "conv_width": 2 * key + value}


def attention_weight_specs(spec, kind: str) -> list:
    h, g = spec.hidden, spec.group_size
    out = [("input_layernorm_gamma", "float16", (1, 1, h))]
    if kind == "linear_attention":
        d = linear_dims(spec)
        return (out + _linear("qkvz_projection", h, d["conv_width"] + d["value_width"], g)
                + [("ba_weight", "float16", (2 * d["v_heads"], h)),   # rows: beta logits (v_heads), then the decay input a (v_heads): small, kept in float16
                   ("conv_weight", "float16", (CONV_TAPS, 1, d["conv_width"])),
                   ("decay_rate", "float16", (1, 1, d["v_heads"])), ("dt_bias", "float16", (1, 1, d["v_heads"])),
                   ("norm_gamma", "float16", (1, 1, 1, d["v_dim"]))]
                + _linear("output_projection", d["value_width"], h, g))
    d = spec.decoder_dims("full_attention")
    return (out + _linear("qkvg_projection", h, 2 * d.attention_width + 2 * d.kv_width, g)
            + [("query_norm_gamma", "float16", (1, 1, 1, d.head_dim)), ("key_norm_gamma", "float16", (1, 1, 1, d.head_dim))]
            + _linear("output_projection", d.attention_width, h, g))


def feed_forward_weight_specs(spec) -> list:
    h, f, g = spec.hidden, spec.ffn, spec.group_size
    return [("post_attention_layernorm_gamma", "float16", (1, 1, h))] + _linear("gate_up_projection", h, 2 * f, g) + _linear("down_projection", f, h, g)


def final_norm_specs(spec) -> list:
    return [("final_norm_gamma", "float16", (1, 1, spec.hidden))]


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def l2_normalize(c: Ctx, x: Value, rank: int, scale: float = 1.0) -> Value:
    """scale * x / sqrt(sum(x^2) + eps) over the last axis, written as the RMS-norm pattern that the NPU compiler recognises and fuses (x * (1 / sqrt(mean(x^2) + eps)) * (scale / sqrt(dim))).
    The plain sum / sqrt / div form (`x * (1 / sqrt(reduceSum(x^2) + eps))`) returned garbage and non-finite values for some rows on the NPU (bench/raw/qwen35probe.html: key, but not the query, which has a
    multiplication behind its normalization)."""
    b, dim = c.b, x.shape[rank - 1]
    mean_square = b.reduce_mean(b.pow(x, c.scalar(2, rank)), axes=[rank - 1], keep_dimensions=True)
    normalized = b.mul(x, b.div(c.scalar(1, rank), b.sqrt(b.add(mean_square, c.scalar(L2_EPS / dim, rank)))))   # sum + eps = dim * (mean + eps / dim)
    return b.mul(normalized, c.scalar(scale / math.sqrt(dim), rank))


GATED_NORM_SCALE = 16.0   # the input of the gated RMSNorm (the output of the recurrence) has an rms of ~1e-2: its squares (1e-4) sit at the bottom of the float16 normal range (6.1e-5) and the NPU's reduceMean of
#                           them lost most of its precision (bench/raw/qwen35probe.html: p_normed 1e-2 off, p_projected 7e-2). RMSNorm does not depend on the scale of its input: scale up, and the eps with it.


def gated_rms_norm_input(c: Ctx, x: Value, gamma: Value) -> Value:
    """gamma * x / sqrt(mean(x^2) + eps) computed on x * 16 with eps * 256: the same value, with squares in the comfortable range of float16."""
    b = c.b
    scaled = b.mul(x, c.scalar(GATED_NORM_SCALE, 4))
    mean_square = b.reduce_mean(b.pow(scaled, c.scalar(2, 4)), axes=[3], keep_dimensions=True)
    rms = b.sqrt(b.add(mean_square, c.scalar(c.dims.eps * GATED_NORM_SCALE ** 2, 4)))
    return b.mul(c.to_compute(gamma), b.mul(scaled, b.div(c.scalar(1, 4), rms)))


def silu(c: Ctx, x: Value) -> Value:
    return c.b.mul(x, c.b.sigmoid(x))


def _ctx(b: GraphBuilder, spec, opt: GraphOptions) -> Ctx:
    return Ctx(b, spec.decoder_dims("full_attention"), opt)


# ---------------------------------------------------------------------------------------------------------------------
# the halves
# ---------------------------------------------------------------------------------------------------------------------
def attention_half_full(c: Ctx, spec, residual: Value, w: dict, seq: int, past_len: int, opt: GraphOptions, mask_dtype: str = "float16"):
    """Full attention with the output gate. -> (attention_output f16 [1,S,hidden], kv_out)"""
    b, d = c.b, c.dims
    c.seq = seq
    decode, R, to_c = seq == 1, opt.staging_rows, c.to_compute
    total = past_len + R if decode else past_len + seq
    cos = c.declare("rope_cos", "float16", [1, 1, seq, d.head_dim], "control")
    sin = c.declare("rope_sin", "float16", [1, 1, seq, d.head_dim], "control")
    mask = c.declare("attention_mask", mask_dtype, [1, 1, seq, total], "control")
    past_key = c.declare("past_key", "float16", [1, d.num_kv_heads, past_len, d.head_dim], "kv")
    past_value = c.declare("past_value", "float16", [1, d.num_kv_heads, past_len, d.head_dim], "kv")

    normalized = rms_norm(c, residual, w["input_layernorm_gamma"], 3)
    qkvg = quantized_linear(c, normalized, w["qkvg_projection_weight"], w["qkvg_projection_scale"], d.hidden, 2 * d.attention_width + 2 * d.kv_width, seq)
    part = lambda start, width: b.slice(qkvg, [0, 0, start], [1, seq, width])  # noqa: E731
    aw, kw = d.attention_width, d.kv_width

    def heads_of(projected, heads, gamma):   # [1,S,heads*d] -> (per-head RMSNorm) -> [1,heads,S,d]
        x = b.reshape(projected, [1, seq, heads, d.head_dim])
        if gamma is not None:
            x = rms_norm(c, x, gamma, 4)
        return b.transpose(x, [0, 2, 1, 3])

    query = apply_rotary(c, heads_of(part(0, aw), d.num_heads, w["query_norm_gamma"]), cos, sin, d.num_heads)
    gate = part(aw, aw)
    key = apply_rotary(c, heads_of(part(2 * aw, kw), d.num_kv_heads, w["key_norm_gamma"]), cos, sin, d.num_kv_heads)
    value = heads_of(part(2 * aw + kw, kw), d.num_kv_heads, None)
    if decode:
        staging_key = c.declare("staging_key", "float16", [1, d.num_kv_heads, R, d.head_dim], "kv")
        staging_value = c.declare("staging_value", "float16", [1, d.num_kv_heads, R, d.head_dim], "kv")
        updated_key, updated_value = update_kv_cache(c, staging_key, staging_value, key, value, R, seq, selector_prefix="staging_")
        all_keys, all_values = b.concat([to_c(past_key), to_c(updated_key)], 2), b.concat([to_c(past_value), to_c(updated_value)], 2)
    else:
        updated_key, updated_value = update_kv_cache(c, past_key, past_value, key, value, past_len, seq)
        all_keys, all_values = b.concat([to_c(past_key), to_c(key)], 2), b.concat([to_c(past_value), to_c(value)], 2)
    scaled = b.mul(query, c.scalar(1.0 / math.sqrt(d.head_dim), 4))
    keys_t = b.transpose(repeat_kv(c, all_keys, [total, d.head_dim]), [0, 1, 3, 2])
    scores = b.add(b.matmul(to_c(scaled), keys_t), to_c(mask))
    context = b.matmul(b.softmax(scores, 3), repeat_kv(c, all_values, [total, d.head_dim]))
    context = b.reshape(b.transpose(context, [0, 2, 1, 3]), [1, seq, aw])
    gated = b.mul(context, b.sigmoid(gate))
    projected = quantized_linear(c, gated, w["output_projection_weight"], w["output_projection_scale"], aw, d.hidden, seq)
    return c.to_f16(projected), {"updated_key": updated_key, "updated_value": updated_value}


def attention_half_linear(c: Ctx, spec, residual: Value, w: dict, seq: int, probes: dict | None = None):
    """Gated DeltaNet. -> (attention_output f16 [1,S,hidden], {new_conv_state, new_recurrent_state})"""
    b, h = c.b, spec.hidden
    d = linear_dims(spec)
    hk, hv, dk, dv = d["k_heads"], d["v_heads"], d["k_dim"], d["v_dim"]
    cw, vw = d["conv_width"], d["value_width"]
    c.seq = seq
    conv_state = c.declare("conv_state", "float16", [1, CONV_TAPS - 1, cw], "kv")
    state = c.declare("recurrent_state", "float16", [1, hv, dk, dv], "kv")

    normalized = rms_norm(c, residual, w["input_layernorm_gamma"], 3)
    qkvz = quantized_linear(c, normalized, w["qkvz_projection_weight"], w["qkvz_projection_scale"], h, cw + vw, seq)
    qkv, z = b.slice(qkvz, [0, 0, 0], [1, seq, cw]), b.slice(qkvz, [0, 0, cw], [1, seq, vw])
    ba = b.reshape(b.gemm(b.reshape(normalized, [seq, h]), c.to_compute(w["ba_weight"]), b_transpose=True), [1, seq, 2 * hv])
    beta = b.sigmoid(b.slice(ba, [0, 0, 0], [1, seq, hv]))
    a = b.slice(ba, [0, 0, hv], [1, seq, hv])
    g = b.mul(b.softplus(b.add(a, c.to_compute(w["dt_bias"]))), b.mul(c.to_compute(w["decay_rate"]), c.scalar(-1.0, 3)))   # -exp(A_log) * softplus(a + dt_bias)  (decay_rate = exp(A_log))

    # causal depthwise conv over the time axis: y[t] = sum_j w[j] * x[t + j] on [state rows | x rows]; the output of row t uses the 3 rows before it
    sequence = b.concat([c.to_compute(conv_state), qkv], 1)                                 # [1, 3 + S, cw]
    if seq > 1:   # prefill: the pad rows of the (left-padded) chunk move in front of the state rows, so that the first real row sees the state, not the pad rows
        sequence = b.matmul(c.to_compute(c.declare("conv_perm", "float16", [1, seq + CONV_TAPS - 1, seq + CONV_TAPS - 1], "control")), sequence)
    new_conv_state = c.to_f16(b.slice(sequence, [0, seq, 0], [1, CONV_TAPS - 1, cw]))
    taps = [c.to_compute(b.slice(w["conv_weight"], [j, 0, 0], [1, 1, cw])) for j in range(CONV_TAPS)]
    conv = None
    for j in range(CONV_TAPS):
        term = b.mul(b.slice(sequence, [0, j, 0], [1, seq, cw]), taps[j])
        conv = term if conv is None else b.add(conv, term)
    conv = silu(c, conv)
    if seq > 1:   # pad rows (valid_mask 0): conv, hence q and k, would be exactly 0 there: add a guard of 1 so that their normalization is finite (real rows get + 0)
        conv = b.add(conv, b.reshape(b.sub(c.scalar(1.0, 4), c.to_compute(c.declare("valid_mask", "float16", [1, 1, seq, 1], "control"))), [1, seq, 1]))

    q = b.reshape(b.slice(conv, [0, 0, 0], [1, seq, hk * dk]), [1, seq, hk, dk])
    k = b.reshape(b.slice(conv, [0, 0, hk * dk], [1, seq, hk * dk]), [1, seq, hk, dk])
    v = b.reshape(b.slice(conv, [0, 0, 2 * hk * dk], [1, seq, vw]), [1, seq, hv, dv])
    q = l2_normalize(c, q, 4, dk ** -0.5)
    k = l2_normalize(c, k, 4)
    if hv != hk:   # repeat_interleave of the key heads: head h of the values uses key head h // (hv / hk)
        rep = hv // hk
        expand = lambda x: b.reshape(b.expand(b.reshape(x, [1, seq, hk, 1, dk]), [1, seq, hk, rep, dk]), [1, seq, hv, dk])  # noqa: E731
        q, k = expand(q), expand(k)
    heads = lambda x: b.transpose(x, [0, 2, 1, 3])  # noqa: E731
    per_head = lambda x: b.reshape(b.transpose(x, [0, 2, 1]), [1, hv, seq, 1])  # noqa: E731    [1,S,hv] -> [1,hv,S,1]
    beta_h, g_h = per_head(beta), per_head(g)
    if seq > 1:   # pad rows: beta = 0 and g = 0 leave the state untouched
        valid = c.to_compute(c.b.graph.values["valid_mask"])
        beta_h, g_h = b.mul(beta_h, valid), b.mul(g_h, valid)
        out, new_state = gated_delta.gated_delta_batched(b, heads(q), heads(k), heads(v), g_h, beta_h, c.to_compute(state), SUB_CHUNK, c.compute, PREFILL_SOLVER)
    else:
        out, new_state = gated_delta.gated_delta_step(b, heads(q), heads(k), heads(v), g_h, beta_h, c.to_compute(state))
    normed = gated_rms_norm_input(c, b.transpose(out, [0, 2, 1, 3]), w["norm_gamma"])      # gated RMSNorm (plain weight): norm(o) * silu(z)
    z4 = b.reshape(z, [1, seq, hv, dv])
    gated = b.reshape(b.mul(normed, silu(c, z4)), [1, seq, vw])
    projected = quantized_linear(c, gated, w["output_projection_weight"], w["output_projection_scale"], vw, h, seq)
    if probes is not None:   # intermediate results as extra outputs (build_head_graph(probe=True)): where does a device deviate from the interpreter? see tools/qwen35_probe.py
        probes.update(normalized=normalized, qkvz=qkvz, ba=ba, beta=beta, g=g, conv=conv, q=heads(q), k=heads(k), v=heads(v), core=out, new_state=new_state, normed=normed, gated=gated, projected=projected)
    return c.to_f16(projected), {"new_conv_state": new_conv_state, "new_recurrent_state": c.to_f16(new_state)}


def attention_half(c: Ctx, spec, kind: str, residual: Value, w: dict, seq: int, past_len: int, opt: GraphOptions):
    if kind == "linear_attention":
        return attention_half_linear(c, spec, residual, w, seq)
    return attention_half_full(c, spec, residual, w, seq, past_len, opt)


def feed_forward_half(c: Ctx, spec, residual: Value, attention_output: Value, w: dict, seq: int) -> Value:
    """x1 = residual + attention_output;  out = x1 + down(silu(gate(norm(x1))) * up(norm(x1)))"""
    b, h, f = c.b, spec.hidden, spec.ffn
    x1 = b.add(residual, attention_output)
    normalized = rms_norm(c, x1, w["post_attention_layernorm_gamma"], 3)
    gate_up = quantized_linear(c, normalized, w["gate_up_projection_weight"], w["gate_up_projection_scale"], h, 2 * f, seq)
    gate, up = b.slice(gate_up, [0, 0, 0], [1, seq, f]), b.slice(gate_up, [0, 0, f], [1, seq, f])
    return b.add(x1, quantized_linear(c, b.mul(silu(c, gate), up), w["down_projection_weight"], w["down_projection_scale"], f, h, seq))


# ---------------------------------------------------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------------------------------------------------
def _outputs(b: GraphBuilder, kv_out: dict) -> None:
    for k, v in kv_out.items():
        b.output(k, v)


def build_head_graph(spec, opt: GraphOptions, *, seq: int, past_len: int, name: str = "head", probe: bool = False):
    """attention half of layer 0 -> residual, attention_output (+ the state outputs of that layer's kind). probe: also every intermediate result of a linear attention half as p_<name>."""
    b = GraphBuilder(name)
    kind = spec.layer_kind(0)
    c = _ctx(b, spec, opt)
    x = c.to_compute(c.declare("hidden_states", "float16", [1, seq, spec.hidden]))
    probes = {} if probe and kind == "linear_attention" else None
    weights = c.weights(attention_weight_specs(spec, kind))
    attention_output, kv_out = attention_half_linear(c, spec, x, weights, seq, probes) if probes is not None else attention_half(c, spec, kind, x, weights, seq, past_len, opt)
    b.output("residual", c.to_f16(x))
    b.output("attention_output", attention_output)
    _outputs(b, kv_out)
    for key, value in (probes or {}).items():
        b.output(f"p_{key}", c.to_f16(value))
    return b.build()


def build_block_graph(spec, opt: GraphOptions, kind: str, *, seq: int, past_len: int, name: str = "block"):
    """feed-forward half of layer i (uniform) + attention half of layer i + 1 of `kind`."""
    b = GraphBuilder(name)
    c = _ctx(b, spec, opt)
    residual = c.declare("residual", "float16", [1, seq, spec.hidden])
    attention_output = c.declare("attention_output", "float16", [1, seq, spec.hidden])
    layer_output = feed_forward_half(c, spec, c.to_compute(residual), c.to_compute(attention_output), c.weights(feed_forward_weight_specs(spec)), seq)
    next_attention, kv_out = attention_half(c, spec, kind, layer_output, c.weights(attention_weight_specs(spec, kind)), seq, past_len, opt)
    b.output("next_residual", c.to_f16(layer_output))
    b.output("next_attention_output", next_attention)
    _outputs(b, kv_out)
    return b.build()


def build_tail_graph(spec, opt: GraphOptions, *, seq: int, name: str = "tail"):
    b = GraphBuilder(name)
    c = _ctx(b, spec, opt)
    residual = c.declare("residual", "float16", [1, seq, spec.hidden])
    attention_output = c.declare("attention_output", "float16", [1, seq, spec.hidden])
    weights = c.weights(feed_forward_weight_specs(spec) + final_norm_specs(spec))
    layer_output = feed_forward_half(c, spec, c.to_compute(residual), c.to_compute(attention_output), weights, seq)
    final = rms_norm(c, layer_output, weights["final_norm_gamma"], 3)
    if seq > 1:
        final = b.slice(final, [0, seq - 1, 0], [1, 1, spec.hidden])
    b.output("final_hidden_states", c.to_f16(final))
    return b.build()


def build_flush_graph(spec, *, staging_rows: int, past_len: int, name: str = "kv_flush"):
    """the main cache <- the staging rows (full attention layers only)."""
    return build_kv_update_graph(spec.decoder_dims("full_attention"), seq=staging_rows, past_len=past_len, name=name)

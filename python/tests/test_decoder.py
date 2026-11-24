"""The ported decoder subgraphs must compute a standard transformer (float32 reference) and the staged / one-hot / tensorScatter KV styles must agree."""
import numpy as np
import pytest

from m2w.frontends.hf_llm.builders import decoder as D
from m2w.ir.interpreter import run

DIMS = D.DecoderDims(hidden=256, num_heads=4, num_kv_heads=2, head_dim=64, ffn=256, vocab=512)
NEG = np.float16(-65504)


def random_layer(rng, dims, opt):
    """(feeds for the graph inputs, dequantized float32 matrices) for one layer: attention half + feed-forward half."""
    feeds, ref = {}, {}
    for specs in (D.attention_weight_specs(dims, opt), D.feed_forward_weight_specs(dims, opt)):
        for name, dtype, shape in specs:
            if dtype == "int4":
                feeds[name] = rng.integers(-8, 8, shape).astype(np.int8)
            elif name.endswith("_scale"):
                feeds[name] = rng.uniform(0.01, 0.05, shape).astype(np.float16)
            else:   # gammas
                feeds[name] = rng.uniform(0.8, 1.2, shape).astype(np.float16)
    for name in list(feeds):
        if name.endswith("_weight"):
            prefix = name[: -len("_weight")]
            q, s = feeds[name].astype(np.float32), feeds[prefix + "_scale"].astype(np.float32)
            w = (q * s).astype(np.float16).astype(np.float32)            # [G, N, 128]
            ref[prefix] = w.transpose(1, 0, 2).reshape(w.shape[1], -1)   # [N, K]
    ref.update({k: v.astype(np.float32).reshape(-1) for k, v in feeds.items() if k.endswith("_gamma")})
    return feeds, ref


def rms(x, gamma, eps):
    return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * gamma


def rotary(x, cos, sin):
    h = x.shape[-1] // 2
    return x * cos + np.concatenate([-x[..., h:], x[..., :h]], -1) * sin


def rope_tables(positions, head_dim, theta=10000.0):
    inv = 1.0 / theta ** (np.arange(0, head_dim, 2) / head_dim)
    ang = np.outer(positions, inv)
    full = np.concatenate([ang, ang], -1)
    return np.cos(full).astype(np.float32), np.sin(full).astype(np.float32)


def ref_attention_half(dims, ref, x, cos, sin, past_k, past_v, valid_past):
    """x [S,E] -> attention output [S, attention_width], new k/v [kvHeads,S,hd]; attends to the valid past rows + the S new rows (causal)."""
    S = x.shape[0]
    n = rms(x, ref["input_layernorm_gamma"], dims.eps)
    if "qkv_projection" in ref:
        qkv = n @ ref["qkv_projection"].T
        q, k, v = np.split(qkv, [dims.attention_width, dims.attention_width + dims.kv_width], axis=1)
    else:
        q, k, v = n @ ref["query"].T, n @ ref["key"].T, n @ ref["value"].T
    q = q.reshape(S, dims.num_heads, dims.head_dim)
    k = k.reshape(S, dims.num_kv_heads, dims.head_dim)
    v = v.reshape(S, dims.num_kv_heads, dims.head_dim)
    if dims.qk_norm:
        q, k = rms(q, ref["query_norm_gamma"], dims.eps), rms(k, ref["key_norm_gamma"], dims.eps)
    q = rotary(q.transpose(1, 0, 2), cos, sin)
    k = rotary(k.transpose(1, 0, 2), cos, sin)
    v = v.transpose(1, 0, 2)
    group = dims.num_heads // dims.num_kv_heads
    out = np.zeros((dims.num_heads, S, dims.head_dim), np.float32)
    for h in range(dims.num_heads):
        kv = h // group
        keys = np.concatenate([past_k[kv][:valid_past], k[kv]], 0)
        vals = np.concatenate([past_v[kv][:valid_past], v[kv]], 0)
        scores = (q[h] / np.sqrt(dims.head_dim)) @ keys.T
        mask = np.zeros_like(scores)
        for i in range(S):
            mask[i, valid_past + i + 1:] = -1e30
        p = np.exp(scores + mask - (scores + mask).max(-1, keepdims=True))
        p /= p.sum(-1, keepdims=True)
        out[h] = p @ vals
    return out.transpose(1, 0, 2).reshape(S, -1), k, v


def ref_ffn_half(dims, ref, residual, attn):
    h = residual + attn @ ref["output_projection"].T
    n = rms(h, ref["post_attention_layernorm_gamma"], dims.eps)
    if "gate_up_projection" in ref:
        gate, up = np.split(n @ ref["gate_up_projection"].T, 2, axis=1)
    else:
        gate, up = n @ ref["gate_projection"].T, n @ ref["up_projection"].T
    return h + ((gate / (1 + np.exp(-gate))) * up) @ ref["down_projection"].T


@pytest.mark.parametrize("merge", [True, False])
@pytest.mark.parametrize("qk_norm", [True, False])
def test_two_layer_prefill_matches_reference(merge, qk_norm):
    dims = D.DecoderDims(256, 4, 2, 64, 256, 512, qk_norm=qk_norm)
    opt = D.GraphOptions(merge_projections=merge, kv_style="onehot")
    S, P = 4, 8
    rng = np.random.default_rng(1)
    layers = [random_layer(rng, dims, opt) for _ in range(2)]
    final_gamma = rng.uniform(0.8, 1.2, (1, 1, dims.hidden)).astype(np.float16)
    hidden = (rng.standard_normal((1, S, dims.hidden)) * 0.5).astype(np.float16)
    cos, sin = rope_tables(np.arange(S), dims.head_dim)
    mask = np.full((1, 1, S, P + S), NEG, np.float16)
    for i in range(S):
        mask[0, 0, i, P: P + i + 1] = 0           # past rows hidden, new rows causal
    rows = np.zeros((1, 1, P, S), np.float16)      # new row j goes to cache row j
    rows[0, 0, np.arange(S), np.arange(S)] = 1
    keep = np.ones((1, 1, P, 1), np.float16)
    keep[0, 0, :S] = 0
    zeros_kv = np.zeros((1, dims.num_kv_heads, P, dims.head_dim), np.float16)
    common = {"rope_cos": cos[None, None].astype(np.float16), "rope_sin": sin[None, None].astype(np.float16), "attention_mask": mask,
              "row_selector": rows, "rows_to_keep": keep}

    head = D.build_head_graph(dims, opt, seq=S, past_len=P)
    block = D.build_block_graph(dims, opt, seq=S, past_len=P)
    tail = D.build_tail_graph(dims, opt, seq=S)

    attn_w = {k: v for k, v in layers[0][0].items() if k in {n for n, _, _ in D.attention_weight_specs(dims, opt)}}
    h_out = run(head, {"hidden_states": hidden, "past_key": zeros_kv, "past_value": zeros_kv, **attn_w, **common})
    ffn0 = {k: v for k, v in layers[0][0].items() if k in {n for n, _, _ in D.feed_forward_weight_specs(dims, opt)}}
    attn1 = {k: v for k, v in layers[1][0].items() if k in {n for n, _, _ in D.attention_weight_specs(dims, opt)}}
    b_out = run(block, {"residual": h_out["residual"], "attention_output": h_out["attention_output"], "past_key": zeros_kv, "past_value": zeros_kv,
                        **ffn0, **attn1, **common})
    ffn1 = {k: v for k, v in layers[1][0].items() if k in {n for n, _, _ in D.feed_forward_weight_specs(dims, opt)}}
    t_out = run(tail, {"residual": b_out["next_residual"], "attention_output": b_out["next_attention_output"], **ffn1, "final_norm_gamma": final_gamma})

    # float32 reference
    x = hidden[0].astype(np.float32)
    empty = np.zeros((dims.num_kv_heads, 0, dims.head_dim), np.float32)
    a0, k0, v0 = ref_attention_half(dims, layers[0][1], x, cos, sin, empty, empty, 0)
    x = ref_ffn_half(dims, layers[0][1], x, a0)
    a1, k1, v1 = ref_attention_half(dims, layers[1][1], x, cos, sin, empty, empty, 0)
    x = ref_ffn_half(dims, layers[1][1], x, a1)
    x = rms(x, final_gamma.astype(np.float32).reshape(-1), dims.eps)[-1]

    got = t_out["final_hidden_states"].astype(np.float32).reshape(-1)
    scale = np.abs(x).max()
    np.testing.assert_allclose(got, x, atol=0.03 * scale)
    # the KV update wrote the new rows of layer 0 / layer 1 at cache rows 0..S-1 and left the rest zero
    np.testing.assert_allclose(h_out["updated_key"][0, :, :S].astype(np.float32), k0, atol=0.03 * np.abs(k0).max())
    np.testing.assert_allclose(b_out["updated_value"][0, :, :S].astype(np.float32), v1, atol=0.03 * np.abs(v1).max())
    assert not h_out["updated_key"][0, :, S:].any()


def test_staged_decode_matches_onehot_decode_and_tensor_scatter():
    """Same logical state three ways: one big cache (one-hot, tensorScatter) vs main cache + staging buffer (staged)."""
    dims = DIMS
    R, P, valid_main, valid_staging = 4, 8, 5, 2          # sequence so far: 5 rows in the main cache, 2 in staging, the new token is row 7
    rng = np.random.default_rng(2)
    feeds_w, ref = random_layer(rng, dims, D.GraphOptions())
    attn_names = {n for n, _, _ in D.attention_weight_specs(dims, D.GraphOptions())}
    attn_w = {k: v for k, v in feeds_w.items() if k in attn_names}
    hidden = (rng.standard_normal((1, 1, dims.hidden)) * 0.5).astype(np.float16)
    past = (rng.standard_normal((2, dims.num_kv_heads, P, dims.head_dim)) * 0.5).astype(np.float16)   # [k|v, kvHeads, rows, hd]; rows 0..6 valid
    position = valid_main + valid_staging
    cos, sin = rope_tables([position], dims.head_dim)
    rope = {"rope_cos": cos[None, None].astype(np.float16), "rope_sin": sin[None, None].astype(np.float16)}

    # (a) one-hot / tensorScatter over a cache of P rows holding the 7 valid rows
    mask_a = np.full((1, 1, 1, P + 1), NEG, np.float16); mask_a[..., :position] = 0; mask_a[..., P] = 0
    sel = np.zeros((1, 1, P, 1), np.float16); sel[0, 0, position] = 1
    keep = 1 - sel
    onehot = D.build_head_graph(dims, D.GraphOptions(kv_style="onehot"), seq=1, past_len=P)
    a = run(onehot, {"hidden_states": hidden, "past_key": past[0:1], "past_value": past[1:2], **attn_w, **rope, "attention_mask": mask_a,
                     "row_selector": sel, "rows_to_keep": keep})
    # (b) staged: main cache rows 0..4, staging rows 0..1 (= sequence rows 5..6); the new token goes to staging row 2
    main_k = np.zeros_like(past[0:1]); main_k[:, :, :valid_main] = past[0:1, :, :valid_main]
    main_v = np.zeros_like(past[1:2]); main_v[:, :, :valid_main] = past[1:2, :, :valid_main]
    stg_k = np.zeros((1, dims.num_kv_heads, R, dims.head_dim), np.float16); stg_k[:, :, :valid_staging] = past[0:1, :, valid_main:position]
    stg_v = np.zeros_like(stg_k); stg_v[:, :, :valid_staging] = past[1:2, :, valid_main:position]
    mask_b = np.full((1, 1, 1, P + R), NEG, np.float16); mask_b[..., :valid_main] = 0; mask_b[..., P: P + valid_staging + 1] = 0
    ssel = np.zeros((1, 1, R, 1), np.float16); ssel[0, 0, valid_staging] = 1
    staged = D.build_head_graph(dims, D.GraphOptions(kv_style="staged", staging_rows=R), seq=1, past_len=P)
    bout = run(staged, {"hidden_states": hidden, "past_key": main_k, "past_value": main_v, "staging_key": stg_k, "staging_value": stg_v, **attn_w, **rope,
                        "attention_mask": mask_b, "staging_row_selector": ssel, "staging_rows_to_keep": 1 - ssel})
    # (c) tensorScatter: attention reads the scattered cache itself (P rows), the mask opens rows 0..position
    mask_c = np.full((1, 1, 1, P), NEG, np.float16); mask_c[..., : position + 1] = 0
    scatter = D.build_head_graph(dims, D.GraphOptions(kv_style="tensorScatter"), seq=1, past_len=P)
    cout = run(scatter, {"hidden_states": hidden, "past_key": past[0:1], "past_value": past[1:2], **attn_w, **rope, "attention_mask": mask_c,
                         "kv_write_index": np.array([position], np.int32)})

    scale = np.abs(a["attention_output"].astype(np.float32)).max()
    np.testing.assert_allclose(bout["attention_output"].astype(np.float32), a["attention_output"].astype(np.float32), atol=0.01 * scale)
    np.testing.assert_allclose(cout["attention_output"].astype(np.float32), a["attention_output"].astype(np.float32), atol=0.01 * scale)
    # the staging buffer got the new row at staging row 2, the scattered cache at row 7
    np.testing.assert_array_equal(bout["updated_key"][0, :, valid_staging], cout["updated_key"][0, :, position])
    np.testing.assert_array_equal(a["updated_key"][0, :, position], cout["updated_key"][0, :, position])


@pytest.mark.parametrize("fmt", ["int8-sym", "uint8-asym"])
def test_lm_head_matches_reference(fmt):
    dims = DIMS
    rng = np.random.default_rng(3)
    sc = rng.uniform(0.001, 0.003, (dims.vocab, 1)).astype(np.float16)
    h = (rng.standard_normal((1, 1, dims.hidden))).astype(np.float16)
    g = D.build_lm_head_graph(dims, D.GraphOptions(lm_head_format=fmt))
    if fmt == "uint8-asym":
        wq = rng.integers(0, 256, (dims.vocab, dims.hidden)).astype(np.uint8)
        zp = rng.integers(100, 150, (dims.vocab, 1)).astype(np.uint8)
        externals = {"lm_head_weight": wq, "lm_head_zero_point": zp, "lm_head_scale": sc}
        deq = ((wq.astype(np.float32) - zp.astype(np.float32)) * sc.astype(np.float32)).astype(np.float16).astype(np.float32)
    else:
        wq = rng.integers(-127, 128, (dims.vocab, dims.hidden)).astype(np.int8)
        externals = {"lm_head_weight": wq, "lm_head_scale": sc}
        deq = (wq.astype(np.float32) * sc.astype(np.float32)).astype(np.float16).astype(np.float32)
    out = run(g, {"final_hidden_states": h}, externals=externals)["logits"]
    ref = h.reshape(1, -1).astype(np.float32) @ deq.T
    np.testing.assert_allclose(out.astype(np.float32).reshape(-1), ref.reshape(-1), atol=0.02 * np.abs(ref).max())


def test_block_group_equals_chained_blocks():
    dims, opt = DIMS, D.GraphOptions(kv_style="onehot")
    S, P = 2, 4
    rng = np.random.default_rng(4)
    lw = [random_layer(rng, dims, opt)[0] for _ in range(3)]
    zeros = np.zeros((1, dims.num_kv_heads, P, dims.head_dim), np.float16)
    cos, sin = rope_tables(np.arange(S), dims.head_dim)
    mask = np.full((1, 1, S, P + S), NEG, np.float16)
    for i in range(S):
        mask[0, 0, i, P: P + i + 1] = 0
    rows = np.zeros((1, 1, P, S), np.float16); rows[0, 0, np.arange(S), np.arange(S)] = 1
    common = {"rope_cos": cos[None, None].astype(np.float16), "rope_sin": sin[None, None].astype(np.float16), "attention_mask": mask,
              "row_selector": rows, "rows_to_keep": 1 - rows.sum(-1, keepdims=True)}
    residual = (rng.standard_normal((1, S, dims.hidden)) * 0.5).astype(np.float16)
    attn_out = (rng.standard_normal((1, S, dims.attention_width)) * 0.5).astype(np.float16)
    ffn_names = {n for n, _, _ in D.feed_forward_weight_specs(dims, opt)}
    att_names = {n for n, _, _ in D.attention_weight_specs(dims, opt)}
    part = lambda w, names: {k: v for k, v in w.items() if k in names}  # noqa: E731

    block = D.build_block_graph(dims, opt, seq=S, past_len=P)
    r, a = residual, attn_out
    for j in range(2):
        out = run(block, {"residual": r, "attention_output": a, "past_key": zeros, "past_value": zeros,
                          **part(lw[j], ffn_names), **part(lw[j + 1], att_names), **common})
        r, a = out["next_residual"], out["next_attention_output"]

    group = D.build_block_group_graph(dims, opt, block_count=2, seq=S, past_len=P)
    feeds = {"residual": residual, "attention_output": attn_out, **common}
    for j in range(2):
        feeds.update({f"b{j}_{k}": v for k, v in {**part(lw[j], ffn_names), **part(lw[j + 1], att_names)}.items()})
        feeds[f"b{j}_past_key"] = zeros
        feeds[f"b{j}_past_value"] = zeros
    g = run(group, feeds)
    # inside a group the residual stays in the compute type (f32) between blocks, a chain of single blocks rounds to f16 after each: close, not identical
    for name, expected in (("next_residual", r), ("next_attention_output", a)):
        expected = expected.astype(np.float32)
        np.testing.assert_allclose(g[name].astype(np.float32), expected, atol=0.01 * np.abs(expected).max())

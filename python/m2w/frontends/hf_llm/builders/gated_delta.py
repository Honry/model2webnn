"""The Gated DeltaNet (linear attention of Qwen3.5 / Qwen3-Next) as a static WebNN graph: no loop, no scan.

The recurrence per head (state S [dk, dv], one token t with L2-normalised q, k, decay g <= 0, write strength beta in (0, 1)):
    S <- exp(g) S ;  S <- S + k (beta (v - S^T k))^T ;  o = S^T q
Decode is that update on one token. Prefill uses the chunkwise-parallel form (flash-linear-attention, HF `torch_chunk_gated_delta_rule`): the state is carried from one
sub-chunk of C tokens to the next, and inside a sub-chunk everything is a matmul except `(I + A)^-1` of a unit lower-triangular matrix, which HF computes with a for loop.
A is strictly lower-triangular, therefore nilpotent, so the inverse is EXACT as a finite product: (I + A)^-1 = (I - A)(I + A^2)(I + A^4)...(I + A^(C/2)), log2(C) factors.
The sub-chunks of a prefill graph are unrolled (the trip count is static), the chunks of a prompt are the runtime's loop (the state is an input and an output of the graph).

A padded token (beta = 0, g = 0) leaves the state untouched: pad rows are exact no-ops of the recurrence.
"""
from __future__ import annotations

import numpy as np

from ....ir import GraphBuilder, Value


def _tri(b: GraphBuilder, c: int, dtype: str, strict: bool) -> Value:
    """A constant lower-triangular 0/1 matrix [C, C] (diagonal included unless strict)."""
    data = np.tril(np.ones((c, c)), -1 if strict else 0).astype(int).reshape(-1).tolist()
    return b.constant(dtype, [c, c], data=data, cache_key=("tri", c, dtype, strict))


def _identity(b: GraphBuilder, c: int, dtype: str) -> Value:
    return b.constant(dtype, [c, c], data=np.eye(c, dtype=int).reshape(-1).tolist(), cache_key=("eye", c, dtype))


def unit_lower_inverse(b: GraphBuilder, n: Value, c: int, dtype: str) -> Value:
    """(I - N)^-1 for a strictly lower-triangular N [.., C, C]: (I + N)(I + N^2)(I + N^4)...(I + N^(C/2)) (exact, N^C = 0)."""
    eye = _identity(b, c, dtype)
    product = b.add(n, eye)
    power, step = n, 2
    while step < c:
        power = b.matmul(power, power)
        product = b.matmul(product, b.add(power, eye))
        step *= 2
    return product


def unit_lower_inverse_blocked(b: GraphBuilder, n: Value, c: int, base: int, dtype: str) -> Value:
    """(I - N)^-1 by block forward substitution: the Neumann product only on the diagonal blocks of `base` x `base`, the rest are matmuls with the rows already inverted.
    The plain product explodes for nearly parallel keys (the powers of N grow like binomial coefficients before they cancel); inside a small block they cannot."""
    if base >= c:
        return unit_lower_inverse(b, n, c, dtype)
    m, heads = c // base, n.shape[1]
    assert c % base == 0
    diagonal = [b.slice(n, [0, 0, i * base, i * base], [1, heads, base, base]) for i in range(m)]
    inverted = unit_lower_inverse(b, b.concat(diagonal, 1), base, dtype)                      # all diagonal blocks at once, as extra "heads"
    t_diag = [b.slice(inverted, [0, i * heads, 0, 0], [1, heads, base, base]) for i in range(m)]
    rows = [t_diag[0]]                                                                          # row i of blocks: [T_i0 .. T_i,i-1, T_ii], width (i + 1) base
    for i in range(1, m):
        width = i * base
        done = b.concat([rows[k] if k == i - 1 else b.concat([rows[k], b.constant(dtype, [1, heads, base, width - (k + 1) * base], fill=0.0)], 3) for k in range(i)], 2)   # T of the first i block rows [i base, i base]
        left = b.matmul(t_diag[i], b.matmul(b.slice(n, [0, 0, i * base, 0], [1, heads, base, width]), done))
        rows.append(b.concat([left, t_diag[i]], 3))
    return b.concat([b.concat([r, b.constant(dtype, [1, heads, base, c - r.shape[3]], fill=0.0)], 3) if r.shape[3] < c else r for r in rows], 2)


def gated_delta_chunk(b: GraphBuilder, q: Value, k: Value, v: Value, gc: Value, beta: Value, state: Value, dtype: str, base: int):
    """One sub-chunk of C tokens. q, k [1,H,C,dk] (normalised, q already scaled), v [1,H,C,dv], gc [1,H,C,1] = cumulative sum of the log decays inside the chunk,
    beta [1,H,C,1], state [1,H,dk,dv] -> (out [1,H,C,dv], new state)."""
    c = q.shape[2]
    incl, strict = _tri(b, c, dtype, False), _tri(b, c, dtype, True)
    gc_row = b.transpose(gc, [0, 1, 3, 2])
    decay = b.mul(b.exp(b.mul(b.sub(gc, gc_row), incl)), incl)                    # exp(gc_i - gc_j) for i >= j (the upper part is masked BEFORE the exp: no inf * 0)
    k_t = b.transpose(k, [0, 1, 3, 2])
    k_beta, v_beta = b.mul(k, beta), b.mul(v, beta)
    a = b.mul(b.mul(b.matmul(k_beta, k_t), decay), strict)                          # strictly lower part of (beta k) k^T * decay
    minus_one = b.scalar(-1.0, dtype, 4)
    t = unit_lower_inverse_blocked(b, b.mul(a, minus_one), c, base, dtype)           # (I + A)^-1
    gate = b.exp(gc)
    value = b.matmul(t, v_beta)
    k_cumdecay = b.matmul(t, b.mul(k_beta, gate))
    v_new = b.sub(value, b.matmul(k_cumdecay, state))                              # the delta: what the state does not predict yet
    inter = b.matmul(b.mul(q, gate), state)
    intra = b.matmul(b.mul(b.matmul(q, k_t), decay), v_new)                         # decay is lower incl. the diagonal
    out = b.add(inter, intra)
    g_last = b.slice(gc, [0, 0, c - 1, 0], [1, gc.shape[1], 1, 1])
    k_decayed = b.mul(k, b.exp(b.sub(g_last, gc)))
    new_state = b.add(b.mul(state, b.exp(g_last)), b.matmul(b.transpose(k_decayed, [0, 1, 3, 2]), v_new))
    return out, new_state


def gated_delta_chunks(b: GraphBuilder, q: Value, k: Value, v: Value, g: Value, beta: Value, state: Value, sub: int, dtype: str, base: int = 16):
    """T = n * sub tokens: q, k [1,H,T,dk], v [1,H,T,dv], g (log decay per token) and beta [1,H,T,1], state [1,H,dk,dv] -> (out [1,H,T,dv], state after the last token)."""
    heads, total = q.shape[1], q.shape[2]
    assert total % sub == 0, f"{total} tokens are not a multiple of the sub-chunk {sub}"
    incl = _tri(b, sub, dtype, False)
    outs = []
    for s in range(total // sub):
        part = lambda x, width: b.slice(x, [0, 0, s * sub, 0], [1, heads, sub, width])
        gs = part(g, 1)
        gc = b.matmul(incl, gs)                                                     # cumulative sum inside the sub-chunk as a matmul with the triangular constant
        out, state = gated_delta_chunk(b, part(q, q.shape[3]), part(k, k.shape[3]), part(v, v.shape[3]), gc, part(beta, 1), state, dtype, base)
        outs.append(out)
    return (b.concat(outs, 2) if len(outs) > 1 else outs[0]), state


def build_core_graph(heads: int, dk: int, dv: int, tokens: int, sub: int, dtype: str = "float32", name: str | None = None, base: int = 16):
    """The recurrence of one layer on its own (inputs after conv / l2norm / gating): the experiment graph, and the part that goes into the prefill BLOCK graph."""
    b = GraphBuilder(name or f"gdn_core_t{tokens}_c{sub}_b{min(base, sub)}_{dtype}")
    q = b.input("q", dtype, [1, heads, tokens, dk])
    k = b.input("k", dtype, [1, heads, tokens, dk])
    v = b.input("v", dtype, [1, heads, tokens, dv])
    g = b.input("g", dtype, [1, heads, tokens, 1])
    beta = b.input("beta", dtype, [1, heads, tokens, 1])
    state = b.input("state", dtype, [1, heads, dk, dv], role="kv")
    out, new_state = gated_delta_chunks(b, q, k, v, g, beta, state, sub, dtype, base)
    b.output("out", out)
    b.output("new_state", new_state)
    return b.build()


def build_recurrent_graph(heads: int, dk: int, dv: int, dtype: str = "float32", name: str | None = None):
    """The decode step: one token, no chunk machinery."""
    b = GraphBuilder(name or f"gdn_step_{dtype}")
    q = b.input("q", dtype, [1, heads, 1, dk])
    k = b.input("k", dtype, [1, heads, 1, dk])
    v = b.input("v", dtype, [1, heads, 1, dv])
    g = b.input("g", dtype, [1, heads, 1, 1])
    beta = b.input("beta", dtype, [1, heads, 1, 1])
    state = b.input("state", dtype, [1, heads, dk, dv], role="kv")
    decayed = b.mul(state, b.exp(g))                                                # g [1,H,1,1] broadcasts over [dk, dv]
    delta = b.mul(b.sub(v, b.matmul(k, decayed)), beta)                             # [1,H,1,dv]
    new_state = b.add(decayed, b.matmul(b.transpose(k, [0, 1, 3, 2]), delta))
    b.output("out", b.matmul(q, new_state))
    b.output("new_state", new_state)
    return b.build()

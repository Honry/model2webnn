"""Numpy oracle of the Gated DeltaNet recurrence, token by token in float64 (the definition the chunkwise graph has to reproduce)."""
from __future__ import annotations

import numpy as np


def recurrent(q, k, v, g, beta, state):
    """q, k [H,T,dk] (L2-normalised, q scaled), v [H,T,dv], g, beta [H,T] (g = log decay <= 0), state [H,dk,dv] -> (out [H,T,dv], state)."""
    q, k, v, g, beta, state = (np.asarray(x, dtype=np.float64) for x in (q, k, v, g, beta, state))
    state = state.copy()
    out = np.zeros(v.shape, dtype=np.float64)
    for t in range(q.shape[1]):
        state = state * np.exp(g[:, t])[:, None, None]
        delta = (v[:, t] - np.einsum("hk,hkv->hv", k[:, t], state)) * beta[:, t][:, None]
        state = state + np.einsum("hk,hv->hkv", k[:, t], delta)
        out[:, t] = np.einsum("hk,hkv->hv", q[:, t], state)
    return out, state


def l2norm(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)

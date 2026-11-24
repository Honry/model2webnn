"""Independent float32 numpy forward of a decoder described by a ModelSpec: the oracle for everything downstream (quantization, graphs, protocol).

Deliberately written from the model definition (HF semantics), not from the graphs: causal attention with an optional sliding window per layer, RMSNorm,
per-head q/k RMSNorm, GQA, SwiGLU, per-layer-type RoPE (incl. YaRN), tied embedding / lm_head. No KV cache: it recomputes the whole sequence."""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..frontends.hf_llm import rope as rope_lib
from ..frontends.hf_llm.arch import ModelSpec


def rms_norm(x, gamma, eps):
    return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps)) * gamma


def rotate_half(x):
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)


class ReferenceModel:
    def __init__(self, spec: ModelSpec, read: Callable[[str], np.ndarray]):
        """read(hf_tensor_name) -> float32 array (the original or a dequantized checkpoint)."""
        self.spec, self.read = spec, read
        self._inv = {t: rope_lib.inv_freq_for(r, spec.head_dim) for t, r in spec.rope.items()}
        self.layer_max_abs: list = []   # max |residual stream| after every layer of the last logits() call (f16 overflow check)

    def _layer(self, i):
        n = self.spec.layer_tensor_names(i)
        return {role: self.read(name) for role, name in n.items() if role not in ("q_norm", "k_norm") or self.spec.qk_norm}

    def visible(self, layer_type: str, q_pos: int, k_pos: int) -> bool:
        if k_pos > q_pos:
            return False
        return layer_type != "sliding_attention" or q_pos - k_pos < self.spec.sliding_window

    def logits(self, token_ids) -> np.ndarray:
        """Logits [T, vocab] for the whole sequence."""
        s = self.spec
        T = len(token_ids)
        embed = self.read(s.tensors["embed"])
        x = embed[np.asarray(token_ids)].astype(np.float32)
        group = s.num_heads // s.num_kv_heads
        self.layer_max_abs = []
        for i in range(s.num_layers):
            w = self._layer(i)
            ltype = s.layer_types[i]
            inv, factor = self._inv[ltype]
            cos, sin = rope_lib.cos_sin(np.arange(T), inv, factor)               # [T, hd]
            n = rms_norm(x, w["input_layernorm"], s.eps)
            q = (n @ w["q_proj"].T).reshape(T, s.num_heads, s.head_dim)
            k = (n @ w["k_proj"].T).reshape(T, s.num_kv_heads, s.head_dim)
            v = (n @ w["v_proj"].T).reshape(T, s.num_kv_heads, s.head_dim)
            if s.qk_norm:
                q, k = rms_norm(q, w["q_norm"], s.eps), rms_norm(k, w["k_norm"], s.eps)
            q = q * cos[:, None] + rotate_half(q) * sin[:, None]
            k = k * cos[:, None] + rotate_half(k) * sin[:, None]
            mask = np.array([[0.0 if self.visible(ltype, qi, ki) else -1e30 for ki in range(T)] for qi in range(T)], np.float32)
            out = np.zeros((T, s.num_heads, s.head_dim), np.float32)
            for h in range(s.num_heads):
                kv = h // group
                scores = (q[:, h] / np.sqrt(s.head_dim)) @ k[:, kv].T + mask
                p = np.exp(scores - scores.max(-1, keepdims=True))
                out[:, h] = (p / p.sum(-1, keepdims=True)) @ v[:, kv]
            x = x + out.reshape(T, -1) @ w["o_proj"].T
            n = rms_norm(x, w["post_attention_layernorm"], s.eps)
            gate, up = n @ w["gate_proj"].T, n @ w["up_proj"].T
            x = x + ((gate / (1.0 + np.exp(-gate))) * up) @ w["down_proj"].T
            self.layer_max_abs.append(float(np.abs(x).max()))
        x = rms_norm(x, self.read(s.tensors["final_norm"]), s.eps)
        self.last_hidden = x            # [T, hidden]: the input of the lm_head
        return x @ embed.T    # tied lm_head

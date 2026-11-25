"""Exploration: a numpy float32 reference of the Gemma 4 text decoder (google/gemma-4-E4B-it), written from transformers' modeling_gemma4.py.

It pins down what the compiler and the runtime have to implement (per-layer head dims, KV sharing, per-layer embeddings, sandwich norms, layer scalar, proportional RoPE,
logit softcapping) before any graph is built, and is the oracle to compare graphs against. No KV cache: every step recomputes the whole sequence (simple, slow).

    python tools/gemma4_ref.py <model dir> "What is the capital of France?" [--steps 8] [--dump-layers]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2w.frontends.hf_llm.safetensors_io import TensorStore  # noqa: E402

PREFIX = "model.language_model."


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def rms_norm(x, weight=None, eps=1e-6):
    out = x * np.power(np.mean(x * x, axis=-1, keepdims=True) + eps, -0.5)
    return out * weight if weight is not None else out


def rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class Gemma4Reference:
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        config = json.loads((self.dir / "config.json").read_text(encoding="utf-8"))
        self.cfg = c = config["text_config"]
        self.store = TensorStore.open(self.dir)
        self.hidden, self.layers, self.heads = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"]
        self.kv_heads, self.eps, self.window = c["num_key_value_heads"], c["rms_norm_eps"], c["sliding_window"]
        self.types = c["layer_types"]
        self.dim = {"sliding_attention": c["head_dim"], "full_attention": c["global_head_dim"]}
        self.ple_dim = c["hidden_size_per_layer_input"]
        self.softcap = c["final_logit_softcapping"]
        first_shared = self.layers - c["num_kv_shared_layers"]
        self.shared = [i >= first_shared for i in range(self.layers)]
        # the last non-shared layer of each type produces the K / V the shared layers read
        self.producer = {t: max(i for i in range(first_shared) if self.types[i] == t) for t in set(self.types)}
        self._cache: dict = {}

    # ------------------------------------------------------------------ weights (float32 on demand; layers are big, so nothing is kept)
    def w(self, name, rows=None):
        return self.store.read(PREFIX + name, rows=rows)

    def rope(self, layer_type, positions):
        c = self.cfg["rope_parameters"][layer_type]
        d = self.dim[layer_type]
        if c["rope_type"] == "default":
            inv = 1.0 / (c["rope_theta"] ** (np.arange(0, d, 2, dtype=np.float64) / d))
        else:   # proportional: only the first `partial_rotary_factor` of the head dim rotates; the exponent still divides by the FULL head dim; the rest has frequency 0
            angles = int(c.get("partial_rotary_factor", 1.0) * d // 2)
            inv = np.concatenate([1.0 / (c["rope_theta"] ** (np.arange(0, 2 * angles, 2, dtype=np.float64) / d)), np.zeros(d // 2 - angles)])
        freqs = np.asarray(positions, np.float64)[:, None] * inv[None, :]
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    # ------------------------------------------------------------------ the model
    def logits(self, ids, dump=None):
        n = len(ids)
        positions = np.arange(n)
        embed_rows = np.stack([self.w("embed_tokens.weight", rows=(int(t), int(t) + 1))[0] for t in ids])
        x = embed_rows * np.float32(self.hidden ** 0.5)                                             # [n, hidden]
        # per-layer embeddings: token identity (a row of the huge table) + a projection of the embedding, normalised, averaged
        ple_rows = np.stack([self.w("embed_tokens_per_layer.weight", rows=(int(t), int(t) + 1))[0] for t in ids]) * np.float32(self.ple_dim ** 0.5)
        ple_rows = ple_rows.reshape(n, self.layers, self.ple_dim)
        projected = (x @ self.w("per_layer_model_projection.weight").T) * np.float32(self.hidden ** -0.5)
        projected = rms_norm(projected.reshape(n, self.layers, self.ple_dim), self.w("per_layer_projection_norm.weight"), self.eps)
        per_layer = (projected + ple_rows) * np.float32(2.0 ** -0.5)                                # [n, layers, ple_dim]
        tables = {t: self.rope(t, positions) for t in set(self.types)}
        store: dict = {}
        causal = positions[:, None] >= positions[None, :]
        masks = {"full_attention": causal, "sliding_attention": causal & (positions[:, None] - positions[None, :] < self.window)}
        for i in range(self.layers):
            t0 = time.time()
            t = self.types[i]
            d, p = self.dim[t], f"layers.{i}."
            cos, sin = tables[t]
            h = rms_norm(x, self.w(p + "input_layernorm.weight"), self.eps)
            q = (h @ self.w(p + "self_attn.q_proj.weight").T).reshape(n, self.heads, d)
            q = rms_norm(q, self.w(p + "self_attn.q_norm.weight"), self.eps)
            q = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
            if self.shared[i]:
                k, v = store[self.producer[t]]
            else:
                k = (h @ self.w(p + "self_attn.k_proj.weight").T).reshape(n, self.kv_heads, d)
                v = (h @ self.w(p + "self_attn.v_proj.weight").T).reshape(n, self.kv_heads, d)
                k = rms_norm(k, self.w(p + "self_attn.k_norm.weight"), self.eps)
                k = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]
                v = rms_norm(v, None, self.eps)                                                       # v_norm has no scale
                store[i] = (k, v)
            group = self.heads // self.kv_heads
            kk, vv = np.repeat(k, group, axis=1), np.repeat(v, group, axis=1)                          # [n, heads, d]
            scores = np.einsum("qhd,khd->hqk", q, kk)                                                  # attention scaling is 1.0
            scores = np.where(masks[t][None], scores, np.float32(-1e30))
            ctx = np.einsum("hqk,khd->qhd", softmax(scores), vv).reshape(n, self.heads * d)
            attn = ctx @ self.w(p + "self_attn.o_proj.weight").T
            x = x + rms_norm(attn, self.w(p + "post_attention_layernorm.weight"), self.eps)
            h = rms_norm(x, self.w(p + "pre_feedforward_layernorm.weight"), self.eps)
            mlp = (gelu_tanh(h @ self.w(p + "mlp.gate_proj.weight").T) * (h @ self.w(p + "mlp.up_proj.weight").T)) @ self.w(p + "mlp.down_proj.weight").T
            x = x + rms_norm(mlp, self.w(p + "post_feedforward_layernorm.weight"), self.eps)
            gate = gelu_tanh(x @ self.w(p + "per_layer_input_gate.weight").T) * per_layer[:, i, :]
            x = x + rms_norm(gate @ self.w(p + "per_layer_projection.weight").T, self.w(p + "post_per_layer_input_norm.weight"), self.eps)
            x = x * self.w(p + "layer_scalar")
            if dump is not None:
                dump.append(float(np.abs(x).max()))
            print(f"  layer {i:2d} {t[:4]} d={d} {'shared' if self.shared[i] else 'own   '} max|x|={np.abs(x).max():9.1f}  {time.time() - t0:.1f}s", file=sys.stderr, flush=True)
        x = rms_norm(x[-1:], self.w("norm.weight"), self.eps)
        logits = x @ self.w("embed_tokens.weight").T
        return np.tanh(logits / self.softcap) * self.softcap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompt")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()
    from jinja2 import Environment
    from tokenizers import Tokenizer

    model_dir = Path(args.model)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    env = Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    template = env.from_string((model_dir / "chat_template.jinja").read_text(encoding="utf-8"))
    text = template.render(messages=[{"role": "user", "content": args.prompt}], add_generation_prompt=True, enable_thinking=False, bos_token="<bos>", tools=None)
    print(repr(text))
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    print("prompt ids", ids)
    ref = Gemma4Reference(model_dir)
    out = []
    for _ in range(args.steps):
        logits = ref.logits(ids + out)[0]
        token = int(np.argmax(logits))
        out.append(token)
        print("token", token, repr(tokenizer.decode(out)), flush=True)
        if token in (1, 106, 50):
            break


if __name__ == "__main__":
    main()

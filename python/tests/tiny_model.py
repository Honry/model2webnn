"""A tiny random HF-style decoder (safetensors + config) for pipeline tests: Aion-like (hybrid layers, two RoPE types, qk-norm, tied embedding, odd vocabulary)."""
import json
from pathlib import Path

import numpy as np

from m2w.frontends.hf_llm.safetensors_io import write_safetensors

CONFIG = {
    "architectures": ["AionInstructForCausalLM"], "vocab_size": 300, "hidden_size": 256, "intermediate_size": 256, "num_hidden_layers": 3,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 64, "hidden_act": "silu", "max_position_embeddings": 4096, "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True, "attention_bias": False, "bos_token_id": 1, "eos_token_id": 2,
    "layer_types": ["sliding_attention", "full_attention", "sliding_attention"], "sliding_window": 16,
    "rope_parameters": {"full_attention": {"rope_type": "yarn", "rope_theta": 10000, "factor": 4.0, "original_max_position_embeddings": 1024, "beta_fast": 32.0, "beta_slow": 1.0},
                        "sliding_attention": {"rope_type": "default", "rope_theta": 10000}},
}


def write_tiny_model(directory, seed: int = 0, dtype: str = "BF16") -> dict:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    c = CONFIG
    rng = np.random.default_rng(seed)
    H, F, hd = c["hidden_size"], c["intermediate_size"], c["head_dim"]
    aw, kw = c["num_attention_heads"] * hd, c["num_key_value_heads"] * hd
    lin = lambda out, inp: (rng.standard_normal((out, inp)) / np.sqrt(inp)).astype(np.float32)  # noqa: E731
    gamma = lambda n: rng.uniform(0.8, 1.2, n).astype(np.float32)  # noqa: E731
    tensors = {"model.embed_tokens.weight": (rng.standard_normal((c["vocab_size"], H)) * 0.5).astype(np.float32), "model.norm.weight": gamma(H)}
    for i in range(c["num_hidden_layers"]):
        p = f"model.layers.{i}."
        tensors.update({
            p + "input_layernorm.weight": gamma(H), p + "post_attention_layernorm.weight": gamma(H),
            p + "self_attn.q_proj.weight": lin(aw, H), p + "self_attn.k_proj.weight": lin(kw, H), p + "self_attn.v_proj.weight": lin(kw, H),
            p + "self_attn.o_proj.weight": lin(H, aw), p + "self_attn.q_norm.weight": gamma(hd), p + "self_attn.k_norm.weight": gamma(hd),
            p + "mlp.gate_proj.weight": lin(F, H), p + "mlp.up_proj.weight": lin(F, H), p + "mlp.down_proj.weight": lin(H, F),
        })
    (d / "config.json").write_text(json.dumps(c))
    write_safetensors(d / "model.safetensors", tensors, dtype)
    return tensors

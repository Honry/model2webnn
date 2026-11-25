"""A tiny random Gemma 4 style model (safetensors + config): hybrid layers with different head dims (sliding 32 / full 64), KV sharing in the last two layers, per-layer embeddings,
sandwich norms, a layer scalar, proportional RoPE on the full attention layers and a logit softcap - everything the real E4B has, at a size the CPU tests can run."""
import json
from pathlib import Path

import numpy as np

from m2w.frontends.hf_llm.safetensors_io import write_safetensors

TEXT = {
    "vocab_size": 300, "vocab_size_per_layer_input": 300, "hidden_size": 128, "hidden_size_per_layer_input": 128, "intermediate_size": 256, "num_hidden_layers": 6,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32, "global_head_dim": 64, "num_kv_shared_layers": 2, "hidden_activation": "gelu_pytorch_tanh",
    "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "tie_word_embeddings": True, "attention_bias": False, "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0,
    "final_logit_softcapping": 30.0, "sliding_window": 16, "enable_moe_block": False, "num_experts": None, "attention_k_eq_v": False, "use_double_wide_mlp": False,
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention", "full_attention", "sliding_attention"],
    "rope_parameters": {"full_attention": {"rope_type": "proportional", "partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
                        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}},
}
CONFIG = {"architectures": ["Gemma4ForConditionalGeneration"], "model_type": "gemma4", "text_config": TEXT, "eos_token_id": [1, 106]}
P = "model.language_model."


def write_tiny_gemma4(directory, seed: int = 0, dtype: str = "BF16") -> dict:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    c = TEXT
    rng = np.random.default_rng(seed)
    H, F, n, ple = c["hidden_size"], c["intermediate_size"], c["num_hidden_layers"], c["hidden_size_per_layer_input"]
    heads, kv = c["num_attention_heads"], c["num_key_value_heads"]
    lin = lambda out, inp: (rng.standard_normal((out, inp)) / np.sqrt(inp)).astype(np.float32)  # noqa: E731
    gamma = lambda m: rng.uniform(0.8, 1.2, m).astype(np.float32)  # noqa: E731
    t = {P + "embed_tokens.weight": (rng.standard_normal((c["vocab_size"], H)) * 0.5).astype(np.float32), P + "norm.weight": gamma(H),
         P + "embed_tokens_per_layer.weight": (rng.standard_normal((c["vocab_size_per_layer_input"], n * ple)) * 0.5).astype(np.float32),
         P + "per_layer_model_projection.weight": lin(n * ple, H), P + "per_layer_projection_norm.weight": gamma(ple)}
    for i in range(n):
        d = c["global_head_dim"] if c["layer_types"][i] == "full_attention" else c["head_dim"]
        p = f"{P}layers.{i}."
        t.update({
            p + "input_layernorm.weight": gamma(H), p + "post_attention_layernorm.weight": gamma(H), p + "pre_feedforward_layernorm.weight": gamma(H),
            p + "post_feedforward_layernorm.weight": gamma(H), p + "post_per_layer_input_norm.weight": gamma(H), p + "layer_scalar": rng.uniform(0.5, 1.5, 1).astype(np.float32),
            p + "self_attn.q_proj.weight": lin(heads * d, H), p + "self_attn.k_proj.weight": lin(kv * d, H), p + "self_attn.v_proj.weight": lin(kv * d, H),
            p + "self_attn.o_proj.weight": lin(H, heads * d), p + "self_attn.q_norm.weight": gamma(d), p + "self_attn.k_norm.weight": gamma(d),
            p + "mlp.gate_proj.weight": lin(F, H), p + "mlp.up_proj.weight": lin(F, H), p + "mlp.down_proj.weight": lin(H, F),
            p + "per_layer_input_gate.weight": lin(ple, H), p + "per_layer_projection.weight": lin(H, ple),
        })
    (root / "config.json").write_text(json.dumps(CONFIG))
    write_safetensors(root / "model.safetensors", t, dtype)
    return t

"""A tiny random Qwen3.5 (hybrid: 3 Gated DeltaNet layers + 1 full attention layer per group), built with transformers so that the same weights can drive the independent reference model.
Needs torch + transformers >= 5.3 (tests skip without them)."""
import json
from pathlib import Path

import numpy as np

from m2w.frontends.hf_llm.safetensors_io import write_safetensors

TEXT = {
    "vocab_size": 300, "hidden_size": 128, "intermediate_size": 256, "num_hidden_layers": 6, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
    "layer_types": ["linear_attention", "linear_attention", "full_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
    "attn_output_gate": True, "attention_bias": False, "attention_dropout": 0.0, "hidden_act": "silu", "max_position_embeddings": 512, "rms_norm_eps": 1e-6,
    "tie_word_embeddings": True, "full_attention_interval": 3, "mlp_only_layers": [], "eos_token_id": 1, "bos_token_id": None, "pad_token_id": 0,
    "rope_parameters": {"mrope_interleaved": True, "mrope_section": [1, 1, 2], "rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25},
}
CONFIG = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "text_config": {**TEXT, "model_type": "qwen3_5_text"}, "eos_token_id": 1}


def hf_model(seed: int = 0):
    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(seed)
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**TEXT)).eval()
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log"):
                p.copy_(torch.log(torch.empty_like(p).uniform_(0.05, 8.0, generator=g)))
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif name.endswith("linear_attn.norm.weight"):
                p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.1)
            elif name.endswith("norm.weight"):   # every (1 + w) RMSNorm
                p.copy_(torch.randn(p.shape, generator=g) * 0.2)
            elif p.dim() >= 2:
                p.copy_(torch.randn(p.shape, generator=g) / np.sqrt(p.shape[-1]) * (0.5 if "embed" in name else 1.0))
    return model


def write_tiny_qwen35(directory, seed: int = 0, dtype: str = "F32") -> dict:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    model = hf_model(seed)
    tensors = {}
    for name, p in model.state_dict().items():
        if name == "lm_head.weight":
            continue   # tied
        tensors["model.language_model." + name[len("model."):]] = p.detach().float().numpy()
    (root / "config.json").write_text(json.dumps(CONFIG))
    write_safetensors(root / "model.safetensors", tensors, dtype)
    return tensors

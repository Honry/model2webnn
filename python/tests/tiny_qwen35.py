"""A tiny random Qwen3.5 (hybrid: 3 Gated DeltaNet layers + 1 full attention layer per group, plus a tiny image tower), built with transformers so that the same weights can drive the independent
reference model (`hf_model`: the text decoder, `hf_vl_model`: text + image tower). Needs torch + transformers >= 5.3 (tests skip without them)."""
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
VISION = {
    "depth": 2, "hidden_size": 128, "num_heads": 8, "intermediate_size": 256, "patch_size": 8, "temporal_patch_size": 2, "spatial_merge_size": 2, "out_hidden_size": 128,
    "num_position_embeddings": 36, "hidden_act": "gelu_pytorch_tanh", "in_channels": 3, "deepstack_visual_indexes": [], "initializer_range": 0.02,
}
TOKENS = {"image_token_id": 290, "video_token_id": 291, "vision_start_token_id": 288, "vision_end_token_id": 289}
GRIDS = ((4, 4), (4, 6), (6, 4))   # the grid presets of the tiny package (patches of 8 pixels: 32 x 32, 32 x 48, 48 x 32 pixels)
CONFIG = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "text_config": {**TEXT, "model_type": "qwen3_5_text"}, "eos_token_id": 1,
          "vision_config": {**VISION, "model_type": "qwen3_5"}, **TOKENS}


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


def vl_config():
    from transformers import Qwen3_5Config

    return Qwen3_5Config(text_config=TEXT, vision_config=VISION, **TOKENS)


def hf_vl_model(seed: int = 0):
    """Qwen3_5ForConditionalGeneration: the text weights of hf_model(seed), random image tower weights (norm scales around 1, small biases, learned position table of 6 x 6)."""
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    text = hf_model(seed)
    model = Qwen3_5ForConditionalGeneration(vl_config()).eval()
    g = torch.Generator().manual_seed(seed + 2)
    with torch.no_grad():
        state = model.state_dict()
        for name, p in text.state_dict().items():
            if name != "lm_head.weight":
                state["model.language_model." + name[len("model."):]].copy_(p)
        for name, p in model.model.visual.named_parameters():
            if name.endswith("norm1.weight") or name.endswith("norm2.weight") or name == "merger.norm.weight":
                p.copy_(1.0 + torch.randn(p.shape, generator=g) * 0.1)
            elif name.endswith("bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.1)
            elif name == "pos_embed.weight":
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            else:
                p.copy_(torch.randn(p.shape, generator=g) / np.sqrt(p[0].numel()))
    return model


def write_tiny_qwen35(directory, seed: int = 0, dtype: str = "F32") -> dict:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    model = hf_vl_model(seed)
    tensors = {}
    for name, p in model.state_dict().items():
        if name == "lm_head.weight":
            continue   # tied
        tensors[name] = p.detach().float().numpy()   # model.language_model.* and model.visual.*
    (root / "config.json").write_text(json.dumps(CONFIG))
    write_safetensors(root / "model.safetensors", tensors, dtype)
    return tensors

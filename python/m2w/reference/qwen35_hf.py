"""The transformers implementation of Qwen3.5 as the independent reference of `m2w verify` (needs torch + transformers >= 5.3): the same forward pass the model authors ship, fed either with the
dequantized package weights (graphs + protocol + packing must agree with it to float16 noise) or with the original weights (the cost of the quantization)."""
from __future__ import annotations

import numpy as np


class Qwen35HFModel:
    """logits(ids) -> float32 [len(ids), vocab]; `tensors`: HF names of the CAUSAL LM layout ("model.layers.0...", "model.embed_tokens.weight") -> arrays (the lm_head is tied)."""

    def __init__(self, text_config: dict, tensors: dict):
        import torch
        from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

        self.layer_max_abs: list = []
        config = Qwen3_5TextConfig(**{k: v for k, v in text_config.items() if k != "model_type"})
        self.torch = torch
        self.model = Qwen3_5ForCausalLM(config).eval()
        state = {k: torch.from_numpy(np.ascontiguousarray(np.asarray(v, dtype=np.float32))) for k, v in tensors.items()}
        missing = self.model.load_state_dict(state, strict=False).missing_keys
        if [k for k in missing if k != "lm_head.weight"]:
            raise KeyError(f"tensors missing for the Qwen3.5 reference: {missing[:5]}")
        self.model.tie_weights()

    def logits(self, ids) -> np.ndarray:
        with self.torch.no_grad():
            return self.model(self.torch.tensor([list(ids)])).logits[0].float().numpy()


def original_tensors(store, spec) -> dict:
    """The tensors of a TensorStore (HF checkpoint layout `model.language_model.*`) renamed to the causal-LM layout of the reference (the vision tower and the MTP head are not read)."""
    prefix = "model.language_model."
    names = [spec.tensors["embed"], spec.tensors["final_norm"]] + [n for i in range(spec.num_layers) for n in spec.layer_tensor_names(i).values()]
    return {"model." + name[len(prefix):]: store.read(name) for name in names if name in store}


class Qwen35VisionHF:
    """transformers' Qwen3_5VisionModel (the image tower of Qwen3.5) as the independent reference of the vision graphs; `tensors`: the names of the full model ("model.visual.blocks.0.attn.qkv.weight" ...)."""

    def __init__(self, vision_config: dict, tensors: dict, prefix: str = "model.visual."):
        import torch
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        self.torch = torch
        self.model = Qwen3_5VisionModel(Qwen3_5VisionConfig(**{k: v for k, v in vision_config.items() if k != "model_type"})).eval()
        state = {k[len(prefix):]: torch.from_numpy(np.ascontiguousarray(np.asarray(v, dtype=np.float32))) for k, v in tensors.items() if k.startswith(prefix)}
        self.model.load_state_dict(state, strict=True)
        self.block_outputs: list = []
        for block in self.model.blocks:
            block.register_forward_hook(lambda _m, _i, out: self.block_outputs.append(out.detach().float().numpy()))

    def forward(self, patches, grid):
        """patches [N, patch_vector] float32 (merge-block order), grid (h, w) -> (image_embeds [N / 4, out_hidden], last hidden [N, hidden]); block_outputs = the hidden state after every block"""
        self.block_outputs.clear()
        with self.torch.no_grad():
            out = self.model(self.torch.from_numpy(np.ascontiguousarray(patches, dtype=np.float32)), grid_thw=self.torch.tensor([[1, grid[0], grid[1]]]))
        return out.pooler_output.float().numpy(), out.last_hidden_state.float().numpy()

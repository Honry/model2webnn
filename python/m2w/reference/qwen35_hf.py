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

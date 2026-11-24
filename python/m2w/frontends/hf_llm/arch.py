"""config.json (+ generation_config.json) -> ModelSpec, the language-neutral description of a decoder (written as model.json in the package).

The architectures are described by DATA (families/*.json: HF tensor names per role, architecture strings), not by classes. Everything a graph cannot see
(per-layer-type RoPE, sliding windows, EOS tokens) is resolved here so that the runtime only reads model.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from importlib import resources
from typing import Optional

from .builders.decoder import DecoderDims

FORMAT = "m2w-model"
VERSION = 0
GROUP_SIZE = 128
VOCAB_ALIGNMENT = 128   # the lm_head matmul wants a vocabulary that is a multiple of 128 (pad rows have scale 0)


class UnsupportedModel(ValueError):
    pass


def families() -> dict:
    out = {}
    root = resources.files(__package__).joinpath("families")
    for entry in sorted(root.iterdir(), key=lambda p: p.name):
        if entry.name.endswith(".json"):
            family = json.loads(entry.read_text(encoding="utf-8"))
            out[family["name"]] = family
    return out


def family_for(architecture: str) -> dict:
    for family in families().values():
        if architecture in family["architectures"]:
            return family
    supported = sorted(a for f in families().values() for a in f["architectures"])
    raise UnsupportedModel(f"architecture '{architecture}' is not supported yet (supported: {', '.join(supported)})")


@dataclass
class ModelSpec:
    architecture: str
    family: str
    hidden: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    ffn: int
    vocab: int                 # the real vocabulary size
    vocab_padded: int          # what the lm_head graph uses
    eps: float
    qk_norm: bool
    tie_embeddings: bool
    activation: str
    max_position_embeddings: int
    layer_types: list          # HF names: "full_attention" | "sliding_attention", one per layer
    sliding_window: Optional[int]
    rope: dict                 # layer type -> {"type": "default"|"linear"|"yarn", "theta": ..., ...}
    eos_token_ids: list
    bos_token_id: Optional[int]
    pad_token_id: Optional[int]
    group_size: int = GROUP_SIZE
    tensors: dict = field(default_factory=dict)   # role -> HF tensor name (templates with {i})
    chat_template_kwargs: dict = field(default_factory=dict)   # e.g. {"enable_thinking": False} for Qwen3

    def decoder_dims(self) -> DecoderDims:
        return DecoderDims(hidden=self.hidden, num_heads=self.num_heads, num_kv_heads=self.num_kv_heads, head_dim=self.head_dim, ffn=self.ffn,
                           vocab=self.vocab_padded, eps=self.eps, group_size=self.group_size, qk_norm=self.qk_norm)

    def layer_tensor_names(self, i: int) -> dict:
        return {role: name.format(i=i) for role, name in self.tensors["layer"].items()}

    def to_dict(self) -> dict:
        return {"format": FORMAT, "version": VERSION, **asdict(self)}

    @staticmethod
    def from_dict(d: dict) -> "ModelSpec":
        if d.get("format") != FORMAT or d.get("version") != VERSION:
            raise ValueError(f"not an {FORMAT} v{VERSION} document")
        return ModelSpec(**{k: v for k, v in d.items() if k not in ("format", "version")})


def _round_up(n: int, multiple: int) -> int:
    return -(-n // multiple) * multiple


def _resolve_rope(config: dict, layer_types: list) -> dict:
    """Normalise the three config spellings (per-layer-type rope_parameters, flat rope_parameters, legacy rope_theta + rope_scaling) to {layer type: entry}."""
    params = config.get("rope_parameters")
    if params is None:
        scaling = dict(config.get("rope_scaling") or {})
        params = {**scaling, "rope_type": scaling.get("rope_type") or scaling.get("type") or "default", "rope_theta": config.get("rope_theta", 10000.0)}
    per_type = params if any(k in ("full_attention", "sliding_attention") for k in params) else {t: params for t in set(layer_types)}
    out = {}
    for layer_type in sorted(set(layer_types)):
        p = per_type[layer_type]
        kind = p.get("rope_type", "default")
        if kind not in ("default", "linear", "yarn"):
            raise UnsupportedModel(f"rope type '{kind}' is not supported yet")
        if p.get("partial_rotary_factor", config.get("partial_rotary_factor", 1.0)) != 1.0:
            raise UnsupportedModel("partial rotary embeddings are not supported yet")
        entry = {"type": kind, "theta": float(p["rope_theta"])}
        for key in ("factor", "original_max_position_embeddings", "beta_fast", "beta_slow", "attention_factor", "mscale", "mscale_all_dim", "truncate"):
            if p.get(key) is not None:
                entry[key] = p[key]
        if kind == "yarn":
            entry.setdefault("original_max_position_embeddings", config["max_position_embeddings"])
            if "factor" not in entry:
                entry["factor"] = config["max_position_embeddings"] / entry["original_max_position_embeddings"]
        out[layer_type] = entry
    return out


def spec_from_config(config: dict, generation_config: Optional[dict] = None) -> ModelSpec:
    architecture = (config.get("architectures") or [""])[0]
    family = family_for(architecture)
    if config.get("attention_bias") or config.get("mlp_bias"):
        raise UnsupportedModel("models with attention / MLP biases are not supported yet")
    activation = config.get("hidden_act", "silu")
    if activation != "silu":
        raise UnsupportedModel(f"activation '{activation}' is not supported yet (only silu)")
    heads = config["num_attention_heads"]
    layers = config["num_hidden_layers"]
    layer_types = list(config.get("layer_types") or ["full_attention"] * layers)
    if len(layer_types) != layers or not set(layer_types) <= {"full_attention", "sliding_attention"}:
        raise UnsupportedModel(f"unsupported layer_types: {sorted(set(layer_types))}")
    sliding_window = config.get("sliding_window") if "sliding_attention" in layer_types else None
    generation_config = generation_config or {}
    eos = generation_config.get("eos_token_id", config.get("eos_token_id"))
    eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    vocab = config["vocab_size"]
    spec = ModelSpec(
        architecture=architecture, family=family["name"], hidden=config["hidden_size"], num_layers=layers, num_heads=heads,
        num_kv_heads=config.get("num_key_value_heads", heads), head_dim=config.get("head_dim") or config["hidden_size"] // heads,
        ffn=config["intermediate_size"], vocab=vocab, vocab_padded=_round_up(vocab, VOCAB_ALIGNMENT), eps=float(config["rms_norm_eps"]),
        qk_norm=bool(family["qk_norm"]), tie_embeddings=bool(config.get("tie_word_embeddings", False)), activation=activation,
        max_position_embeddings=config["max_position_embeddings"], layer_types=layer_types, sliding_window=sliding_window,
        rope=_resolve_rope(config, layer_types), eos_token_ids=eos, bos_token_id=config.get("bos_token_id"), pad_token_id=config.get("pad_token_id"),
        tensors=family["tensors"], chat_template_kwargs=dict(family.get("chat_template_kwargs", {}).get(architecture, {})),
    )
    spec.decoder_dims().check()
    if layers < 2:
        raise UnsupportedModel("the HEAD / BLOCK / TAIL split needs at least 2 layers")
    return spec

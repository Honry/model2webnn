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
    # --- what Gemma-style models add (all defaults describe the plain decoders above) ---
    builder: str = "decoder"                                    # which graph builder: "decoder" (Qwen3-like) | "gemma4"
    layer_dims: dict = field(default_factory=dict)              # layer type -> {"head_dim", "num_heads", "num_kv_heads"} where it differs from the defaults above
    kv_producer: list = field(default_factory=list)             # per layer: the layer whose K / V this layer attends over (KV sharing), -1 = its own; empty = nothing shared
    ple_dim: int = 0                                            # per-layer embedding size (0 = none)
    embed_scale: float = 1.0                                    # the embedding is multiplied by this (Gemma: sqrt(hidden))
    attention_scale: Optional[float] = None                     # None = 1 / sqrt(head_dim)
    logit_softcap: float = 0.0                                  # logits = cap * tanh(logits / cap) when > 0
    linear: dict = field(default_factory=dict)                  # Gated DeltaNet layers (layer type "linear_attention", Qwen3.5): {"num_k_heads", "num_v_heads", "k_dim", "v_dim", "conv_kernel"}
    attn_output_gate: bool = False                              # the full attention output is multiplied by sigmoid(gate), the gate comes out of q_proj next to the query (Qwen3.5)

    def decoder_dims(self, layer_type: Optional[str] = None) -> DecoderDims:
        """The dims of the attention of a layer of this type (the defaults for the types that do not override them)."""
        o = self.layer_dims.get(layer_type, {}) if layer_type else {}
        return DecoderDims(hidden=self.hidden, num_heads=o.get("num_heads", self.num_heads), num_kv_heads=o.get("num_kv_heads", self.num_kv_heads),
                           head_dim=o.get("head_dim", self.head_dim), ffn=self.ffn, vocab=self.vocab_padded, eps=self.eps, group_size=self.group_size, qk_norm=self.qk_norm)

    def layer_kind(self, i: int) -> str:
        """The kind of the attention half of layer i: '<layer type>_own' | '<layer type>_shared' (shared = reads the K / V of another layer)."""
        if self.builder == "qwen35":
            return self.layer_types[i]   # linear_attention | full_attention: every layer with K / V has its own
        return f"{self.layer_types[i]}_{'shared' if self.kv_producer and self.kv_producer[i] >= 0 else 'own'}"

    def mask_types(self) -> list:
        """The layer types that attend over a KV cache (and so have masks and rope tables): the linear attention layers have neither."""
        return sorted(t for t in set(self.layer_types) if t != "linear_attention")

    def layer_tensor_names(self, i: int) -> dict:
        return {role: name.format(i=i) for role, name in self.tensors["layer"].items()}

    def to_dict(self) -> dict:
        d = asdict(self)
        for key in ("linear", "attn_output_gate"):   # Qwen3.5 additions: absent from the model.json of every other model
            if not d[key]:
                del d[key]
        return {"format": FORMAT, "version": VERSION, **d}

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
    for layer_type in sorted(set(layer_types) - {"linear_attention"}):   # Gated DeltaNet layers have no positions
        p = per_type[layer_type]
        kind = p.get("rope_type", "default")
        if kind not in ("default", "linear", "yarn", "proportional"):
            raise UnsupportedModel(f"rope type '{kind}' is not supported yet")
        partial = p.get("partial_rotary_factor", config.get("partial_rotary_factor", 1.0))
        if partial != 1.0 and kind not in ("proportional", "default"):
            raise UnsupportedModel("partial rotary embeddings are only supported for the default and the proportional rope types")
        entry = {"type": kind, "theta": float(p["rope_theta"])}
        if kind == "proportional" or partial != 1.0:
            entry["partial_rotary_factor"] = float(partial)
        for key in ("factor", "original_max_position_embeddings", "beta_fast", "beta_slow", "attention_factor", "mscale", "mscale_all_dim", "truncate"):
            if p.get(key) is not None:
                entry[key] = p[key]
        if kind == "yarn":
            entry.setdefault("original_max_position_embeddings", config["max_position_embeddings"])
            if "factor" not in entry:
                entry["factor"] = config["max_position_embeddings"] / entry["original_max_position_embeddings"]
        out[layer_type] = entry
    return out


def _kv_producers(layer_types: list, shared_layers: int) -> list:
    """Gemma-style KV sharing: from layer (N - shared_layers) on, a layer reads the K / V of the last layer before that point that has the same type. -1 = its own K / V."""
    if not shared_layers:
        return []
    first = len(layer_types) - shared_layers
    producers = {t: max(i for i in range(first) if layer_types[i] == t) for t in set(layer_types[:first])}
    if set(layer_types) - set(producers):
        raise UnsupportedModel("a layer type that shares K / V has no layer of the same type before the shared layers")
    return [-1 if i < first else producers[layer_types[i]] for i in range(len(layer_types))]


def _gemma4_spec(config: dict, generation_config: Optional[dict], family: dict, architecture: str) -> ModelSpec:
    t = config.get("text_config", config)
    if t.get("enable_moe_block") or t.get("num_experts"):
        raise UnsupportedModel("Gemma 4 models with a mixture-of-experts block are not supported (E2B / E4B have none)")
    if t.get("attention_k_eq_v") or t.get("use_double_wide_mlp") or t.get("use_bidirectional_attention"):
        raise UnsupportedModel("Gemma 4 options attention_k_eq_v / use_double_wide_mlp / use_bidirectional_attention are not supported yet")
    activation = {"gelu_pytorch_tanh": "gelu_tanh"}.get(t.get("hidden_activation"))
    if activation is None:
        raise UnsupportedModel(f"activation '{t.get('hidden_activation')}' is not supported")
    heads, layers = t["num_attention_heads"], t["num_hidden_layers"]
    layer_types = list(t["layer_types"])
    if len(layer_types) != layers or not set(layer_types) <= {"full_attention", "sliding_attention"}:
        raise UnsupportedModel(f"unsupported layer_types: {sorted(set(layer_types))}")
    generation_config = generation_config or {}
    eos = generation_config.get("eos_token_id", config.get("eos_token_id", t.get("eos_token_id")))
    eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    vocab = t["vocab_size"]
    kv_heads = t["num_key_value_heads"]
    spec = ModelSpec(
        architecture=architecture, family=family["name"], hidden=t["hidden_size"], num_layers=layers, num_heads=heads, num_kv_heads=kv_heads, head_dim=t["head_dim"],
        ffn=t["intermediate_size"], vocab=vocab, vocab_padded=_round_up(vocab, VOCAB_ALIGNMENT), eps=float(t["rms_norm_eps"]), qk_norm=True,
        tie_embeddings=bool(t.get("tie_word_embeddings", config.get("tie_word_embeddings", True))), activation=activation,
        max_position_embeddings=t["max_position_embeddings"], layer_types=layer_types, sliding_window=t.get("sliding_window") if "sliding_attention" in layer_types else None,
        rope=_resolve_rope(t, layer_types), eos_token_ids=eos, bos_token_id=t.get("bos_token_id"), pad_token_id=t.get("pad_token_id"), tensors=family["tensors"],
        builder=family["builder"], layer_dims={"full_attention": {"head_dim": t["global_head_dim"], "num_heads": heads, "num_kv_heads": kv_heads}} if "full_attention" in layer_types else {},
        kv_producer=_kv_producers(layer_types, t.get("num_kv_shared_layers") or 0), ple_dim=t.get("hidden_size_per_layer_input") or 0, embed_scale=float(t["hidden_size"] ** 0.5),
        attention_scale=1.0, logit_softcap=float(t.get("final_logit_softcapping") or 0.0),
    )
    for layer_type in set(layer_types):
        spec.decoder_dims(layer_type).check()
    return spec


def _qwen35_spec(config: dict, generation_config: Optional[dict], family: dict, architecture: str) -> ModelSpec:
    t = config.get("text_config", config)
    if t.get("num_experts") or t.get("moe_intermediate_size"):
        raise UnsupportedModel("Qwen3.5 mixture-of-experts models are not supported (the dense models have no experts)")
    if t.get("attention_bias") or t.get("mlp_only_layers"):
        raise UnsupportedModel("attention biases / mlp_only_layers are not supported")
    if t.get("hidden_act", "silu") != "silu":
        raise UnsupportedModel(f"activation '{t.get('hidden_act')}' is not supported (only silu)")
    heads, layers = t["num_attention_heads"], t["num_hidden_layers"]
    layer_types = list(t["layer_types"])
    if len(layer_types) != layers or not set(layer_types) <= {"full_attention", "linear_attention"}:
        raise UnsupportedModel(f"unsupported layer_types: {sorted(set(layer_types))}")
    generation_config = generation_config or {}
    eos = generation_config.get("eos_token_id", config.get("eos_token_id", t.get("eos_token_id")))
    eos = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
    vocab = t["vocab_size"]
    linear = {"num_k_heads": t["linear_num_key_heads"], "num_v_heads": t["linear_num_value_heads"], "k_dim": t["linear_key_head_dim"], "v_dim": t["linear_value_head_dim"],
              "conv_kernel": t["linear_conv_kernel_dim"]} if "linear_attention" in layer_types else {}
    if linear and (linear["k_dim"] != 128 or linear["v_dim"] != 128 or linear["num_v_heads"] % linear["num_k_heads"] or linear["conv_kernel"] != 4):
        raise UnsupportedModel(f"unsupported Gated DeltaNet dims {linear} (head dims 128, kernel 4)")
    spec = ModelSpec(
        architecture=architecture, family=family["name"], hidden=t["hidden_size"], num_layers=layers, num_heads=heads, num_kv_heads=t["num_key_value_heads"],
        head_dim=t.get("head_dim") or t["hidden_size"] // heads, ffn=t["intermediate_size"], vocab=vocab, vocab_padded=_round_up(vocab, VOCAB_ALIGNMENT),
        eps=float(t["rms_norm_eps"]), qk_norm=True, tie_embeddings=bool(t.get("tie_word_embeddings", config.get("tie_word_embeddings", True))), activation="silu",
        max_position_embeddings=t["max_position_embeddings"], layer_types=layer_types, sliding_window=None, rope=_resolve_rope(t, layer_types), eos_token_ids=eos,
        bos_token_id=t.get("bos_token_id"), pad_token_id=t.get("pad_token_id"), tensors=family["tensors"], builder=family["builder"],
        chat_template_kwargs=dict(family.get("chat_template_kwargs", {}).get(architecture, {})), linear=linear, attn_output_gate=bool(t.get("attn_output_gate", False)),
    )
    spec.decoder_dims("full_attention").check()
    if spec.hidden % 128:
        raise UnsupportedModel("the hidden size must be a multiple of the quantization group size")
    return spec


def spec_from_config(config: dict, generation_config: Optional[dict] = None) -> ModelSpec:
    architecture = (config.get("architectures") or [""])[0]
    family = family_for(architecture)
    if family.get("builder") == "gemma4":
        return _gemma4_spec(config, generation_config, family, architecture)
    if family.get("builder") == "qwen35":
        return _qwen35_spec(config, generation_config, family, architecture)
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

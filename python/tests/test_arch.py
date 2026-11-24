import json
from pathlib import Path

import pytest

from m2w.frontends.hf_llm.arch import ModelSpec, UnsupportedModel, spec_from_config

FIXTURES = Path(__file__).parent / "fixtures"
load = lambda name: json.loads((FIXTURES / name).read_text())  # noqa: E731


def test_qwen3_spec():
    spec = spec_from_config(load("qwen3_config.json"), load("qwen3_generation_config.json"))
    assert (spec.hidden, spec.num_layers, spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.ffn) == (2560, 36, 32, 8, 128, 9728)
    assert spec.vocab == spec.vocab_padded == 151936 and spec.tie_embeddings and spec.qk_norm
    assert set(spec.layer_types) == {"full_attention"} and spec.sliding_window is None
    assert spec.rope == {"full_attention": {"type": "default", "theta": 1000000.0}}
    assert spec.eos_token_ids == [151645, 151643]


def test_aion_spec_has_hybrid_layers_and_two_rope_tables():
    spec = spec_from_config(load("aion_config.json"), load("aion_generation_config.json"))
    assert (spec.hidden, spec.num_layers, spec.num_heads, spec.num_kv_heads, spec.head_dim, spec.ffn) == (2048, 28, 24, 8, 128, 8192)
    assert spec.vocab == 200029 and spec.vocab_padded == 200064 and spec.tie_embeddings
    assert [i for i, t in enumerate(spec.layer_types) if t == "full_attention"] == [2, 3, 4, 9, 13, 15, 16]
    assert spec.sliding_window == 2048
    assert spec.rope["sliding_attention"] == {"type": "default", "theta": 32000.0}
    assert spec.rope["full_attention"] == {"type": "yarn", "theta": 32000.0, "factor": 16.0, "original_max_position_embeddings": 2048, "beta_fast": 32.0, "beta_slow": 1.0}
    assert spec.eos_token_ids == [200020, 199999]
    dims = spec.decoder_dims()
    assert dims.vocab == 200064 and dims.attention_width == 3072 and dims.kv_width == 1024
    assert spec.layer_tensor_names(3)["q_norm"] == "model.layers.3.self_attn.q_norm.weight"


def test_spec_roundtrips_through_json():
    spec = spec_from_config(load("aion_config.json"), load("aion_generation_config.json"))
    assert ModelSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_unsupported_models_fail_with_a_clear_message():
    config = load("qwen3_config.json")
    with pytest.raises(UnsupportedModel, match="not supported yet"):
        spec_from_config({**config, "architectures": ["MysteryForCausalLM"]})
    with pytest.raises(UnsupportedModel, match="bias"):
        spec_from_config({**config, "attention_bias": True})
    with pytest.raises(UnsupportedModel, match="activation"):
        spec_from_config({**config, "hidden_act": "gelu"})
    with pytest.raises(UnsupportedModel, match="rope type"):
        spec_from_config({**config, "rope_parameters": {"rope_type": "longrope", "rope_theta": 1e4}})


def test_legacy_rope_spelling_is_accepted():
    config = {**load("qwen3_config.json")}
    del config["rope_parameters"]
    config.update(rope_theta=500000.0, rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 8192})
    spec = spec_from_config(config)
    assert spec.rope["full_attention"]["type"] == "yarn" and spec.rope["full_attention"]["theta"] == 500000.0 and spec.rope["full_attention"]["factor"] == 4.0

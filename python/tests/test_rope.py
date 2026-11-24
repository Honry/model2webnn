import json
from pathlib import Path

import numpy as np
import pytest

from m2w.frontends.hf_llm import rope

FIXTURES = Path(__file__).parent / "fixtures"


def test_default_rope_tables_use_the_rotate_half_layout():
    inv = rope.default_inv_freq(10000.0, 8)
    cos, sin = rope.cos_sin([0, 3], inv)
    assert cos.shape == (2, 8)
    np.testing.assert_allclose(cos[:, :4], cos[:, 4:])       # both halves hold the same angles
    np.testing.assert_allclose(cos[0], 1) and np.testing.assert_allclose(sin[0], 0)
    np.testing.assert_allclose(sin[1, 0], np.sin(3.0), rtol=1e-6)


def test_yarn_attention_factor_is_applied_to_cos_and_sin():
    inv, att = rope.yarn_inv_freq(32000, 128, 16.0, 2048)
    assert att == pytest.approx(0.1 * np.log(16) + 1)         # ~1.2773
    cos, sin = rope.cos_sin([0], inv, att)
    np.testing.assert_allclose(cos[0], att, rtol=1e-6)         # angle 0: cos = 1 * factor, sin = 0


def test_yarn_matches_transformers():
    torch = pytest.importorskip("torch")   # noqa: F841
    transformers = pytest.importorskip("transformers")
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    c = json.loads((FIXTURES / "aion_config.json").read_text())
    cfg = transformers.PretrainedConfig(hidden_size=c["hidden_size"], num_attention_heads=c["num_attention_heads"], head_dim=c["head_dim"],
                                        max_position_embeddings=c["max_position_embeddings"], rope_parameters=c["rope_parameters"], layer_types=c["layer_types"])
    reference, reference_factor = ROPE_INIT_FUNCTIONS["yarn"](cfg, layer_type="full_attention")
    mine, factor = rope.inv_freq_for({"type": "yarn", "theta": 32000, "factor": 16.0, "original_max_position_embeddings": 2048, "beta_fast": 32.0, "beta_slow": 1.0}, 128)
    np.testing.assert_allclose(mine, reference.numpy(), rtol=1e-6, atol=1e-12)
    assert factor == pytest.approx(reference_factor)

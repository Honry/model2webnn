"""Qwen3.5 (Gated DeltaNet + gated full attention): the compiled graphs, run by the numpy interpreter following the runtime protocol, against the transformers model with the SAME (dequantized)
package weights: logits of a prefill (multi-chunk, left-padded last chunk) and of the decode steps after it."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers", minversion="5.3")
sys.path.insert(0, str(Path(__file__).parent))
import tiny_qwen35 as T  # noqa: E402

from m2w.compile import compile_model  # noqa: E402
from m2w.frontends.hf_llm import arch  # noqa: E402
from m2w.frontends.hf_llm.pack_qwen35 import rope_permutation  # noqa: E402
from m2w.frontends.hf_llm.rope import partial_inv_freq  # noqa: E402
from m2w.reference import protocol  # noqa: E402
from m2w.reference.qwen35_runner import Qwen35PackageReader, Qwen35Runner  # noqa: E402


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    root = tmp_path_factory.mktemp("q35")
    T.write_tiny_qwen35(root / "hf")
    out = compile_model(str(root / "hf"), root / "pkg", max_lengths=(128,), chunks=(16, 32), staging_rows=(8,), compute_types=("float32",), progress=lambda m: None)
    spec = arch.ModelSpec.from_dict(json.loads((out / "model.json").read_text()))
    return out, spec


@pytest.fixture(scope="module")
def reference(package):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    out, spec = package
    state = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in Qwen35PackageReader(spec, out / "weights").hf_state_dict().items()}
    model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**T.TEXT)).eval()
    assert not model.load_state_dict(state, strict=False).missing_keys
    return model


def test_spec_and_plan(package):
    out, spec = package
    assert spec.builder == "qwen35" and spec.layer_types.count("linear_attention") == 4 and spec.linear["num_v_heads"] == 4 and spec.attn_output_gate
    assert set(spec.rope) == {"full_attention"} and spec.rope["full_attention"]["partial_rotary_factor"] == 0.25
    plan = json.loads((out / "plan.json").read_text())
    assert plan["architecture_kind"] == "qwen35" and plan["state"]["layers"] == [0, 1, 3, 4]
    assert set(plan["graphs"][plan["default_variant"]]["decode"]["128"]["blocks"]) == {"linear_attention", "full_attention"}


def test_rope_permutation_keeps_the_rotation():
    """Rotating the permuted head dims with the whole-head rotate_half and partial_inv_freq equals HF's partial rotate_half on the original dims."""
    head_dim, rot, theta = 32, 8, 1e7
    perm = rope_permutation(head_dim, rot)
    assert sorted(perm) == list(range(head_dim))
    x = np.random.default_rng(0).standard_normal(head_dim).astype(np.float32)
    pos = 5
    freqs = pos * (1.0 / theta ** (np.arange(0, rot, 2) / rot))
    cos, sin = np.cos(np.concatenate([freqs, freqs])), np.sin(np.concatenate([freqs, freqs]))
    hf = x.copy()
    hf[:rot] = x[:rot] * cos + np.concatenate([-x[rot // 2:rot], x[:rot // 2]]) * sin
    inv = partial_inv_freq(theta, head_dim, rot / head_dim)
    angles = pos * inv
    c, s = np.cos(np.concatenate([angles, angles])), np.sin(np.concatenate([angles, angles]))
    xp = x[perm]
    rotated = xp * c + np.concatenate([-xp[head_dim // 2:], xp[:head_dim // 2]]) * s
    assert np.allclose(rotated, hf[perm], atol=1e-5)


def test_pad_controls():
    mask, perm = protocol.linear_prefill_controls(6, 4)
    assert mask.reshape(-1).tolist() == [0, 0, 1, 1, 1, 1] and perm[0].argmax(1).tolist() == [3, 4, 0, 1, 2, 5, 6, 7, 8]
    assert protocol.linear_prefill_controls(6, 6)[1][0].argmax(1).tolist() == list(range(9))


@pytest.mark.parametrize("n", [5, 16, 21, 37])   # 5, 21: a padded single chunk; 37: two chunks with a padded last one; 16: exactly full
def test_logits_match_transformers(package, reference, n):
    out, spec = package
    runner = Qwen35Runner.from_package(out, spec, chunk_sizes=[16, 32])
    rng = np.random.default_rng(n)
    tokens = [int(t) for t in rng.integers(2, 300, n)]
    logits = runner.prefill(tokens)
    with torch.no_grad():
        want = reference(torch.tensor([tokens])).logits[0, -1].numpy()
    assert np.abs(logits - want).max() < 0.05 and int(logits.argmax()) == int(want.argmax())
    sequence = list(tokens)
    for _ in range(10):   # decode steps: the staging buffer fills and flushes (8 rows), the states carry over
        token = int(want.argmax())
        sequence.append(token)
        logits = runner.step(token)
        with torch.no_grad():
            want = reference(torch.tensor([sequence])).logits[0, -1].numpy()
        assert np.abs(logits - want).max() < 0.1 and int(logits.argmax()) == int(want.argmax())

"""Qwen3.5 with an image in the prompt: the Python reference runner (image tower graphs + embedding injection + interleaved 3-position MRoPE + decode with rope_delta, all on the compiled graphs with the
numpy interpreter) against transformers' Qwen3_5ForConditionalGeneration fed with the SAME dequantized package weights: prefill logits and decode steps."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers", minversion="5.3")
sys.path.insert(0, str(Path(__file__).parent))
import tiny_qwen35 as T  # noqa: E402
from test_qwen35_vision import PATCH, pixels  # noqa: E402

from m2w.compile import compile_model  # noqa: E402
from m2w.frontends.hf_llm import arch, vision as V  # noqa: E402
from m2w.reference import protocol  # noqa: E402
from m2w.reference.qwen35_runner import Qwen35PackageReader, Qwen35Runner  # noqa: E402

START, END, PAD = T.TOKENS["vision_start_token_id"], T.TOKENS["vision_end_token_id"], T.TOKENS["image_token_id"]


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    root = tmp_path_factory.mktemp("q35img")
    T.write_tiny_qwen35(root / "hf")
    out = compile_model(str(root / "hf"), root / "pkg", max_lengths=(128,), chunks=(16, 32), staging_rows=(8,), vision_grids=T.GRIDS, progress=lambda m: None)
    return out, arch.ModelSpec.from_dict(json.loads((out / "model.json").read_text()))


@pytest.fixture(scope="module")
def reference(package):
    from transformers import Qwen3_5ForConditionalGeneration

    out, spec = package
    reader = Qwen35PackageReader(spec, out / "weights")
    state = {**reader.hf_state_dict(prefix="model.language_model."), **reader.vision_hf_state_dict()}
    model = Qwen3_5ForConditionalGeneration(T.vl_config()).eval()
    assert not model.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in state.items()}, strict=False).missing_keys
    return model


def hf_logits(model, tokens, images):
    ids = torch.tensor([tokens])
    with torch.no_grad():
        out = model(input_ids=ids, pixel_values=torch.from_numpy(np.concatenate([p for p, _ in images]).astype(np.float32)),
                    image_grid_thw=torch.tensor([[1, *g] for _, g in images]), mm_token_type_ids=(ids == PAD).int())
    return out.logits[0, -1].numpy()


def image(grid, seed):
    return V.patchify(pixels(grid, seed), grid[0], grid[1], PATCH, 2, 2).astype(np.float16), grid


def test_mrope_positions_equal_get_rope_index(package, reference):
    """the (t, h, w) positions and rope_delta of a prompt with two images against transformers' get_rope_index; the tables against the text model's rotary embedding (permutation undone)"""
    out, spec = package
    images = [image(T.GRIDS[1], 0), image(T.GRIDS[2], 1)]
    short = [4, 5, START, PAD, END, 6, START, PAD, END, 7, 8]
    tokens = protocol.expand_image_tokens(spec, short, [g for _, g in images])
    assert len(tokens) == len(short) - 2 + 6 + 6
    mine, delta = protocol.mrope_positions(spec, tokens, [g for _, g in images])
    ids = torch.tensor([tokens])
    want, want_delta = reference.model.get_rope_index(ids, mm_token_type_ids=(ids == PAD).int(), image_grid_thw=torch.tensor([[1, *g] for _, g in images]))
    np.testing.assert_array_equal(mine, want[:, 0].numpy())
    assert delta == int(want_delta[0, 0])
    # the tables against the text model's own rotary embedding. theta 1e7 makes every frequency but the first almost 0 in a tiny prompt: use theta 10 so that all three axes matter
    import dataclasses

    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    from m2w.frontends.hf_llm.pack_qwen35 import rope_permutation

    text = {**T.TEXT, "rope_parameters": {**T.TEXT["rope_parameters"], "rope_theta": 10}}
    cos_hf, sin_hf = Qwen3_5TextRotaryEmbedding(Qwen3_5TextConfig(**text))(torch.zeros(1, len(tokens), 1), want)
    tens = dataclasses.replace(spec, rope={"full_attention": {**spec.rope["full_attention"], "theta": 10.0}})
    cos, sin = protocol.rope_tables_mrope(tens, mine, len(tokens), 0)["full_attention"]
    head_dim, rot = spec.head_dim, int(spec.head_dim * 0.25)
    assert cos.shape == (1, 1, len(tokens), head_dim)
    for new, old in enumerate(rope_permutation(head_dim, rot)):   # new head-dim index -> original index
        if old < rot:   # a rotating dim: the same angle as the original layout
            np.testing.assert_allclose(cos[0, 0, :, new].astype(np.float32), cos_hf[0, :, old].numpy(), atol=1e-3)
            np.testing.assert_allclose(sin[0, 0, :, new].astype(np.float32), sin_hf[0, :, old].numpy(), atol=1e-3)
        else:           # a dim that does not rotate: cos 1, sin 0
            assert (cos[0, 0, :, new] == 1).all() and (sin[0, 0, :, new] == 0).all()
    assert np.abs(cos[0, 0, :, 0].astype(np.float32) - 1).max() > 0.1 and len({tuple(np.round(cos[0, 0, k].astype(np.float32), 3)) for k in range(len(tokens))}) > 8   # the tables are not trivial


def test_expand_image_tokens():
    spec = arch.ModelSpec(architecture="x", family="qwen3.5", hidden=1, num_layers=2, num_heads=1, num_kv_heads=1, head_dim=1, ffn=1, vocab=1, vocab_padded=1, eps=1e-6, qk_norm=True,
                          tie_embeddings=True, activation="silu", max_position_embeddings=1, layer_types=["full_attention"] * 2, sliding_window=None, rope={}, eos_token_ids=[], bos_token_id=None,
                          pad_token_id=None, vision={"image_token_id": 9, "merge_size": 2})
    assert protocol.expand_image_tokens(spec, [1, 8, 9, 7, 2], [(4, 6)]) == [1, 8] + [9] * 6 + [7, 2]
    assert protocol.expand_image_tokens(spec, [1, 9, 9, 9, 9, 2], [(4, 4)]) == [1] + [9] * 4 + [2]   # already expanded
    with pytest.raises(ValueError):
        protocol.expand_image_tokens(spec, [1, 9, 9, 2], [(4, 4)])
    with pytest.raises(ValueError):
        protocol.expand_image_tokens(spec, [1, 9, 2, 9], [(4, 4)])


@pytest.mark.parametrize("sizes,grids,text_before,text_after", [
    ([16, 32], [T.GRIDS[0]], 3, 4),                  # one image, one padded chunk
    ([16], [T.GRIDS[1]], 12, 9),                     # 12 + start + 6 image tokens + end + 9: two chunks of 16, the image straddles the chunk boundary
    ([16, 32], [T.GRIDS[1], T.GRIDS[2]], 5, 3),      # two images (different grids) in one prompt
])
def test_prefill_and_decode_match_transformers(package, reference, sizes, grids, text_before, text_after):
    out, spec = package
    runner = Qwen35Runner.from_package(out, spec, chunk_sizes=sizes)
    rng = np.random.default_rng(len(grids) * 100 + text_before)
    images = [image(g, i) for i, g in enumerate(grids)]
    tokens = [int(t) for t in rng.integers(2, 280, text_before)]
    for k in range(len(grids)):
        tokens += [START, PAD, END] + [int(t) for t in rng.integers(2, 280, 2 + k)]
    tokens += [int(t) for t in rng.integers(2, 280, text_after)]
    logits = runner.prefill(tokens, images=images)
    assert len(runner.prompt) == len(tokens) - len(grids) + sum(protocol.image_token_count(spec, g) for g in grids)
    sequence = list(runner.prompt)
    want = hf_logits(reference, sequence, images)
    assert np.abs(logits - want).max() < 0.1 and int(logits.argmax()) == int(want.argmax()), np.abs(logits - want).max()
    assert runner.rope_delta != 0
    for _ in range(10):   # decode: the staging buffer flushes (8 rows), the positions are index + rope_delta
        token = int(want.argmax())
        sequence.append(token)
        logits = runner.step(token)
        want = hf_logits(reference, sequence, images)
        assert np.abs(logits - want).max() < 0.15 and int(logits.argmax()) == int(want.argmax()), np.abs(logits - want).max()


def test_text_only_prompt_is_unchanged_and_images_need_a_tower(package, reference):
    """a prompt without images gives the same logits as before (rope_delta 0, plain tables); an image without a tower is an error"""
    out, spec = package
    runner = Qwen35Runner.from_package(out, spec, chunk_sizes=[16])
    tokens = [5, 9, 100, 7, 33, 2, 8]
    logits = runner.prefill(tokens)
    assert runner.rope_delta == 0
    with torch.no_grad():
        want = reference(input_ids=torch.tensor([tokens])).logits[0, -1].numpy()
    assert np.abs(logits - want).max() < 0.05
    runner.vision = None
    with pytest.raises(ValueError, match="no image tower"):
        runner.prefill([1, PAD, 2], images=[image(T.GRIDS[0], 0)])

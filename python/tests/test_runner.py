"""End to end on the CPU: HF weights -> package -> compiled graphs run by the interpreter (the runtime protocol) vs an independent float32 reference."""
import numpy as np
import pytest

from m2w.frontends.hf_llm import pack
from m2w.frontends.hf_llm.arch import spec_from_config
from m2w.frontends.hf_llm.safetensors_io import TensorStore
from m2w.reference.graph_runner import PackageRunner
from m2w.reference.package_reader import PackageReader
from m2w.reference.transformer_ref import ReferenceModel

from tiny_model import CONFIG, write_tiny_model


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    write_tiny_model(root / "hf", dtype="F32")
    spec = spec_from_config(CONFIG)
    store = TensorStore.open(root / "hf")
    pack.pack_model(spec, store, root / "pkg", workers=2)
    return spec, store, root / "pkg"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def make_runner(tiny, **kw):
    spec, _, pkg = tiny
    return PackageRunner(spec, pkg, max_length=kw.pop("max_length", 64), chunk_size=kw.pop("chunk_size", 8), staging_rows=kw.pop("staging_rows", 4), **kw)


def test_runner_matches_the_reference_on_the_same_quantized_weights(tiny):
    """Graphs + protocol vs the independent reference fed with the DEQUANTIZED package weights: the only differences left are f16 roundings."""
    spec, store, pkg = tiny
    quantized = ReferenceModel(spec, PackageReader(spec, pkg).read)
    tokens = list(np.random.default_rng(0).integers(3, spec.vocab, 21))      # 3 chunks of 8: the last one is left-padded; sliding window is 16
    expected = quantized.logits(tokens)[-1]
    got = make_runner(tiny).prefill(tokens)
    assert got.shape == (spec.vocab,)
    assert cosine(got, expected) > 0.9999, cosine(got, expected)
    assert np.abs(got - expected).max() < 0.002 * np.abs(expected).max()
    # int4 quantization noise of this RANDOM tiny model against the original float32 weights (real models are far more benign): a sanity floor only
    original = ReferenceModel(spec, store.read).logits(tokens)[-1]
    assert cosine(expected, original) > 0.93, cosine(expected, original)


def test_decode_with_staging_and_flush_equals_prefill(tiny):
    """Prefill of n tokens must give the same logits as prefill of fewer tokens followed by single-token staged decode steps,
    across several flushes of the staging buffer (4 rows) and beyond the sliding window (16)."""
    spec = tiny[0]
    tokens = [int(t) for t in np.random.default_rng(1).integers(3, spec.vocab, 44)]
    expected = make_runner(tiny).prefill(tokens)
    runner = make_runner(tiny)
    runner.prefill(tokens[:30])
    for t in tokens[30:-1]:
        runner.step(t)
    got = runner.step(tokens[-1])
    assert runner.base > 30                       # at least one flush happened
    assert cosine(got, expected) > 0.999, cosine(got, expected)
    assert int(np.argmax(got)) == int(np.argmax(expected))


def test_sliding_window_layers_ignore_old_tokens(tiny):
    """Layers 0 and 2 use a 16-token window, layer 1 sees everything. On a prompt much longer than the window the runner must still match the reference
    (mask logic of the chunks), and changing the FIRST token must still change the output through the full-attention layer."""
    spec, _, pkg = tiny
    quantized = ReferenceModel(spec, PackageReader(spec, pkg).read)
    tokens = [int(t) for t in np.random.default_rng(2).integers(3, spec.vocab, 40)]
    got = make_runner(tiny, chunk_size=16).prefill(tokens)
    assert cosine(got, quantized.logits(tokens)[-1]) > 0.9999
    changed = list(tokens)
    changed[0] = (tokens[0] + 7) % spec.vocab
    assert cosine(make_runner(tiny, chunk_size=16).prefill(changed), got) < 0.9999     # the full-attention layer still sees token 0


def test_generate_is_greedy_and_stops_at_eos(tiny):
    spec = tiny[0]
    tokens = [5, 9, 11, 4, 8]
    out, logits = make_runner(tiny).generate(tokens, max_new_tokens=6)
    assert len(out) == 6 and len(logits) == 6
    assert out[0] == int(np.argmax(logits[0]))
    stopped, _ = make_runner(tiny).generate(tokens, max_new_tokens=6, eos=(out[0],))
    assert stopped == [out[0]]                                   # the eos token itself is returned, nothing after it
    never, _ = make_runner(tiny).generate(tokens, max_new_tokens=3, eos=(spec.vocab - 1,))
    assert never == out[:3]


def test_runner_with_the_uint8_asymmetric_lm_head(tmp_path):
    from m2w.frontends.hf_llm.builders import decoder as D

    write_tiny_model(tmp_path / "hf", dtype="F32")
    spec = spec_from_config(CONFIG)
    store = TensorStore.open(tmp_path / "hf")
    opt = D.GraphOptions(kv_style="staged", staging_rows=4, lm_head_format="uint8-asym")
    pack.pack_model(spec, store, tmp_path / "pkg", opt, workers=2)
    tokens = [int(t) for t in np.random.default_rng(9).integers(3, spec.vocab, 21)]
    got = PackageRunner(spec, tmp_path / "pkg", max_length=64, chunk_size=8, opt=opt).prefill(tokens)
    expected = ReferenceModel(spec, PackageReader(spec, tmp_path / "pkg").read).logits(tokens)[-1]
    assert cosine(got, expected) > 0.9999


# ------------------------------------------------------------------------------------------------------------- option variants (KV protocol, chunk plans, groups)
@pytest.mark.parametrize("style", ["staged", "onehot", "tensorScatter"])
@pytest.mark.parametrize("chunks", [(8,), (4, 8)])
def test_every_kv_protocol_matches_the_reference_in_prefill_and_decode(tiny, style, chunks):
    """Prefill + single-token decode steps with each KV protocol and a single / several chunk sizes: logits = the independent reference on the dequantized weights."""
    spec, _, pkg = tiny
    quantized = ReferenceModel(spec, PackageReader(spec, pkg).read)
    tokens = [int(t) for t in np.random.default_rng(3).integers(3, spec.vocab, 27)]   # beyond the 16-token sliding window; padded last chunk / front padding
    runner = PackageRunner(spec, pkg, max_length=64, chunk_sizes=chunks, staging_rows=4, kv_style=style, prefill_fixed_tokens=2)
    got = runner.prefill(tokens[:21])
    assert cosine(got, quantized.logits(tokens[:21])[-1]) > 0.9999, (style, chunks, runner.plan(21))
    for i in range(21, 27):   # decode steps; staged: crosses a flush of the 4-row staging buffer
        got = runner.step(tokens[i])
        assert cosine(got, quantized.logits(tokens[: i + 1])[-1]) > 0.9999, (style, i)


def test_tensor_scatter_pads_the_front_and_stops_when_the_cache_is_full(tiny):
    spec, _, pkg = tiny
    runner = PackageRunner(spec, pkg, max_length=24, chunk_sizes=(8,), kv_style="tensorScatter")   # 23 cache rows
    tokens = [int(t) for t in np.random.default_rng(4).integers(3, spec.vocab, 13)]               # 2 chunks = 16 rows, 3 pad rows in front
    runner.prefill(tokens)
    assert runner.pad == 3 and not runner.cache_full
    out, _ = runner.generate(tokens, 50)
    assert len(out) == 23 - 3 - 13 + 1       # rows left for decode steps + the token the last prefill logits gave


def test_float16_compute_agrees_with_float32(tiny):
    spec, _, pkg = tiny
    tokens = [int(t) for t in np.random.default_rng(5).integers(3, spec.vocab, 19)]
    expected = PackageRunner(spec, pkg, max_length=64, chunk_size=8, staging_rows=4).prefill(tokens)
    f16 = PackageRunner(spec, pkg, max_length=64, chunk_size=8, staging_rows=4, compute_type="float16").prefill(tokens)
    assert cosine(f16, expected) > 0.999


@pytest.mark.parametrize("style", ["staged", "onehot", "tensorScatter"])
def test_block_groups_equal_single_blocks_for_a_single_layer_type_model(tiny, style):
    """Blocks of a group share one mask / RoPE table, so the graphs are only valid for a model with ONE layer type (here: the tiny model with full attention everywhere)."""
    import dataclasses

    spec, _, pkg = tiny
    spec = dataclasses.replace(spec, layer_types=["full_attention"] * spec.num_layers)
    tokens = [int(t) for t in np.random.default_rng(6).integers(3, spec.vocab, 19)]
    make = lambda **kw: PackageRunner(spec, pkg, max_length=64, chunk_size=8, staging_rows=4, kv_style=style, **kw)   # noqa: E731
    single, grouped = make(), make(decode_group=2, prefill_group=2)
    assert cosine(grouped.prefill(tokens), single.prefill(tokens)) > 0.99999
    for t in (5, 9, 4, 7, 3, 11):
        assert cosine(grouped.step(t), single.step(t)) > 0.99999


def test_plan_prefill_picks_the_cheapest_chunk_sequence():
    from m2w.reference.protocol import plan_prefill

    assert plan_prefill([64], 20) == [64] and plan_prefill([64], 130) == [64, 64, 64]
    assert plan_prefill([64, 256], 20, 200) == [64]            # one small dispatch is cheapest
    assert plan_prefill([64, 256], 300, 200) == [256, 64]      # bigger chunks first, the padded one is the small one
    assert plan_prefill([64, 256], 200, 200) == [256]          # 1 x (200+256) < 4 x (200+64)

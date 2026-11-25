"""Gemma 4 on a tiny random model: spec, packing, the compiled graphs run by the interpreter (all four attention kinds, per-layer embeddings, KV sharing, softcap) vs the numpy reference."""
import json

import numpy as np
import pytest

from m2w import compile as C
from m2w.frontends.hf_llm import arch, pack_gemma4
from m2w.frontends.hf_llm.builders import decoder as D
from m2w.frontends.hf_llm.builders import gemma4 as G
from m2w.frontends.hf_llm.safetensors_io import TensorStore
from m2w.reference.gemma4_package import Gemma4PackageReader
from m2w.reference.gemma4_ref import Gemma4Reference
from m2w.reference.gemma4_runner import Gemma4Runner
from m2w.verify import verify_package

from tiny_gemma4 import CONFIG, TEXT, write_tiny_gemma4


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny_gemma4")
    write_tiny_gemma4(root / "hf", dtype="F32")
    spec = arch.spec_from_config(CONFIG)
    manifest = pack_gemma4.pack_model(spec, TensorStore.open(root / "hf"), root / "pkg" / "weights", workers=2)
    reader = Gemma4PackageReader(spec, root / "pkg" / "weights")
    return spec, manifest, root, Gemma4Reference(TEXT, reader.read)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_spec_of_the_real_e4b_config():
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    spec = arch.spec_from_config(json.loads((fixtures / "gemma4_e4b_config.json").read_text()), json.loads((fixtures / "gemma4_e4b_generation_config.json").read_text()))
    assert spec.builder == "gemma4" and spec.num_layers == 42 and spec.ple_dim == 256 and spec.logit_softcap == 30.0 and spec.eos_token_ids == [1, 106, 50]
    assert spec.layer_dims == {"full_attention": {"head_dim": 512, "num_heads": 8, "num_kv_heads": 2}} and spec.decoder_dims("sliding_attention").head_dim == 256
    assert spec.kv_producer[:24] == [-1] * 24 and set(spec.kv_producer[24:]) == {22, 23}
    assert [spec.layer_kind(i) for i in (22, 23, 24, 29)] == ["sliding_attention_own", "full_attention_own", "sliding_attention_shared", "full_attention_shared"]
    assert spec.rope["full_attention"] == {"type": "proportional", "theta": 1000000.0, "partial_rotary_factor": 0.25} and spec.rope["sliding_attention"]["theta"] == 10000.0


def test_moe_and_unsupported_options_are_refused():
    for key, value in (("enable_moe_block", True), ("attention_k_eq_v", True), ("hidden_activation", "silu")):
        config = json.loads(json.dumps(CONFIG))
        config["text_config"][key] = value
        with pytest.raises(arch.UnsupportedModel):
            arch.spec_from_config(config)


def test_the_shards_have_what_the_graphs_bind(tiny):
    spec, manifest, _, _ = tiny
    layers = {f["layer"]: f for f in manifest["files"] if f["role"] == "layer"}
    for i in range(spec.num_layers):
        kind = spec.layer_kind(i)
        names = {t["name"] for t in layers[i]["tensors"]}
        assert names == {n for n, _, _ in G.attention_weight_specs(spec, kind) + G.feed_forward_weight_specs(spec)}, i
        assert ("key_norm_gamma" in names) == (not G.kind_shared(kind))   # a layer that shares K / V has no k / v projection
    assert {f["role"] for f in manifest["files"]} == {"layer", "final_norm", "head_extra", "embedding", "ple_table"}
    ple = next(f for f in manifest["files"] if f["role"] == "ple_table")
    assert [t["dtype"] for t in ple["tensors"]] == ["int8", "float16"] and ple["tensors"][0]["shape"] == [300, 6 * 128]


def test_graph_interfaces(tiny):
    spec, _, _, _ = tiny
    opt = D.GraphOptions(staging_rows=4)
    P = 63
    for kind in G.KINDS:
        d = spec.decoder_dims(G.kind_type(kind))
        decode = G.build_block_graph(spec, opt, kind, seq=1, past_len=P)
        assert decode.inputs["past_key"].shape == (1, 2, P, d.head_dim) and decode.inputs["rope_cos"].shape == (1, 1, 1, d.head_dim)
        assert ("updated_key" in decode.outputs) == (not G.kind_shared(kind))
        prefill = G.build_block_graph(spec, opt, kind, seq=8, past_len=P)
        assert prefill.inputs["attention_mask"].shape == (1, 1, 8, P if G.kind_shared(kind) else P + 8)   # a shared layer reads the updated cache: mask columns = cache rows
        assert ("staging_key" in decode.inputs) and ("staging_key" not in prefill.inputs)
    head = G.build_head_graph(spec, opt, seq=8, past_len=P)
    assert [n for n in head.outputs if n.startswith("ple_")] == [f"ple_{i}" for i in range(6)] and head.inputs["ple_token"].shape == (1, 8, 6 * 128)
    lm = G.build_lm_head_graph(spec, opt)
    assert lm.values[lm.outputs["logits"]].shape == (1, 1, spec.vocab_padded)


@pytest.mark.parametrize("compute_type", ["float32", "float16"])
@pytest.mark.parametrize("chunks", [(8,), (4, 8)])
def test_compiled_graphs_match_the_numpy_reference_in_prefill_and_decode(tiny, chunks, compute_type):
    """Prefill (one / several chunk sizes, left-padded last chunk, sliding window crossed) and decode steps across flushes of the staging buffer: logits = the reference on the dequantized weights."""
    spec, _, root, ref = tiny
    tokens = [int(t) for t in np.random.default_rng(1).integers(3, spec.vocab, 27)]
    runner = Gemma4Runner(spec, root / "pkg" / "weights", max_length=64, chunk_sizes=chunks, opt=D.GraphOptions(staging_rows=4, compute_type=compute_type), prefill_fixed_tokens=2)
    tolerance = 0.9999 if compute_type == "float32" else 0.999
    got = runner.prefill(tokens[:21])
    assert cosine(got, ref.logits(tokens[:21])[0][: spec.vocab]) > tolerance, (chunks, runner.plan(21))
    for i in range(21, 27):   # six steps: the 4-row staging buffer is flushed once; layers 4 and 5 read the caches of layers 2 and 3
        got = runner.step(tokens[i])
        assert cosine(got, ref.logits(tokens[: i + 1])[0][: spec.vocab]) > tolerance, i


def test_compile_a_gemma4_model_into_a_package_and_verify_it(tmp_path, capsys):
    write_tiny_gemma4(tmp_path / "hf", dtype="BF16")
    out = C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(64,), chunks=(8, 16), staging_rows=(4,), compute_types=("float32", "float16"), workers=2, progress=lambda m: None)
    index = json.loads((out / "m2w.json").read_text())
    plan = json.loads((out / "plan.json").read_text())
    assert plan["architecture_kind"] == "gemma4" and [v["id"] for v in plan["variants"]] == ["staged-r4-float32", "staged-r4-float16"]
    assert "tanh" in index["requires"]["webnn"]["ops"]
    table = plan["graphs"]["staged-r4-float32"]
    assert set(table["decode"]["64"]["blocks"]) == {"sliding_attention_own", "full_attention_own", "sliding_attention_shared", "full_attention_shared"}
    assert set(table["decode"]["64"]["flush"]) == {"sliding_attention", "full_attention"} and set(table["prefill"]["64"]) == {"8", "16"}
    assert plan["chain"]["kinds"][4:] == ["full_attention_shared", "sliding_attention_shared"]
    with pytest.raises(ValueError, match="staged"):
        C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg2", kv_styles=("onehot",), workers=2, progress=lambda m: None)
    ids = [int(t) for t in np.random.default_rng(0).integers(3, 300, 19)]
    assert verify_package(out, str(tmp_path / "hf"), prompt="", steps=4, token_ids=ids, variants=["staged-r4-float32"], chunks=[8, 16], skip_float_reference=True) == 0
    assert "RESULT: pipeline OK" in capsys.readouterr().out

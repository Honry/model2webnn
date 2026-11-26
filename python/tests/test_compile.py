import json

import numpy as np

from m2w import compile as C
from m2w.frontends.hf_llm.arch import ModelSpec
from m2w.ir import serialize
from m2w.verify import verify_package

from tiny_model import write_tiny_model


def test_compile_a_local_model_into_a_package_and_verify_every_variant(tmp_path, capsys):
    write_tiny_model(tmp_path / "hf", dtype="BF16")
    out = C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(64,), chunks=(8, 16), kv_styles=("staged", "onehot", "tensorScatter", "scatterND"), staging_rows=(4, 8),
                          compute_types=("float32", "float16"), blocks_per_graph=(1, 2), prefill_blocks_per_graph=(1, 2), workers=2, progress=lambda m: None)
    index = json.loads((out / "m2w.json").read_text())
    assert index["format"] == "m2w-package" and index["target"] == "webnn-ov-npu" and index["chat_format"] is None   # the tiny model has no chat template
    assert "tensorScatter" in index["requires"]["webnn"]["ops"] and "scatterND" in index["requires"]["webnn"]["ops"]
    plan = json.loads((out / "plan.json").read_text())
    ids = [v["id"] for v in plan["variants"]]
    assert plan["default_variant"] == "staged-r4-float32" == ids[0]
    assert plan["block_group_controls"] == "per_block"   # the tiny model is hybrid: every block of a group gets its own rope / mask inputs
    assert ids == ["staged-r4-float32", "staged-r8-float32", "staged-r4-float16", "staged-r8-float16", "onehot-float32", "onehot-float16", "tensorScatter-float32", "tensorScatter-float16",
                   "scatterND-float32", "scatterND-float16"]
    table = plan["graphs"]["staged-r4-float32"]
    assert table["decode"]["64"]["block_groups"].keys() == {"2"} and table["prefill"]["64"]["8"]["block_groups"].keys() == {"2"} and "16" in table["prefill"]["64"]
    assert "flush" in table["decode"]["64"] and "flush" not in plan["graphs"]["onehot-float32"]["decode"]["64"]
    spec = ModelSpec.from_dict(json.loads((out / "model.json").read_text()))
    assert spec.layer_types == ["sliding_attention", "full_attention", "sliding_attention"]
    # every file of the package is listed with its checksum, and every graph loads
    listed = {f["path"] for f in index["files"]}
    assert "weights/manifest.json" in listed and "graphs/staged-r4-float32/lm_head.json" in listed and "model.json" in listed
    for path in (out / "graphs").rglob("*.json"):
        assert serialize.load(path).outputs
    # the CPU verification of the package files (graphs + protocol + weights) passes for every variant, with two chunk sizes
    ids_tokens = [int(t) for t in np.random.default_rng(0).integers(3, spec.vocab, 19)]
    assert verify_package(out, str(tmp_path / "hf"), prompt="", steps=4, token_ids=ids_tokens, variants=["all"], chunks=[8, 16], skip_float_reference=True) == 0
    assert capsys.readouterr().out.count("RESULT: pipeline OK") == 1


def test_rebuild_graphs_changes_the_variants_and_keeps_the_weights(tmp_path):
    write_tiny_model(tmp_path / "hf", dtype="BF16")
    out = C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(64,), chunks=(8,), staging_rows=(4,), workers=2, progress=lambda m: None)
    weights_before = {f["path"]: f["sha256"] for f in json.loads((out / "m2w.json").read_text())["files"] if f["path"].startswith("weights/")}
    C.rebuild_graphs(out, kv_styles=("onehot",), chunks=(8, 16), progress=lambda m: None)
    index = json.loads((out / "m2w.json").read_text())
    assert index["summary"]["variants"] == ["onehot-float32"] and index["summary"]["prefill_chunks"] == [8, 16]
    assert {f["path"]: f["sha256"] for f in index["files"] if f["path"].startswith("weights/")} == weights_before
    assert not (out / "graphs" / "staged-r4-float32").exists()


def test_committed_tiny_package_is_what_the_compiler_writes_today(tmp_path):
    """tests/golden/tiny-package is the fixture of the TS runtime tests: regenerate it with scripts/make-tiny-package.py when the package format changes."""
    from pathlib import Path

    committed = Path(__file__).resolve().parents[2] / "tests" / "golden" / "tiny-package"
    write_tiny_model(tmp_path / "hf", dtype="BF16")
    out = C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(128,), chunks=(16, 8), kv_styles=("staged", "onehot", "tensorScatter", "scatterND"), staging_rows=(8,),
                          compute_types=("float32", "float16"), blocks_per_graph=(1, 2), prefill_blocks_per_graph=(1, 2), workers=2, progress=lambda m: None)
    fresh = {f["path"]: f["sha256"] for f in json.loads((out / "m2w.json").read_text())["files"]}
    old = {f["path"]: f["sha256"] for f in json.loads((committed / "m2w.json").read_text())["files"]}
    assert fresh == old, "tests/golden/tiny-package is stale: run `python scripts/make-tiny-package.py`"


def test_committed_tiny_gemma4_package_is_what_the_compiler_writes_today(tmp_path):
    from pathlib import Path

    from tiny_gemma4 import write_tiny_gemma4

    committed = Path(__file__).resolve().parents[2] / "tests" / "golden" / "tiny-gemma4-package"
    write_tiny_gemma4(tmp_path / "hf", dtype="BF16")
    out = C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(128,), chunks=(16, 8), staging_rows=(8,), compute_types=("float32", "float16"), blocks_per_graph=(1, 2, 3, 4, 5), workers=2,
                          progress=lambda m: None)
    fresh = {f["path"]: f["sha256"] for f in json.loads((out / "m2w.json").read_text())["files"]}
    old = {f["path"]: f["sha256"] for f in json.loads((committed / "m2w.json").read_text())["files"]}
    assert fresh == old, "tests/golden/tiny-gemma4-package is stale: run `python scripts/make-tiny-package.py`"


def test_recommended_session_options_of_the_families(tmp_path):
    from pathlib import Path
    import shutil

    from m2w.frontends.hf_llm import arch

    assert arch.recommended_session("qwen3-like")["blocks_per_graph"] == 5
    assert arch.recommended_session("gemma4")["blocks_per_graph"] == 3 and arch.recommended_session("qwen3.5") == {"blocks_per_graph": 4, "prefill_blocks_per_graph": 1, "prefill_chunks": [64, 128], "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5}}
    spec = arch.ModelSpec.from_dict(json.loads((Path(__file__).resolve().parents[2] / "tests" / "golden" / "tiny-gemma4-package" / "model.json").read_text()))
    assert C.resolve_blocks_per_graph(spec, None) == (1, 3) and C.resolve_blocks_per_graph(spec, (1, 2)) == (1, 2)   # not given: single blocks + the recommended K; given: as asked
    # a package that has no groups of the recommended size recommends single blocks; set_recommended rewrites an existing package from the family file
    pkg = tmp_path / "pkg"
    shutil.copytree(Path(__file__).resolve().parents[2] / "tests" / "golden" / "tiny-gemma4-package", pkg)
    plan = json.loads((pkg / "plan.json").read_text())
    assert plan["recommended"] == {"blocks_per_graph": 3, "prefill_blocks_per_graph": 1, "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}}   # the golden package has groups of 1 .. 5
    plan["blocks_per_graph"] = [1, 2]
    (pkg / "plan.json").write_text(json.dumps(plan))
    C.set_recommended(pkg, progress=lambda m: None)
    assert json.loads((pkg / "plan.json").read_text())["recommended"] == {"blocks_per_graph": 1, "prefill_blocks_per_graph": 1, "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 64}}
    assert any(f["path"] == "plan.json" for f in json.loads((pkg / "m2w.json").read_text())["files"])   # the file index was refreshed

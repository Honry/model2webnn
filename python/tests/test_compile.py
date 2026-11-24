import json

import numpy as np

from m2w import compile as C
from m2w.frontends.hf_llm.arch import ModelSpec
from m2w.ir import serialize
from m2w.verify import verify_package

from tiny_model import write_tiny_model


def test_compile_a_local_model_into_a_package_and_verify_it(tmp_path, capsys):
    write_tiny_model(tmp_path / "hf", dtype="BF16")
    out = C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(64,), chunks=(8,), staging_rows=4, blocks_per_graph=(1, 2), workers=2, progress=lambda m: None)
    index = json.loads((out / "m2w.json").read_text())
    assert index["format"] == "m2w-package" and index["target"] == "webnn-ov-npu" and index["chat_format"] is None   # the tiny model has no chat template
    plan = json.loads((out / "plan.json").read_text())
    assert plan["graphs"]["decode"]["64"]["block_groups"].keys() == {"2"} and "8" in plan["graphs"]["prefill"]["64"]
    spec = ModelSpec.from_dict(json.loads((out / "model.json").read_text()))
    assert spec.layer_types == ["sliding_attention", "full_attention", "sliding_attention"]
    # every file of the package is listed with its checksum, and every graph loads
    listed = {f["path"] for f in index["files"]}
    assert "weights/manifest.json" in listed and "graphs/lm_head.json" in listed and "model.json" in listed
    for path in (out / "graphs").glob("*.json"):
        graph = serialize.load(path)
        assert graph.outputs
    # the CPU verification of the package (graphs + protocol + weights) passes
    ids = [int(t) for t in np.random.default_rng(0).integers(3, spec.vocab, 19)]
    assert verify_package(out, str(tmp_path / "hf"), prompt="", steps=4, token_ids=ids) == 0
    assert "RESULT: pipeline OK" in capsys.readouterr().out


def test_committed_tiny_package_is_what_the_compiler_writes_today(tmp_path):
    """tests/golden/tiny-package is the fixture of the TS runtime tests: regenerate it with scripts/make-tiny-package.py when the package format changes."""
    from pathlib import Path

    committed = Path(__file__).resolve().parents[2] / "tests" / "golden" / "tiny-package"
    write_tiny_model(tmp_path / "hf", dtype="BF16")
    out = C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(128,), chunks=(16,), staging_rows=8, workers=2, progress=lambda m: None)
    fresh = {f["path"]: f["sha256"] for f in json.loads((out / "m2w.json").read_text())["files"]}
    old = {f["path"]: f["sha256"] for f in json.loads((committed / "m2w.json").read_text())["files"]}
    assert fresh == old, "tests/golden/tiny-package is stale: run `python scripts/make-tiny-package.py`"

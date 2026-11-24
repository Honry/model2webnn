import os
from pathlib import Path

import numpy as np

from m2w.ir import serialize
from m2w.ir.interpreter import run

from golden_graphs import tiny_all_ops

GOLDEN = Path(__file__).resolve().parents[2] / "tests" / "golden" / "tiny_all_ops.json"


def test_golden_graph_is_current():
    text = serialize.dumps(tiny_all_ops(), indent=1)
    if os.environ.get("UPDATE_GOLDEN"):
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(text + "\n", encoding="utf-8", newline="\n")
    assert GOLDEN.read_text(encoding="utf-8").strip() == text.strip(), "golden graph is stale: run `UPDATE_GOLDEN=1 pytest tests/test_golden.py`"


def test_golden_graph_loads_and_runs():
    g = serialize.load(GOLDEN)
    rng = np.random.default_rng(0)
    out = run(g, {"x": rng.standard_normal((1, 4, 8)).astype(np.float32), "q": rng.integers(-8, 8, (2, 6, 4)).astype(np.int8),
                  "s": rng.uniform(0.05, 0.2, (2, 6, 1)).astype(np.float16), "cache": np.zeros((1, 2, 6, 4), np.float16),
                  "rows": np.ones((1, 2, 2, 4), np.float16), "index": np.array([1], np.int32)})
    assert out["e"].shape == (1, 6, 8) and out["updated"][0, :, 1:3].min() == 1


def test_golden_and_generated_graphs_satisfy_the_json_schema():
    import json

    import pytest

    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).resolve().parents[2] / "spec" / "graph.schema.json").read_text(encoding="utf-8"))
    from m2w.frontends.hf_llm.builders import decoder as D
    from m2w.presets import PRESETS

    dims = PRESETS["qwen3-4b"]["dims"]
    opt = D.GraphOptions()
    docs = [json.loads(GOLDEN.read_text(encoding="utf-8")),
            D.build_block_graph(dims, opt, seq=1, past_len=1151).to_dict(),
            D.build_lm_head_graph(dims, opt).to_dict(),
            D.build_tail_graph(dims, opt, seq=64).to_dict()]
    for doc in docs:
        jsonschema.validate(doc, schema)

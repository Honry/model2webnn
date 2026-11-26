"""The JSON Schemas in spec/ describe what the compiler really writes: validate the committed tiny package (and a derived chat format) against them."""
import json
from pathlib import Path

import pytest

from m2w.frontends.hf_llm import chat_format as C

jsonschema = pytest.importorskip("jsonschema")
SPEC = Path(__file__).resolve().parents[2] / "spec"
PACKAGE = Path(__file__).resolve().parents[2] / "tests" / "golden" / "tiny-package"
FIXTURES = Path(__file__).parent / "fixtures"


def schema(name):
    return json.loads((SPEC / name).read_text(encoding="utf-8"))


def test_every_schema_is_a_valid_json_schema():
    for path in SPEC.glob("*.schema.json"):
        jsonschema.Draft202012Validator.check_schema(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("name,schema_file", [("m2w.json", "package.schema.json"), ("model.json", "model.schema.json"), ("plan.json", "plan.schema.json"),
                                              ("weights/manifest.json", "weights.schema.json")])
def test_package_files_satisfy_their_schema(name, schema_file):
    jsonschema.validate(json.loads((PACKAGE / name).read_text(encoding="utf-8")), schema(schema_file))


@pytest.mark.parametrize("name,schema_file", [("m2w.json", "package.schema.json"), ("model.json", "model.schema.json"), ("plan.json", "plan.schema.json"),
                                              ("weights/manifest.json", "weights.schema.json")])
def test_the_qwen35_package_with_an_image_tower_satisfies_the_schemas(name, schema_file):
    """hybrid layer types, per-kind block graphs, and the `vision` entries of model.json / plan.json / the weights manifest"""
    package = PACKAGE.parent / "tiny-qwen35-package"
    jsonschema.validate(json.loads((package / name).read_text(encoding="utf-8")), schema(schema_file))


def test_chat_formats_satisfy_the_schema():
    for template, kwargs in (("aion_chat_template.jinja", {}), ("qwen3_chat_template.jinja", {"enable_thinking": False})):
        fmt = C.derive_chat_format((FIXTURES / template).read_text(encoding="utf-8"), kwargs)
        jsonschema.validate(fmt, schema("chat-format.schema.json"))


def test_the_package_index_lists_real_files_with_matching_checksums():
    import hashlib

    index = json.loads((PACKAGE / "m2w.json").read_text(encoding="utf-8"))
    for entry in index["files"]:
        data = (PACKAGE / entry["path"]).read_bytes()
        assert len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]

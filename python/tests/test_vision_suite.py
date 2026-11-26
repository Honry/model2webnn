"""bench/raw/vision-suite: cases.json is consistent with its generator (python/tools/make_vision_suite.py) and with the PNGs next to it"""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python" / "tools"))
import make_vision_suite as M  # noqa: E402

SUITE = ROOT / "bench" / "raw" / "vision-suite"


def test_cases_match_the_generator_and_the_images():
    cases = json.loads((SUITE / "cases.json").read_text(encoding="utf-8"))["cases"]
    assert [c["id"] for c in cases] == [c[0] for c in M.CASES] and len(cases) == 9
    for case in cases:
        with Image.open(SUITE / case["image"]) as image:
            assert image.size == (case["width"], case["height"]) and 256 <= max(image.size) <= 768
        assert case["question"] and (case["expect_all"] or case["expect_any"] or case["expect_regex"])


def test_generator_is_deterministic(tmp_path):
    for case_id, make, *_ in M.CASES[2:6]:   # the shape cases do not depend on a font
        a, b = make(), make()
        assert a.tobytes() == b.tobytes()
        assert a.tobytes() == Image.open(SUITE / f"{case_id}.png").convert("RGB").tobytes()

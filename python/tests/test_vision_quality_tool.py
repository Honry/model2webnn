"""python/tools/qwen35_vision_quality.py: the scoring rules (shared with bench/raw/vision-suite.js), the token comparison and the report on made-up results"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import qwen35_vision_quality as Q  # noqa: E402


def test_score_rules():
    case = {"expect_all": ["webnn", "2026"], "expect_any": [], "expect_regex": []}
    assert Q.score(case, "The text is WebNN 2026.") and not Q.score(case, "WebNN")
    assert Q.score({"expect_all": [], "expect_any": ["5", "five"], "expect_regex": []}, "There are Five squares") and not Q.score({"expect_all": [], "expect_any": ["5", "five"]}, "six")
    bar = {"expect_all": [], "expect_any": [], "expect_regex": [r"\bD\b"]}
    assert Q.score(bar, "The tallest bar is D.") and Q.score(bar, "d") and not Q.score(bar, "The tallest is C, not bad") and not Q.score(bar, "Dad")
    assert Q.score({"expect_all": [], "expect_any": [], "expect_regex": []}, "anything") is None   # an ad-hoc case: nothing to check


def test_every_suite_case_is_satisfiable_by_its_own_wording():
    answers = {"ocr-large": "WebNN 2026", "ocr-small": "48213 and 127.50", "color-circle": "red", "count-squares": "5", "left-right": "Red", "bar-chart": "D", "banner-wide": "BANNER", "tall-triangle": "a green triangle", "ocr-micro": "93.05"}
    for case in Q.load_cases():
        assert Q.score(case, answers[case["id"]]), case["id"]
        assert not Q.score(case, "I do not know")


def test_compare_tokens_and_report():
    rng = np.random.default_rng(0)
    want = rng.standard_normal((16, 32)).astype(np.float32)
    same = Q.compare_tokens(want, want)
    assert same["cos_min"] > 0.9999 and same["rel_l2"] < 1e-6
    noisy = Q.compare_tokens(want + 0.3 * rng.standard_normal(want.shape).astype(np.float32), want)
    assert 0.9 < noisy["cos_mean"] < 0.99 and noisy["rel_l2"] > 0.2
    results = {"ocr-large": {"tokens": {"grid": [16, 24], "int8": same, "int8g128": noisy},
                             "answers": {"hf-package": {"int8": {"answer": "WebNN 2026", "pass": True, "nll": 0.1, "top1": 1.0}, "int8g128": {"answer": "x", "pass": False, "nll": 0.5, "top1": 0.5}}}}}
    text = Q.report(results, ["int8", "int8g128"])
    assert "ocr-large" in text and "16x24" in text and "PASS" in text and "FAIL" in text and "1/1" in text and "Teacher-forced" in text

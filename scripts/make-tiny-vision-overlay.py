"""The image tower of the tiny Qwen3.5 in the int8g128 weight format (int8 data in blocks of 128 input channels with a scale per block), as an OVERLAY of tests/golden/tiny-qwen35-package:
tests/golden/tiny-qwen35-vision-int8g128/{vision.json, graphs/vision/*, weights/vision_*}; src/runtime/llm-qwen35-vision.test.ts merges it into the plan / manifest of the committed package.
python scripts/make-tiny-vision-overlay.py   (needs torch + transformers >= 5.3 like make-tiny-package.py; the text part of the package is not touched)"""
import json
import shutil
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "python"
sys.path[:0] = [str(root), str(root / "tests")]
from m2w.compile import compile_model  # noqa: E402
from tiny_qwen35 import GRIDS, write_tiny_qwen35  # noqa: E402

work = root / "_out" / "tiny-qwen35-g128"
overlay = root.parent / "tests" / "golden" / "tiny-qwen35-vision-int8g128"
write_tiny_qwen35(work / "hf")
pkg = compile_model(str(work / "hf"), work / "pkg", max_lengths=(64,), chunks=(16,), staging_rows=(8,), vision_weights="int8g128", vision_grids=GRIDS, workers=2, progress=lambda m: None)
plan, manifest = json.loads((pkg / "plan.json").read_text()), json.loads((pkg / "weights" / "manifest.json").read_text())
shutil.rmtree(overlay, ignore_errors=True)
(overlay / "weights").mkdir(parents=True)
shutil.copytree(pkg / "graphs" / "vision", overlay / "graphs" / "vision")
files = [f for f in manifest["files"] if f["role"].startswith("vision_")]
for f in files:
    shutil.copy(pkg / "weights" / f["file"], overlay / "weights" / f["file"])
(overlay / "vision.json").write_text(json.dumps({"plan": plan["vision"], "quant": manifest["quant"]["vision"], "files": files}, indent=1), encoding="utf-8", newline="\n")
print("wrote", overlay)

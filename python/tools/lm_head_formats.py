"""Accuracy of the lm_head formats on REAL final hidden states: float table vs uint8 per-row asymmetric vs int8 per-row symmetric.

    python tools/lm_head_formats.py <package dir> <hf model dir> [--rows N]

The hidden states are the post-final-norm outputs of the float32 reference model for the prompt (+ generated tokens) of the package's verification.json; for every row the logits
from each table are compared with the float table: cosine, top-1 agreement, top-5 overlap, largest |difference|.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2w.frontends.hf_llm import arch  # noqa: E402
from m2w.frontends.hf_llm.safetensors_io import TensorStore  # noqa: E402
from m2w.quantize import dequantize_u8_rowwise, quantize_u8_rowwise  # noqa: E402
from m2w.reference.transformer_ref import ReferenceModel  # noqa: E402


def quantize_i8_symmetric_rowwise(w: np.ndarray):
    """per row: scale = max|w| / 127 (float16), q = round(w / scale) in [-127, 127]; an all-zero row gets scale 1e-3"""
    scale = (np.abs(w).max(axis=1, keepdims=True) / 127.0).astype(np.float16)
    scale = np.where(scale == 0, np.float16(1e-3), scale)
    q = np.clip(np.rint(w / scale.astype(np.float32)), -127, 127).astype(np.int8)
    return q, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("source")
    ap.add_argument("--rows", type=int, default=64)
    args = ap.parse_args()
    package, source = Path(args.package), Path(args.source)
    index = json.loads((package / "m2w.json").read_text())
    spec = arch.ModelSpec.from_dict(json.loads((package / index["model"]).read_text()))
    ver = json.loads((package / "verification.json").read_text())
    ids = (ver["prompt_ids"] + ver["generated_ids"])[: args.rows]
    store = TensorStore.open(source)
    model = ReferenceModel(spec, store.read)
    model.logits(ids)                                    # runs the whole float32 model once
    hidden = model.last_hidden.astype(np.float32)        # [T, E]
    table = store.read(spec.tensors["embed"])            # [V, E] float32
    print(f"{spec.architecture}: {len(ids)} hidden states, table {table.shape}")
    exact = hidden @ table.T

    qa, za, sa = quantize_u8_rowwise(table)
    asym = dequantize_u8_rowwise(qa, za, sa)
    qs, ss = quantize_i8_symmetric_rowwise(table)
    sym = qs.astype(np.float32) * ss.astype(np.float32)
    for name, deq in (("uint8 asymmetric (per-row zero point)", asym), ("int8 symmetric (per-row, scale only)", sym)):
        approx = hidden @ deq.T
        top1 = float(np.mean(approx.argmax(1) == exact.argmax(1)))
        top5 = float(np.mean([len(set(np.argsort(a)[-5:]) & set(np.argsort(e)[-5:])) / 5 for a, e in zip(approx, exact)]))
        cos = [float(np.dot(a, e) / np.linalg.norm(a) / np.linalg.norm(e)) for a, e in zip(approx, exact)]
        rel = float(np.linalg.norm(deq - table) / np.linalg.norm(table))
        print(f"{name:40s} table rel. error {rel:.5f} | logits cosine min {min(cosines := cos):.6f} mean {np.mean(cos):.6f} | top-1 agrees {top1 * 100:.1f}% | top-5 overlap {top5 * 100:.1f}% | max |dlogit| {np.abs(approx - exact).max():.3f} (logit range {exact.max() - exact.min():.1f})")


if __name__ == "__main__":
    main()

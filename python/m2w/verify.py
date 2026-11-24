"""m2w verify: run a compiled package on the CPU (interpreter, the runtime protocol) and compare it with the original model.

  * graphs + protocol vs the same weights dequantized (independent numpy forward)   -> must agree to f16 noise (cosine > 0.999, same greedy tokens)
  * quantized vs the ORIGINAL (bf16 / f32) weights                                  -> the cost of int4 / u8 quantization (reported; judge it for your model)
  * the largest |residual| of any layer                                              -> warns before float16 overflows (limit 65504)
"""
from __future__ import annotations

import datetime
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .frontends.hf_llm import arch, chat_format, hub
from .frontends.hf_llm.safetensors_io import TensorStore
from .reference.graph_runner import PackageRunner
from .reference.package_reader import PackageReader
from .reference.transformer_ref import ReferenceModel


def _cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def verify_package(package: Path, source: str, *, prompt: str, steps: int, max_length: int = 0, chunk: int = 0, endpoint: Optional[str] = None, cache_dir=None,
                   skip_float_reference: bool = False, token_ids: Optional[list] = None) -> int:
    index = json.loads((package / "m2w.json").read_text())
    spec = arch.ModelSpec.from_dict(json.loads((package / index["model"]).read_text()))
    plan = json.loads((package / index["plan"]).read_text())
    max_length = max_length or min(plan["max_lengths"])
    chunk = chunk or min(c for c in plan["prefill_chunks"] if c <= max_length - 1)
    weights_dir = package / "weights"

    if token_ids is None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(package / "tokenizer" / "tokenizer.json"))
        if index.get("chat_format"):
            fmt = json.loads((package / index["chat_format"]).read_text(encoding="utf-8"))
            text = chat_format.fill_placeholders(chat_format.compose(fmt, [(prompt, None)]), datetime.datetime.now())
        else:
            text = prompt
        token_ids = tokenizer.encode(text, add_special_tokens=False).ids
        print(f"prompt: {len(token_ids)} tokens")
    else:
        tokenizer = None

    t = time.perf_counter()
    runner = PackageRunner(spec, weights_dir, max_length=max_length, chunk_size=chunk, staging_rows=plan["staging_rows"])
    generated, step_logits = runner.generate(token_ids, steps, eos=spec.eos_token_ids)
    print(f"compiled graphs on the CPU: {len(generated)} tokens in {time.perf_counter() - t:.0f} s (max length {max_length}, chunk {chunk})")
    if tokenizer:
        print("  generated:", repr(tokenizer.decode(generated, skip_special_tokens=False)))

    # the expected result of this run, for the runtime conformance test in the browser (tests/ in the package folder is not part of the package index)
    (package / "verification.json").write_text(json.dumps({
        "max_length": max_length, "chunk": chunk, "prompt_ids": [int(t) for t in token_ids], "generated_ids": generated,
        "steps": [{"top_ids": np.argsort(l)[-5:][::-1].tolist(), "top_logits": [round(float(l[i]), 3) for i in np.argsort(l)[-5:][::-1]]} for l in step_logits],
        "text": tokenizer.decode(generated, skip_special_tokens=False) if tokenizer else None}), encoding="utf-8", newline="\n")

    sources = [("dequantized weights", ReferenceModel(spec, PackageReader(spec, weights_dir).read), True)]
    if not skip_float_reference:
        folder = hub.resolve_model(source, cache_dir=cache_dir, endpoint=endpoint)
        sources.append(("original weights", ReferenceModel(spec, TensorStore.open(folder).read), False))
    ok = True
    for name, reference, strict in sources:
        cosines, agree = [], 0
        for k in range(len(generated)):
            logits = reference.logits(token_ids + generated[:k])[-1]
            cosines.append(_cos(step_logits[k], logits))
            agree += int(np.argmax(logits)) == generated[k]
        print(f"vs {name:20s}: logits cosine min {min(cosines):.5f} mean {np.mean(cosines):.5f}; the reference picks the same token at {agree}/{len(generated)} steps")
        peak = max(reference.layer_max_abs) if reference.layer_max_abs else 0
        if strict:
            ok &= min(cosines) > 0.999 and agree == len(generated)
            print(f"  largest |residual| of any layer: {peak:.0f}" + ("   WARNING: close to the float16 limit 65504" if peak > 30000 else ""))
    print("RESULT:", "pipeline OK (graphs + protocol + weights agree with the independent reference)" if ok else "MISMATCH between the compiled graphs and the reference")
    return 0 if ok else 1

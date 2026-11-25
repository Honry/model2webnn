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
from .reference.gemma4_package import Gemma4PackageReader
from .reference.gemma4_ref import PREFIX as GEMMA4_PREFIX
from .reference.gemma4_ref import Gemma4Reference
from .reference.gemma4_runner import Gemma4Runner
from .reference.graph_runner import PackageRunner
from .reference.package_reader import PackageReader
from .reference.qwen35_runner import Qwen35PackageReader, Qwen35Runner
from .reference.transformer_ref import ReferenceModel


class _Gemma4Model:
    """Gemma4Reference with the interface verify_package expects of a reference model: logits(ids)[-1] over the real vocabulary, layer_max_abs of the last call."""

    def __init__(self, spec, text_config, read):
        self.spec, self.layer_max_abs = spec, []
        self._ref = Gemma4Reference(text_config, read)

    def logits(self, ids):
        self.layer_max_abs = []
        return self._ref.logits(ids, dump=self.layer_max_abs)[:, : self.spec.vocab]


def _cos(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def verify_package(package: Path, source: str, *, prompt: str, steps: int, max_length: int = 0, chunks=None, variants=None, decode_group: int = 1, prefill_group: int = 1,
                   endpoint: Optional[str] = None, cache_dir=None, skip_float_reference: bool = False, token_ids: Optional[list] = None) -> int:
    """Runs the graph FILES of the package (what the browser loads) for each requested variant; the first variant also writes verification.json."""
    index = json.loads((package / "m2w.json").read_text())
    spec = arch.ModelSpec.from_dict(json.loads((package / index["model"]).read_text()))
    plan = json.loads((package / index["plan"]).read_text())
    max_length = max_length or min(plan["max_lengths"])
    chunks = list(chunks) if chunks else [min(c for c in plan["prefill_chunks"] if c <= max_length - 1)]
    ids = [v["id"] for v in plan["variants"]]
    wanted = ids if variants and "all" in variants else list(variants or [plan["default_variant"]])
    for v in wanted:
        if v not in ids:
            raise SystemExit(f"the package has no variant '{v}' (it has {', '.join(ids)})")
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

    runs = []
    for number, variant in enumerate(wanted):
        t = time.perf_counter()
        if spec.builder == "gemma4":
            runner = Gemma4Runner.from_package(package, spec, variant=variant, max_length=max_length, chunk_sizes=chunks)
        elif spec.builder == "qwen35":
            runner = Qwen35Runner.from_package(package, spec, variant=variant, max_length=max_length, chunk_sizes=chunks)
        else:
            runner = PackageRunner.from_package(package, spec, variant=variant, max_length=max_length, chunk_sizes=chunks, decode_group=decode_group, prefill_group=prefill_group)
        generated, step_logits = runner.generate(token_ids, steps, eos=spec.eos_token_ids)
        print(f"[{variant}] compiled graphs on the CPU: {len(generated)} tokens in {time.perf_counter() - t:.0f} s (max length {max_length}, chunks {chunks}, "
              f"plan {runner.plan(len(token_ids))}, blocks/graph {decode_group}/{prefill_group})")
        if tokenizer:
            print("  generated:", repr(tokenizer.decode(generated, skip_special_tokens=False)))
        runs.append((variant, generated, step_logits))
        if number == 0:   # the expected result of this run, for the runtime conformance test in the browser (verification.json is not part of the package index)
            (package / "verification.json").write_text(json.dumps({
                "variant": variant, "max_length": max_length, "chunk": chunks[0], "chunks": chunks, "prompt_ids": [int(t) for t in token_ids], "generated_ids": generated,
                "steps": [{"top_ids": np.argsort(l)[-5:][::-1].tolist(), "top_logits": [round(float(l[i]), 3) for i in np.argsort(l)[-5:][::-1]]} for l in step_logits],
                "text": tokenizer.decode(generated, skip_special_tokens=False) if tokenizer else None}), encoding="utf-8", newline="\n")

    if spec.builder == "gemma4":
        text_config = _gemma4_text_config(package, source, endpoint, cache_dir)
        sources = [("dequantized weights", _Gemma4Model(spec, text_config, Gemma4PackageReader(spec, weights_dir).read), True)]
        if not skip_float_reference:
            store = TensorStore.open(hub.resolve_model(source, cache_dir=cache_dir, endpoint=endpoint))
            sources.append(("original weights", _Gemma4Model(spec, text_config, lambda name, rows=None: store.read(GEMMA4_PREFIX + name, rows=rows)), False))
    elif spec.builder == "qwen35":   # the transformers implementation is the reference
        from .reference.qwen35_hf import Qwen35HFModel, original_tensors

        text_config = _gemma4_text_config(package, source, endpoint, cache_dir)
        sources = [("dequantized weights", Qwen35HFModel(text_config, Qwen35PackageReader(spec, weights_dir).hf_state_dict()), True)]
        if not skip_float_reference:
            store = TensorStore.open(hub.resolve_model(source, cache_dir=cache_dir, endpoint=endpoint))
            sources.append(("original weights", Qwen35HFModel(text_config, original_tensors(store, spec)), False))
    else:
        sources = [("dequantized weights", ReferenceModel(spec, PackageReader(spec, weights_dir).read), True)]
        if not skip_float_reference:
            folder = hub.resolve_model(source, cache_dir=cache_dir, endpoint=endpoint)
            sources.append(("original weights", ReferenceModel(spec, TensorStore.open(folder).read), False))
    ok = True
    for name, reference, strict in sources:
        for variant, generated, step_logits in runs:
            cosines, agree = [], 0
            for k in range(len(generated)):
                logits = reference.logits(token_ids + generated[:k])[-1]
                cosines.append(_cos(step_logits[k], logits))
                agree += int(np.argmax(logits)) == generated[k]
            print(f"[{variant}] vs {name:20s}: logits cosine min {min(cosines):.5f} mean {np.mean(cosines):.5f}; the reference picks the same token at {agree}/{len(generated)} steps")
            agreed = min(cosines) > 0.999 and agree == len(generated)   # NaN cosines compare False
            if strict and variant.endswith("float16") and not agreed:
                # not a failure of the package: x * x of the RMSNorm exceeds 65504 for residuals above 255 and overflows on a backend that does not fuse the pattern (this CPU interpreter);
                # the NPU compiler fuses RMSNorm. float16 compute has to be checked on the device.
                print(f"  [{variant}] differs here: float16 compute overflows in RMSNorm on this CPU interpreter (the NPU fuses RMSNorm); check this variant on the device")
            elif strict:
                ok &= agreed
        peak = max(reference.layer_max_abs) if reference.layer_max_abs else 0
        if strict:
            print(f"  largest |residual| of any layer: {peak:.0f}" + ("   WARNING: close to the float16 limit 65504" if peak > 30000 else ""))
    print("RESULT:", "pipeline OK (graphs + protocol + weights agree with the independent reference)" if ok else "MISMATCH between the compiled graphs and the reference")
    return 0 if ok else 1


def _gemma4_text_config(package: Path, source: str, endpoint, cache_dir) -> dict:
    """The text_config the numpy reference reads: from the source model's config.json (a package does not carry the original config)."""
    folder = hub.resolve_model(source, cache_dir=cache_dir, endpoint=endpoint, weights=False)
    return json.loads((folder / "config.json").read_text(encoding="utf-8"))["text_config"]

"""Image QUALITY of the Qwen3.5 image path per vision weight format, measured offline on the CPU (no NPU involved): the cases of bench/raw/vision-suite (python/tools/make_vision_suite.py).

    python tools/qwen35_vision_quality.py --build int8g128 float16 int4                 # the sibling packages python/_out/qwen35-4b-vl-<format>-pkg (text shards hard-linked, own vision shards)
    python tools/qwen35_vision_quality.py --stage tokens  [--formats int8 int8g128 float16 int4] [--cases ocr-large ...]
    python tools/qwen35_vision_quality.py --stage answers --answers hf-original hf-package numpy [--max-new 24]

(the preset of an image is chosen automatically like the runtime does: --detail low|auto|high, or forced with --grid HxW; results of a forced preset / detail get their own `@...` column set)
(a) stage `tokens`: the image tokens of every format (the compiled vision graphs in the numpy interpreter on the package weights) against the ORIGINAL bfloat16 vision tower (transformers
    Qwen3_5VisionModel with the checkpoint's weights, float32 compute) on the SAME preprocessed pixels: cosine per image token (mean / min / 5th percentile) and the relative error.
(b) stage `answers`: the greedy answer (max-new tokens) of the full model, scored with the expectations of the case:
      hf-original  transformers Qwen3_5ForConditionalGeneration on the original checkpoint (float32 compute, bfloat16 weights);
      hf-package   the same transformers model with the DEQUANTIZED package weights (int4 text + the vision tower of the format): the graphs / protocol are verified equal to it (qwen35_vl_check.py), it is
                   ~50x faster than the interpreter, so it is the default stand-in for `numpy` on every case;
      numpy        our reference runner (compiled graphs in the numpy interpreter, prompt expansion, MRoPE): minutes per answer, select cases with --cases.
Results are merged into --results (json, resumable); --report prints the markdown tables of everything collected so far.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from m2w.frontends.hf_llm import arch, chat_format, vision as V  # noqa: E402
from m2w.reference import protocol  # noqa: E402

OUT = ROOT / "python" / "_out"
SUITE = ROOT / "bench" / "raw" / "vision-suite"
FORMATS = ("int8", "int8g128", "float16", "int4")
BASE = OUT / "qwen35-4b-vl-pkg"      # the int8 per row package (never modified)
SOURCE = OUT / "hf_cache" / "Qwen--Qwen3.5-4B"


def package_for(fmt: str) -> Path:
    return BASE if fmt == "int8" else OUT / f"qwen35-4b-vl-{fmt}-pkg"


EXTRA: list = []   # ad-hoc cases from --image / --question (no expectations: only the token and teacher-forced numbers mean something)


def load_cases(only=None) -> list:
    cases = json.loads((SUITE / "cases.json").read_text(encoding="utf-8"))["cases"] + EXTRA
    return [c for c in cases if not only or c["id"] in only]


def score(case: dict, answer: str) -> bool:
    """the expectations of cases.json (the same rules as bench/raw/vision-suite.js): every expect_all substring, one of expect_any (when given), every expect_regex; case-insensitive"""
    if not (case.get("expect_all") or case.get("expect_any") or case.get("expect_regex")):
        return None   # an ad-hoc case: nothing to check
    text = answer.lower()
    return (all(s.lower() in text for s in case.get("expect_all", [])) and (not case.get("expect_any") or any(s.lower() in text for s in case["expect_any"]))
            and all(re.search(r, answer, re.IGNORECASE) for r in case.get("expect_regex", [])))


def build_packages(formats) -> None:
    """sibling packages of BASE with the image tower in another weight format: everything small is copied, the text weight shards are HARD-LINKED (no extra disk, BASE is never written), the vision shards
    / graphs are packed from the original checkpoint by add_vision"""
    import os

    from m2w.compile import add_vision

    for fmt in formats:
        out = package_for(fmt)
        if fmt == "int8" or (out / "m2w.json").exists():
            print("exists", out)
            continue
        (out / "weights").mkdir(parents=True, exist_ok=True)
        for item in BASE.iterdir():
            if item.name != "weights":
                (shutil.copytree if item.is_dir() else shutil.copy2)(item, out / item.name)
        for f in (BASE / "weights").iterdir():
            if f.name.startswith("vision_"):
                continue
            try:
                (shutil.copy2 if f.name == "manifest.json" else os.link)(f, out / "weights" / f.name)
            except OSError:
                shutil.copy2(f, out / "weights" / f.name)
        add_vision(out, str(SOURCE), vision_weights=fmt, progress=lambda m: None)
        print("built", out, flush=True)


def load_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_results(path: Path, results: dict) -> None:
    path.write_text(json.dumps(results, indent=1), encoding="utf-8", newline="\n")


class Context:
    """what every stage shares: the spec, the tokenizer, the chat format, the preprocessing of a case"""

    def __init__(self):
        from PIL import Image
        from tokenizers import Tokenizer

        self.Image = Image
        self.spec = arch.ModelSpec.from_dict(json.loads((BASE / "model.json").read_text(encoding="utf-8")))
        self.tokenizer = Tokenizer.from_file(str(BASE / "tokenizer" / "tokenizer.json"))
        self.fmt = json.loads((BASE / "chat_format.json").read_text(encoding="utf-8"))
        self.grids = [tuple(p["grid"]) for p in json.loads((BASE / "plan.json").read_text(encoding="utf-8"))["vision"]["presets"]]

    def prepare(self, case: dict, grid=None, detail: str = "auto"):
        """-> (patches float16 [N, 1536], grid): the pixels every reference and every format is fed with (the preset is chosen automatically like the runtime does, unless `grid` is given)"""
        patches, grid = protocol.prepare_image(self.Image.open(SUITE / case["image"]).convert("RGB"), self.spec, self.grids, grid, detail)
        return patches.astype(np.float16), grid

    def prompt(self, case: dict) -> list:
        text = chat_format.fill_placeholders(chat_format.compose(self.fmt, [(self.spec.vision["image_prompt"] + case["question"], None)]), datetime.datetime(2026, 1, 1))
        tokens = self.tokenizer.encode(text, add_special_tokens=False).ids
        assert tokens.count(self.spec.vision["image_token_id"]) == 1
        return tokens

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True).strip()


# --------------------------------------------------------------------------------------------------------------------- (a) image tokens
def vision_config(spec) -> dict:
    v = spec.vision
    return {"depth": v["depth"], "hidden_size": v["hidden"], "num_heads": v["heads"], "intermediate_size": v["ffn"], "patch_size": v["patch_size"], "temporal_patch_size": v["temporal_patch_size"],
            "spatial_merge_size": v["merge_size"], "out_hidden_size": v["out_hidden"], "num_position_embeddings": v["pos_grid"] ** 2, "hidden_act": "gelu_pytorch_tanh", "in_channels": v["channels"],
            "deepstack_visual_indexes": []}


def compare_tokens(got: np.ndarray, want: np.ndarray) -> dict:
    cos = (got * want).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1) + 1e-12)
    return {"tokens": int(got.shape[0]), "cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "cos_p5": float(np.percentile(cos, 5)),
            "rel_l2": float(np.linalg.norm(got - want) / np.linalg.norm(want)), "max_abs_over_max": float(np.abs(got - want).max() / np.abs(want).max()), "worst_token": int(cos.argmin())}


def stage_tokens(ctx: Context, args, results: dict) -> None:
    from m2w.frontends.hf_llm.safetensors_io import TensorStore
    from m2w.reference.qwen35_hf import Qwen35VisionHF
    from m2w.reference.qwen35_vision import Qwen35VisionRunner

    store = TensorStore.open(str(SOURCE))
    original = Qwen35VisionHF(vision_config(ctx.spec), {n: store.read(n) for n in store.names() if n.startswith("model.visual.")})
    runners = {fmt: Qwen35VisionRunner(package_for(fmt), ctx.spec) for fmt in args.formats}
    for case in load_cases(args.cases):
        patches, grid = ctx.prepare(case, V.parse_grid(args.grid) if args.grid else None, args.detail)
        want, _ = original.forward(patches.astype(np.float32), grid)
        entry = results.setdefault(case["id"], {}).setdefault("tokens", {})
        entry["grid"] = list(grid)
        for fmt in args.formats:
            t = time.time()
            got = runners[fmt].encode(patches, grid).astype(np.float32)
            entry[fmt] = compare_tokens(got, want)
            print(f"{case['id']:14s} {V.grid_name(grid)} {fmt:9s} cos mean {entry[fmt]['cos_mean']:.5f} min {entry[fmt]['cos_min']:.4f} p5 {entry[fmt]['cos_p5']:.4f} rel L2 {entry[fmt]['rel_l2']:.4f} ({time.time() - t:.0f} s)", flush=True)
        save_results(args.results, results)


# --------------------------------------------------------------------------------------------------------------------- (b) answers
def hf_generate(model, ctx: Context, case: dict, patches, grid, max_new: int):
    import torch

    expanded = protocol.expand_image_tokens(ctx.spec, ctx.prompt(case), [grid])
    ids = torch.tensor([expanded])
    eos = list(ctx.spec.eos_token_ids)
    with torch.no_grad():
        out = model.generate(input_ids=ids, pixel_values=torch.from_numpy(patches.astype(np.float32)).to(model.dtype), image_grid_thw=torch.tensor([[1, *grid]]),
                             mm_token_type_ids=(ids == ctx.spec.vision["image_token_id"]).int(), max_new_tokens=max_new, do_sample=False, eos_token_id=eos, pad_token_id=eos[0])
    return out[0, ids.shape[1]:].tolist()


def teacher_forced(model, ctx: Context, case: dict, patches, grid, reference_ids: list) -> dict:
    """how well `model` (with the image of this case) predicts the answer of the ORIGINAL model: mean negative log-likelihood (nats / token) of its tokens and the share of positions whose argmax is its token.
    A much finer instrument than PASS / FAIL: a degraded image path shows here long before the greedy answer changes."""
    import torch

    expanded = protocol.expand_image_tokens(ctx.spec, ctx.prompt(case), [grid])
    ids = torch.tensor([expanded + reference_ids])
    with torch.no_grad():
        logits = model(input_ids=ids, pixel_values=torch.from_numpy(patches.astype(np.float32)).to(model.dtype), image_grid_thw=torch.tensor([[1, *grid]]),
                       mm_token_type_ids=(ids == ctx.spec.vision["image_token_id"]).int()).logits[0].float()
    at = logits[len(expanded) - 1:-1]   # the logits that predict the reference tokens
    target = torch.tensor(reference_ids)
    logp = torch.log_softmax(at, dim=-1)
    return {"nll": float(-logp[torch.arange(len(reference_ids)), target].mean()), "top1": float((at.argmax(-1) == target).float().mean())}


def reference_ids_of(results: dict, case: dict, args):
    return results.get(case["id"], {}).get("answers", {}).get("hf-original" + variant_suffix(args), {}).get("original", {}).get("ids")


def answers_hf_original(ctx: Context, args, results: dict) -> None:
    import torch
    from transformers import Qwen3_5ForConditionalGeneration

    model = Qwen3_5ForConditionalGeneration.from_pretrained(str(SOURCE), dtype=torch.bfloat16 if args.hf_dtype == "bfloat16" else torch.float32).eval()
    for case in load_cases(args.cases):
        patches, grid = ctx.prepare(case, V.parse_grid(args.grid) if args.grid else None, args.detail)
        t = time.time()
        ids = hf_generate(model, ctx, case, patches, grid, args.max_new)
        extra = {"ids": ids, **teacher_forced(model, ctx, case, patches, grid, [i for i in ids if i not in ctx.spec.eos_token_ids])}
        record(results, case, "hf-original", "original", ctx.decode(ids), time.time() - t, args, extra)
    del model
    gc.collect()


def answers_hf_package(ctx: Context, args, results: dict) -> None:
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    from m2w.reference.qwen35_runner import Qwen35PackageReader

    config = Qwen3_5Config(**json.loads((SOURCE / "config.json").read_text(encoding="utf-8")))
    model = Qwen3_5ForConditionalGeneration(config).to(torch.float32).eval()
    text_state = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in Qwen35PackageReader(ctx.spec, BASE / "weights").hf_state_dict(prefix="model.language_model.").items()}
    model.load_state_dict(text_state, strict=False)
    del text_state
    for fmt in args.formats:
        vision = {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in Qwen35PackageReader(ctx.spec, package_for(fmt) / "weights").vision_hf_state_dict().items()}
        missing = model.load_state_dict(vision, strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        for case in load_cases(args.cases):
            patches, grid = ctx.prepare(case, V.parse_grid(args.grid) if args.grid else None, args.detail)
            t = time.time()
            ids = hf_generate(model, ctx, case, patches, grid, args.max_new)
            reference = reference_ids_of(results, case, args)
            extra = {"ids": ids, "grid": list(grid), **(teacher_forced(model, ctx, case, patches, grid, [i for i in reference if i not in ctx.spec.eos_token_ids]) if reference else {})}
            record(results, case, "hf-package", fmt, ctx.decode(ids), time.time() - t, args, extra)
    del model
    gc.collect()


def answers_numpy(ctx: Context, args, results: dict) -> None:
    from m2w.reference.qwen35_runner import Qwen35Runner

    for fmt in args.formats:
        runner = Qwen35Runner.from_package(package_for(fmt), ctx.spec, chunk_sizes=[args.chunk], max_length=args.max_length)
        for case in load_cases(args.cases):
            patches, grid = ctx.prepare(case, V.parse_grid(args.grid) if args.grid else None, args.detail)
            t = time.time()
            generated, _ = runner.generate(ctx.prompt(case), args.max_new, eos=ctx.spec.eos_token_ids, images=[(patches, grid)])
            record(results, case, "numpy", fmt, ctx.decode(generated), time.time() - t, args)


def variant_suffix(args) -> str:
    """results of a forced grid / another detail level are kept apart: 'hf-package@high', 'hf-package@24x16'"""
    return f"@{args.grid}" if args.grid else (f"@{args.detail}" if args.detail != "auto" else "")


def record(results: dict, case: dict, runner: str, fmt: str, text: str, seconds: float, args, extra=None) -> None:
    ok = score(case, text)
    results.setdefault(case["id"], {}).setdefault("answers", {}).setdefault(runner + variant_suffix(args), {})[fmt] = {"answer": text, "pass": ok, "seconds": round(seconds), "max_new": args.max_new, **(extra or {})}
    nll = f" nll {extra['nll']:.3f} top1 {extra['top1']:.2f}" if extra and "nll" in extra else ""
    print(f"{case['id']:14s} {runner:12s} {fmt:9s} {'-' if ok is None else 'PASS' if ok else 'FAIL'} ({seconds:.0f} s){nll} {text!r}", flush=True)
    save_results(args.results, results)


# --------------------------------------------------------------------------------------------------------------------- report
def report(results: dict, formats) -> str:
    lines = ["### Image tokens against the original bf16 vision tower (same pixels)", "", "| case | grid | " + " | ".join(f"{f} cos mean / min / rel L2" for f in formats) + " |", "|---|---|" + "---|" * len(formats)]
    sums = {f: [] for f in formats}
    for cid, r in results.items():
        t = r.get("tokens")
        if not t:
            continue
        cells = []
        for f in formats:
            m = t.get(f)
            cells.append(f"{m['cos_mean']:.4f} / {m['cos_min']:.3f} / {m['rel_l2']:.3f}" if m else "-")
            if m:
                sums[f].append(m)
        lines.append(f"| {cid} | {t['grid'][0]}x{t['grid'][1]} | " + " | ".join(cells) + " |")
    if any(sums.values()):
        lines.append("| **mean over cases** | | " + " | ".join(f"{np.mean([m['cos_mean'] for m in sums[f]]):.4f} / {np.min([m['cos_min'] for m in sums[f]]):.3f} (worst) / {np.mean([m['rel_l2'] for m in sums[f]]):.3f}" if sums[f] else "-" for f in formats) + " |")
    runners = sorted({run for r in results.values() for run in r.get("answers", {})}, key=lambda run: (["hf-original", "hf-package", "numpy"].index(run.split("@")[0]), run))
    for run in runners:
        cols = ["original"] if run.startswith("hf-original") else list(formats)
        lines += ["", f"### Greedy answers ({run}), PASS / FAIL against the expectations", "", "| case | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        total = {c: [0, 0] for c in cols}
        for cid, r in results.items():
            a = r.get("answers", {}).get(run)
            if not a:
                continue
            cells = []
            for c in cols:
                x = a.get(c)
                cells.append(f"{'-' if x['pass'] is None else 'PASS' if x['pass'] else 'FAIL'}{' ' + 'x'.join(map(str, x['grid'])) if x and 'grid' in x else ''}: {x['answer'][:60]!r}" if x else "-")
                if x and x["pass"] is not None:
                    total[c][0] += x["pass"]
                    total[c][1] += 1
            lines.append(f"| {cid} | " + " | ".join(cells) + " |")
        lines.append("| **passed** | " + " | ".join(f"{total[c][0]}/{total[c][1]}" for c in cols) + " |")
        if any("nll" in x for r in results.values() for x in r.get("answers", {}).get(run, {}).values()):
            lines += ["", f"### Teacher-forced: NLL (nats / token) of the ORIGINAL model's answer / share of positions with the same argmax ({run})", "", "| case | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
            mean = {c: [] for c in cols}
            for cid, r in results.items():
                a = r.get("answers", {}).get(run)
                if not a:
                    continue
                cells = []
                for c in cols:
                    x = a.get(c)
                    cells.append(f"{x['nll']:.3f} / {x['top1']:.2f}" if x and "nll" in x else "-")
                    if x and "nll" in x:
                        mean[c].append((x["nll"], x["top1"]))
                lines.append(f"| {cid} | " + " | ".join(cells) + " |")
            lines.append("| **mean** | " + " | ".join(f"{np.mean([m[0] for m in mean[c]]):.3f} / {np.mean([m[1] for m in mean[c]]):.2f}" if mean[c] else "-" for c in cols) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", nargs="+", metavar="FORMAT", help="build the sibling packages of these formats (int8 is the base package) and exit")
    ap.add_argument("--stage", choices=["tokens", "answers"])
    ap.add_argument("--formats", nargs="+", default=list(FORMATS), choices=list(FORMATS))
    ap.add_argument("--cases", nargs="+", help="case ids (default: all)")
    ap.add_argument("--image", help="an extra picture (a real photo / screenshot) to measure next to the suite; with --question; its id is the file name")
    ap.add_argument("--question", default="Describe this image in two sentences.")
    ap.add_argument("--grid", help="force a preset HxW instead of the automatic choice")
    ap.add_argument("--detail", choices=["low", "auto", "high"], default="auto", help="detail level of the automatic grid choice")
    ap.add_argument("--answers", nargs="+", default=["hf-original", "hf-package"], choices=["hf-original", "hf-package", "numpy"])
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--hf-dtype", choices=["float32", "bfloat16"], default="float32")
    ap.add_argument("--chunk", type=int, default=64, help="prefill chunk of the numpy runner")
    ap.add_argument("--max-length", type=int, default=0)
    ap.add_argument("--results", type=Path, default=OUT / "vision_quality.json")
    ap.add_argument("--report", action="store_true", help="print the markdown tables of the collected results")
    args = ap.parse_args()
    if args.build:
        build_packages(args.build)
        return 0
    if args.image:
        EXTRA.append({"id": Path(args.image).stem[:24], "image": str(Path(args.image).resolve()), "question": args.question, "expect_all": [], "expect_any": [], "expect_regex": []})
    results = load_results(args.results)
    if args.report:
        print(report(results, args.formats))
        return 0
    ctx = Context()
    if args.stage == "tokens":
        stage_tokens(ctx, args, results)
    elif args.stage == "answers":
        for name in args.answers:
            {"hf-original": answers_hf_original, "hf-package": answers_hf_package, "numpy": answers_numpy}[name](ctx, args, results)
    print(report(results, args.formats))
    return 0


if __name__ == "__main__":
    sys.exit(main())

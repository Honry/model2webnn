"""An experiment sibling of a package with sliding window caches: `<package>-window` shares the weights of `<package>` (a directory junction on Windows, a symlink elsewhere: nothing is copied),
has its own small files (model.json, tokenizer, chat format) and graphs / plan.json rebuilt with `--sliding-cache window` for the max lengths and variants of the original (the window caches
support the staged KV protocol only, so the one-hot / scatter variants of the original are left out).  Prints the KV cache memory of both for every max length.

    python tools/window_sibling.py _out/gemma4-e4b-pkg            # -> _out/gemma4-e4b-pkg-window
    python tools/window_sibling.py _out/aion-2.3b-i8 --report-only  # only the memory table of an existing sibling

Never delete the sibling directory with rmtree / rm -r while the junction is in it: remove the `weights` junction first (`rmdir`)."""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2w import compile as C  # noqa: E402
from m2w.frontends.hf_llm import arch  # noqa: E402
from m2w.frontends.hf_llm.builders import decoder as D  # noqa: E402


def link(target: Path, source: Path) -> None:
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(target), str(source)], check=True, stdout=subprocess.DEVNULL)
    else:
        target.symlink_to(source, target_is_directory=True)


def staged_variants(plan: dict):
    """(staging rows, compute types) of the staged variants of the package, in the order of the original"""
    staged = [v for v in plan["variants"] if v["kv_style"] == "staged"]
    return list(dict.fromkeys(v["staging_rows"] for v in staged)), list(dict.fromkeys(v["compute_type"] for v in staged))


def make(source: Path, sibling: Path, progress=print) -> None:
    plan = json.loads((source / "plan.json").read_text(encoding="utf-8"))
    rows, computes = staged_variants(plan)
    sibling.mkdir(exist_ok=True)
    for name in ("model.json", "plan.json", "m2w.json", "chat_format.json", "verification.json"):
        if (source / name).exists():
            shutil.copy2(source / name, sibling / name)
    if not (sibling / "tokenizer").exists():
        shutil.copytree(source / "tokenizer", sibling / "tokenizer")
    if not (sibling / "weights").exists():
        link(sibling / "weights", (source / "weights").resolve())
    C.rebuild_graphs(sibling, kv_styles=("staged",), staging_rows=tuple(rows), compute_types=tuple(computes), blocks_per_graph=tuple(plan.get("blocks_per_graph", [1])),
                     prefill_blocks_per_graph=tuple(plan.get("prefill_blocks_per_graph", [1])), sliding_cache="window", progress=progress)


def kv_report(package: Path) -> list:
    """per max length: the cache rows of every layer type, the K / V bytes (f16, double buffered: what the page estimates) of the package"""
    plan = json.loads((package / "plan.json").read_text(encoding="utf-8"))
    spec = arch.ModelSpec.from_dict(json.loads((package / "model.json").read_text(encoding="utf-8")))
    opt = D.GraphOptions(**{k: v for k, v in plan["profile"].items() if k in D.GraphOptions.__dataclass_fields__})
    out = []
    for L in plan["max_lengths"]:
        rows, total = {}, 0
        for i, t in enumerate(spec.layer_types):
            if spec.kv_producer and spec.kv_producer[i] >= 0:
                continue
            dims = spec.decoder_dims(t)
            r = D.cache_rows(opt, t, L - 1)
            rows[t] = r
            total += 2 * 2 * dims.num_kv_heads * r * dims.head_dim * 2
        out.append({"max_length": L, "rows": rows, "mb": total / 2 ** 20})
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("package")
    p.add_argument("--suffix", default="-window")
    p.add_argument("--report-only", action="store_true")
    args = p.parse_args(argv)
    source = Path(args.package)
    sibling = source.with_name(source.name + args.suffix)
    if not args.report_only:
        make(source, sibling)
    full, window = kv_report(source), kv_report(sibling)
    print(f"\nKV cache (K + V, f16, double buffered; layers with a cache of their own): {source.name} -> {sibling.name}")
    print(f"{'max length':>10}  {'rows (full)':>26}  {'rows (window)':>26}  {'full MB':>9}  {'window MB':>9}  {'saved':>6}")
    for a, b in zip(full, window):
        show = lambda r: " ".join(f"{t.split('_')[0][:4]} {n}" for t, n in r.items())   # noqa: E731
        print(f"{a['max_length']:>10}  {show(a['rows']):>26}  {show(b['rows']):>26}  {a['mb']:9.0f}  {b['mb']:9.0f}  {100 * (1 - b['mb'] / a['mb']):5.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())

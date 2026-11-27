"""Write the reference trace of a Qwen3.5 package for the on-device layer-by-layer comparison (bench/raw/qwen35check.html): the hidden state entering every layer in the LAST prefill chunk and in
the first decode step, the final hidden state and the logits, computed by the numpy interpreter on the compiled graphs (the executable specification of the runtime).

  python tools/qwen35_trace.py <package> --out ../bench/data/qwen35-trace [--prompt "..."] [--tokens 1,2,3] [--chunk 64] [--max-length 1024] [--variant staged-r64-float32]
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2w.frontends.hf_llm import arch, chat_format  # noqa: E402
from m2w.reference import protocol  # noqa: E402
from m2w.reference.qwen35_runner import Qwen35Runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--tokens", help="comma separated token ids instead of a prompt")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=0)
    ap.add_argument("--variant")
    args = ap.parse_args()
    package = Path(args.package)
    index = json.loads((package / "m2w.json").read_text())
    spec = arch.ModelSpec.from_dict(json.loads((package / "model.json").read_text()))
    if args.tokens:
        ids = [int(t) for t in args.tokens.split(",")]
    else:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(package / "tokenizer" / "tokenizer.json"))
        fmt = json.loads((package / index["chat_format"]).read_text(encoding="utf-8"))
        ids = tokenizer.encode(chat_format.fill_placeholders(chat_format.compose(fmt, [(args.prompt, None)]), datetime.datetime.now()), add_special_tokens=False).ids
    runner = Qwen35Runner.from_package(package, spec, variant=args.variant, max_length=args.max_length, chunk_sizes=[args.chunk])
    plan = runner.plan(len(ids))
    logits = runner.prefill(ids, plan)
    prefill_trace = [np.asarray(t, np.float16) for t in runner.trace]
    token = int(np.argmax(logits))
    step_logits = runner.step(token)
    decode_trace = [np.asarray(t, np.float16) for t in runner.trace]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blob, entries = bytearray(), {}

    def add(name, arrays):
        entries[name] = []
        for a in arrays:
            raw = np.ascontiguousarray(a).tobytes()
            entries[name].append({"shape": list(a.shape), "offset": len(blob), "bytes": len(raw)})
            blob.extend(raw)

    add("prefill", prefill_trace)
    add("decode", decode_trace)
    last_valid = len(ids) - sum(plan[:-1])
    top = lambda l: [int(i) for i in np.argsort(l)[-5:][::-1]]  # noqa: E731
    (out / "trace.bin").write_bytes(bytes(blob))
    (out / "trace.json").write_text(json.dumps({
        "tokens": ids, "plan": plan, "chunk": args.chunk, "last_chunk_valid": last_valid, "decode_token": token, "layers": spec.num_layers, "hidden": spec.hidden,
        "variant": args.variant or "default", "entries": entries, "prefill_top": top(logits), "decode_top": top(step_logits),
        "prefill_top_logits": [float(logits[i]) for i in top(logits)], "decode_top_logits": [float(step_logits[i]) for i in top(step_logits)]}), encoding="utf-8")
    print(f"wrote {out}: {len(ids)} prompt tokens, plan {plan}, decode token {token}, {len(blob) / 2**20:.1f} MB")


if __name__ == "__main__":
    main()

"""Intermediate results of the linear attention half of layer 0 of a Qwen3.5 package for the real prompt, from the numpy interpreter, for bench/raw/qwen35probe.html (which runs the same graph
with the same weights and inputs ON THE DEVICE and shows the first intermediate result that deviates).

  python tools/qwen35_probe.py <package> --out ../bench/data/qwen35-probe [--chunk 128]
Writes probe_prefill.* (a prefill chunk of --chunk rows, the prompt left-padded) and probe_decode.* (one decode step after that prefill) and probe.json (where the weights of layer 0 are).
"""
import argparse
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2w.frontends.hf_llm import arch, chat_format  # noqa: E402
from m2w.frontends.hf_llm.builders import decoder as D  # noqa: E402
from m2w.frontends.hf_llm.builders import qwen35 as Q  # noqa: E402
from m2w.ir import serialize  # noqa: E402
from m2w.ir.interpreter import run  # noqa: E402
from m2w.reference import protocol  # noqa: E402
from m2w.reference.qwen35_runner import Qwen35Runner  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--compute-type", default="float32")
    ap.add_argument("--decode-token", type=int, default=760, help="the token of the decode step (the argmax of the reference prefill of the default prompt)")
    args = ap.parse_args()
    package = Path(args.package)
    index = json.loads((package / "m2w.json").read_text())
    spec = arch.ModelSpec.from_dict(json.loads((package / "model.json").read_text()))
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(package / "tokenizer" / "tokenizer.json"))
    fmt = json.loads((package / index["chat_format"]).read_text(encoding="utf-8"))
    ids = tokenizer.encode(chat_format.fill_placeholders(chat_format.compose(fmt, [(args.prompt, None)]), datetime.datetime.now()), add_special_tokens=False).ids
    runner = Qwen35Runner.from_package(package, spec, chunk_sizes=[args.chunk])   # for the weights / embedding access and the layer-0 states after the prefill
    opt = D.GraphOptions(kv_style="staged", staging_rows=runner.R, compute_type=args.compute_type)
    P = runner.P
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shard = runner.reader._layer[0]
    weights = runner._weights(0, runner.attention_names["linear_attention"])

    def export(name, seq, feeds):
        graph = Q.build_head_graph(spec, opt, seq=seq, past_len=P, name=name, probe=True)
        serialize.dump(graph, out / f"{name}.json")
        result = run(graph, feeds)
        blob, entries = bytearray(), {"inputs": [], "expected": [], "weights": [n for n in graph.inputs if graph.inputs[n].role == "weight"]}
        for n, spec_in in graph.inputs.items():
            if spec_in.role == "weight":
                continue
            raw = np.ascontiguousarray(feeds[n]).tobytes()
            entries["inputs"].append({"name": n, "dataType": spec_in.dtype, "shape": list(spec_in.shape), "offset": len(blob), "bytes": len(raw)})
            blob.extend(raw)
        for n in graph.outputs:
            a = np.ascontiguousarray(result[n].astype(np.float32))
            raw = a.tobytes()
            entries["expected"].append({"name": n, "dataType": "float32", "shape": list(a.shape), "offset": len(blob), "bytes": len(raw)})
            blob.extend(raw)
        (out / f"{name}.bin").write_bytes(bytes(blob))
        (out / f"{name}.io.json").write_text(json.dumps(entries), encoding="utf-8")
        print(name, "graph nodes", len(graph.nodes), "outputs", len(graph.outputs), flush=True)
        return result

    C, n = args.chunk, len(ids)
    pad = C - n
    hidden = np.zeros((1, C, spec.hidden), np.float16)
    for i in range(n):
        hidden[0, pad + i] = runner._embed(ids[i])
    mask, perm = protocol.linear_prefill_controls(C, n)
    ld = Q.linear_dims(spec)
    zeros_conv = np.zeros((1, 3, ld["conv_width"]), np.float16)
    zeros_rec = np.zeros((1, ld["v_heads"], ld["k_dim"], ld["v_dim"]), np.float16)
    after = export("probe_prefill", C, {"hidden_states": hidden, "conv_state": zeros_conv, "recurrent_state": zeros_rec, "valid_mask": mask, "conv_perm": perm, **weights})
    export("probe_decode", 1, {"hidden_states": runner._embed(args.decode_token).reshape(1, 1, -1), "conv_state": after["new_conv_state"], "recurrent_state": after["new_recurrent_state"], **weights})
    (out / "probe.json").write_text(json.dumps({"shard": f"weights/{shard['file']}", "pad": pad, "rows": C, "tokens": n, "tensors": shard["tensors"]}), encoding="utf-8")


if __name__ == "__main__":
    main()

"""The causal depthwise conv of the Gated DeltaNet layers (kernel 4 over 8192 channels, silu) as graphs of their own, in the ways it can be written, for bench/raw/gdn.html (dir=../data/conv/):

  slices      4 shifted multiply-adds over concat(conv_state, x) [1, S + 3, C]            (what qwen35.py does today; OpenVINO's IR uses a GroupConvolution)
  conv2d      WebNN conv2d, depthwise (groups = C): [1, S + 3, C] -> transpose / reshape [1, C, 1, S + 3] -> conv2d (filter [C, 1, 1, 4]) -> back to [1, S, C]
  conv2d_cf   the same conv2d on a channels-first input [1, C, 1, S + 3] (no transposes: the cost of the convolution itself)

  python tools/conv_experiment.py --export ../bench/data/conv [--tokens 128 64 1]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2w.ir import GraphBuilder, interpreter, serialize  # noqa: E402

C, TAPS = 8192, 4   # --channels overrides C (the depthwise conv2d of 8192 channels did not compile on the NPU: how does it scale?)


def silu(b, x):
    return b.mul(x, b.sigmoid(x))


def build(style: str, tokens: int, dtype: str):
    b = GraphBuilder(f"conv_{style}_c{C}_s{tokens}_{dtype}")
    w = b.input("w", dtype, [TAPS, 1, C], "weight")                                         # the layout of the package today
    if style == "perm_matmul":   # the pad permutation of the prefill conv: [S+3, S+3] @ [S+3, C]
        n = tokens + TAPS - 1
        p = b.input("perm", dtype, [1, n, n])
        x = b.input("x", dtype, [1, n, C])
        b.output("y", b.matmul(p, x))
        return b.build()
    if style == "silu_only":   # what the activation alone costs: [1, S, C] -> silu
        x = b.input("x", dtype, [1, tokens + TAPS - 1, C])
        b.output("y", silu(b, b.slice(x, [0, 0, 0], [1, tokens, C])))
        return b.build()
    if style == "conv2d_cf_nosilu":
        filt = b.reshape(b.transpose(b.reshape(w, [TAPS, C]), [1, 0]), [C, 1, 1, TAPS])
        x = b.input("x", dtype, [1, C, 1, tokens + TAPS - 1])
        b.output("y", b.conv2d(x, filt, groups=C))
        return b.build()
    if style == "slices":
        x = b.input("x", dtype, [1, tokens + TAPS - 1, C])
        y = None
        for j in range(TAPS):
            term = b.mul(b.slice(x, [0, j, 0], [1, tokens, C]), b.slice(w, [j, 0, 0], [1, 1, C]))
            y = term if y is None else b.add(y, term)
        b.output("y", silu(b, y))
    else:
        filt = b.reshape(b.transpose(b.reshape(w, [TAPS, C]), [1, 0]), [C, 1, 1, TAPS])
        if style == "conv2d":
            x = b.input("x", dtype, [1, tokens + TAPS - 1, C])
            x4 = b.reshape(b.transpose(x, [0, 2, 1]), [1, C, 1, tokens + TAPS - 1])
            y = b.transpose(b.reshape(b.conv2d(x4, filt, groups=C), [1, C, tokens]), [0, 2, 1])
        else:   # conv2d_cf
            x = b.input("x", dtype, [1, C, 1, tokens + TAPS - 1])
            y = b.conv2d(x, filt, groups=C)
        b.output("y", silu(b, y))
    return b.build()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--tokens", type=int, nargs="+", default=[128, 64, 1])
    ap.add_argument("--channels", type=int, default=8192)
    ap.add_argument("--styles", nargs="+", default=["slices", "conv2d", "conv2d_cf"])
    args = ap.parse_args()
    global C
    C = args.channels
    out = Path(args.export)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    names = []
    for tokens in args.tokens:
        x = rng.standard_normal((1, tokens + TAPS - 1, C)).astype(np.float32)
        w = (rng.standard_normal((TAPS, 1, C)) * 0.5).astype(np.float32)
        reference = None
        for style in args.styles:
            for dtype in ("float16",):
                graph = build(style, tokens, dtype)
                feed_x = x if style not in ("conv2d_cf", "conv2d_cf_nosilu") else np.ascontiguousarray(np.transpose(x, (0, 2, 1))[:, :, None, :])
                if reference is None:
                    reference = interpreter.run(build("slices", tokens, "float32"), {"x": x, "w": w})["y"]   # float32 interpreter: the expected result
                want = reference if style != "conv2d_cf" else np.ascontiguousarray(np.transpose(reference, (0, 2, 1))[:, :, None, :])
                if style == "perm_matmul":
                    want = x
                if style == "silu_only":
                    want = reference[:, :, :]   # not the same function: the accuracy column of these two timing-only graphs is meaningless
                if style == "conv2d_cf_nosilu":
                    want = np.ascontiguousarray(np.transpose(reference, (0, 2, 1))[:, :, None, :])
                dt = np.float16
                serialize.dump(graph, out / f"{graph.name}.json")
                blob, entries = bytearray(), {"inputs": [], "expected": [], "weights": []}
                extra = [("perm", np.eye(tokens + TAPS - 1, dtype=np.float32)[None])] if style == "perm_matmul" else []
                for n, arr in [(n, a.astype(dt)) for n, a in extra] + [("x", feed_x.astype(dt))] + ([] if style == "perm_matmul" else [("w", w.astype(dt))]):
                    raw = np.ascontiguousarray(arr).tobytes()
                    entries["inputs"].append({"name": n, "dataType": "float16", "shape": list(arr.shape), "offset": len(blob), "bytes": len(raw)})
                    blob.extend(raw)
                raw = np.ascontiguousarray(want.astype(np.float32)).tobytes()
                entries["expected"].append({"name": "y", "dataType": "float32", "shape": list(want.shape), "offset": len(blob), "bytes": len(raw)})
                blob.extend(raw)
                (out / f"{graph.name}.mixed.bin").write_bytes(bytes(blob))
                (out / f"{graph.name}.mixed.io.json").write_text(json.dumps(entries), encoding="utf-8")
                names.append(graph.name)
    (out / "index.json").write_text(json.dumps({"graphs": names, "regimes": ["mixed"]}), encoding="utf-8")
    print("wrote", names)


if __name__ == "__main__":
    main()

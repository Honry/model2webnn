"""Experiment: can the Gated DeltaNet recurrence run as static, loop-free WebNN graphs with chunkwise prefill?  (Qwen3.5 linear attention)

  python tools/gdn_experiment.py                      accuracy of the chunkwise graphs against the token-by-token float64 oracle (interpreter: float32 and float16 graphs)
  python tools/gdn_experiment.py --export ../bench/data/gdn   also write the graphs + inputs + expected outputs for the browser page bench/raw/gdn.html (NPU time and accuracy)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2w.frontends.hf_llm.builders.gated_delta import build_core_graph, build_recurrent_graph, unit_lower_inverse, unit_lower_inverse_blocked  # noqa: E402
from m2w.ir import GraphBuilder  # noqa: E402
from m2w.ir import interpreter, serialize  # noqa: E402
from m2w.reference.gated_delta import l2norm, recurrent  # noqa: E402

DK = DV = 128


def make_inputs(rng, heads, tokens, regime, dtype_in="float32"):
    """Inputs of the recurrence as the model produces them: L2-normalised q / k, q scaled by 1/sqrt(dk), v ~ N(0, 1), beta = sigmoid, g = -exp(A_log) softplus(a + dt_bias)."""
    corr = regime.endswith("-corr")
    q = l2norm(rng.standard_normal((heads, tokens, DK))) / np.sqrt(DK)
    k = l2norm(rng.standard_normal((heads, tokens, DK)))
    if regime.endswith("-corr"):   # adversarial: nearly parallel keys (repeated text) and beta close to 1: the entries of (I + A)^-1 are largest
        k = l2norm(rng.standard_normal((heads, 1, DK)) + 0.15 * rng.standard_normal((heads, tokens, DK)))
        regime = regime[:-5]
    v = rng.standard_normal((heads, tokens, DV))
    beta = 1 / (1 + np.exp(-rng.standard_normal((heads, tokens))))
    if corr:
        beta = 1 / (1 + np.exp(-(3.0 + rng.standard_normal((heads, tokens)))))
    a_rate = {"fast": rng.uniform(0.5, 16.0, (heads, 1)), "slow": rng.uniform(0.002, 0.2, (heads, 1)), "mixed": np.exp(rng.uniform(np.log(0.002), np.log(16.0), (heads, 1)))}[regime]
    softplus = np.log1p(np.exp(rng.standard_normal((heads, tokens))))
    g = -a_rate * softplus
    return q, k, v, g, beta


def graph_feeds(inputs, state, dtype):
    q, k, v, g, beta = inputs
    dt = np.float16 if dtype == "float16" else np.float32
    return {"q": q[None].astype(dt), "k": k[None].astype(dt), "v": v[None].astype(dt), "g": g[None, :, :, None].astype(dt), "beta": beta[None, :, :, None].astype(dt), "state": state[None].astype(dt)}


def error(got, want):
    got, want = np.asarray(got, dtype=np.float64), np.asarray(want, dtype=np.float64)
    return float(np.abs(got - want).max()), float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30))


def accuracy(heads, tokens, subs, regimes, seed):
    print(f"accuracy against the float64 token loop (H={heads}, T={tokens}); max abs error / relative L2 error of the output and of the final state")
    for regime in regimes:
        rng = np.random.default_rng(seed)
        inputs = make_inputs(rng, heads, tokens, regime)
        state0 = rng.standard_normal((heads, DK, DV)) * 0.1
        out_ref, state_ref = recurrent(*inputs, state0)
        print(f"  decay regime '{regime}' (|out| max {np.abs(out_ref).max():.2f}, |state| max {np.abs(state_ref).max():.2f}, mean g {inputs[3].mean():.3f}, min g {inputs[3].min():.2f})")
        for dtype in ("float32", "float16"):
            for sub, base in subs:
                graph = build_core_graph(heads, DK, DV, tokens, sub, dtype, base=base)
                got = interpreter.run(graph, graph_feeds(inputs, state0, dtype))
                eo, es = error(got["out"][0], out_ref), error(got["new_state"][0], state_ref)
                finite = bool(np.isfinite(np.asarray(got["out"], dtype=np.float32)).all())
                print(f"    {dtype:8} sub-chunk {sub:3} base {min(base, sub):3} ({len(graph.nodes):4} nodes): out abs {eo[0]:.2e} rel {eo[1]:.2e} | state abs {es[0]:.2e} rel {es[1]:.2e}{'' if finite else '   NOT FINITE'}")


def long_run(heads, calls, tokens, config, regime, seed):
    sub, base = config
    """The state travels from call to call in the precision of the graph (float16: stored as float16): how does the error grow over a long prompt?"""
    print(f"long prompt: {calls} calls of {tokens} tokens (sub-chunk {sub}, regime '{regime}', H={heads}); relative error of the output / of the state after each call")
    rng = np.random.default_rng(seed)
    state_ref = rng.standard_normal((heads, DK, DV)) * 0.1
    states = {dtype: state_ref.astype(np.float16 if dtype == "float16" else np.float32)[None] for dtype in ("float32", "float16")}
    graphs = {dtype: build_core_graph(heads, DK, DV, tokens, sub, dtype, base=base) for dtype in states}
    for call in range(calls):
        inputs = make_inputs(rng, heads, tokens, regime)
        out_ref, state_ref = recurrent(*inputs, state_ref)
        line = f"  call {call + 1:2}:"
        for dtype, graph in graphs.items():
            feeds = graph_feeds(inputs, states[dtype][0], dtype)
            got = interpreter.run(graph, feeds)
            states[dtype] = got["new_state"]
            line += f"  {dtype} out {error(got['out'][0], out_ref)[1]:.2e} state {error(got['new_state'][0], state_ref)[1]:.2e}"
        print(line)


def matmul_shape_graphs(dtype):
    """Which matmul shapes does the NPU accept? (the 8 x 8 blocks of the blocked inverse did not compile)"""
    out = []
    for batch, m, k, n in [(32, 8, 8, 8), (128, 8, 8, 8), (32, 4, 4, 4), (32, 16, 16, 16), (32, 1, 16, 256), (32, 16, 16, 256), (32, 1, 32, 256), (32, 32, 32, 256), (32, 8, 8, 256), (128, 16, 16, 16)]:
        b = GraphBuilder(f"part_matmul_{batch}x{m}x{k}x{n}_{dtype}")
        b.output("y", b.matmul(b.input("x", dtype, [1, batch, m, k]), b.constant(dtype, [1, batch, k, n], fill=0.01)))
        out.append(b.build())
    return out


def part_graphs(heads, dtype):
    """Pieces of the blocked triangular inverse as graphs of their own: the NPU failed to create a session for the whole thing, which operator pattern is it?"""
    out = []

    def graph(name, shape, body):
        b = GraphBuilder(f"part_{name}_{dtype}")
        x = b.input("x", dtype, shape)
        b.output("y", body(b, x))
        out.append(b.build())

    out += matmul_shape_graphs(dtype)

    def graph_multi(name, shapes, body):
        b = GraphBuilder(f"part_{name}_{dtype}")
        values = {n: b.input(n, dtype, shape) for n, shape in shapes.items()}
        for output_name, value in body(b, values).items():
            b.output(output_name, value)
        out.append(b.build())

    # bisect 1: which pattern of the Neumann inverse makes the NPU fail to create a session (8 x 8 batched 128 did, 16 x 16 and 32 x 32 did not)?
    for c in (4, 8, 16):
        for batch in (32, 128):
            graph(f"inv_c{c}_b{batch}", [1, batch, c, c], lambda b, x, c=c: unit_lower_inverse(b, x, c, dtype))
    eye8 = lambda b: _identity_constant(b, 8, dtype)
    graph("sq_c8", [1, 32, 8, 8], lambda b, x: b.matmul(x, x))
    graph("sqadd_c8", [1, 32, 8, 8], lambda b, x: b.add(b.matmul(x, x), eye8(b)))
    graph("prod_c8", [1, 32, 8, 8], lambda b, x: b.matmul(b.add(x, eye8(b)), b.add(x, eye8(b))))
    graph("prod2_c8", [1, 32, 8, 8], lambda b, x: b.matmul(b.add(x, eye8(b)), b.add(b.matmul(x, x), eye8(b))))
    # bisect 2: the decode step is WRONG on the NPU when decayed = state * exp(g) feeds the prediction and the new state (style A); which part of that pattern?
    h, step = 32, {"state": [1, 32, 128, 128], "k": [1, 32, 1, 128], "g": [1, 32, 1, 1], "delta": [1, 32, 1, 128]}
    graph_multi("step_pred_only", {n: step[n] for n in ("state", "k", "g")}, lambda b, v: {"y": b.matmul(v["k"], b.mul(v["state"], b.exp(v["g"])))})
    graph_multi("step_new_only", step, lambda b, v: {"y": b.add(b.mul(v["state"], b.exp(v["g"])), b.matmul(b.transpose(v["k"], [0, 1, 3, 2]), v["delta"]))})
    graph_multi("step_shared", step, lambda b, v: (lambda d: {"y": b.matmul(v["k"], d), "z": b.add(d, b.matmul(b.transpose(v["k"], [0, 1, 3, 2]), v["delta"]))})(b.mul(v["state"], b.exp(v["g"]))))
    graph_multi("step_shared_noexp", step, lambda b, v: (lambda d: {"y": b.matmul(v["k"], d), "z": b.add(d, b.matmul(b.transpose(v["k"], [0, 1, 3, 2]), v["delta"]))})(b.mul(v["state"], v["g"])))
    graph_multi("step_shared_chain", step, lambda b, v: (lambda d: {"z": b.add(d, b.matmul(b.transpose(v["k"], [0, 1, 3, 2]), b.sub(v["delta"], b.matmul(v["k"], d))))})(b.mul(v["state"], b.exp(v["g"]))))
    graph_multi("step_scale_only", {"state": step["state"], "g": step["g"]}, lambda b, v: {"y": b.mul(v["state"], b.exp(v["g"]))})
    graph("diag_slices_c32", [1, heads, 32, 32], lambda b, x: b.concat([b.slice(x, [0, 0, i * 8, i * 8], [1, heads, 8, 8]) for i in range(4)], 1))
    graph("slice_heads", [1, 4 * heads, 8, 8], lambda b, x: b.slice(x, [0, heads, 0, 0], [1, heads, 8, 8]))
    graph("zero_concat", [1, heads, 8, 8], lambda b, x: b.concat([x, b.constant(dtype, [1, heads, 8, 24], fill=0.0)], 3))
    graph("inverse8_batched", [1, 4 * heads, 8, 8], lambda b, x: unit_lower_inverse(b, x, 8, dtype))
    graph("inverse16", [1, heads, 16, 16], lambda b, x: unit_lower_inverse(b, x, 16, dtype))
    graph("inverse32_plain", [1, heads, 32, 32], lambda b, x: unit_lower_inverse(b, x, 32, dtype))
    graph("inverse32_blocked8", [1, heads, 32, 32], lambda b, x: unit_lower_inverse_blocked(b, x, 32, 8, dtype))
    graph("inverse16_blocked8", [1, heads, 16, 16], lambda b, x: unit_lower_inverse_blocked(b, x, 16, 8, dtype))
    return out


def _identity_constant(b, c, dtype):
    from m2w.frontends.hf_llm.builders.gated_delta import _identity
    return _identity(b, c, dtype)


def build_part_twin(name):
    base = name.rsplit("_", 1)[0]
    return next(g for g in part_graphs(32, "float32") if g.name == f"{base}_float32")


def build_graph_like(graph, dtype):
    """The float32 twin of a probe graph (parameters read back from its inputs and meta)."""
    heads, tokens = graph.inputs["q"].shape[1], graph.inputs["q"].shape[2]
    if tokens == 1:
        return build_recurrent_graph(heads, DK, DV, dtype, probe=True, style=graph.name[len("gdn_step")] if graph.name[len("gdn_step")] in "ABC" else "A")
    return build_core_graph(heads, DK, DV, tokens, graph.meta["sub"], dtype, base=graph.meta["base"], probe=True, solve=graph.meta["solve"])


def export(directory: Path, heads, tokens, subs, regimes, seed):
    directory.mkdir(parents=True, exist_ok=True)
    index = {"graphs": [], "regimes": regimes}
    for regime in regimes:
        rng = np.random.default_rng(seed)
        inputs = make_inputs(rng, heads, tokens, regime)
        state0 = rng.standard_normal((heads, DK, DV)) * 0.1
        out_ref, state_ref = recurrent(*inputs, state0)
        decode_inputs = tuple(x[:, :1] for x in inputs)
        out1, state1 = recurrent(*decode_inputs, state0)
        for dtype in ("float32", "float16"):
            cases = [(build_core_graph(heads, DK, DV, tokens, sub, dtype, base=base), inputs, out_ref, state_ref) for sub, base in subs] + [(build_recurrent_graph(heads, DK, DV, dtype, style=style), decode_inputs, out1, state1) for style in "ABCDEFG"] + [(build_core_graph(heads, DK, DV, tokens, sub, dtype, solve="substitution"), inputs, out_ref, state_ref) for sub in (16, 32)] + [(build_core_graph(heads, DK, DV, tokens, sub, dtype, solve="batched"), inputs, out_ref, state_ref) for sub in (16, 32, 64)]
            # probe graphs: the same graph with the intermediate results of the first sub-chunk as extra outputs; the expected values are the float32 interpreter's
            probes = [(build_core_graph(heads, DK, DV, tokens, sub, dtype, base=base, probe=True), inputs, out_ref, state_ref) for sub, base in subs[:1] + subs[-1:]] + [(build_recurrent_graph(heads, DK, DV, dtype, probe=True, style=style), decode_inputs, out1, state1) for style in "AB"] + [(build_core_graph(heads, DK, DV, tokens, 16, dtype, probe=True, solve="substitution"), inputs, out_ref, state_ref)]
            parts = [(g, None, None, None) for g in part_graphs(heads, dtype)]
            for graph, ins, want_out, want_state in cases + probes + parts:
                serialize.dump(graph, directory / f"{graph.name}.json")
                if ins is None:   # a part: random inputs (strictly lower-triangular x, negative g), the expected results are the float32 interpreter's
                    f32 = {}
                    for name, spec in graph.inputs.items():
                        r = rng.standard_normal(spec.shape) * 0.5
                        f32[name] = np.tril(r * 0.4, -1) if name == "x" else (-np.abs(r) if name == "g" else r)
                    feeds = {n: x.astype(np.float16 if dtype == "float16" else np.float32) for n, x in f32.items()}
                    reference = interpreter.run(build_part_twin(graph.name), {n: x.astype(np.float32) for n, x in f32.items()})
                    want_out = want_state = None
                else:
                    feeds = graph_feeds(ins, state0, dtype)
                blob, entries = bytearray(), {"inputs": [], "expected": []}
                for name in graph.inputs:
                    raw = np.ascontiguousarray(feeds[name]).tobytes()
                    entries["inputs"].append({"name": name, "dataType": dtype, "shape": list(feeds[name].shape), "offset": len(blob), "bytes": len(raw)}); blob += raw
                expected = {"out": want_out[None], "new_state": want_state[None]} if ins is not None else dict(reference)
                if any(n.startswith("p_") for n in graph.outputs):
                    reference = interpreter.run(build_graph_like(graph, "float32"), graph_feeds(ins, state0, "float32"))
                    expected.update({n: reference[n] for n in graph.outputs if n.startswith("p_")})
                for name, want in expected.items():
                    raw = np.ascontiguousarray(want.astype(np.float32)).tobytes()
                    entries["expected"].append({"name": name, "dataType": "float32", "shape": list(want.shape), "offset": len(blob), "bytes": len(raw)}); blob += raw
                (directory / f"{graph.name}.{regime}.bin").write_bytes(bytes(blob))
                (directory / f"{graph.name}.{regime}.io.json").write_text(json.dumps(entries), encoding="utf-8")
                if graph.name not in index["graphs"]:
                    index["graphs"].append(graph.name)
    (directory / "index.json").write_text(json.dumps(index), encoding="utf-8")
    print(f"wrote {len(index['graphs'])} graphs x {len(regimes)} datasets to {directory}: {index['graphs']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=128)
    ap.add_argument("--subs", default="16,32,64", help="sub-chunk sizes, each optionally with the size of the Neumann base blocks: 64:8")
    ap.add_argument("--long-regime", default="mixed")
    ap.add_argument("--regimes", default="fast,mixed,slow")
    ap.add_argument("--long", type=int, default=0, help="number of consecutive calls for the long-prompt drift test")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    subs = [(int(x.split(":")[0]), int(x.split(":")[1]) if ":" in x else 16) for x in args.subs.split(",")]
    if args.export:
        export(Path(args.export), 32, args.tokens, subs, ["mixed", "mixed-corr"], args.seed)
        return
    accuracy(args.heads, args.tokens, subs, args.regimes.split(","), args.seed)
    if args.long:
        long_run(args.heads, args.long, args.tokens, subs[-1], args.long_regime, args.seed)


if __name__ == "__main__":
    main()

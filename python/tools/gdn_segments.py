"""Time the pieces of the chunkwise Gated DeltaNet core (gated_delta_batched, 128 tokens x 32 heads, sub-chunk 16) as graphs of their own, on the device, to see which piece costs what:

  decay       gc = cumulative sum of g (matmul with a lower-triangular constant), D = exp(gc_i - gc_j) masked                       [all 256 head x sub-chunk batches]
  a_matrix    A = strict-lower((beta k) k^T * D)
  rhs         [beta v | beta k exp(gc)]
  subst       (I + A)^-1 rhs by 16 dependent row steps
  products    q_gate = q exp(gc), attn = (q k^T) * D, k_decayed^T, total decay
  state_loop  the 8 dependent sub-chunk steps carrying the state (value / k_cumdecay / q_gate / attn / k_decayed^T / total -> out, new state)
  whole       the complete core (what the package has)
  overhead    a one-operator graph: the fixed cost of a dispatch
Every segment gets its inputs from the float32 interpreter run of the previous ones, so the data is realistic; the accuracy column is against the float32 interpreter.

  python tools/gdn_segments.py --export ../bench/data/gdn_seg      then   bench/raw/gdn.html?dir=../data/gdn_seg/&data=mixed&chain=20
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gdn_experiment import DK, DV, graph_feeds, make_inputs  # noqa: E402
from m2w.frontends.hf_llm.builders.gated_delta import _tri, build_core_graph, forward_substitution, solve_block_diagonal  # noqa: E402
from m2w.ir import GraphBuilder, interpreter, serialize  # noqa: E402

H, T, SUB = 32, 128, 16
N = T // SUB
HN = H * N


def folded(b, name, width, dtype):
    return b.input(name, dtype, [1, HN, SUB, width])


def segment_graphs(dtype):
    out = {}

    b = GraphBuilder(f"seg_decay_{dtype}")
    g = b.input("g", dtype, [1, HN, SUB, 1])
    incl = _tri(b, SUB, dtype, False)
    gc = b.matmul(incl, g)
    decay = b.mul(b.exp(b.mul(b.sub(gc, b.transpose(gc, [0, 1, 3, 2])), incl)), incl)
    b.output("gc", gc); b.output("decay", decay)
    out["decay"] = b.build()

    b = GraphBuilder(f"seg_a_matrix_{dtype}")
    k, beta, decay = folded(b, "k", DK, dtype), folded(b, "beta", 1, dtype), b.input("decay", dtype, [1, HN, SUB, SUB])
    k_beta = b.mul(k, beta)
    b.output("a", b.mul(b.mul(b.matmul(k_beta, b.transpose(k, [0, 1, 3, 2])), decay), _tri(b, SUB, dtype, True)))
    out["a_matrix"] = b.build()

    b = GraphBuilder(f"seg_rhs_{dtype}")
    k, v, beta, gc = folded(b, "k", DK, dtype), folded(b, "v", DV, dtype), folded(b, "beta", 1, dtype), folded(b, "gc", 1, dtype)
    k_beta = b.mul(k, beta)
    b.output("rhs", b.concat([b.mul(v, beta), b.mul(k_beta, b.exp(gc))], 3))
    out["rhs"] = b.build()

    b = GraphBuilder(f"seg_subst_{dtype}")
    a, rhs = b.input("a", dtype, [1, HN, SUB, SUB]), b.input("rhs", dtype, [1, HN, SUB, DV + DK])
    b.output("solved", forward_substitution(b, a, rhs, dtype))
    out["subst"] = b.build()

    b = GraphBuilder(f"seg_subst_blockdiag_{dtype}")   # the same solve without the 16 dependent steps
    a, rhs = b.input("a", dtype, [1, HN, SUB, SUB]), b.input("rhs", dtype, [1, HN, SUB, DV + DK])
    b.output("solved", solve_block_diagonal(b, a, rhs, dtype, SUB // 2))
    out["subst_blockdiag"] = b.build()

    for variant in ("dup", "aug", "probed", "inputs"):   # ways to write the same solve; which one does the NPU compile correctly?
        b = GraphBuilder(f"seg_bd_{variant}_{dtype}")
        a, rhs = b.input("a", dtype, [1, HN, SUB, SUB]), b.input("rhs", dtype, [1, HN, SUB, DV + DK])
        probes = [] if variant == "probed" else None
        split = (b.input("n", dtype, [1, HN, SUB, SUB]), b.input("al", dtype, [1, HN, SUB, SUB])) if variant == "inputs" else None
        solved = solve_block_diagonal(b, a, rhs, dtype, SUB // 2, style="plain" if variant in ("probed", "inputs") else variant, probes=probes, split_inputs=split)
        b.output("solved", solved)
        for i, p in enumerate(probes or []):   # every intermediate result is also an output: an extra consumer of each (what made the earlier wrong graphs right)
            b.output(f"probe{i}", p)
        out[f"bd_{variant}"] = b.build()

    b = GraphBuilder(f"seg_products_{dtype}")
    q, k, gc, decay = folded(b, "q", DK, dtype), folded(b, "k", DK, dtype), folded(b, "gc", 1, dtype), b.input("decay", dtype, [1, HN, SUB, SUB])
    k_t = b.transpose(k, [0, 1, 3, 2])
    g_last = b.slice(gc, [0, 0, SUB - 1, 0], [1, HN, 1, 1])
    b.output("q_gate", b.mul(q, b.exp(gc)))
    b.output("attn", b.mul(b.matmul(q, k_t), decay))
    b.output("k_decayed_t", b.transpose(b.mul(k, b.exp(b.sub(g_last, gc))), [0, 1, 3, 2]))
    b.output("total", b.exp(g_last))
    out["products"] = b.build()

    b = GraphBuilder(f"seg_state_loop_{dtype}")
    value, kcd = b.input("value", dtype, [1, H, N, SUB, DV]), b.input("k_cumdecay", dtype, [1, H, N, SUB, DK])
    qg, attn = b.input("q_gate", dtype, [1, H, N, SUB, DK]), b.input("attn", dtype, [1, H, N, SUB, SUB])
    kdt, total = b.input("k_decayed_t", dtype, [1, H, N, DK, SUB]), b.input("total", dtype, [1, H, N, 1, 1])
    state = b.reshape(b.input("state", dtype, [1, H, DK, DV], "kv"), [1, H, 1, DK, DV])
    pick = lambda x, s: b.slice(x, [0, 0, s, 0, 0], [1, H, 1] + list(x.shape[3:]))  # noqa: E731
    outs = []
    for s in range(N):
        v_new = b.sub(pick(value, s), b.matmul(pick(kcd, s), state))
        outs.append(b.add(b.matmul(pick(qg, s), state), b.matmul(pick(attn, s), v_new)))
        state = b.add(b.mul(state, pick(total, s)), b.matmul(pick(kdt, s), v_new))
    b.output("out", b.concat(outs, 2))
    b.output("new_state", b.reshape(state, [1, H, DK, DV]))
    out["state_loop"] = b.build()

    out["whole"] = build_core_graph(H, DK, DV, T, SUB, dtype, name=f"seg_whole_{dtype}", solve="batched")
    out["whole_blockdiag"] = build_core_graph(H, DK, DV, T, SUB, dtype, name=f"seg_whole_blockdiag_{dtype}", solve="blockdiag")

    b = GraphBuilder(f"seg_overhead_{dtype}")
    b.output("y", b.add(b.input("x", dtype, [1, 1, 16, 16]), b.scalar(1.0, dtype, 4)))
    out["overhead"] = b.build()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    args = ap.parse_args()
    out = Path(args.export)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    q, k, v, g, beta = make_inputs(rng, H, T, "mixed")
    state0 = rng.standard_normal((H, DK, DV)) * 0.1
    f32, f16 = segment_graphs("float32"), segment_graphs("float16")
    fold = lambda x, w: x.reshape(1, H * N, SUB, w).astype(np.float32)   # [H, T, w] -> [1, H*n, C, w]
    values = {"q": fold(q, DK), "k": fold(k, DK), "v": fold(v, DV), "g": fold(g[:, :, None], 1), "beta": fold(beta[:, :, None], 1)}
    run = lambda name, feeds: interpreter.run(f32[name], feeds)   # noqa: E731
    r = run("decay", {"g": values["g"]}); values.update(r)
    r = run("a_matrix", {"k": values["k"], "beta": values["beta"], "decay": values["decay"]}); values.update(r)
    r = run("rhs", {n: values[n] for n in ("k", "v", "beta", "gc")}); values.update(r)
    r = run("subst", {"a": values["a"], "rhs": values["rhs"]}); values.update(r)
    solved = values["solved"]
    unfold = lambda x: x.reshape((1, H, N) + x.shape[2:])   # [1, H*n, r, c] -> [1, H, n, r, c]
    values["value"], values["k_cumdecay"] = solved[..., :DV], solved[..., DV:]
    r = run("products", {n: values[n] for n in ("q", "k", "gc", "decay")}); values.update(r)
    loop_feeds = {"value": unfold(values["value"]), "k_cumdecay": unfold(values["k_cumdecay"]), "q_gate": unfold(values["q_gate"]), "attn": unfold(values["attn"]),
                  "k_decayed_t": unfold(values["k_decayed_t"]), "total": unfold(values["total"]), "state": state0[None].astype(np.float32)}
    sub_mask = np.zeros((SUB, SUB), np.float32)
    for lo in (0, SUB // 2):
        sub_mask[lo:lo + SUB // 2, lo:lo + SUB // 2] = np.tril(np.ones((SUB // 2, SUB // 2)), -1)
    ll_mask = np.zeros((SUB, SUB), np.float32)
    ll_mask[SUB // 2:, :SUB // 2] = 1
    split_feeds = {"a": values["a"], "rhs": values["rhs"], "n": -values["a"] * sub_mask, "al": values["a"] * ll_mask}   # only the graph that declares n / al uses them
    feeds_by_segment = {
        "decay": {"g": values["g"]}, "a_matrix": {n: values[n] for n in ("k", "beta", "decay")}, "rhs": {n: values[n] for n in ("k", "v", "beta", "gc")},
        "subst": {"a": values["a"], "rhs": values["rhs"]}, "subst_blockdiag": {"a": values["a"], "rhs": values["rhs"]}, **{f"bd_{v}": split_feeds for v in ("dup", "aug", "probed", "inputs")}, "products": {n: values[n] for n in ("q", "k", "gc", "decay")}, "state_loop": loop_feeds,
        "whole": graph_feeds((q, k, v, g, beta), state0, "float32"), "whole_blockdiag": graph_feeds((q, k, v, g, beta), state0, "float32"), "overhead": {"x": np.zeros((1, 1, 16, 16), np.float32)},
    }
    names = []
    for segment, feeds in feeds_by_segment.items():
        graph = f16[segment]
        feeds = {n: v for n, v in feeds.items() if n in graph.inputs}
        want = interpreter.run(f32[segment], feeds)
        serialize.dump(graph, out / f"{graph.name}.json")
        blob, entries = bytearray(), {"inputs": [], "expected": [], "weights": []}
        for n in graph.inputs:
            raw = np.ascontiguousarray(feeds[n]).astype(np.float16).tobytes()
            entries["inputs"].append({"name": n, "dataType": "float16", "shape": list(graph.inputs[n].shape), "offset": len(blob), "bytes": len(raw)})
            blob.extend(raw)
        for n, a in want.items():
            raw = np.ascontiguousarray(a.astype(np.float32)).tobytes()
            entries["expected"].append({"name": n, "dataType": "float32", "shape": list(a.shape), "offset": len(blob), "bytes": len(raw)})
            blob.extend(raw)
        (out / f"{graph.name}.mixed.bin").write_bytes(bytes(blob))
        (out / f"{graph.name}.mixed.io.json").write_text(json.dumps(entries), encoding="utf-8")
        names.append(graph.name)
        print(f"{segment:12} {len(graph.nodes):4} nodes")
    (out / "index.json").write_text(json.dumps({"graphs": names, "regimes": ["mixed"]}), encoding="utf-8")


if __name__ == "__main__":
    main()

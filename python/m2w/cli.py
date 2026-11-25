from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .frontends.hf_llm.builders import decoder as D
from .ir import serialize
from .presets import PRESETS


def cmd_compile(args) -> int:
    from .compile import compile_model

    out = compile_model(args.model, args.out, max_lengths=args.max_length, chunks=args.chunk, kv_styles=args.kv_style, staging_rows=args.staging_rows,
                        compute_types=args.compute_type, blocks_per_graph=args.blocks_per_graph, prefill_blocks_per_graph=args.prefill_blocks_per_graph,
                        merge_projections=not args.no_merge, lm_head_format=args.lm_head_format, search=not args.no_search, workers=args.workers,
                        endpoint=args.endpoint, cache_dir=args.cache_dir)
    print(f"\nNext: m2w verify {out} --source {args.model}")
    return 0


def cmd_rebuild_graphs(args) -> int:
    from .compile import rebuild_graphs

    rebuild_graphs(args.package, max_lengths=args.max_length, chunks=args.chunk, kv_styles=args.kv_style, staging_rows=args.staging_rows, compute_types=args.compute_type,
                   blocks_per_graph=args.blocks_per_graph, prefill_blocks_per_graph=args.prefill_blocks_per_graph)
    return 0


def cmd_inspect(args) -> int:
    from .frontends.hf_llm import arch, hub

    source = hub.resolve_model(args.model, cache_dir=args.cache_dir, endpoint=args.endpoint, weights=False)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    generation = json.loads((source / "generation_config.json").read_text(encoding="utf-8")) if (source / "generation_config.json").exists() else {}
    try:
        spec = arch.spec_from_config(config, generation)
    except arch.UnsupportedModel as e:
        print(f"NOT SUPPORTED: {e}")
        return 1
    dims = spec.decoder_dims()
    per_layer = (dims.hidden * (dims.attention_width + 2 * dims.kv_width) + dims.attention_width * dims.hidden + 3 * dims.hidden * dims.ffn)
    weights = spec.num_layers * per_layer * (0.5 + 2 / dims.group_size) + spec.vocab_padded * spec.hidden * 1.0
    kinds = {t: spec.layer_types.count(t) for t in sorted(set(spec.layer_types))}
    print(f"{spec.architecture}  ->  supported (family {spec.family})")
    print(f"  layers {spec.num_layers}  hidden {spec.hidden}  heads {spec.num_heads}/{spec.num_kv_heads}  head_dim {spec.head_dim}  ffn {spec.ffn}")
    print(f"  vocab {spec.vocab} (padded {spec.vocab_padded})  tied embedding {spec.tie_embeddings}  qk-norm {spec.qk_norm}")
    print(f"  layer types {kinds}" + (f"  sliding window {spec.sliding_window}" if spec.sliding_window else ""))
    print(f"  rope {json.dumps(spec.rope)}")
    print(f"  eos {spec.eos_token_ids}  max_position_embeddings {spec.max_position_embeddings}")
    print(f"  estimated package: ~{weights / 2**20:.0f} MB of weights (int4 g{spec.group_size} + u8 embedding), {spec.num_layers + 4} weight files")
    print(f"  compile with:  m2w compile {args.model} --out <dir> --max-length 1152")
    return 0


def cmd_verify(args) -> int:
    from .verify import verify_package

    ids = [int(t) for t in args.token_ids.split(",")] if args.token_ids else None
    return verify_package(Path(args.package), args.source, prompt=args.prompt, steps=args.steps, max_length=args.max_length, chunks=args.chunk, variants=args.variant,
                          decode_group=args.blocks_per_graph, prefill_group=args.prefill_blocks_per_graph, endpoint=args.endpoint, cache_dir=args.cache_dir,
                          skip_float_reference=args.skip_float_reference, token_ids=ids)


def cmd_graphs(args) -> int:
    """Write the subgraphs of a preset as graph JSON (spike: no weights, no plan)."""
    preset = PRESETS[args.preset]
    dims = preset["dims"]
    opt = D.GraphOptions(compute_type=args.compute_type, merge_projections=not args.no_merge, kv_style=args.kv_style, staging_rows=args.staging_rows)
    past = args.max_length if opt.kv_style == "tensorScatter" else args.max_length - 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    graphs = {
        "decode_head": D.build_head_graph(dims, opt, seq=1, past_len=past, name="decode_head"),
        "decode_tail": D.build_tail_graph(dims, opt, seq=1, name="decode_tail"),
        "lm_head": D.build_lm_head_graph(dims, opt),
        "kv_flush": D.build_kv_update_graph(dims, seq=args.staging_rows, past_len=past, name="kv_flush"),
        f"prefill_head_c{args.chunk}": D.build_head_graph(dims, opt, seq=args.chunk, past_len=past, name=f"prefill_head_c{args.chunk}"),
        f"prefill_tail_c{args.chunk}": D.build_tail_graph(dims, opt, seq=args.chunk, name=f"prefill_tail_c{args.chunk}"),
    }
    for k in {1, *args.blocks_per_graph}:
        graphs["decode_block" if k == 1 else f"decode_block_x{k}"] = (
            D.build_block_graph(dims, opt, seq=1, past_len=past, name="decode_block") if k == 1
            else D.build_block_group_graph(dims, opt, block_count=k, seq=1, past_len=past))
    graphs[f"prefill_block_c{args.chunk}"] = D.build_block_graph(dims, opt, seq=args.chunk, past_len=past, name=f"prefill_block_c{args.chunk}")
    for name, g in graphs.items():
        serialize.dump(g, out / f"{name}.json")
        print(f"{name}: {len(g.nodes)} nodes, {len(g.inputs)} inputs, {len(g.outputs)} outputs")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="m2w", description="Compile Hugging Face LLMs into static-shape WebNN graphs + weights")
    sub = p.add_subparsers(dest="command", required=True)

    def graph_options(sp, defaults=True):
        """The graph variants of a package: kv style x staging rows (staged only) x compute type; the first of each list is the default. Weights are shared by all variants."""
        sp.add_argument("--max-length", type=int, nargs="+", default=[1152] if defaults else None, help="KV cache capacities to build graphs for (static shapes)")
        sp.add_argument("--chunk", type=int, nargs="+", default=[64] if defaults else None, help="prefill chunk sizes (several: the runtime plans the chunk sequence of a prompt)")
        sp.add_argument("--kv-style", nargs="+", choices=["staged", "onehot", "tensorScatter"], default=["staged"], help="KV cache protocols (see plan.json kv_protocols)")
        sp.add_argument("--staging-rows", type=int, nargs="+", default=[64], help="rows of the staging buffer (staged only)")
        sp.add_argument("--compute-type", nargs="+", choices=["float32", "float16"], default=["float32"], help="type of the computation inside the graphs (the boundaries stay f16)")
        sp.add_argument("--blocks-per-graph", type=int, nargs="+", default=[1], help="also build decode graphs with K blocks each (fewer dispatches, longer compile)")
        sp.add_argument("--prefill-blocks-per-graph", type=int, nargs="+", default=[1], help="the same for the prefill graphs")

    def source_args(sp):
        sp.add_argument("--endpoint", help="Hugging Face endpoint (default: $HF_ENDPOINT or https://huggingface.co; e.g. https://hf-mirror.com)")
        sp.add_argument("--cache-dir", help="download cache (default: ~/.cache/m2w)")

    c = sub.add_parser("compile", help="HF model (repo id or local directory) -> model package")
    c.add_argument("model")
    c.add_argument("--out", required=True)
    graph_options(c)
    c.add_argument("--no-merge", action="store_true", help="keep q/k/v and gate/up as separate projections")
    c.add_argument("--lm-head-format", choices=["int8-sym", "uint8-asym"], default="int8-sym", help="tied embedding / lm_head table: int8 symmetric per row (default) or uint8 asymmetric per row (the OpenVINO export); same accuracy, see tools/lm_head_formats.py")
    c.add_argument("--no-search", action="store_true", help="plain absmax int4 scales (faster, a little less accurate)")
    c.add_argument("--workers", type=int, default=4)
    source_args(c)
    c.set_defaults(func=cmd_compile)

    rb = sub.add_parser("rebuild-graphs", help="rewrite the graphs + plan of an existing package with other variants (the weights stay: no quantization)")
    rb.add_argument("package")
    graph_options(rb, defaults=False)
    rb.set_defaults(func=cmd_rebuild_graphs)

    i = sub.add_parser("inspect", help="what would be built for this model (downloads only config files)")
    i.add_argument("model")
    source_args(i)
    i.set_defaults(func=cmd_inspect)

    v = sub.add_parser("verify", help="run the compiled package on the CPU and compare with the original model")
    v.add_argument("package")
    v.add_argument("--source", required=True, help="the HF model the package was compiled from (repo id or directory)")
    v.add_argument("--prompt", default="What is the capital of France?")
    v.add_argument("--steps", type=int, default=6, help="greedy tokens to compare")
    v.add_argument("--token-ids", help="comma separated prompt token ids (instead of --prompt; for models without a tokenizer)")
    v.add_argument("--max-length", type=int, default=0, help="0 = the smallest max length of the package")
    v.add_argument("--chunk", type=int, nargs="+", default=None, help="prefill chunk sizes to use (default: the smallest chunk of the package); several = planned per prompt")
    v.add_argument("--variant", nargs="+", default=None, help="graph variants to run (default: the default variant of the package); 'all' = every variant")
    v.add_argument("--blocks-per-graph", type=int, default=1, help="run the decode graphs with K blocks each (the package must contain them)")
    v.add_argument("--prefill-blocks-per-graph", type=int, default=1, help="the same for the prefill graphs")
    v.add_argument("--skip-float-reference", action="store_true", help="only compare with the dequantized weights (skips the slow float32 reference)")
    source_args(v)
    v.set_defaults(func=cmd_verify)

    g = sub.add_parser("graphs", help="write the subgraphs of a preset as graph JSON (spike)")
    g.add_argument("--preset", choices=sorted(PRESETS), default="qwen3-4b")
    g.add_argument("--out", required=True)
    g.add_argument("--max-length", type=int, default=1152)
    g.add_argument("--chunk", type=int, default=64)
    g.add_argument("--staging-rows", type=int, default=64)
    g.add_argument("--kv-style", choices=["staged", "onehot", "tensorScatter"], default="staged")
    g.add_argument("--compute-type", choices=["float32", "float16"], default="float32")
    g.add_argument("--blocks-per-graph", type=int, nargs="*", default=[])
    g.add_argument("--no-merge", action="store_true")
    g.set_defaults(func=cmd_graphs)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

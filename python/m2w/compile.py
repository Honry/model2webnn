"""m2w compile: a Hugging Face decoder -> a model package for the static NPUW-style WebNN runtime.

    <out>/m2w.json            the single entry point: format, compiler version, target, requires, file index (sizes, sha256), model + plan summary
    <out>/model.json          ModelSpec (architecture, dims, layer types, RoPE per layer type, tokens)
    <out>/plan.json           which graph for which phase / max length / chunk, how weights and per-layer-type inputs are bound
    <out>/chat_format.json    prompt pieces derived from the chat template
    <out>/graphs/*.json       m2w-graph documents (spec/graph.schema.json)
    <out>/weights/            manifest.json + per-layer shards (hf_llm/pack.py)
    <out>/tokenizer/          tokenizer.json, tokenizer_config.json, generation_config.json, ...
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from . import __version__
from .frontends.hf_llm import arch, chat_format, hub, pack, pack_gemma4, pack_qwen35
from .frontends.hf_llm.builders import decoder as D
from .frontends.hf_llm.builders import gemma4 as G
from .frontends.hf_llm.builders import qwen35 as Q
from .frontends.hf_llm.safetensors_io import TensorStore
from .ir import check_tensor_limit, serialize

TARGET = "webnn-ov-npu"
PACKAGE_FORMAT = "m2w-package"
PACKAGE_VERSION = 0
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "chat_template.jinja", "chat_template.json",
                   "vocab.json", "merges.txt")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _sequence(value) -> tuple:
    return tuple(value) if isinstance(value, (list, tuple)) else (value,)


def variant_id(kv_style: str, staging_rows: int, compute_type: str) -> str:
    return f"staged-r{staging_rows}-{compute_type}" if kv_style == "staged" else f"{kv_style}-{compute_type}"


def expand_variants(kv_styles, staging_rows, compute_types) -> list:
    """The graph variants of a package: kv style x staging rows (staged only) x compute type. Weights are shared by all of them."""
    variants = []
    for kv in _sequence(kv_styles):
        if kv not in ("staged", "onehot", "tensorScatter", "scatterND"):
            raise ValueError(f"unknown kv style '{kv}' (staged | onehot | tensorScatter | scatterND)")
        for ct in _sequence(compute_types):
            for rows in (_sequence(staging_rows) if kv == "staged" else (_sequence(staging_rows)[0],)):
                v = {"id": variant_id(kv, rows, ct), "kv_style": kv, "compute_type": ct}
                if kv == "staged":
                    v["staging_rows"] = int(rows)
                if kv == "tensorScatter":
                    v["scatter_index_type"] = "int32"
                if kv == "onehot":
                    v["value_layout"] = "transposed"   # the layout of the hand-written one-hot graphs; faster than natural on the NPU (staged / tensorScatter need natural)
                variants.append(v)
    return variants


def write_graphs(spec: arch.ModelSpec, opt: D.GraphOptions, out: Path, variants, max_lengths, chunks, blocks_per_graph, prefill_blocks_per_graph=(1,)) -> dict:
    """Write the graphs of every variant (graphs/<variant id>/) and return the plan's 'graphs' table: variant id -> lm_head / decode / prefill (names relative to the package root)."""
    dims = spec.decoder_dims()
    counts = lambda k: sorted({k, (spec.num_layers - 1) % k} - {0})   # noqa: E731
    per_block = len(set(spec.layer_types)) > 1   # block groups of a hybrid model take the rope / mask of every block as separate inputs
    tables: dict = {}
    for variant in variants:
        vopt = D.with_options(opt, kv_style=variant["kv_style"], compute_type=variant["compute_type"], staging_rows=variant.get("staging_rows", opt.staging_rows),
                           value_layout=variant.get("value_layout", "natural"))
        gdir = out / "graphs" / variant["id"]
        gdir.mkdir(parents=True, exist_ok=True)

        def emit(graph) -> str:
            check_tensor_limit(graph)   # WebNN rejects tensors above 2 GiB
            serialize.dump(graph, gdir / f"{graph.name}.json")
            return f"graphs/{variant['id']}/{graph.name}.json"

        table: dict = {"lm_head": emit(D.build_lm_head_graph(dims, vopt)), "decode": {}, "prefill": {}}
        for L in max_lengths:
            past = L - 1
            decode = {"head": emit(D.build_head_graph(dims, vopt, seq=1, past_len=past, name=f"decode_head_L{L}")),
                      "block": emit(D.build_block_graph(dims, vopt, seq=1, past_len=past, name=f"decode_block_L{L}")),
                      "tail": emit(D.build_tail_graph(dims, vopt, seq=1, name="decode_tail")), "block_groups": {}}
            if vopt.kv_style == "staged":
                decode["flush"] = emit(D.build_kv_update_graph(dims, seq=vopt.staging_rows, past_len=past, name=f"kv_flush_L{L}"))
            for k in blocks_per_graph:
                if k > 1:
                    for count in counts(k):
                        decode["block_groups"][str(count)] = emit(D.build_block_group_graph(dims, vopt, block_count=count, seq=1, past_len=past, name=f"decode_block_x{count}_L{L}",
                                                                                                  per_block_controls=per_block))
            table["decode"][str(L)] = decode
            table["prefill"][str(L)] = {}
            for C in chunks:
                if C > past:
                    continue
                prefill = {"head": emit(D.build_head_graph(dims, vopt, seq=C, past_len=past, name=f"prefill_head_L{L}_C{C}")),
                           "block": emit(D.build_block_graph(dims, vopt, seq=C, past_len=past, name=f"prefill_block_L{L}_C{C}")),
                           "tail": emit(D.build_tail_graph(dims, vopt, seq=C, name=f"prefill_tail_C{C}")), "block_groups": {}}
                for k in prefill_blocks_per_graph:
                    if k > 1:
                        for count in counts(k):
                            prefill["block_groups"][str(count)] = emit(D.build_block_group_graph(dims, vopt, block_count=count, seq=C, past_len=past,
                                                                                                name=f"prefill_block_x{count}_L{L}_C{C}", per_block_controls=per_block))
                table["prefill"][str(L)][str(C)] = prefill
        tables[variant["id"]] = table
    return tables


def write_graphs_gemma4(spec: arch.ModelSpec, opt: D.GraphOptions, out: Path, variants, max_lengths, chunks) -> dict:
    """The graphs of a Gemma-4-style model, per variant: head, one block per attention kind of layers 1.., tail, flush per layer type (own K / V), lm_head."""
    block_kinds = sorted({spec.layer_kind(i) for i in range(1, spec.num_layers)})
    cache_types = sorted({spec.layer_types[i] for i in range(spec.num_layers) if not spec.kv_producer or spec.kv_producer[i] < 0})
    tables: dict = {}
    for variant in variants:
        vopt = D.with_options(opt, kv_style="staged", compute_type=variant["compute_type"], staging_rows=variant["staging_rows"])
        gdir = out / "graphs" / variant["id"]
        gdir.mkdir(parents=True, exist_ok=True)

        def emit(graph) -> str:
            check_tensor_limit(graph)
            serialize.dump(graph, gdir / f"{graph.name}.json")
            return f"graphs/{variant['id']}/{graph.name}.json"

        table: dict = {"lm_head": emit(G.build_lm_head_graph(spec, vopt)), "decode": {}, "prefill": {}}
        for L in max_lengths:
            past = L - 1
            table["decode"][str(L)] = {
                "head": emit(G.build_head_graph(spec, vopt, seq=1, past_len=past, name=f"decode_head_L{L}")),
                "blocks": {k: emit(G.build_block_graph(spec, vopt, k, seq=1, past_len=past, name=f"decode_block_{k}_L{L}")) for k in block_kinds},
                "tail": emit(G.build_tail_graph(spec, vopt, seq=1, name="decode_tail")),
                "flush": {t: emit(G.build_flush_graph(spec, t, staging_rows=vopt.staging_rows, past_len=past, name=f"kv_flush_{t}_L{L}")) for t in cache_types}}
            table["prefill"][str(L)] = {}
            for C in chunks:
                if C > past:
                    continue
                table["prefill"][str(L)][str(C)] = {
                    "head": emit(G.build_head_graph(spec, vopt, seq=C, past_len=past, name=f"prefill_head_L{L}_C{C}")),
                    "blocks": {k: emit(G.build_block_graph(spec, vopt, k, seq=C, past_len=past, name=f"prefill_block_{k}_L{L}_C{C}")) for k in block_kinds},
                    "tail": emit(G.build_tail_graph(spec, vopt, seq=C, name=f"prefill_tail_C{C}"))}
        tables[variant["id"]] = table
    return tables


def build_plan_gemma4(spec: arch.ModelSpec, opt: D.GraphOptions, variants, graphs: dict, max_lengths, chunks) -> dict:
    kinds = sorted({spec.layer_kind(i) for i in range(spec.num_layers)})
    return {
        "format": "m2w-plan", "version": 0, "target": TARGET, "architecture_kind": "gemma4",
        "weights": {
            "description": "tensor names of the weights manifest: attention = by attention kind (the layer type and whether the layer has K / V of its own), ffn = the feed-forward half of every layer "
                           "(incl. the per-layer-embedding branch and the layer scalar), head_extra = the PLE projection bound to HEAD, ple_table = the per-layer embedding table (rows are read on the host)",
            "attention": {k: [n for n, _, _ in G.attention_weight_specs(spec, k)] for k in kinds},
            "ffn": [n for n, _, _ in G.feed_forward_weight_specs(spec)],
            "head_extra": [n for n, _, _ in G.head_extra_weight_specs(spec)],
            "final_norm": ["final_norm_gamma"],
            "embedding": [n for n, _, _ in G.lm_head_weight_specs(spec)],
            "ple_table": ["ple_weight", "ple_scale"],
        },
        "profile": {k: v for k, v in asdict(opt).items() if k not in ("kv_style", "staging_rows", "compute_type", "scatter_index_type", "value_layout")},
        "variants": variants, "default_variant": variants[0]["id"],
        "max_lengths": list(max_lengths), "prefill_chunks": list(chunks), "blocks_per_graph": [1], "prefill_blocks_per_graph": [1],
        "graphs": graphs,
        "chain": {
            "description": "HEAD = attention half of layer 0 + the per-layer-embedding prologue (outputs ple_0 .. ple_{N-1}); BLOCK i (i = 0 .. N-2) = feed-forward half of layer i (input per_layer_input = ple_i) + "
                           "attention half of layer i + 1, the graph of its KIND (plan.graphs[..].blocks[kind]); TAIL = feed-forward half of layer N-1 + final norm; lm_head (with the logit softcap). The kind of "
                           "layer i is model.json layer_types[i] + '_own' | '_shared' (kv_producer[i] >= 0 = it attends over the cache of that layer).",
            "num_layers": spec.num_layers, "kinds": [spec.layer_kind(i) for i in range(spec.num_layers)],
        },
        "per_layer_type_inputs": {
            "description": "rope_cos / rope_sin [1,1,S,head_dim of the type] / attention_mask of an attention half are bound with the tables of the layer type of ITS layer; a shared layer's mask has the columns of "
                           "the cache it reads (prefill: P, decode: P + staging rows)",
            "inputs": ["rope_cos", "rope_sin", "attention_mask"],
        },
        "kv_protocols": {
            "description": "only the staged protocol: prefill = chunks (the last one left-padded) updating the main caches of the layers with K / V of their own; decode = the new row goes into a staging buffer, "
                           "attention reads concat(main cache, staging), the flush graph (one per layer type) moves the staging rows into the main cache every staging_rows steps. Layers that share K / V read "
                           "the cache of their producer: decode = its main cache + its staging buffer after this step's update; prefill = its cache after the update of the chunk.",
        },
        "host": {"embed_scale": spec.embed_scale, "ple": "per prompt token a row of the ple_table (int8 * row scale, float16) is bound to HEAD's ple_token input; the graph scales it by sqrt(ple_dim)",
                 "logit_softcap": spec.logit_softcap},
        "prefill_planning": "several chunk sizes: protocol plan_prefill() picks the chunk sequence of a prompt with the lowest sum of (fixed tokens + size); fixed tokens default 200",
        "logits": {"padded_vocab": spec.vocab_padded, "vocab": spec.vocab, "note": "argmax / sampling over the first `vocab` entries only"},
    }


def write_graphs_qwen35(spec: arch.ModelSpec, opt: D.GraphOptions, out: Path, variants, max_lengths, chunks) -> dict:
    """The graphs of a Qwen3.5-style hybrid model, per variant: head, one block per kind of attention half (linear_attention | full_attention) of layers 1.., tail, one flush (the full
    attention layers' caches), lm_head."""
    bad = [C for C in chunks if C % Q.SUB_CHUNK]
    if bad:
        raise ValueError(f"the prefill chunk sizes of a Gated DeltaNet model must be multiples of {Q.SUB_CHUNK} (got {bad})")
    block_kinds = sorted({spec.layer_kind(i) for i in range(1, spec.num_layers)})
    tables: dict = {}
    for variant in variants:
        vopt = D.with_options(opt, kv_style="staged", compute_type=variant["compute_type"], staging_rows=variant["staging_rows"])
        gdir = out / "graphs" / variant["id"]
        gdir.mkdir(parents=True, exist_ok=True)

        def emit(graph) -> str:
            check_tensor_limit(graph)
            serialize.dump(graph, gdir / f"{graph.name}.json")
            return f"graphs/{variant['id']}/{graph.name}.json"

        table: dict = {"lm_head": emit(Q.build_lm_head_graph(spec, vopt)), "decode": {}, "prefill": {}}
        for L in max_lengths:
            past = L - 1
            table["decode"][str(L)] = {
                "head": emit(Q.build_head_graph(spec, vopt, seq=1, past_len=past, name=f"decode_head_L{L}")),
                "blocks": {k: emit(Q.build_block_graph(spec, vopt, k, seq=1, past_len=past, name=f"decode_block_{k}_L{L}")) for k in block_kinds},
                "tail": emit(Q.build_tail_graph(spec, vopt, seq=1, name="decode_tail")),
                "flush": emit(Q.build_flush_graph(spec, staging_rows=vopt.staging_rows, past_len=past, name=f"kv_flush_L{L}"))}
            table["prefill"][str(L)] = {}
            for C in chunks:
                if C > past:
                    continue
                table["prefill"][str(L)][str(C)] = {
                    "head": emit(Q.build_head_graph(spec, vopt, seq=C, past_len=past, name=f"prefill_head_L{L}_C{C}")),
                    "blocks": {k: emit(Q.build_block_graph(spec, vopt, k, seq=C, past_len=past, name=f"prefill_block_{k}_L{L}_C{C}")) for k in block_kinds},
                    "tail": emit(Q.build_tail_graph(spec, vopt, seq=C, name=f"prefill_tail_C{C}"))}
        tables[variant["id"]] = table
    return tables


def build_plan_qwen35(spec: arch.ModelSpec, opt: D.GraphOptions, variants, graphs: dict, max_lengths, chunks) -> dict:
    kinds = sorted(set(spec.layer_types))
    dims = Q.linear_dims(spec) if spec.linear else {}
    return {
        "format": "m2w-plan", "version": 0, "target": TARGET, "architecture_kind": "qwen35",
        "weights": {
            "description": "tensor names of the weights manifest: attention = by kind of the layer (linear_attention = Gated DeltaNet, full_attention), ffn = the feed-forward half of every layer",
            "attention": {k: [n for n, _, _ in Q.attention_weight_specs(spec, k)] for k in kinds},
            "ffn": [n for n, _, _ in Q.feed_forward_weight_specs(spec)],
            "final_norm": ["final_norm_gamma"],
            "embedding": [n for n, _, _ in Q.lm_head_weight_specs(spec)],
        },
        "profile": {k: v for k, v in asdict(opt).items() if k not in ("kv_style", "staging_rows", "compute_type", "scatter_index_type", "value_layout")},
        "variants": variants, "default_variant": variants[0]["id"],
        "max_lengths": list(max_lengths), "prefill_chunks": list(chunks), "blocks_per_graph": [1], "prefill_blocks_per_graph": [1],
        "graphs": graphs,
        "chain": {
            "description": "HEAD = attention half of layer 0; BLOCK i (i = 0 .. N-2) = feed-forward half of layer i + attention half of layer i + 1, the graph of its KIND (plan.graphs[..].blocks[kind]); "
                           "TAIL = feed-forward half of layer N-1 + final norm; lm_head. The kind of layer i is model.json layer_types[i].",
            "num_layers": spec.num_layers, "kinds": [spec.layer_kind(i) for i in range(spec.num_layers)],
        },
        "per_layer_type_inputs": {
            "description": "full attention layers: rope_cos / rope_sin [1,1,S,head_dim] / attention_mask (partial rotary: the head dims are permuted at packing time, see model.json rope + pack_qwen35.py)",
            "inputs": ["rope_cos", "rope_sin", "attention_mask"],
        },
        "state": {
            "description": "linear attention (Gated DeltaNet) layers carry a state through the graphs instead of a KV cache: conv_state [1,3,conv_channels] and recurrent_state [1,v_heads,k_dim,v_dim] (float16) are "
                           "inputs of the layer's attention half and come back as new_conv_state / new_recurrent_state (the runtime swaps a state tensor with its spare); zero at the start of a prompt. "
                           "Prefill graphs (S > 1) take two controls: valid_mask [1,1,S,1] (0 for the pad rows of the left-padded last chunk: beta and g are zeroed, the pad rows do nothing to the state) and "
                           "conv_perm [1,S+3,S+3] (a permutation matrix applied to [conv_state rows | chunk rows]: the pad rows move in front of the state rows; the identity for a full chunk). "
                           "protocol.linear_prefill_controls() builds both.",
            "conv_channels": dims.get("conv_width"), "v_heads": dims.get("v_heads"), "k_dim": dims.get("k_dim"), "v_dim": dims.get("v_dim"),
            "layers": [i for i in range(spec.num_layers) if spec.layer_types[i] == "linear_attention"],
        },
        "kv_protocols": {
            "description": "only the staged protocol, for the full attention layers (a Gated DeltaNet layer has no KV cache): prefill = chunks (the last one left-padded) updating the main caches; decode = the new row "
                           "goes into a staging buffer, attention reads concat(main cache, staging), the flush graph moves the staging rows into the main cache every staging_rows steps.",
        },
        "prefill_planning": "several chunk sizes: protocol plan_prefill() picks the chunk sequence of a prompt with the lowest sum of (fixed tokens + size); fixed tokens default 200",
        "logits": {"padded_vocab": spec.vocab_padded, "vocab": spec.vocab, "note": "argmax / sampling over the first `vocab` entries only"},
    }


def _graphs_and_plan(spec: arch.ModelSpec, opt: D.GraphOptions, out: Path, variants, max_lengths, chunks, blocks_per_graph, prefill_blocks_per_graph):
    if spec.builder == "qwen35":
        if any(v["kv_style"] != "staged" for v in variants):
            raise ValueError("Qwen3.5 packages support the staged KV protocol only")
        if any(k > 1 for k in tuple(blocks_per_graph) + tuple(prefill_blocks_per_graph)):
            raise ValueError("block groups are not available for Qwen3.5 packages yet")
        graphs = write_graphs_qwen35(spec, opt, out, variants, max_lengths, chunks)
        return graphs, build_plan_qwen35(spec, opt, variants, graphs, max_lengths, chunks)
    if spec.builder == "gemma4":
        if any(v["kv_style"] != "staged" for v in variants):
            raise ValueError("Gemma 4 packages support the staged KV protocol only")
        if any(k > 1 for k in tuple(blocks_per_graph) + tuple(prefill_blocks_per_graph)):
            raise ValueError("block groups are not available for Gemma 4 packages yet")
        graphs = write_graphs_gemma4(spec, opt, out, variants, max_lengths, chunks)
        return graphs, build_plan_gemma4(spec, opt, variants, graphs, max_lengths, chunks)
    graphs = write_graphs(spec, opt, out, variants, max_lengths, chunks, blocks_per_graph, prefill_blocks_per_graph)
    return graphs, build_plan(spec, opt, variants, graphs, max_lengths, chunks, blocks_per_graph, prefill_blocks_per_graph)


def build_plan(spec: arch.ModelSpec, opt: D.GraphOptions, variants, graphs: dict, max_lengths, chunks, blocks_per_graph, prefill_blocks_per_graph=(1,)) -> dict:
    dims = spec.decoder_dims()
    return {
        "format": "m2w-plan", "version": 0, "target": TARGET,
        "weights": {
            "description": "tensor names of the weights manifest, by the half of the layer they feed (a graph input with such a name is bound to that tensor of the layer)",
            "attention": [n for n, _, _ in D.attention_weight_specs(dims, opt)],
            "ffn": [n for n, _, _ in D.feed_forward_weight_specs(dims, opt)],
            "final_norm": ["final_norm_gamma"],
            "embedding": [n for n, _, _ in D.lm_head_weight_specs(dims, opt)],
        },
        "profile": {k: v for k, v in asdict(opt).items() if k not in ("kv_style", "staging_rows", "compute_type", "scatter_index_type")},
        "variants": variants, "default_variant": variants[0]["id"],
        "block_group_controls": "per_block" if len(set(spec.layer_types)) > 1 else "shared",
        "max_lengths": list(max_lengths), "prefill_chunks": list(chunks), "blocks_per_graph": list(blocks_per_graph), "prefill_blocks_per_graph": list(prefill_blocks_per_graph),
        "graphs": graphs,
        "chain": {
            "description": "HEAD = attention half of layer 0; BLOCK i (i = 0 .. N-2) = feed-forward half of layer i + attention half of layer i+1; TAIL = feed-forward half of layer N-1 "
                           "+ final norm; lm_head. Per layer the runtime binds the weights of the 'attention_weights' layer and the 'ffn_weights' layer from the weights manifest.",
            "num_layers": spec.num_layers,
            "head": {"attention_layer": 0}, "block": {"ffn_layer": "i", "attention_layer": "i+1"}, "tail": {"ffn_layer": "N-1"},
        },
        "per_layer_type_inputs": {
            "description": "rope_cos / rope_sin / attention_mask of an attention half are bound with the tables of the layer type of ITS layer (see model.json: layer_types, rope, sliding_window)",
            "inputs": ["rope_cos", "rope_sin", "attention_mask"],
        },
        "kv_protocols": {
            "description": "how the KV cache is written and attended, by the kv_style of a variant (cache rows P = max_length - 1; row j holds position j; protocol.py / protocol.ts)",
            "staged": "prefill: chunks of the chunk size, the last chunk is left-padded, one-hot row selectors write the real rows; decode: the new row goes into a staging buffer "
                      "(staging_rows rows), attention reads concat(main cache, staging), the flush graph moves the staging rows into the main cache every staging_rows steps",
            "onehot": "prefill as staged; decode: the graph writes the new row into the main cache with one-hot selectors (rows_to_keep / row_selector), attention reads concat(cache, new row)",
            "scatterND": "prefill and decode: scatterND(cache, kv_indices, new rows) with the (0, head, cache row) of every new row, attention reads the updated cache; the pad rows of the "
                         "left-padded last chunk are parked behind the real rows",
            "tensorScatter": "prefill and decode: tensorScatter(cache, new rows, kv_write_index, axis 2), attention reads the updated cache; every chunk is full, the prompt padding is in FRONT "
                             "of the prompt (cache rows 0..pad-1)",
        },
        "prefill_planning": "several chunk sizes: protocol plan_prefill() picks the chunk sequence of a prompt with the lowest sum of (fixed tokens + size); fixed tokens default 200",
        "logits": {"padded_vocab": spec.vocab_padded, "vocab": spec.vocab, "note": "argmax / sampling over the first `vocab` entries only"},
    }


def _write_index(out: Path, spec: arch.ModelSpec, source_name: str, plan: dict, manifest: dict, chat_file, tokenizer_files) -> dict:
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in ("m2w.json", "verification.json"))   # verification.json is the output of `m2w verify`, not part of the package
    index = {
        "format": PACKAGE_FORMAT, "version": PACKAGE_VERSION, "compiler": f"m2w {__version__}", "target": TARGET, "source": source_name,
        "model": "model.json", "plan": "plan.json", "chat_format": chat_file, "weights_manifest": "weights/manifest.json",
        "tokenizer": [f"tokenizer/{n}" for n in tokenizer_files],
        "requires": {"webnn": {"ops": ["dequantizeLinear", "gemm", "matmul", "softmax", "concat", "slice", "expand", "reduceMean", "transpose", "reshape", "cast"]
                               + (["tensorScatter"] if any(v["kv_style"] == "tensorScatter" for v in plan["variants"]) else [])
                               + (["scatterND"] if any(v["kv_style"] == "scatterND" for v in plan["variants"]) else [])
                               + (["tanh"] if plan.get("architecture_kind") == "gemma4" else [])
                               + (["exp", "softplus", "reduceSum"] if plan.get("architecture_kind") == "qwen35" else []),
                               "dataTypes": ["float16", "float32", "int4", "uint8"], "inputs_may_be_int4": True}},
        "files": [{"path": p.relative_to(out).as_posix(), "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files],
        "summary": {"layers": spec.num_layers, "max_lengths": plan["max_lengths"], "prefill_chunks": plan["prefill_chunks"], "variants": [v["id"] for v in plan["variants"]],
                    "weights_bytes": sum(f["bytes"] for f in manifest["files"])},
    }
    (out / "m2w.json").write_text(json.dumps(index, indent=1), encoding="utf-8", newline="\n")
    return index


def _bos_token(source: Path) -> str:
    """The text of the BOS token from tokenizer_config.json ('' when there is none): templates of models like Gemma start with it."""
    try:
        token = json.loads((source / "tokenizer_config.json").read_text(encoding="utf-8")).get("bos_token")
    except (OSError, ValueError):
        return ""
    return token.get("content", "") if isinstance(token, dict) else token or ""


def _check_lengths(spec: arch.ModelSpec, max_lengths, progress) -> None:
    for L in max_lengths:
        if L % 64:
            progress(f"warning: max length {L} is not a multiple of 64 (the NPU compiler returned NaN for some attention lengths that are not)")
        if L > spec.max_position_embeddings:
            raise ValueError(f"max length {L} exceeds the model's max_position_embeddings {spec.max_position_embeddings}")


def compile_model(model: str, out_dir, *, max_lengths=(1152,), chunks=(64,), kv_styles=("staged",), staging_rows=(64,), compute_types=("float32",), blocks_per_graph=(1,),
                  prefill_blocks_per_graph=(1,), merge_projections: bool = True, lm_head_format: str = "int8-sym", search: bool = True, workers: int = 4,
                  endpoint: Optional[str] = None, cache_dir=None, progress: Callable[[str], None] = print) -> Path:
    """Graph variants = kv_styles x staging_rows (staged only) x compute_types; the first one is the default variant. Weights are shared by all variants."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for owned in ("graphs", "weights", "tokenizer"):   # directories the compiler owns: files of an earlier compile (another format, fewer layers ...) must not linger in the package
        shutil.rmtree(out / owned, ignore_errors=True)
    source = hub.resolve_model(model, cache_dir=cache_dir, endpoint=endpoint, progress=progress)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    generation_config = json.loads((source / "generation_config.json").read_text(encoding="utf-8")) if (source / "generation_config.json").exists() else {}
    spec = arch.spec_from_config(config, generation_config)
    _check_lengths(spec, max_lengths, progress)
    hybrid = spec.builder in ("gemma4", "qwen35")
    opt = D.GraphOptions(merge_projections=merge_projections or hybrid, lm_head_format="int8-sym" if hybrid else lm_head_format)
    variants = expand_variants(kv_styles, staging_rows, compute_types)
    progress(f"{spec.architecture}: {spec.num_layers} layers, hidden {spec.hidden}, {spec.num_heads}/{spec.num_kv_heads} heads, vocab {spec.vocab} (padded {spec.vocab_padded}), "
             f"layer types {sorted(set(spec.layer_types))}")

    store = TensorStore.open(source)
    packer = {"gemma4": pack_gemma4, "qwen35": pack_qwen35}.get(spec.builder, pack)
    manifest = packer.pack_model(spec, store, out / "weights", opt, grid=pack.DEFAULT_GRID if search else None, workers=workers, progress=progress)
    graphs, plan = _graphs_and_plan(spec, opt, out, variants, max_lengths, chunks, blocks_per_graph, prefill_blocks_per_graph)
    (out / "model.json").write_text(json.dumps(spec.to_dict(), indent=1), encoding="utf-8", newline="\n")
    (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8", newline="\n")

    chat_file = None
    try:
        fmt = chat_format.derive_chat_format(chat_format.load_template(source), spec.chat_template_kwargs, _bos_token(source))
        (out / "chat_format.json").write_text(json.dumps(fmt, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
        chat_file = "chat_format.json"
    except (chat_format.ChatTemplateError, OSError, KeyError) as e:
        progress(f"warning: no chat format derived ({e}); the runtime can only complete raw text")
    (out / "tokenizer").mkdir(exist_ok=True)
    tokenizer_files = [n for n in TOKENIZER_FILES if (source / n).exists() and not shutil.copy2(source / n, out / "tokenizer" / n)]

    source_name = Path(model).name if Path(model).is_dir() else model   # a local folder is recorded by its name, never by its absolute path
    if "/" not in source_name and source_name.count("--") == 1:
        source_name = source_name.replace("--", "/")   # the folder of the Hugging Face cache (org--name) is the model id org/name
    index = _write_index(out, spec, source_name, plan, manifest, chat_file, tokenizer_files)
    progress(f"wrote {out} ({index['summary']['weights_bytes'] / 2**20:.0f} MB of weights, {len(index['files'])} files, variants: {', '.join(index['summary']['variants'])})")
    return out


def rebuild_graphs(package_dir, *, max_lengths=None, chunks=None, kv_styles=("staged",), staging_rows=(64,), compute_types=("float32",), blocks_per_graph=(1,),
                   prefill_blocks_per_graph=(1,), progress: Callable[[str], None] = print) -> Path:
    """Rewrite graphs/, plan.json and the file index of an existing package with other graph variants; the weights, tokenizer and model.json stay (no quantization).
    max_lengths / chunks default to those of the package."""
    out = Path(package_dir)
    old = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    index_old = json.loads((out / "m2w.json").read_text(encoding="utf-8"))
    spec = arch.ModelSpec.from_dict(json.loads((out / "model.json").read_text(encoding="utf-8")))
    manifest = json.loads((out / "weights" / "manifest.json").read_text(encoding="utf-8"))
    max_lengths, chunks = tuple(max_lengths or old["max_lengths"]), tuple(chunks or old["prefill_chunks"])
    _check_lengths(spec, max_lengths, progress)
    opt = D.GraphOptions(**{k: v for k, v in old["profile"].items() if k in D.GraphOptions.__dataclass_fields__})
    variants = expand_variants(kv_styles, staging_rows, compute_types)
    shutil.rmtree(out / "graphs", ignore_errors=True)
    graphs, plan = _graphs_and_plan(spec, opt, out, variants, max_lengths, chunks, blocks_per_graph, prefill_blocks_per_graph)
    (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8", newline="\n")
    tokenizer_files = [Path(f).name for f in index_old.get("tokenizer", [])]
    index = _write_index(out, spec, index_old.get("source", ""), plan, manifest, index_old.get("chat_format"), tokenizer_files)
    progress(f"rewrote the graphs of {out}: variants {', '.join(index['summary']['variants'])}")
    return out


def repack_package(package_dir, progress: Callable[[str], None] = print) -> Path:
    """Convert the weight shards of an existing package to .safetensors (no quantization) and refresh the file index m2w.json."""
    out = Path(package_dir)
    manifest = pack.repack_to_safetensors(out / "weights")
    index_old = json.loads((out / "m2w.json").read_text(encoding="utf-8"))
    spec = arch.ModelSpec.from_dict(json.loads((out / "model.json").read_text(encoding="utf-8")))
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    index = _write_index(out, spec, index_old.get("source", ""), plan, manifest, index_old.get("chat_format"), [Path(f).name for f in index_old.get("tokenizer", [])])
    progress(f"{out}: {len(manifest['files'])} weight shards are .safetensors files now ({index['summary']['weights_bytes'] / 2**20:.0f} MB)")
    return out

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
from .frontends.hf_llm import arch, chat_format, hub, pack, pack_gemma4, pack_qwen35, vision as V
from .frontends.hf_llm.builders import decoder as D
from .frontends.hf_llm.builders import gemma4 as G
from .frontends.hf_llm.builders import qwen35 as Q
from .frontends.hf_llm.builders import qwen35_vision as QV
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


def write_graphs_gemma4(spec: arch.ModelSpec, opt: D.GraphOptions, out: Path, variants, max_lengths, chunks, blocks_per_graph=(1,)) -> dict:
    """The graphs of a Gemma-4-style model, per variant: head, one block per attention kind of layers 1.., tail, flush per layer type (own K / V), lm_head;
    decode only: for every K > 1 of blocks_per_graph the block groups (decode.block_groups[K] = the schedule of the N - 1 blocks, one graph per distinct sequence of kinds)."""
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
            groups = {str(k): _gemma4_block_groups(spec, vopt, k, past, L, emit) for k in blocks_per_graph if k > 1}
            if groups:
                table["decode"][str(L)]["block_groups"] = groups
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


def _gemma4_block_groups(spec: arch.ModelSpec, vopt: D.GraphOptions, count: int, past: int, L: int, emit) -> list:
    """The schedule of the decode block groups of `count` blocks: [{first_block, blocks, key, graph}]; the graphs are shared by the groups with the same sequence of kinds (the key)."""
    if count > spec.num_layers - 1:
        raise ValueError(f"{count} blocks per graph: the model has {spec.num_layers - 1} blocks")
    schedule, built = [], {}
    for first, blocks in G.block_group_schedule(spec, count):
        layout = G.block_group_layout(spec, first, blocks)
        key = G.block_group_key(layout)
        if key not in built:
            built[key] = (layout, emit(G.build_block_group_graph(spec, vopt, layout, seq=1, past_len=past, name=f"decode_group_{key}_L{L}")))
        elif built[key][0] != layout:   # the same kinds but another cache sharing: would need its own graph
            raise ValueError(f"block groups with the kinds {key} differ in which block reads which cache: {built[key][0]} vs {layout}")
        schedule.append({"first_block": first, "blocks": blocks, "key": key, "graph": built[key][1]})
    return schedule


def build_plan_gemma4(spec: arch.ModelSpec, opt: D.GraphOptions, variants, graphs: dict, max_lengths, chunks, blocks_per_graph=(1,)) -> dict:
    kinds = sorted({spec.layer_kind(i) for i in range(spec.num_layers)})
    plan = {
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
        "max_lengths": list(max_lengths), "prefill_chunks": list(chunks), "blocks_per_graph": list(blocks_per_graph), "prefill_blocks_per_graph": [1],
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
    if any(k > 1 for k in blocks_per_graph):
        plan["chain"]["block_groups"] = {
            "description": "decode only: graphs.<variant>.decode.<max length>.block_groups[K] = the N - 1 BLOCKs cut into runs of K ([{first_block, blocks, key, graph}], the last run may be shorter), one "
                           "graph per distinct sequence of kinds (key: the kinds joined by '-', so / fo = sliding / full attention with K / V of its own, ss / fs = sliding / full attention sharing K / V). A group graph takes "
                           "residual + attention_output of its first block and returns next_residual + next_attention_output of its last block. Block j (BLOCK first_block + j) has the prefix b<j>_: "
                           "b<j>_per_layer_input (ple_<first_block + j>), the weights of the feed-forward half of layer first_block + j and of the attention half of layer first_block + j + 1, "
                           "and b<j>_past_key / b<j>_past_value / b<j>_staging_key / b<j>_staging_value = the main cache and staging buffer of the layer whose K / V that attention half reads "
                           "(model.json kv_producer); a block that reads the cache of a layer already read by an earlier block of the group has no cache inputs (it reuses that block's), and a block "
                           "that reads the cache of a layer INSIDE the group attends over the staging buffer after that layer's update, computed in the graph. Output b<j>_updated_key / "
                           "b<j>_updated_value (blocks with K / V of their own) = the spare staging buffer, to be swapped in. <layer type>_rope_cos / _rope_sin / _attention_mask: one per layer type "
                           "(the decode masks of own and shared layers are the same). Prefill has no groups (prefill_blocks_per_graph = [1]).",
            "kind_codes": {code: kind for kind, code in G.KIND_CODES.items()}}
    return plan


def write_graphs_qwen35(spec: arch.ModelSpec, opt: D.GraphOptions, out: Path, variants, max_lengths, chunks, blocks_per_graph=(1,)) -> dict:
    """The graphs of a Qwen3.5-style hybrid model, per variant: head, one block per kind of attention half (linear_attention | full_attention) of layers 1.., tail, one flush (the full
    attention layers' caches), lm_head; decode only: for every K > 1 of blocks_per_graph the block groups (decode.block_groups[K] = the schedule of the N - 1 blocks, one graph per distinct
    sequence of kinds)."""
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
            groups = {str(k): _qwen35_block_groups(spec, vopt, k, past, L, emit) for k in blocks_per_graph if k > 1}
            if groups:
                table["decode"][str(L)]["block_groups"] = groups
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


def _qwen35_block_groups(spec: arch.ModelSpec, vopt: D.GraphOptions, count: int, past: int, L: int, emit) -> list:
    """The schedule of the decode block groups of `count` blocks: [{first_block, blocks, key, graph}]; the graphs are shared by the groups with the same sequence of kinds (the key)."""
    if count > spec.num_layers - 1:
        raise ValueError(f"{count} blocks per graph: the model has {spec.num_layers - 1} blocks")
    schedule, built = [], {}
    for first, blocks in Q.block_group_schedule(spec, count):
        kinds = Q.block_group_kinds(spec, first, blocks)
        key = Q.block_group_key(kinds)
        if key not in built:
            built[key] = emit(Q.build_block_group_graph(spec, vopt, kinds, seq=1, past_len=past, name=f"decode_group_{key}_L{L}"))
        schedule.append({"first_block": first, "blocks": blocks, "key": key, "graph": built[key]})
    return schedule


def build_plan_qwen35(spec: arch.ModelSpec, opt: D.GraphOptions, variants, graphs: dict, max_lengths, chunks, blocks_per_graph=(1,)) -> dict:
    kinds = sorted(set(spec.layer_types))
    dims = Q.linear_dims(spec) if spec.linear else {}
    plan = {
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
        "max_lengths": list(max_lengths), "prefill_chunks": list(chunks), "blocks_per_graph": list(blocks_per_graph), "prefill_blocks_per_graph": [1],
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
    if any(k > 1 for k in blocks_per_graph):
        plan["chain"]["block_groups"] = {
            "description": "decode only: graphs.<variant>.decode.<max length>.block_groups[K] = the N - 1 BLOCKs cut into runs of K ([{first_block, blocks, key, graph}], the last run may be shorter), one "
                           "graph per distinct sequence of kinds (key: the kinds joined by '-', l / f = linear / full attention). A group graph takes residual + attention_output of its first block and returns "
                           "next_residual + next_attention_output of its last block. Block j (BLOCK first_block + j) has the prefix b<j>_: the weights of the feed-forward half of layer first_block + j and of the "
                           "attention half of layer first_block + j + 1 and its state: b<j>_conv_state / b<j>_recurrent_state (linear attention), b<j>_past_key / _past_value / _staging_key / _staging_value "
                           "(full attention). Outputs b<j>_new_conv_state / b<j>_new_recurrent_state / b<j>_updated_key / b<j>_updated_value = the spare tensors, to be swapped in. rope_cos / rope_sin / "
                           "attention_mask / staging_rows_to_keep / staging_row_selector are shared by the full attention blocks. Prefill has no groups (prefill_blocks_per_graph = [1]).",
            "kind_codes": {code: kind for kind, code in Q.KIND_CODES.items()}}
    return plan


def write_vision_graphs(spec: arch.ModelSpec, out: Path, grids, weights: str, compute_type: str) -> dict:
    """The image tower graphs of every grid preset (graphs/vision/<grid>_head|block|merger.json; independent of the text variants) -> plan.json `vision`."""
    gdir = out / "graphs" / "vision"
    gdir.mkdir(parents=True, exist_ok=True)

    def emit(graph) -> str:
        check_tensor_limit(graph)
        serialize.dump(graph, gdir / f"{graph.name}.json")
        return f"graphs/vision/{graph.name}.json"

    presets = []
    for grid in grids:
        V.check_grid(grid, spec.vision["merge_size"])
        g = V.grid_name(grid)
        graphs = {"head": emit(QV.build_head_graph(spec, grid, weights, compute_type, name=f"{g}_head")), "block": emit(QV.build_block_graph(spec, grid, weights, compute_type, name=f"{g}_block")),
                  "merger": emit(QV.build_merger_graph(spec, grid, weights, compute_type, name=f"{g}_merger"))}
        patches = grid[0] * grid[1]
        presets.append({"grid": list(grid), "patches": patches, "tokens": patches // spec.vision["merge_size"] ** 2, "graphs": graphs})
    return {
        "description": "the image tower in the NPUW-style split, one set of graphs per grid preset (grid = patches in height x width): HEAD (patches [N, patch_vector] -> hidden_states [1,N,hidden]: patch matmul + bias + the "
                       "position embedding of the grid), BLOCK x depth (ONE graph, weights bound per layer: hidden_states -> next_hidden_states), MERGER (hidden_states -> image_embeds [1,N/4,out_hidden]). "
                       "Patches are in merge-block order and in the (channel, temporal, kh, kw) order of the Conv3d kernel (protocol patchify). The external constants pos_embed (HEAD) and rope_cos / rope_sin "
                       "(BLOCK) are the tensors of the manifest file with role vision_grid and this grid; the weights are the manifest files vision_head / vision_block (layer i) / vision_merger.",
        "weights": weights, "compute_type": compute_type, "depth": spec.vision["depth"], "image_token_id": spec.vision["image_token_id"],
        "tensors": {"head": [n for n, _, _ in QV.head_weight_specs(spec, weights)], "block": [n for n, _, _ in QV.block_weight_specs(spec, weights)],
                    "merger": [n for n, _, _ in QV.merger_weight_specs(spec, weights)], "tables": list(QV.TABLES)},
        "presets": presets,
    }


def resolve_blocks_per_graph(spec: arch.ModelSpec, blocks_per_graph) -> tuple:
    """--blocks-per-graph not given: the single blocks and the recommended group size of the model family (so that the package contains the graphs the runtime defaults to)."""
    if blocks_per_graph is not None:
        return _sequence(blocks_per_graph)
    k = int(arch.recommended_session(spec).get("blocks_per_graph", 1))
    return (1, k) if k > 1 else (1,)


def recommended_for(spec: arch.ModelSpec, plan: dict) -> dict:
    """plan.recommended: the family's recommendation, restricted to what this package contains (a group size that was not built falls back to 1)."""
    rec = arch.recommended_session(spec)
    built = set(plan.get("blocks_per_graph", [1]))
    k = int(rec.get("blocks_per_graph", 1))
    out = {"blocks_per_graph": k if k in built else 1, "prefill_blocks_per_graph": int(rec.get("prefill_blocks_per_graph", 1)) if int(rec.get("prefill_blocks_per_graph", 1)) in set(plan.get("prefill_blocks_per_graph", [1])) else 1}
    chunks = [int(c) for c in rec.get("prefill_chunks", []) if int(c) in set(plan.get("prefill_chunks", []))]   # the prefill chunk sizes of the family's choice that the package has (none of them: the runtime picks the smallest)
    if chunks:
        out["prefill_chunks"] = chunks
    if rec.get("sampling"):   # the sampling parameters of the model card (greedy decoding of these models falls into repetition loops): the runtime uses them when the caller does not choose
        out["sampling"] = dict(rec["sampling"])
    return out


def _graphs_and_plan(spec: arch.ModelSpec, opt: D.GraphOptions, out: Path, variants, max_lengths, chunks, blocks_per_graph, prefill_blocks_per_graph, vision: Optional[dict] = None):
    """vision: {"weights", "grids", "compute_type"} of the image tower to write graphs for (the spec has a `vision` entry), else None."""
    graphs, plan = _graphs_and_plan_core(spec, opt, out, variants, max_lengths, chunks, blocks_per_graph, prefill_blocks_per_graph)
    plan["recommended"] = recommended_for(spec, plan)
    if vision:
        plan["vision"] = write_vision_graphs(spec, out, vision["grids"], vision["weights"], vision["compute_type"])
    return graphs, plan


def _graphs_and_plan_core(spec: arch.ModelSpec, opt: D.GraphOptions, out: Path, variants, max_lengths, chunks, blocks_per_graph, prefill_blocks_per_graph):
    if spec.builder == "qwen35":
        if any(v["kv_style"] != "staged" for v in variants):
            raise ValueError("Qwen3.5 packages support the staged KV protocol only")
        if any(k > 1 for k in prefill_blocks_per_graph):
            raise ValueError("block groups of Qwen3.5 packages are for decode only: prefill blocks per graph must be 1")
        graphs = write_graphs_qwen35(spec, opt, out, variants, max_lengths, chunks, blocks_per_graph)
        return graphs, build_plan_qwen35(spec, opt, variants, graphs, max_lengths, chunks, blocks_per_graph)
    if spec.builder == "gemma4":
        if any(v["kv_style"] != "staged" for v in variants):
            raise ValueError("Gemma 4 packages support the staged KV protocol only")
        if any(k > 1 for k in prefill_blocks_per_graph):
            raise ValueError("block groups of Gemma 4 packages are for decode only: prefill blocks per graph must be 1 (prefill grouping gave no TTFT gain)")
        graphs = write_graphs_gemma4(spec, opt, out, variants, max_lengths, chunks, blocks_per_graph)
        return graphs, build_plan_gemma4(spec, opt, variants, graphs, max_lengths, chunks, blocks_per_graph)
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
                               + (["tanh"] if plan.get("architecture_kind") == "gemma4" or "vision" in plan else [])
                               + (["clamp"] if "vision" in plan else [])
                               + (["exp", "softplus", "reduceSum"] if plan.get("architecture_kind") == "qwen35" else []),
                               "dataTypes": ["float16", "float32", "int4", "uint8"] + (["int8"] if "vision" in plan else []), "inputs_may_be_int4": True}},
        "files": [{"path": p.relative_to(out).as_posix(), "bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files],
        "summary": {"layers": spec.num_layers, "max_lengths": plan["max_lengths"], "prefill_chunks": plan["prefill_chunks"], "variants": [v["id"] for v in plan["variants"]],
                    "weights_bytes": sum(f["bytes"] for f in manifest["files"])},
    }
    (out / "m2w.json").write_text(json.dumps(index, indent=1), encoding="utf-8", newline="\n")
    return index


def _add_end_of_turn_token(out: Path, spec: arch.ModelSpec) -> bool:
    """Adds the token that ends an assistant turn of the chat format to the eos tokens of the spec (True when it was missing)."""
    try:
        fmt = json.loads((out / "chat_format.json").read_text(encoding="utf-8"))
        token = chat_format.end_of_turn_token_id(fmt, json.loads((out / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return False
    if token is None or token in spec.eos_token_ids:
        return False
    spec.eos_token_ids = [token, *spec.eos_token_ids]
    return True


def refresh_stop_tokens(package_dir, progress: Callable[[str], None] = print) -> bool:
    """Adds the end-of-turn token of the chat format to the eos tokens of an existing package (compiled before this was done: it stopped only at <|endoftext|>); model.json and the file index are rewritten."""
    out = Path(package_dir)
    spec = arch.ModelSpec.from_dict(json.loads((out / "model.json").read_text(encoding="utf-8")))
    if not _add_end_of_turn_token(out, spec):
        return False
    (out / "model.json").write_text(json.dumps(spec.to_dict(), indent=1), encoding="utf-8", newline="\n")
    index_old = json.loads((out / "m2w.json").read_text(encoding="utf-8"))
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "weights" / "manifest.json").read_text(encoding="utf-8"))
    _write_index(out, spec, index_old.get("source", ""), plan, manifest, index_old.get("chat_format"), [Path(f).name for f in index_old.get("tokenizer", [])])
    progress(f"{out.name}: eos tokens {spec.eos_token_ids}")
    return True


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


def _load_preprocessor(source: Path) -> Optional[dict]:
    path = source / "preprocessor_config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _vision_options(spec: arch.ModelSpec, weights: str, grids, compute_type: str) -> Optional[dict]:
    if not spec.vision:
        return None
    if weights not in QV.WEIGHT_FORMATS:
        raise ValueError(f"unknown vision weight format '{weights}' ({' | '.join(QV.WEIGHT_FORMATS)})")
    return {"weights": weights, "grids": [tuple(g) for g in (grids or V.DEFAULT_GRIDS)], "compute_type": compute_type}


def compile_model(model: str, out_dir, *, max_lengths=(1152,), chunks=(64,), kv_styles=("staged",), staging_rows=(64,), compute_types=("float32",), blocks_per_graph=None,
                  prefill_blocks_per_graph=(1,), merge_projections: bool = True, lm_head_format: str = "int8-sym", search: bool = True, workers: int = 4,
                  endpoint: Optional[str] = None, cache_dir=None, progress: Callable[[str], None] = print, vision: bool = True, vision_weights: str = "int8g128",
                  vision_grids=None, vision_compute_type: str = "float32") -> Path:
    """Graph variants = kv_styles x staging_rows (staged only) x compute_types; the first one is the default variant. Weights are shared by all variants.
    A Qwen3.5 checkpoint with a vision tower gets the image tower too (images only): `vision_weights` int8 | int8g128 | int4 | float16, `vision_grids` the (grid_h, grid_w) presets in patches
    (default vision.DEFAULT_GRIDS), `vision_compute_type` of its graphs; vision=False leaves it out."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for owned in ("graphs", "weights", "tokenizer"):   # directories the compiler owns: files of an earlier compile (another format, fewer layers ...) must not linger in the package
        shutil.rmtree(out / owned, ignore_errors=True)
    source = hub.resolve_model(model, cache_dir=cache_dir, endpoint=endpoint, progress=progress)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    generation_config = json.loads((source / "generation_config.json").read_text(encoding="utf-8")) if (source / "generation_config.json").exists() else {}
    spec = arch.spec_from_config(config, generation_config, _load_preprocessor(source))
    if not vision:
        spec.vision = {}
    vision_opt = _vision_options(spec, vision_weights, vision_grids, vision_compute_type)
    blocks_per_graph = resolve_blocks_per_graph(spec, blocks_per_graph)
    _check_lengths(spec, max_lengths, progress)
    if spec.builder == "gemma4" and any(k > 1 for k in prefill_blocks_per_graph):   # before the (long) quantization
        raise ValueError("block groups of Gemma 4 packages are for decode only: prefill blocks per graph must be 1")
    hybrid = spec.builder in ("gemma4", "qwen35")
    opt = D.GraphOptions(merge_projections=merge_projections or hybrid, lm_head_format="int8-sym" if hybrid else lm_head_format)
    variants = expand_variants(kv_styles, staging_rows, compute_types)
    progress(f"{spec.architecture}: {spec.num_layers} layers, hidden {spec.hidden}, {spec.num_heads}/{spec.num_kv_heads} heads, vocab {spec.vocab} (padded {spec.vocab_padded}), "
             f"layer types {sorted(set(spec.layer_types))}")

    store = TensorStore.open(source)
    packer = {"gemma4": pack_gemma4, "qwen35": pack_qwen35}.get(spec.builder, pack)
    manifest = packer.pack_model(spec, store, out / "weights", opt, grid=pack.DEFAULT_GRID if search else None, workers=workers, progress=progress,
                                 **({"vision": {"weights": vision_opt["weights"], "grids": vision_opt["grids"]}} if vision_opt else {}))
    graphs, plan = _graphs_and_plan(spec, opt, out, variants, max_lengths, chunks, blocks_per_graph, prefill_blocks_per_graph, vision_opt)
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
    _add_end_of_turn_token(out, spec)
    (out / "model.json").write_text(json.dumps(spec.to_dict(), indent=1), encoding="utf-8", newline="\n")

    source_name = Path(model).name if Path(model).is_dir() else model   # a local folder is recorded by its name, never by its absolute path
    if "/" not in source_name and source_name.count("--") == 1:
        source_name = source_name.replace("--", "/")   # the folder of the Hugging Face cache (org--name) is the model id org/name
    index = _write_index(out, spec, source_name, plan, manifest, chat_file, tokenizer_files)
    progress(f"wrote {out} ({index['summary']['weights_bytes'] / 2**20:.0f} MB of weights, {len(index['files'])} files, variants: {', '.join(index['summary']['variants'])})")
    return out


def rebuild_graphs(package_dir, *, max_lengths=None, chunks=None, kv_styles=("staged",), staging_rows=(64,), compute_types=("float32",), blocks_per_graph=None,
                   prefill_blocks_per_graph=(1,), progress: Callable[[str], None] = print) -> Path:
    """Rewrite graphs/, plan.json and the file index of an existing package with other graph variants; the weights, tokenizer and model.json stay (no quantization).
    max_lengths / chunks default to those of the package."""
    out = Path(package_dir)
    old = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    index_old = json.loads((out / "m2w.json").read_text(encoding="utf-8"))
    spec = arch.ModelSpec.from_dict(json.loads((out / "model.json").read_text(encoding="utf-8")))
    manifest = json.loads((out / "weights" / "manifest.json").read_text(encoding="utf-8"))
    max_lengths, chunks = tuple(max_lengths or old["max_lengths"]), tuple(chunks or old["prefill_chunks"])
    blocks_per_graph = tuple(blocks_per_graph) if blocks_per_graph is not None else tuple(old.get("blocks_per_graph", [1]))   # not given: keep the groups the package has
    _check_lengths(spec, max_lengths, progress)
    opt = D.GraphOptions(**{k: v for k, v in old["profile"].items() if k in D.GraphOptions.__dataclass_fields__})
    variants = expand_variants(kv_styles, staging_rows, compute_types)
    shutil.rmtree(out / "graphs", ignore_errors=True)
    old_vision = old.get("vision")   # the image tower graphs of the package are rewritten too (same weights format and grids; the compute type is kept)
    vision_opt = {"weights": old_vision["weights"], "grids": [tuple(p["grid"]) for p in old_vision["presets"]], "compute_type": old_vision["compute_type"]} if old_vision and spec.vision else None
    graphs, plan = _graphs_and_plan(spec, opt, out, variants, max_lengths, chunks, blocks_per_graph, prefill_blocks_per_graph, vision_opt)
    (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8", newline="\n")
    tokenizer_files = [Path(f).name for f in index_old.get("tokenizer", [])]
    index = _write_index(out, spec, index_old.get("source", ""), plan, manifest, index_old.get("chat_format"), tokenizer_files)
    progress(f"rewrote the graphs of {out}: variants {', '.join(index['summary']['variants'])}")
    return out


def add_vision(package_dir, model: str, *, vision_weights: str = "int8g128", vision_grids=None, vision_compute_type: str = "float32", search: bool = True, workers: int = 4,
               endpoint: Optional[str] = None, cache_dir=None, progress: Callable[[str], None] = print) -> Path:
    """Add the image tower of the source checkpoint to an existing Qwen3.5 package (compiled without it, or with other options): vision weights + grid tables are packed, vision graphs and the plan are
    written, model.json / manifest / m2w.json are updated. The text weights and graphs stay (no text quantization)."""
    out = Path(package_dir)
    spec = arch.ModelSpec.from_dict(json.loads((out / "model.json").read_text(encoding="utf-8")))
    if spec.builder != "qwen35":
        raise ValueError("only Qwen3.5 packages have an image tower")
    source = hub.resolve_model(model, cache_dir=cache_dir, endpoint=endpoint, progress=progress)
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    fresh = arch.spec_from_config(config, None, _load_preprocessor(source))
    if not fresh.vision:
        raise ValueError(f"{model} has no vision tower")
    spec.vision, spec.mrope, spec.tensors = fresh.vision, fresh.mrope, {**spec.tensors, "vision": fresh.tensors["vision"]}
    vision_opt = _vision_options(spec, vision_weights, vision_grids, vision_compute_type)
    manifest = json.loads((out / "weights" / "manifest.json").read_text(encoding="utf-8"))
    for entry in [f for f in manifest["files"] if f["role"].startswith("vision_")]:
        (out / "weights" / entry["file"]).unlink(missing_ok=True)
    manifest["files"] = [f for f in manifest["files"] if not f["role"].startswith("vision_")]
    manifest["files"] += pack_qwen35.pack_vision(spec, TensorStore.open(source), out / "weights", vision_opt["weights"], vision_opt["grids"], pack.DEFAULT_GRID if search else None, workers, progress)
    manifest["quant"]["vision"] = pack_qwen35.vision_quant_entry(vision_opt["weights"])
    (out / "weights" / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    shutil.rmtree(out / "graphs" / "vision", ignore_errors=True)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    plan["vision"] = write_vision_graphs(spec, out, vision_opt["grids"], vision_opt["weights"], vision_opt["compute_type"])
    (out / "model.json").write_text(json.dumps(spec.to_dict(), indent=1), encoding="utf-8", newline="\n")
    (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8", newline="\n")
    index_old = json.loads((out / "m2w.json").read_text(encoding="utf-8"))
    index = _write_index(out, spec, index_old.get("source", ""), plan, manifest, index_old.get("chat_format"), [Path(f).name for f in index_old.get("tokenizer", [])])
    progress(f"added the image tower to {out}: {len(plan['vision']['presets'])} grids, {vision_opt['weights']} weights ({index['summary']['weights_bytes'] / 2**20:.0f} MB of weights in total)")
    return out


def set_recommended(package_dir, progress: Callable[[str], None] = print) -> Path:
    """Write plan.recommended of an existing package from the family's recommendation (no graph is rebuilt) and refresh the file index."""
    out = Path(package_dir)
    spec = arch.ModelSpec.from_dict(json.loads((out / "model.json").read_text(encoding="utf-8")))
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    plan["recommended"] = recommended_for(spec, plan)
    (out / "plan.json").write_text(json.dumps(plan, indent=1), encoding="utf-8", newline="\n")
    index_old = json.loads((out / "m2w.json").read_text(encoding="utf-8"))
    manifest = json.loads((out / "weights" / "manifest.json").read_text(encoding="utf-8"))
    _write_index(out, spec, index_old.get("source", ""), plan, manifest, index_old.get("chat_format"), [Path(f).name for f in index_old.get("tokenizer", [])])
    progress(f"{out}: recommended {plan['recommended']} (family {spec.family}; groups in the package: {plan.get('blocks_per_graph')})")
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

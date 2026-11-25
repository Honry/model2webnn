"""Gemma 4 HF weights -> the weights package (same container and manifest as pack.py, see there): per-layer .safetensors shards with the tensors of the attention half (of the KIND of the layer:
a layer that shares K / V has no k / v projection) and of the feed-forward half (with the per-layer-embedding branch and the layer scalar), plus
    head_extra.safetensors   the projection that turns the embedding into the per-layer inputs (int4) and its norm        (role head_extra)
    ple_embed.safetensors    the per-layer embedding table [vocab, layers * ple_dim], int8 per row symmetric (+ scale)   (role ple_table): read one row per token, never loaded whole
    final_norm / embed*      as pack.py (the tied embedding / lm_head table, int8 per row symmetric)
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ...quantize import DEFAULT_GRID, dequantize_int4, pack_int4, quantize_i8_symmetric_rowwise, quantize_int4_symmetric
from . import pack
from .arch import ModelSpec
from .builders import decoder as D
from .builders import gemma4 as G
from .pack import ShardWriter
from .safetensors_io import TensorStore


def _gamma(store: TensorStore, name: str, shape) -> np.ndarray:
    return store.read(name).astype(np.float16).reshape(shape)


def pack_layer(spec: ModelSpec, store: TensorStore, layer: int, out: Path, grid=DEFAULT_GRID) -> dict:
    names = spec.layer_tensor_names(layer)
    kind = spec.layer_kind(layer)
    weights, errors = {}, {}

    def quantize(prefix: str, roles: list) -> None:
        w = np.concatenate([store.read(names[r]) for r in roles], axis=0) if len(roles) > 1 else store.read(names[roles[0]])
        q, s = quantize_int4_symmetric(w, spec.group_size, grid)
        weights[prefix] = (q, s)
        errors[prefix] = float(np.square(dequantize_int4(q, s) - w).sum() / max(np.square(w).sum(), 1e-30))

    if G.kind_shared(kind):
        quantize("query_projection", ["q_proj"])
    else:
        quantize("qkv_projection", ["q_proj", "k_proj", "v_proj"])
    quantize("output_projection", ["o_proj"])
    quantize("gate_up_projection", ["gate_proj", "up_proj"])
    quantize("down_projection", ["down_proj"])
    quantize("per_layer_input_gate", ["per_layer_input_gate"])
    quantize("per_layer_projection", ["per_layer_projection"])
    gammas = {"input_layernorm_gamma": "input_layernorm", "post_attention_layernorm_gamma": "post_attention_layernorm", "query_norm_gamma": "q_norm", "key_norm_gamma": "k_norm",
              "pre_feedforward_layernorm_gamma": "pre_feedforward_layernorm", "post_feedforward_layernorm_gamma": "post_feedforward_layernorm",
              "post_per_layer_input_norm_gamma": "post_per_layer_input_norm", "layer_scalar": "layer_scalar"}
    writer = ShardWriter(out / f"layer_{layer:03d}.safetensors")
    for name, dtype, shape in G.attention_weight_specs(spec, kind) + G.feed_forward_weight_specs(spec):
        if name.endswith("_weight"):
            q, _ = weights[name[: -len("_weight")]]
            assert tuple(q.shape) == tuple(shape), (name, q.shape, shape)
            writer.add(name, dtype, shape, pack_int4(q))
        elif name.endswith("_scale"):
            _, s = weights[name[: -len("_scale")]]
            assert tuple(s.shape) == tuple(shape), (name, s.shape, shape)
            writer.add(name, dtype, shape, s)
        else:
            writer.add(name, dtype, shape, _gamma(store, names[gammas[name]], shape))
    entry = writer.close()
    entry.update(role="layer", layer=layer, kind=kind, relative_error={k: round(v, 6) for k, v in errors.items()})
    return entry


def pack_head_extra(spec: ModelSpec, store: TensorStore, out: Path, grid=DEFAULT_GRID) -> dict:
    w = store.read(spec.tensors["ple_projection"])
    q, s = quantize_int4_symmetric(w, spec.group_size, grid)
    writer = ShardWriter(out / "head_extra.safetensors")
    for name, dtype, shape in G.head_extra_weight_specs(spec):
        if name == "ple_model_projection_weight":
            writer.add(name, dtype, shape, pack_int4(q))
        elif name == "ple_model_projection_scale":
            writer.add(name, dtype, shape, s)
        else:
            writer.add(name, dtype, shape, _gamma(store, spec.tensors["ple_projection_norm"], shape))
    return {**writer.close(), "role": "head_extra"}


def pack_ple_table(spec: ModelSpec, store: TensorStore, out: Path, chunk_rows: int = 8192) -> dict:
    """int8 per row symmetric, quantized in row chunks (the bf16 table is 5.6 GB as it is)."""
    name = spec.tensors["ple_embed"]
    rows, width = store.shape(name)
    q = np.empty((rows, width), np.int8)
    scale = np.empty((rows, 1), np.float16)
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        q[start:stop], scale[start:stop] = quantize_i8_symmetric_rowwise(store.read(name, rows=(start, stop)))
    writer = ShardWriter(out / "ple_embed.safetensors")
    writer.add("ple_weight", "int8", (rows, width), q)
    writer.add("ple_scale", "float16", (rows, 1), scale)
    return {**writer.close(), "role": "ple_table"}


def pack_model(spec: ModelSpec, store: TensorStore, out_dir, opt: Optional[D.GraphOptions] = None, grid=DEFAULT_GRID, workers: int = 4,
               progress: Optional[Callable[[str], None]] = None) -> dict:
    opt = opt or D.GraphOptions()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = progress or (lambda _m: None)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(pack_layer, spec, store, i, out, grid) for i in range(spec.num_layers)]
        layer_entries = []
        for i, f in enumerate(futures):
            layer_entries.append(f.result())
            log(f"layer {i + 1}/{spec.num_layers} quantized")
    final = ShardWriter(out / "final_norm.safetensors")
    final.add("final_norm_gamma", "float16", (1, 1, spec.hidden), _gamma(store, spec.tensors["final_norm"], (1, 1, spec.hidden)))
    files = layer_entries + [{**final.close(), "role": "final_norm"}, pack_head_extra(spec, store, out, grid)] + pack.pack_embedding(spec, store, out, opt)
    log("embedding quantized")
    files.append(pack_ple_table(spec, store, out))
    log("per-layer embedding table quantized")
    manifest = {
        "format": pack.WEIGHTS_FORMAT, "version": pack.WEIGHTS_VERSION, "container": "safetensors",
        "quant": {"bits": 4, "group_size": spec.group_size, "symmetric": True, "signed_scale": True, "layout": "group-major", "pack": "low-nibble-first",
                  "scale_dtype": "float16", "gamma_dtype": "float16", "merge_projections": True,
                  "embedding": {"bits": 8, "asymmetric": False, "granularity": "row", "padded_vocab": spec.vocab_padded},
                  "ple_table": {"bits": 8, "granularity": "row", "rows": spec.vocab_padded, "width": spec.num_layers * spec.ple_dim}},
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return manifest

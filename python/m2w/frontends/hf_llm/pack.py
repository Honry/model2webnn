"""HF weights -> the weights package: per-layer shards of int4 / f16 tensors named exactly like the graph inputs, plus a manifest.

Layout (manifest v0), same binary conventions the browser runtime already reads (tensors concatenated in a shard, offsets in the manifest):
    weights/layer_000.bin ...   attention-half + feed-forward-half tensors of ONE layer (names = decoder.attention_weight_specs / feed_forward_weight_specs)
    weights/final_norm.bin      final_norm_gamma
    weights/embed.bin | embed_zero_point.bin | embed_scale.bin   the tied embedding / lm_head table, uint8 per-row asymmetric, vocabulary padded
    weights/manifest.json
int4 weights are nibble-packed (low nibble first) in the graph layout [groups, out, 128]; scales and gammas are float16.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ...quantize.int4 import ZERO_SCALE_REPLACEMENT
from ...quantize import DEFAULT_GRID, pack_int4, quantize_i8_symmetric_rowwise, quantize_int4_symmetric, quantize_u8_rowwise
from .arch import ModelSpec
from .builders import decoder as D
from .safetensors_io import TensorStore

WEIGHTS_FORMAT = "m2w-weights"
WEIGHTS_VERSION = 0


class ShardWriter:
    def __init__(self, path: Path):
        self.path, self.tensors, self.offset = path, [], 0
        self._file = open(path, "wb")
        self._hash = hashlib.sha256()

    def add(self, name: str, dtype: str, shape, data: np.ndarray) -> None:
        blob = np.ascontiguousarray(data).tobytes()
        self._file.write(blob)
        self._hash.update(blob)
        self.tensors.append({"name": name, "dtype": dtype, "shape": list(shape), "offset": self.offset, "bytes": len(blob)})
        self.offset += len(blob)

    def close(self) -> dict:
        self._file.close()
        return {"file": self.path.name, "bytes": self.offset, "sha256": self._hash.hexdigest(), "tensors": self.tensors}


def _gamma(store: TensorStore, name: str, shape) -> np.ndarray:
    return store.read(name).astype(np.float16).reshape(shape)


def pack_layer(spec: ModelSpec, store: TensorStore, layer: int, opt: D.GraphOptions, out: Path, grid=DEFAULT_GRID) -> dict:
    """Quantize one layer into layer_XXX.bin. Returns the manifest entry (with the per-layer quantization error)."""
    dims, names = spec.decoder_dims(), spec.layer_tensor_names(layer)
    weights = {}   # projection prefix -> (q [G,N,128] int8, scale [G,N,1] f16)
    errors = {}

    def quantize(prefix: str, roles: list) -> None:
        w = np.concatenate([store.read(names[r]) for r in roles], axis=0) if len(roles) > 1 else store.read(names[roles[0]])
        q, s = quantize_int4_symmetric(w, spec.group_size, grid)
        weights[prefix] = (q, s)
        from ...quantize import dequantize_int4
        errors[prefix] = float(np.square(dequantize_int4(q, s) - w).sum() / max(np.square(w).sum(), 1e-30))

    if opt.merge_projections:
        quantize("qkv_projection", ["q_proj", "k_proj", "v_proj"])
        quantize("gate_up_projection", ["gate_proj", "up_proj"])
    else:
        for prefix, role in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj"), ("gate_projection", "gate_proj"), ("up_projection", "up_proj")):
            quantize(prefix, [role])
    quantize("output_projection", ["o_proj"])
    quantize("down_projection", ["down_proj"])
    gammas = {"input_layernorm_gamma": "input_layernorm", "post_attention_layernorm_gamma": "post_attention_layernorm"}
    if spec.qk_norm:
        gammas.update({"query_norm_gamma": "q_norm", "key_norm_gamma": "k_norm"})

    writer = ShardWriter(out / f"layer_{layer:03d}.bin")
    for name, dtype, shape in D.attention_weight_specs(dims, opt) + D.feed_forward_weight_specs(dims, opt):
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
    entry.update(role="layer", layer=layer, relative_error={k: round(v, 6) for k, v in errors.items()})
    return entry


def pack_embedding(spec: ModelSpec, store: TensorStore, out: Path, opt: Optional[D.GraphOptions] = None) -> list:
    """The tied embedding / lm_head table: uint8, per-row asymmetric, rows padded to ``vocab_padded`` (pad rows: q = 0, zero point 0, scale 1e-3 - they dequantize to exactly 0; a scale of 0 can make the NPU lm_head dramatically slower)."""
    embed_name = spec.tensors["embed"]
    if not spec.tie_embeddings and spec.tensors["lm_head"] in store:
        raise NotImplementedError("untied lm_head is not supported yet")
    weight = store.read(embed_name)
    asym = (opt or D.GraphOptions()).lm_head_format == "uint8-asym"
    if asym:
        q, zp, scale = quantize_u8_rowwise(weight)
    else:
        q, scale = quantize_i8_symmetric_rowwise(weight)
    pad = spec.vocab_padded - spec.vocab
    if pad:
        q = np.concatenate([q, np.zeros((pad, q.shape[1]), q.dtype)])
        scale = np.concatenate([scale, np.full((pad, 1), ZERO_SCALE_REPLACEMENT, np.float16)])   # never 0: see quantize/int4.py
        if asym:
            zp = np.concatenate([zp, np.zeros((pad, 1), np.uint8)])
    parts = ([("embed.bin", "lm_head_weight", "uint8", q), ("embed_zero_point.bin", "lm_head_zero_point", "uint8", zp)] if asym else [("embed.bin", "lm_head_weight", "int8", q)])
    parts.append(("embed_scale.bin", "lm_head_scale", "float16", scale))
    entries = []
    for file, name, dtype, array in parts:
        w = ShardWriter(out / file)
        w.add(name, dtype, array.shape, array)
        entries.append({**w.close(), "role": "embedding"})
    return entries


def pack_model(spec: ModelSpec, store: TensorStore, out_dir, opt: Optional[D.GraphOptions] = None, grid=DEFAULT_GRID, workers: int = 4,
               progress: Optional[Callable[[str], None]] = None) -> dict:
    opt = opt or D.GraphOptions()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = progress or (lambda _m: None)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(pack_layer, spec, store, i, opt, out, grid) for i in range(spec.num_layers)]
        layer_entries = []
        for i, f in enumerate(futures):
            layer_entries.append(f.result())
            log(f"layer {i + 1}/{spec.num_layers} quantized")
    final = ShardWriter(out / "final_norm.bin")
    final.add("final_norm_gamma", "float16", (1, 1, spec.hidden), _gamma(store, spec.tensors["final_norm"], (1, 1, spec.hidden)))
    files = layer_entries + [{**final.close(), "role": "final_norm"}] + pack_embedding(spec, store, out, opt)
    log("embedding quantized")
    manifest = {
        "format": WEIGHTS_FORMAT, "version": WEIGHTS_VERSION,
        "quant": {"bits": 4, "group_size": spec.group_size, "symmetric": True, "signed_scale": True, "layout": "group-major", "pack": "low-nibble-first",
                  "scale_dtype": "float16", "gamma_dtype": "float16", "merge_projections": opt.merge_projections,
                  "embedding": {"bits": 8, "asymmetric": opt.lm_head_format == "uint8-asym", "granularity": "row", "padded_vocab": spec.vocab_padded}},
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return manifest

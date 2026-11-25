"""HF weights -> the weights package: per-layer shards of int4 / f16 tensors named exactly like the graph inputs, plus a manifest.

Layout (manifest v0): every shard is a standard .safetensors file (readable by the `safetensors` library and the Hugging Face tools); the manifest repeats where each tensor is
(absolute file offsets, so a runtime that downloads a whole shard needs no header parsing):
    weights/layer_000.safetensors ...   attention-half + feed-forward-half tensors of ONE layer (names = decoder.attention_weight_specs / feed_forward_weight_specs)
    weights/final_norm.safetensors      final_norm_gamma
    weights/embed.safetensors | embed_zero_point.safetensors | embed_scale.safetensors   the tied embedding / lm_head table (int8 per row symmetric or uint8 asymmetric), vocabulary padded
    weights/manifest.json
int4 weights are nibble-packed (low nibble first) in the graph layout [groups, out, 128]: a safetensors header has no int4 dtype, so they are stored as U8 with the last axis halved
([groups, out, 64]); the file metadata (`dtypes`, `shapes`, JSON) and the manifest carry the logical dtype / shape. Scales and gammas are F16.
"""
from __future__ import annotations

import hashlib
import json
import struct
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


SAFETENSORS_DTYPES = {"int4": "U8", "uint8": "U8", "int8": "I8", "float16": "F16"}


class ShardWriter:
    """Collects the tensors of one shard and writes them as a .safetensors file on close(); the returned manifest entry has absolute file offsets."""

    def __init__(self, path: Path):
        self.path, self._items = path, []

    def add(self, name: str, dtype: str, shape, data: np.ndarray) -> None:
        self._items.append((name, dtype, list(shape), np.ascontiguousarray(data).tobytes()))

    def close(self) -> dict:
        header: dict = {}
        offset = 0
        for name, dtype, shape, blob in self._items:
            stored = list(shape)
            if dtype == "int4":   # nibble-packed: two values per byte along the last axis
                assert len(blob) * 2 == int(np.prod(shape, dtype=np.int64)), (name, shape, len(blob))
                stored[-1] //= 2
            header[name] = {"dtype": SAFETENSORS_DTYPES[dtype], "shape": stored, "data_offsets": [offset, offset + len(blob)]}
            offset += len(blob)
        header = {"__metadata__": {"format": "m2w", "dtypes": json.dumps({n: d for n, d, _, _ in self._items}), "shapes": json.dumps({n: s for n, _, s, _ in self._items})}, **header}
        text = json.dumps(header, separators=(",", ":")).encode("utf-8")
        text += b" " * (-len(text) % 8)   # the data starts 8-byte aligned
        start = 8 + len(text)
        digest = hashlib.sha256()
        with open(self.path, "wb") as f:
            for chunk in (struct.pack("<Q", len(text)), text):
                f.write(chunk)
                digest.update(chunk)
            for _, _, _, blob in self._items:
                f.write(blob)
                digest.update(blob)
        tensors, at = [], start
        for name, dtype, shape, blob in self._items:
            tensors.append({"name": name, "dtype": dtype, "shape": shape, "offset": at, "bytes": len(blob)})
            at += len(blob)
        return {"file": self.path.name, "bytes": at, "sha256": digest.hexdigest(), "tensors": tensors}


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

    writer = ShardWriter(out / f"layer_{layer:03d}.safetensors")
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
    parts = ([("embed.safetensors", "lm_head_weight", "uint8", q), ("embed_zero_point.safetensors", "lm_head_zero_point", "uint8", zp)] if asym
             else [("embed.safetensors", "lm_head_weight", "int8", q)])
    parts.append(("embed_scale.safetensors", "lm_head_scale", "float16", scale))
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
    final = ShardWriter(out / "final_norm.safetensors")
    final.add("final_norm_gamma", "float16", (1, 1, spec.hidden), _gamma(store, spec.tensors["final_norm"], (1, 1, spec.hidden)))
    files = layer_entries + [{**final.close(), "role": "final_norm"}] + pack_embedding(spec, store, out, opt)
    log("embedding quantized")
    manifest = {
        "format": WEIGHTS_FORMAT, "version": WEIGHTS_VERSION, "container": "safetensors",
        "quant": {"bits": 4, "group_size": spec.group_size, "symmetric": True, "signed_scale": True, "layout": "group-major", "pack": "low-nibble-first",
                  "scale_dtype": "float16", "gamma_dtype": "float16", "merge_projections": opt.merge_projections,
                  "embedding": {"bits": 8, "asymmetric": opt.lm_head_format == "uint8-asym", "granularity": "row", "padded_vocab": spec.vocab_padded}},
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return manifest


def repack_to_safetensors(weights_dir) -> dict:
    """Convert the shards of an older package (concatenated tensors in *.bin) to .safetensors files without quantizing again; rewrites manifest.json. Returns the manifest."""
    directory = Path(weights_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("container") == "safetensors":
        return manifest
    files = []
    for entry in manifest["files"]:
        old = directory / entry["file"]
        raw = old.read_bytes()
        writer = ShardWriter(directory / (Path(entry["file"]).stem + ".safetensors"))
        for t in entry["tensors"]:
            writer.add(t["name"], t["dtype"], t["shape"], np.frombuffer(raw, np.uint8, count=t["bytes"], offset=t["offset"]))
        files.append({**writer.close(), **{k: v for k, v in entry.items() if k not in ("file", "bytes", "sha256", "tensors")}})
        old.unlink()
    manifest = {**manifest, "container": "safetensors", "files": files}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return manifest

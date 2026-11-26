"""Qwen3.5 HF weights -> the weights package (same container and manifest as pack.py): per-layer .safetensors shards with the tensors of the attention half of the layer's KIND
(linear_attention = Gated DeltaNet, full_attention) and of the feed-forward half, plus final_norm / embed* as in pack.py.

Transformations done here so that the graphs stay plain:
  * every RMSNorm weight w is stored as (1 + w) (Qwen3.5 computes x * (1 + w)); the gated norm of the Gated DeltaNet is a plain weight;
  * full attention: q_proj holds, per head, [query | gate] (2 * head_dim rows): they are split into the query rows and the gate rows; with partial rotary embeddings the head dims of the query
    and key rows (and of the q / k norm weights) are PERMUTED so that the rotating pairs (i, i + rot/2) sit at (i, i + head_dim/2): the graph rotates the whole head dim (rotate_half) with
    frequencies that are 0 for the dims that do not rotate (rope.partial_inv_freq). Attention scores do not depend on a consistent permutation of the head dims;
  * Gated DeltaNet: in_proj_qkv and in_proj_z are one int4 projection, in_proj_b / in_proj_a stay float16 (64 outputs), the depthwise conv kernel is stored as [taps, 1, channels],
    `decay_rate` = exp(A_log).
The image tower (`pack_vision`, only when model.json has a `vision` entry): weights/vision_head.safetensors, vision_block_NNN.safetensors (one per ViT block), vision_merger.safetensors, one
vision_grid_<h>x<w>.safetensors per grid preset (the position embedding interpolated to the grid, the rotary cos / sin) and vision_pos_embed.safetensors (the learned table itself: only the Python
reference reads it). Linear weights are int8 per row symmetric by default (`vision_weights`: int8 | int8g128 | int4 | float16), the Conv3d kernel is a [hidden, 1536] matrix in the (channel, temporal, kh, kw)
order of the patch vector, biases and LayerNorm are float16.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ...quantize import DEFAULT_GRID, dequantize_i8_symmetric_rowwise, dequantize_int4, pack_int4, quantize_i8_symmetric_grouped, quantize_i8_symmetric_rowwise, quantize_int4_symmetric
from . import pack, vision as V
from .arch import ModelSpec
from .builders import decoder as D
from .builders import qwen35 as Q
from .builders import qwen35_vision as QV
from .pack import ShardWriter
from .safetensors_io import TensorStore


def rope_permutation(head_dim: int, rotary_dim: int) -> np.ndarray:
    """new head-dim index -> old index: the rotating pairs (i, i + rot/2) of the first `rot` dims move to (i, i + head_dim/2), the dims that do not rotate keep their order."""
    half, quarter = head_dim // 2, rotary_dim // 2
    old = np.arange(head_dim)
    passing = old[rotary_dim:]
    return np.concatenate([old[:quarter], passing[: half - quarter], old[quarter:rotary_dim], passing[half - quarter:]])


def _f16(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).astype(np.float16)


def pack_layer(spec: ModelSpec, store: TensorStore, layer: int, out: Path, grid=DEFAULT_GRID) -> dict:
    names = spec.layer_tensor_names(layer)
    kind = spec.layer_kind(layer)
    read = lambda role: store.read(names[role]).astype(np.float32)  # noqa: E731
    plus_one = lambda role: 1.0 + read(role)  # noqa: E731
    weights, errors, plain = {}, {}, {}

    def quantize(prefix: str, matrix: np.ndarray) -> None:
        q, s = quantize_int4_symmetric(matrix, spec.group_size, grid)
        weights[prefix] = (q, s)
        errors[prefix] = float(np.square(dequantize_int4(q, s) - matrix).sum() / max(np.square(matrix).sum(), 1e-30))

    plain["input_layernorm_gamma"] = plus_one("input_layernorm")
    plain["post_attention_layernorm_gamma"] = plus_one("post_attention_layernorm")
    gate_proj, up_proj = read("gate_proj"), read("up_proj")
    quantize("gate_up_projection", np.concatenate([gate_proj, up_proj], axis=0))
    quantize("down_projection", read("down_proj"))
    if kind == "linear_attention":
        quantize("qkvz_projection", np.concatenate([read("in_proj_qkv"), read("in_proj_z")], axis=0))
        quantize("output_projection", read("out_proj"))
        plain["ba_weight"] = np.concatenate([read("in_proj_b"), read("in_proj_a")], axis=0)
        plain["conv_weight"] = np.transpose(read("conv1d")[:, 0, :], (1, 0))[:, None, :]
        plain["decay_rate"] = np.exp(read("A_log"))
        plain["dt_bias"] = read("dt_bias")
        plain["norm_gamma"] = read("linear_norm")
    else:
        d = spec.decoder_dims("full_attention")
        rot = int(d.head_dim * float(spec.rope["full_attention"].get("partial_rotary_factor", 1.0)))
        perm = rope_permutation(d.head_dim, rot)
        q_rows = read("q_proj").reshape(d.num_heads, 2 if spec.attn_output_gate else 1, d.head_dim, spec.hidden)
        query = q_rows[:, 0][:, perm]
        gate = q_rows[:, 1] if spec.attn_output_gate else None
        key = read("k_proj").reshape(d.num_kv_heads, d.head_dim, spec.hidden)[:, perm]
        parts = [query.reshape(-1, spec.hidden)] + ([gate.reshape(-1, spec.hidden)] if gate is not None else []) + [key.reshape(-1, spec.hidden), read("v_proj")]
        quantize("qkvg_projection", np.concatenate(parts, axis=0))
        quantize("output_projection", read("o_proj"))
        plain["query_norm_gamma"] = (1.0 + read("q_norm"))[perm]
        plain["key_norm_gamma"] = (1.0 + read("k_norm"))[perm]

    writer = ShardWriter(out / f"layer_{layer:03d}.safetensors")
    for name, dtype, shape in Q.attention_weight_specs(spec, kind) + Q.feed_forward_weight_specs(spec):
        if name.endswith("_weight") and name[: -len("_weight")] in weights:
            q, _ = weights[name[: -len("_weight")]]
            assert tuple(q.shape) == tuple(shape), (name, q.shape, shape)
            writer.add(name, dtype, shape, pack_int4(q))
        elif name.endswith("_scale") and name[: -len("_scale")] in weights:
            _, s = weights[name[: -len("_scale")]]
            assert tuple(s.shape) == tuple(shape), (name, s.shape, shape)
            writer.add(name, dtype, shape, s)
        else:
            value = plain[name]
            assert int(np.prod(value.shape)) == int(np.prod(shape)), (name, value.shape, shape)
            writer.add(name, dtype, shape, _f16(value).reshape(shape))
    entry = writer.close()
    entry.update(role="layer", layer=layer, kind=kind, relative_error={k: round(v, 6) for k, v in errors.items()})
    return entry


def _vision_linear(name: str, matrix: np.ndarray, weights: str, grid, errors: dict) -> dict:
    """matrix [out, in] -> {tensor name: array} of the weight format (int8 per row symmetric | int8g128 = int8 with a scale per block of 128 input channels | int4 group 128 | float16); the relative quantization error goes to `errors`."""
    if weights == "int8":
        q, scale = quantize_i8_symmetric_rowwise(matrix)
        errors[name] = float(np.square(dequantize_i8_symmetric_rowwise(q, scale) - matrix).sum() / max(np.square(matrix).sum(), 1e-30))
        return {f"{name}_weight": q, f"{name}_scale": scale}
    if weights == "int8g128":
        q, scale = quantize_i8_symmetric_grouped(matrix, 128)
        errors[name] = float(np.square(dequantize_int4(q, scale) - matrix).sum() / max(np.square(matrix).sum(), 1e-30))   # dequantize_int4 = q * scale per block, any integer q
        return {f"{name}_weight": q, f"{name}_scale": scale}
    if weights == "int4":
        q, scale = quantize_int4_symmetric(matrix, 128, grid)
        errors[name] = float(np.square(dequantize_int4(q, scale) - matrix).sum() / max(np.square(matrix).sum(), 1e-30))
        return {f"{name}_weight": pack_int4(q), f"{name}_scale": scale}
    return {f"{name}_weight": _f16(matrix)}


def _write_vision_shard(path: Path, specs: list, tensors: dict, **entry) -> dict:
    writer = ShardWriter(path)
    for name, dtype, shape in specs:
        value = np.asarray(tensors[name])
        if dtype == "int4":   # nibble-packed bytes: the logical shape is the unpacked one
            assert value.size * 2 == int(np.prod(shape)), (name, value.shape, shape)
            writer.add(name, dtype, shape, value)
        else:
            assert value.size == int(np.prod(shape)), (name, value.shape, shape)
            writer.add(name, dtype, shape, value.reshape(shape))
    return {**writer.close(), **entry}


def pack_vision(spec: ModelSpec, store: TensorStore, out: Path, weights: str = "int8", grids=V.DEFAULT_GRIDS, search_grid=None, workers: int = 4,
                progress: Optional[Callable[[str], None]] = None) -> list:
    """The image tower of the checkpoint -> shards of `out` (see the module docstring); returns their manifest entries."""
    t, v = spec.tensors["vision"], spec.vision
    log = progress or (lambda _m: None)
    read = lambda name: store.read(name).astype(np.float32)  # noqa: E731
    rounded = lambda errors: {k: round(e, 6) for k, e in errors.items()}  # noqa: E731

    def block(i: int) -> dict:
        names, errors, tensors = {k: n.format(i=i) for k, n in t["block"].items()}, {}, {}
        for key in ("norm1", "norm2"):
            tensors[f"{key}_gamma"], tensors[f"{key}_beta"] = _f16(read(names[f"{key}_weight"])), _f16(read(names[f"{key}_bias"]))
        for key in ("qkv", "proj", "fc1", "fc2"):
            tensors.update(_vision_linear(key, read(names[f"{key}_weight"]), weights, search_grid, errors))
            tensors[f"{key}_bias"] = _f16(read(names[f"{key}_bias"]))
        return _write_vision_shard(out / f"vision_block_{i:03d}.safetensors", QV.block_weight_specs(spec, weights), tensors, role="vision_block", layer=i, relative_error=rounded(errors))

    entries = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(block, i) for i in range(v["depth"])]
        for i, f in enumerate(futures):
            entries.append(f.result())
            log(f"vision block {i + 1}/{v['depth']} packed")
    errors, tensors = {}, {}
    kernel = read(t["patch_embed_weight"])   # Conv3d [hidden, channels, temporal, patch, patch] -> the matrix over the flattened patch (channel, temporal, kh, kw)
    tensors.update(_vision_linear("patch_embed", kernel.reshape(kernel.shape[0], -1), weights, search_grid, errors))
    tensors["patch_embed_bias"] = _f16(read(t["patch_embed_bias"]))
    entries.append(_write_vision_shard(out / "vision_head.safetensors", QV.head_weight_specs(spec, weights), tensors, role="vision_head", relative_error=rounded(errors)))
    errors, tensors = {}, {"merger_norm_gamma": _f16(read(t["merger_norm_weight"])), "merger_norm_beta": _f16(read(t["merger_norm_bias"]))}
    for key in ("merger_fc1", "merger_fc2"):
        tensors.update(_vision_linear(key, read(t[f"{key}_weight"]), weights, search_grid, errors))
        tensors[f"{key}_bias"] = _f16(read(t[f"{key}_bias"]))
    entries.append(_write_vision_shard(out / "vision_merger.safetensors", QV.merger_weight_specs(spec, weights), tensors, role="vision_merger", relative_error=rounded(errors)))
    table, side = read(t["pos_embed"]), v["pos_grid"]
    entries.append(_write_vision_shard(out / "vision_pos_embed.safetensors", [("pos_embed_table", "float16", (side * side, v["hidden"]))], {"pos_embed_table": _f16(table)}, role="vision_pos_embed"))
    for grid in grids:
        V.check_grid(grid, v["merge_size"])
        cos, sin = V.rotary_tables(v["hidden"] // v["heads"], grid[0], grid[1], v["merge_size"])
        tables = {"pos_embed": _f16(V.interpolated_pos_embed(table, grid[0], grid[1], v["merge_size"])), "rope_cos": _f16(cos), "rope_sin": _f16(sin)}
        entries.append(_write_vision_shard(out / f"vision_grid_{V.grid_name(grid)}.safetensors", QV.table_specs(spec, grid), tables, role="vision_grid", grid=list(grid)))
    log(f"vision tower packed ({weights}, grids {', '.join(V.grid_name(g) for g in grids)})")
    return entries


def vision_quant_entry(weights: str) -> dict:
    return {"weights": weights, "granularity": {"int8": "row", "int8g128": "group", "int4": "group"}.get(weights, "none"), "scale_dtype": "float16", "bias_dtype": "float16",
            "conv3d": "reshaped to a [hidden, channels * temporal * patch * patch] matrix"}


def pack_model(spec: ModelSpec, store: TensorStore, out_dir, opt: Optional[D.GraphOptions] = None, grid=DEFAULT_GRID, workers: int = 4,
               progress: Optional[Callable[[str], None]] = None, vision: Optional[dict] = None) -> dict:
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
    final.add("final_norm_gamma", "float16", (1, 1, spec.hidden), _f16(1.0 + store.read(spec.tensors["final_norm"]).astype(np.float32)).reshape(1, 1, spec.hidden))
    files = layer_entries + [{**final.close(), "role": "final_norm"}] + pack.pack_embedding(spec, store, out, opt)
    log("embedding quantized")
    if vision:   # {"weights": "int8" | "int8g128" | "int4" | "float16", "grids": [(h, w), ...]}
        files += pack_vision(spec, store, out, vision["weights"], vision["grids"], grid, workers, progress)
    manifest = {
        "format": pack.WEIGHTS_FORMAT, "version": pack.WEIGHTS_VERSION, "container": "safetensors",
        "quant": {"bits": 4, "group_size": spec.group_size, "symmetric": True, "signed_scale": True, "layout": "group-major", "pack": "low-nibble-first",
                  "scale_dtype": "float16", "gamma_dtype": "float16", "merge_projections": True, "rmsnorm_weight": "stored as 1 + w",
                  "embedding": {"bits": 8, "asymmetric": False, "granularity": "row", "padded_vocab": spec.vocab_padded}},
        "files": files,
    }
    if vision:
        manifest["quant"]["vision"] = vision_quant_entry(vision["weights"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8", newline="\n")
    return manifest

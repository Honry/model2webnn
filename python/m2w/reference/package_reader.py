"""Read a weights package back as float32 tensors under the ORIGINAL HF names (dequantized): feeds ReferenceModel with exactly what the device will use,
so that 'graphs + protocol' can be checked separately from 'int4 quantization noise'."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..frontends.hf_llm.arch import ModelSpec
from ..quantize import dequantize_i8_symmetric_rowwise, dequantize_int4, dequantize_u8_rowwise, unpack_int4


class PackageReader:
    def __init__(self, spec: ModelSpec, weights_dir):
        self.spec, self.dir = spec, Path(weights_dir)
        manifest = json.loads((self.dir / "manifest.json").read_text())
        self.merged = manifest["quant"]["merge_projections"]
        self.files = manifest["files"]
        self._cache: dict = {}
        by_hf = {spec.tensors["embed"]: ("embed", None, None), spec.tensors["final_norm"]: ("final_norm", None, None)}
        for i in range(spec.num_layers):
            for role, name in spec.layer_tensor_names(i).items():
                by_hf[name] = ("layer", i, role)
        self._by_hf = by_hf

    def _raw(self, shard: dict, entry: dict) -> np.ndarray:
        raw = np.fromfile(self.dir / shard["file"], dtype=np.uint8, count=entry["bytes"], offset=entry["offset"])
        shape = tuple(entry["shape"])
        if entry["dtype"] == "int4":
            return unpack_int4(raw, shape)
        return raw.view({"float16": np.float16, "uint8": np.uint8, "int8": np.int8}[entry["dtype"]]).reshape(shape)

    def _layer(self, i: int) -> dict:
        if ("layer", i) not in self._cache:
            shard = next(f for f in self.files if f["role"] == "layer" and f["layer"] == i)
            t = {e["name"]: self._raw(shard, e) for e in shard["tensors"]}
            s, aw, kw = self.spec, self.spec.num_heads * self.spec.head_dim, self.spec.num_kv_heads * self.spec.head_dim
            out = {}
            deq = lambda prefix: dequantize_int4(t[prefix + "_weight"], t[prefix + "_scale"])  # noqa: E731
            if self.merged:
                qkv, gu = deq("qkv_projection"), deq("gate_up_projection")
                out.update(q_proj=qkv[:aw], k_proj=qkv[aw:aw + kw], v_proj=qkv[aw + kw:], gate_proj=gu[:s.ffn], up_proj=gu[s.ffn:])
            else:
                out.update(q_proj=deq("query"), k_proj=deq("key"), v_proj=deq("value"), gate_proj=deq("gate_projection"), up_proj=deq("up_projection"))
            out.update(o_proj=deq("output_projection"), down_proj=deq("down_projection"),
                       input_layernorm=t["input_layernorm_gamma"].astype(np.float32).reshape(-1),
                       post_attention_layernorm=t["post_attention_layernorm_gamma"].astype(np.float32).reshape(-1))
            if s.qk_norm:
                out.update(q_norm=t["query_norm_gamma"].astype(np.float32).reshape(-1), k_norm=t["key_norm_gamma"].astype(np.float32).reshape(-1))
            self._cache[("layer", i)] = out
        return self._cache[("layer", i)]

    def read(self, hf_name: str) -> np.ndarray:
        kind, layer, role = self._by_hf[hf_name]
        if kind == "layer":
            return self._layer(layer)[role]
        if kind == "final_norm":
            shard = next(f for f in self.files if f["role"] == "final_norm")
            return self._raw(shard, shard["tensors"][0]).astype(np.float32).reshape(-1)
        if "embed" not in self._cache:
            parts = {e["name"]: self._raw(f, e) for f in self.files if f["role"] == "embedding" for e in f["tensors"]}
            table = (dequantize_u8_rowwise(parts["lm_head_weight"], parts["lm_head_zero_point"], parts["lm_head_scale"]) if "lm_head_zero_point" in parts
                     else dequantize_i8_symmetric_rowwise(parts["lm_head_weight"], parts["lm_head_scale"]))
            self._cache["embed"] = table[: self.spec.vocab]
        return self._cache["embed"]

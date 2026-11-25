"""Read the weights of a Gemma 4 package back as float32 tensors under the ORIGINAL HF names (relative to 'model.language_model.'), dequantized: feeds Gemma4Reference with exactly what the
device will use, so that 'graphs + protocol' can be checked separately from the quantization noise. Also gives the runner the raw tensors of the manifest."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..frontends.hf_llm.arch import ModelSpec
from ..quantize import dequantize_int4, unpack_int4


class Gemma4PackageReader:
    def __init__(self, spec: ModelSpec, weights_dir):
        self.spec, self.dir = spec, Path(weights_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.files = self.manifest["files"]
        self._layer = {f["layer"]: f for f in self.files if f["role"] == "layer"}
        self._role = {f["role"]: f for f in self.files if f["role"] in ("final_norm", "head_extra", "ple_table")}
        self._embedding = {t["name"]: (f, t) for f in self.files if f["role"] == "embedding" for t in f["tensors"]}
        self._cache: dict = {}

    # ----------------------------------------------------------------------------------------------- raw access
    def raw(self, shard: dict, name: str, mmap: bool = False) -> np.ndarray:
        entry = next(t for t in shard["tensors"] if t["name"] == name)
        shape = tuple(entry["shape"])
        if mmap:   # big tables: rows are read on demand
            return np.memmap(self.dir / shard["file"], np.uint8, "r", entry["offset"], entry["bytes"]).view({"int8": np.int8, "float16": np.float16, "uint8": np.uint8}[entry["dtype"]]).reshape(shape)
        data = np.fromfile(self.dir / shard["file"], np.uint8, count=entry["bytes"], offset=entry["offset"])
        if entry["dtype"] == "int4":
            return unpack_int4(data, shape)
        return data.view({"float16": np.float16, "uint8": np.uint8, "int8": np.int8}[entry["dtype"]]).reshape(shape)

    def has(self, shard: dict, name: str) -> bool:
        return any(t["name"] == name for t in shard["tensors"])

    def _dequant(self, shard: dict, prefix: str) -> np.ndarray:
        return dequantize_int4(self.raw(shard, prefix + "_weight"), self.raw(shard, prefix + "_scale"))

    def embedding_rows(self, rows) -> np.ndarray:
        (fw, tw), (fs, ts) = self._embedding["lm_head_weight"], self._embedding["lm_head_scale"]
        if "table" not in self._cache:
            self._cache["table"] = (self.raw(fw, "lm_head_weight", mmap=True), self.raw(fs, "lm_head_scale", mmap=True))
        q, s = self._cache["table"]
        return q[rows].astype(np.float32) * s[rows].astype(np.float32)

    def embedding_rows_raw(self):
        """(weight int8 [V, hidden], scale float16 [V, 1]) of the tied embedding / lm_head table, memory-mapped."""
        self.embedding_rows(slice(0, 1))
        return self._cache["table"]

    def ple_rows(self, rows) -> np.ndarray:
        shard = self._role["ple_table"]
        if "ple" not in self._cache:
            self._cache["ple"] = (self.raw(shard, "ple_weight", mmap=True), self.raw(shard, "ple_scale", mmap=True))
        q, s = self._cache["ple"]
        return q[rows].astype(np.float32) * s[rows].astype(np.float32)

    # ----------------------------------------------------------------------------------------------- HF-named view
    def read(self, name: str, rows=None) -> np.ndarray:
        spec = self.spec
        if name == "embed_tokens.weight":
            return self.embedding_rows(slice(*rows) if rows else slice(None))
        if name == "embed_tokens_per_layer.weight":
            return self.ple_rows(slice(*rows) if rows else slice(None))
        if name == "norm.weight":
            return self.raw(self._role["final_norm"], "final_norm_gamma").astype(np.float32).reshape(-1)
        if name == "per_layer_model_projection.weight":
            return self._dequant(self._role["head_extra"], "ple_model_projection")
        if name == "per_layer_projection_norm.weight":
            return self.raw(self._role["head_extra"], "ple_projection_norm_gamma").astype(np.float32).reshape(-1)
        m = re.fullmatch(r"layers\.(\d+)\.(.+)", name)
        if not m:
            raise KeyError(name)
        i, tail = int(m.group(1)), m.group(2)
        shard = self._layer[i]
        dims = spec.decoder_dims(spec.layer_types[i])
        aw, kw, f = dims.attention_width, dims.kv_width, spec.ffn
        gammas = {"input_layernorm.weight": "input_layernorm_gamma", "post_attention_layernorm.weight": "post_attention_layernorm_gamma",
                  "pre_feedforward_layernorm.weight": "pre_feedforward_layernorm_gamma", "post_feedforward_layernorm.weight": "post_feedforward_layernorm_gamma",
                  "post_per_layer_input_norm.weight": "post_per_layer_input_norm_gamma", "self_attn.q_norm.weight": "query_norm_gamma", "self_attn.k_norm.weight": "key_norm_gamma",
                  "layer_scalar": "layer_scalar"}
        if tail in gammas:
            return self.raw(shard, gammas[tail]).astype(np.float32).reshape(-1)
        if tail in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"):
            merged = self._dequant(shard, "query_projection") if self.has(shard, "query_projection_weight") else self._dequant(shard, "qkv_projection")
            return {"q": merged[:aw], "k": merged[aw:aw + kw], "v": merged[aw + kw:]}[tail[10]]
        if tail in ("mlp.gate_proj.weight", "mlp.up_proj.weight"):
            merged = self._dequant(shard, "gate_up_projection")
            return merged[:f] if tail.startswith("mlp.gate") else merged[f:]
        flat = {"self_attn.o_proj.weight": "output_projection", "mlp.down_proj.weight": "down_projection", "per_layer_input_gate.weight": "per_layer_input_gate",
                "per_layer_projection.weight": "per_layer_projection"}
        if tail in flat:
            return self._dequant(shard, flat[tail])
        raise KeyError(name)

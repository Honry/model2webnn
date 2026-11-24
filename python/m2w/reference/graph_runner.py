"""Executes the COMPILED graphs (HEAD / BLOCK / TAIL / lm_head / flush) with the numpy interpreter and the packed weights, following the runtime protocol:
chunked prefill (last chunk left-padded), staged-KV decode, flush of the staging buffer. It is the executable specification of the browser runtime
and checks graphs + weights + per-layer-type masks / RoPE end to end on the CPU.

Layer types: the BLOCK graph for layer i also holds the attention half of layer i+1, so rope_cos / rope_sin / attention_mask are bound with the tables of the type
of layer i+1 (HEAD: layer 0). A sliding-attention layer differs from a full one only in that mask (and its RoPE table): the graph is the same.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..frontends.hf_llm.arch import ModelSpec
from ..frontends.hf_llm.builders import decoder as D
from ..ir.interpreter import run
from ..quantize import unpack_int4
from . import protocol


class PackageRunner:
    def __init__(self, spec: ModelSpec, weights_dir, *, max_length: int = 128, chunk_size: int = 16, staging_rows: int = 8,
                 opt: Optional[D.GraphOptions] = None):
        self.spec, self.dir = spec, Path(weights_dir)   # a directory with manifest.json and the shards
        self.opt = opt or D.GraphOptions(kv_style="staged", staging_rows=staging_rows)
        self.R, self.C, self.P = self.opt.staging_rows, chunk_size, max_length - 1
        if self.P < self.C:
            raise ValueError("max_length is smaller than the prefill chunk")
        manifest = json.loads((self.dir / "manifest.json").read_text())
        self.files = {(f["role"], f.get("layer")): f for f in manifest["files"]}
        dims = spec.decoder_dims()
        self.g = {
            "head1": D.build_head_graph(dims, self.opt, seq=1, past_len=self.P), "block1": D.build_block_graph(dims, self.opt, seq=1, past_len=self.P),
            "tail1": D.build_tail_graph(dims, self.opt, seq=1),
            "headC": D.build_head_graph(dims, self.opt, seq=self.C, past_len=self.P), "blockC": D.build_block_graph(dims, self.opt, seq=self.C, past_len=self.P),
            "tailC": D.build_tail_graph(dims, self.opt, seq=self.C), "lm_head": D.build_lm_head_graph(dims, self.opt),
            "flush": D.build_kv_update_graph(dims, seq=self.R, past_len=self.P),
        }
        self.attention_names = {n for n, _, _ in D.attention_weight_specs(dims, self.opt)}
        self.ffn_names = {n for n, _, _ in D.feed_forward_weight_specs(dims, self.opt)}
        self.externals = {t["name"]: self._tensor(f, t) for f in manifest["files"] if f["role"] == "embedding" for t in f["tensors"]}
        self.final_gamma = self._tensor(self.files[("final_norm", None)], self.files[("final_norm", None)]["tensors"][0])
        self.reset()

    # -------------------------------------------------------------------------------------------------- weights
    def _tensor(self, shard: dict, entry: dict) -> np.ndarray:
        raw = np.fromfile(self.dir / shard["file"], dtype=np.uint8, count=entry["bytes"], offset=entry["offset"])
        shape = tuple(entry["shape"])
        if entry["dtype"] == "int4":
            return unpack_int4(raw, shape)
        return raw.view({"float16": np.float16, "uint8": np.uint8, "int8": np.int8}[entry["dtype"]]).reshape(shape)

    def _layer_weights(self, layer: int, names: set) -> dict:
        shard = self.files[("layer", layer)]
        return {t["name"]: self._tensor(shard, t) for t in shard["tensors"] if t["name"] in names}

    def _embed_row(self, token: int) -> np.ndarray:
        q, s = self.externals["lm_head_weight"], self.externals["lm_head_scale"]
        zero_point = float(self.externals["lm_head_zero_point"][token, 0]) if "lm_head_zero_point" in self.externals else 0.0   # int8-sym: no zero point
        return ((q[token].astype(np.float32) - zero_point) * float(s[token, 0])).astype(np.float16)

    # -------------------------------------------------------------------------------------------------- state
    def reset(self) -> None:
        s = self.spec
        zeros = lambda rows: np.zeros((1, s.num_kv_heads, rows, s.head_dim), np.float16)  # noqa: E731
        self.k = [zeros(self.P) for _ in range(s.num_layers)]
        self.v = [zeros(self.P) for _ in range(s.num_layers)]
        self.sk = [zeros(self.R) for _ in range(s.num_layers)]
        self.sv = [zeros(self.R) for _ in range(s.num_layers)]
        self.base = self.filled = 0

    def _chain(self, kind: str, hidden, rope, masks, selectors, last: bool):
        """One forward pass over HEAD, the BLOCKs and (last chunk / decode) TAIL + lm_head. kind: 'C' (prefill chunk) or '1' (decode)."""
        s, decode = self.spec, kind == "1"
        types = s.layer_types

        def attention_feeds(layer):
            cos, sin = rope[types[layer]]
            feeds = {"rope_cos": cos, "rope_sin": sin, "attention_mask": masks[types[layer]], "past_key": self.k[layer], "past_value": self.v[layer],
                     **selectors, **self._layer_weights(layer, self.attention_names)}
            if decode:
                feeds.update(staging_key=self.sk[layer], staging_value=self.sv[layer])
            return feeds

        def store(layer, out):
            if decode:
                self.sk[layer], self.sv[layer] = out["updated_key"], out["updated_value"]
            else:
                self.k[layer], self.v[layer] = out["updated_key"], out["updated_value"]

        out = run(self.g["head" + kind], {"hidden_states": hidden, **attention_feeds(0)})
        store(0, out)
        residual, attention = out["residual"], out["attention_output"]
        for layer in range(s.num_layers - 1):
            out = run(self.g["block" + kind], {"residual": residual, "attention_output": attention, **attention_feeds(layer + 1),
                                               **self._layer_weights(layer, self.ffn_names)})
            store(layer + 1, out)
            residual, attention = out["next_residual"], out["next_attention_output"]
        if not last:
            return None
        last_layer = s.num_layers - 1
        tail = run(self.g["tail" + kind], {"residual": residual, "attention_output": attention, **self._layer_weights(last_layer, self.ffn_names),
                                           "final_norm_gamma": self.final_gamma})
        logits = run(self.g["lm_head"], {"final_hidden_states": tail["final_hidden_states"]}, externals=self.externals)["logits"]
        return logits.reshape(-1)[: s.vocab].astype(np.float32)

    # -------------------------------------------------------------------------------------------------- prefill
    def prefill(self, tokens: list) -> np.ndarray:
        """Chunked prefill of the whole prompt; returns the logits of the last prompt token."""
        s, n, C, P = self.spec, len(tokens), self.C, self.P
        if n >= P:
            raise ValueError(f"prompt of {n} tokens does not fit into the {P}-row cache")
        self.reset()
        chunks = [(start, tokens[start:start + C]) for start in range(0, n, C)]
        logits = None
        for index, (start, chunk) in enumerate(chunks):
            valid = len(chunk)
            pad = C - valid
            hidden = np.zeros((1, C, s.hidden), np.float16)
            for i, t in enumerate(chunk):
                hidden[0, pad + i] = self._embed_row(t)
            rope = protocol.rope_tables(s, list(range(start, start + valid)), C, pad)
            masks = protocol.prefill_masks(s, P, C, start, valid)
            keep, selector = protocol.prefill_selectors(P, C, start, valid)
            last = index == len(chunks) - 1
            logits = self._chain("C", hidden, rope, masks, {"rows_to_keep": keep, "row_selector": selector}, last)
        self.base, self.filled = n, 0     # the prompt is in the main cache, the staging buffer is empty
        self.position = n
        return logits

    # -------------------------------------------------------------------------------------------------- decode
    def _flush(self) -> None:
        keep, selector = protocol.flush_selectors(self.P, self.R, self.base)
        for layer in range(self.spec.num_layers):
            out = run(self.g["flush"], {"past_key": self.k[layer], "past_value": self.v[layer], "new_key": self.sk[layer], "new_value": self.sv[layer],
                                        "rows_to_keep": keep, "row_selector": selector})
            self.k[layer], self.v[layer] = out["updated_key"], out["updated_value"]
        self.base += self.R
        self.filled = 0

    def step(self, token: int) -> np.ndarray:
        """Feed one token at the current position (staged KV); returns the logits for the next one."""
        s, P, R = self.spec, self.P, self.R
        if self.position >= P:
            raise ValueError("the KV cache is full")
        if self.filled == R:
            self._flush()
        pos = self.position
        hidden = self._embed_row(token).reshape(1, 1, -1)
        rope = protocol.rope_tables(s, [pos], 1, 0)
        masks = protocol.decode_masks(s, P, R, pos, self.base, self.filled)
        keep, selector = protocol.decode_selectors(R, self.filled)
        logits = self._chain("1", hidden, rope, masks, {"staging_rows_to_keep": keep, "staging_row_selector": selector}, True)
        self.filled += 1
        self.position += 1
        return logits

    def generate(self, tokens: list, max_new_tokens: int, eos=()) -> tuple:
        """Greedy decoding. Returns (generated token ids, [logits of every step])."""
        logits = self.prefill(tokens)
        out, all_logits = [], [logits]
        for _ in range(max_new_tokens):
            token = int(np.argmax(logits))
            out.append(token)
            if token in eos or len(out) == max_new_tokens or self.position >= self.P:
                break
            logits = self.step(token)
            all_logits.append(logits)
        return out, all_logits

"""Executes the COMPILED graphs (HEAD / BLOCK / TAIL / lm_head / flush) with the numpy interpreter and the packed weights, following the runtime protocol:
chunked prefill (one or several chunk sizes, the last chunk left-padded), decode with one of the KV protocols (staged / onehot / tensorScatter, see protocol.py).
It is the executable specification of the browser runtime and checks graphs + weights + per-layer-type masks / RoPE end to end on the CPU.

Layer types: the BLOCK graph for layer i also holds the attention half of layer i+1, so rope_cos / rope_sin / attention_mask are bound with the tables of the type
of layer i+1 (HEAD: layer 0). A sliding-attention layer differs from a full one only in that mask (and its RoPE table): the graph is the same.

Two ways to get the graphs: built from GraphOptions (unit tests), or read from a package (`PackageRunner.from_package`: the files `m2w compile` wrote, the ones the browser loads).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..frontends.hf_llm.arch import ModelSpec
from ..frontends.hf_llm.builders import decoder as D
from ..ir import serialize
from ..ir.interpreter import run
from ..quantize import unpack_int4
from . import protocol


class PackageRunner:
    def __init__(self, spec: ModelSpec, weights_dir, *, max_length: int = 128, chunk_size: int = 16, chunk_sizes=None, staging_rows: int = 8,
                 opt: Optional[D.GraphOptions] = None, kv_style: Optional[str] = None, compute_type: Optional[str] = None, decode_group: int = 1, prefill_group: int = 1,
                 prefill_fixed_tokens: int = 200, graphs: Optional[dict] = None, per_block_controls: Optional[bool] = None):
        self.spec, self.dir = spec, Path(weights_dir)   # a directory with manifest.json and the shards
        opt = opt or D.GraphOptions(kv_style="staged", staging_rows=staging_rows)
        changes = {k: v for k, v in (("kv_style", kv_style), ("compute_type", compute_type)) if v}
        self.opt = D.with_options(opt, **changes) if changes else opt
        self.R, self.P, self.style = self.opt.staging_rows, max_length - 1, self.opt.kv_style
        self.sizes = sorted(set(chunk_sizes or [chunk_size]))
        self.fixed_tokens = prefill_fixed_tokens
        self.decode_group, self.prefill_group = decode_group, prefill_group
        self.per_block = len(set(spec.layer_types)) > 1 if per_block_controls is None else per_block_controls   # block groups: rope / mask per block (hybrid models)
        if self.P < min(self.sizes):
            raise ValueError("max_length is smaller than the prefill chunk")
        manifest = json.loads((self.dir / "manifest.json").read_text())
        self.files = {(f["role"], f.get("layer")): f for f in manifest["files"]}
        dims = spec.decoder_dims()
        self.g = graphs or self._build_graphs(dims)
        self.attention_names = {n for n, _, _ in D.attention_weight_specs(dims, self.opt)}
        self.ffn_names = {n for n, _, _ in D.feed_forward_weight_specs(dims, self.opt)}
        self.externals = {t["name"]: self._tensor(f, t) for f in manifest["files"] if f["role"] == "embedding" for t in f["tensors"]}
        self.final_gamma = self._tensor(self.files[("final_norm", None)], self.files[("final_norm", None)]["tensors"][0])
        self.reset()

    # -------------------------------------------------------------------------------------------------- graphs
    def _build_graphs(self, dims) -> dict:
        opt, P, n = self.opt, self.P, self.spec.num_layers
        counts = lambda k: sorted({k, (n - 1) % k} - {0})   # noqa: E731
        g = {"decode": {"head": D.build_head_graph(dims, opt, seq=1, past_len=P), "block": D.build_block_graph(dims, opt, seq=1, past_len=P),
                        "tail": D.build_tail_graph(dims, opt, seq=1), "groups": {}},
             "prefill": {}, "lm_head": D.build_lm_head_graph(dims, opt)}
        if self.style == "staged":
            g["decode"]["flush"] = D.build_kv_update_graph(dims, seq=self.R, past_len=P)
        if self.decode_group > 1:
            g["decode"]["groups"] = {c: D.build_block_group_graph(dims, opt, block_count=c, seq=1, past_len=P, per_block_controls=self.per_block) for c in counts(self.decode_group)}
        for C in self.sizes:
            g["prefill"][C] = {"head": D.build_head_graph(dims, opt, seq=C, past_len=P), "block": D.build_block_graph(dims, opt, seq=C, past_len=P),
                               "tail": D.build_tail_graph(dims, opt, seq=C), "groups": {}}
            if self.prefill_group > 1:
                g["prefill"][C]["groups"] = {c: D.build_block_group_graph(dims, opt, block_count=c, seq=C, past_len=P, per_block_controls=self.per_block) for c in counts(self.prefill_group)}
        return g

    @classmethod
    def from_package(cls, package, spec: ModelSpec, *, variant: Optional[str] = None, max_length: int = 0, chunk_sizes=None, decode_group: int = 1, prefill_group: int = 1,
                     prefill_fixed_tokens: int = 200) -> "PackageRunner":
        """The runner of one variant of a package, with the graph files of its plan.json (what the browser runtime loads)."""
        package = Path(package)
        plan = json.loads((package / "plan.json").read_text(encoding="utf-8"))
        info = next(v for v in plan["variants"] if v["id"] == (variant or plan["default_variant"]))
        table = plan["graphs"][info["id"]]
        L = max_length or min(plan["max_lengths"])
        sizes = sorted(chunk_sizes or [min(int(c) for c in table["prefill"][str(L)])])
        load = lambda path: serialize.load(package / path)   # noqa: E731
        decode_files = table["decode"][str(L)]
        decode = {"head": load(decode_files["head"]), "block": load(decode_files["block"]), "tail": load(decode_files["tail"]),
                  "groups": {int(c): load(path) for c, path in decode_files.get("block_groups", {}).items()} if decode_group > 1 else {}}
        if decode_files.get("flush"):
            decode["flush"] = load(decode_files["flush"])
        prefill = {}
        for C in sizes:
            files = table["prefill"][str(L)][str(C)]
            prefill[C] = {"head": load(files["head"]), "block": load(files["block"]), "tail": load(files["tail"]),
                          "groups": {int(c): load(path) for c, path in files.get("block_groups", {}).items()} if prefill_group > 1 else {}}
        opt = D.GraphOptions(**{**plan["profile"], "compute_type": info["compute_type"], "kv_style": info["kv_style"], "staging_rows": info.get("staging_rows", 64),
                                       "scatter_index_type": info.get("scatter_index_type", "int32")})
        return cls(spec, package / "weights", max_length=L, chunk_sizes=sizes, opt=opt, decode_group=decode_group, prefill_group=prefill_group,
                   prefill_fixed_tokens=prefill_fixed_tokens, graphs={"decode": decode, "prefill": prefill, "lm_head": load(table["lm_head"])},
                   per_block_controls=plan.get("block_group_controls") == "per_block")

    # -------------------------------------------------------------------------------------------------- weights
    def _tensor(self, shard: dict, entry: dict) -> np.ndarray:
        raw = np.fromfile(self.dir / shard["file"], dtype=np.uint8, count=entry["bytes"], offset=entry["offset"])
        shape = tuple(entry["shape"])
        if entry["dtype"] == "int4":
            return unpack_int4(raw, shape)
        return raw.view({"float16": np.float16, "uint8": np.uint8, "int8": np.int8}[entry["dtype"]]).reshape(shape)

    def _layer_weights(self, layer: int, names: set, prefix: str = "") -> dict:
        shard = self.files[("layer", layer)]
        return {prefix + t["name"]: self._tensor(shard, t) for t in shard["tensors"] if t["name"] in names}

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
        self.base = self.filled = self.pad = self.position = 0

    @property
    def cache_full(self) -> bool:
        """No row left for the next decode step (tensorScatter: the padding rows in front of the prompt are used too)."""
        return self.position + (self.pad if self.style == "tensorScatter" else 0) >= self.P

    def _chain(self, graphs: dict, decode: bool, hidden, rope, masks, selectors, last: bool, group: int):
        """One forward pass over HEAD, the BLOCKs (single or in groups) and (last chunk / decode) TAIL + lm_head."""
        s = self.spec
        types = s.layer_types
        staged_decode = decode and self.style == "staged"

        def controls(layer, prefix=""):
            cos, sin = rope[types[layer]]
            return {prefix + "rope_cos": cos, prefix + "rope_sin": sin, prefix + "attention_mask": masks[types[layer]]}

        def shared(layer):
            return {**controls(layer), **selectors}

        def cache_feeds(layer, prefix=""):
            feeds = {prefix + "past_key": self.k[layer], prefix + "past_value": self.v[layer]}
            if staged_decode:
                feeds.update({prefix + "staging_key": self.sk[layer], prefix + "staging_value": self.sv[layer]})
            return feeds

        def store(layer, out, prefix=""):
            if staged_decode:
                self.sk[layer], self.sv[layer] = out[prefix + "updated_key"], out[prefix + "updated_value"]
            else:
                self.k[layer], self.v[layer] = out[prefix + "updated_key"], out[prefix + "updated_value"]

        out = run(graphs["head"], {"hidden_states": hidden, **shared(0), **cache_feeds(0), **self._layer_weights(0, self.attention_names)})
        store(0, out)
        residual, attention = out["residual"], out["attention_output"]
        layer = 0
        while layer < s.num_layers - 1:
            count = min(group, s.num_layers - 1 - layer) if group > 1 else 1
            if count == 1:
                out = run(graphs["block"], {"residual": residual, "attention_output": attention, **shared(layer + 1), **cache_feeds(layer + 1),
                                            **self._layer_weights(layer + 1, self.attention_names), **self._layer_weights(layer, self.ffn_names)})
                store(layer + 1, out)
            else:
                feeds = {"residual": residual, "attention_output": attention, **(selectors if self.per_block else shared(layer + 1))}
                for j in range(count):
                    prefix = f"b{j}_"
                    if self.per_block:
                        feeds.update(controls(layer + j + 1, prefix))
                    feeds.update(self._layer_weights(layer + j, self.ffn_names, prefix))
                    feeds.update(self._layer_weights(layer + j + 1, self.attention_names, prefix))
                    feeds.update(cache_feeds(layer + j + 1, prefix))
                out = run(graphs["groups"][count], feeds)
                for j in range(count):
                    store(layer + j + 1, out, f"b{j}_")
            residual, attention = out["next_residual"], out["next_attention_output"]
            layer += count
        if not last:
            return None
        last_layer = s.num_layers - 1
        tail = run(graphs["tail"], {"residual": residual, "attention_output": attention, **self._layer_weights(last_layer, self.ffn_names),
                                    "final_norm_gamma": self.final_gamma})
        logits = run(self.g["lm_head"], {"final_hidden_states": tail["final_hidden_states"]}, externals=self.externals)["logits"]
        return logits.reshape(-1)[: s.vocab].astype(np.float32)

    # -------------------------------------------------------------------------------------------------- prefill
    def plan(self, token_count: int) -> list:
        return protocol.plan_prefill(self.sizes, token_count, self.fixed_tokens)

    def prefill(self, tokens: list, plan: Optional[list] = None) -> np.ndarray:
        """Chunked prefill of the whole prompt; returns the logits of the last prompt token."""
        s, n, P = self.spec, len(tokens), self.P
        plan = plan or self.plan(n)
        scatter = self.style == "tensorScatter"
        rows = sum(plan)
        if rows < n or sum(plan[:-1]) >= n:
            raise ValueError(f"the chunk plan {plan} does not fit a prompt of {n} tokens")
        if n >= P or (scatter and rows > P):
            raise ValueError(f"the prompt ({n} tokens, {rows} rows of chunks) does not fit into the {P}-row cache")
        self.reset()
        pad_total = rows - n if scatter else 0
        logits, start = None, 0
        for index, C in enumerate(plan):
            last = index == len(plan) - 1
            hidden = np.zeros((1, C, s.hidden), np.float16)
            if scatter:   # the chunk covers cache rows start..start+C-1, token i is in row pad_total + i
                for r in range(C):
                    if start + r - pad_total >= 0:
                        hidden[0, r] = self._embed_row(tokens[start + r - pad_total])
                rope = protocol.scatter_prefill_rope(s, C, start, pad_total)
                masks = protocol.scatter_prefill_masks(s, P, C, start, pad_total)
                selectors = {"kv_write_index": np.array([start], np.dtype(self.opt.scatter_index_type))}
                start += C
            else:
                valid = min(C, n - start)
                pad = C - valid
                for i in range(valid):
                    hidden[0, pad + i] = self._embed_row(tokens[start + i])
                rope = protocol.rope_tables(s, list(range(start, start + valid)), C, pad)
                masks = protocol.prefill_masks(s, P, C, start, valid)
                keep, selector = protocol.prefill_selectors(P, C, start, valid)
                selectors = {"rows_to_keep": keep, "row_selector": selector}
                start += valid
            logits = self._chain(self.g["prefill"][C], False, hidden, rope, masks, selectors, last, self.prefill_group)
        self.base, self.filled, self.pad = n, 0, pad_total   # staged: the prompt is in the main cache, the staging buffer is empty
        self.position = n
        return logits

    # -------------------------------------------------------------------------------------------------- decode
    def _flush(self) -> None:
        keep, selector = protocol.flush_selectors(self.P, self.R, self.base)
        for layer in range(self.spec.num_layers):
            out = run(self.g["decode"]["flush"], {"past_key": self.k[layer], "past_value": self.v[layer], "new_key": self.sk[layer], "new_value": self.sv[layer],
                                                  "rows_to_keep": keep, "row_selector": selector})
            self.k[layer], self.v[layer] = out["updated_key"], out["updated_value"]
        self.base += self.R
        self.filled = 0

    def step(self, token: int) -> np.ndarray:
        """Feed one token at the current position; returns the logits for the next one."""
        s, P, R = self.spec, self.P, self.R
        if self.cache_full:
            raise ValueError("the KV cache is full")
        pos = self.position
        hidden = self._embed_row(token).reshape(1, 1, -1)
        rope = protocol.rope_tables(s, [pos], 1, 0)
        if self.style == "staged":
            if self.filled == R:
                self._flush()
            masks = protocol.decode_masks(s, P, R, pos, self.base, self.filled)
            keep, selector = protocol.decode_selectors(R, self.filled)
            selectors = {"staging_rows_to_keep": keep, "staging_row_selector": selector}
        elif self.style == "onehot":
            masks = protocol.decode_masks_onehot(s, P, pos)
            keep, selector = protocol.decode_selectors_onehot(P, pos)
            selectors = {"rows_to_keep": keep, "row_selector": selector}
        else:
            masks = protocol.scatter_decode_masks(s, P, pos, self.pad)
            selectors = {"kv_write_index": np.array([self.pad + pos], np.dtype(self.opt.scatter_index_type))}
        logits = self._chain(self.g["decode"], True, hidden, rope, masks, selectors, True, self.decode_group)
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
            if token in eos or len(out) == max_new_tokens or self.cache_full:
                break
            logits = self.step(token)
            all_logits.append(logits)
        return out, all_logits

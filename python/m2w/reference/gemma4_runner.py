"""Executes the COMPILED Gemma 4 graphs (HEAD / BLOCK per attention kind / TAIL / lm_head / flush) with the numpy interpreter and the packed weights, following the runtime protocol
(staged KV, chunked prefill): the executable specification of how the browser runtime drives them.

Per layer i the attention half has the kind `spec.layer_kind(i)` and the BLOCK graph that runs it is the one of the kind of layer i + 1 (BLOCK i = feed-forward half of layer i +
attention half of layer i + 1); HEAD is the attention half of layer 0 plus the per-layer-embedding prologue (it outputs ple_0 .. ple_{N-1}). A layer that shares K / V reads the cache
of its producer layer: in a decode step the main cache and the staging buffer AFTER this step's update, in a prefill chunk the cache after the update of the chunk (mask columns = cache rows).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..frontends.hf_llm.arch import ModelSpec
from ..frontends.hf_llm.builders import decoder as D
from ..frontends.hf_llm.builders import gemma4 as G
from ..ir import serialize
from ..ir.interpreter import run
from . import protocol
from .gemma4_package import Gemma4PackageReader


class Gemma4Runner:
    def __init__(self, spec: ModelSpec, weights_dir, *, max_length: int = 128, chunk_sizes=(16,), staging_rows: int = 8, opt: Optional[D.GraphOptions] = None,
                 graphs: Optional[dict] = None, prefill_fixed_tokens: int = 200):
        self.spec = spec
        self.opt = opt or D.GraphOptions(kv_style="staged", staging_rows=staging_rows)
        self.R, self.P = self.opt.staging_rows, max_length - 1
        self.sizes = sorted(set(chunk_sizes))
        self.fixed_tokens = prefill_fixed_tokens
        self.reader = Gemma4PackageReader(spec, weights_dir)
        self.g = graphs or self._build_graphs()
        self.kinds = [spec.layer_kind(i) for i in range(spec.num_layers)]
        self.types = spec.layer_types
        self.producer = [p if p >= 0 else i for i, p in enumerate(spec.kv_producer or [-1] * spec.num_layers)]   # the layer whose cache layer i attends over
        self.attention_names = {k: [n for n, _, _ in G.attention_weight_specs(spec, k)] for k in G.KINDS}
        self.ffn_names = [n for n, _, _ in G.feed_forward_weight_specs(spec)]
        self.head_names = [n for n, _, _ in G.head_extra_weight_specs(spec)]
        self.externals = {"lm_head_weight": self.reader.embedding_rows_raw()[0], "lm_head_scale": self.reader.embedding_rows_raw()[1]}
        self.final_gamma = self.reader.raw(self.reader._role["final_norm"], "final_norm_gamma")
        self.reset()

    # ----------------------------------------------------------------------------------------------- graphs
    def _build_graphs(self) -> dict:
        s, opt, P = self.spec, self.opt, self.P
        flush = {t: G.build_flush_graph(s, t, staging_rows=self.R, past_len=P) for t in sorted(set(s.layer_types))}
        decode = {"head": G.build_head_graph(s, opt, seq=1, past_len=P), "blocks": {k: G.build_block_graph(s, opt, k, seq=1, past_len=P) for k in sorted(set(s.layer_kind(i) for i in range(s.num_layers)))},
                  "tail": G.build_tail_graph(s, opt, seq=1), "flush": flush}
        prefill = {C: {"head": G.build_head_graph(s, opt, seq=C, past_len=P), "tail": G.build_tail_graph(s, opt, seq=C),
                       "blocks": {k: G.build_block_graph(s, opt, k, seq=C, past_len=P) for k in sorted(set(s.layer_kind(i) for i in range(s.num_layers)))}} for C in self.sizes}
        return {"decode": decode, "prefill": prefill, "lm_head": G.build_lm_head_graph(s, opt)}

    @classmethod
    def from_package(cls, package, spec: ModelSpec, *, variant: Optional[str] = None, max_length: int = 0, chunk_sizes=None, prefill_fixed_tokens: int = 200) -> "Gemma4Runner":
        package = Path(package)
        plan = json.loads((package / "plan.json").read_text(encoding="utf-8"))
        info = next(v for v in plan["variants"] if v["id"] == (variant or plan["default_variant"]))
        table = plan["graphs"][info["id"]]
        L = max_length or min(plan["max_lengths"])
        sizes = sorted(chunk_sizes or [min(int(c) for c in table["prefill"][str(L)])])
        load = lambda path: serialize.load(package / path)   # noqa: E731
        d = table["decode"][str(L)]
        decode = {"head": load(d["head"]), "tail": load(d["tail"]), "blocks": {k: load(p) for k, p in d["blocks"].items()}, "flush": {t: load(p) for t, p in d["flush"].items()}}
        prefill = {}
        for C in sizes:
            f = table["prefill"][str(L)][str(C)]
            prefill[C] = {"head": load(f["head"]), "tail": load(f["tail"]), "blocks": {k: load(p) for k, p in f["blocks"].items()}}
        opt = D.GraphOptions(**{**{k: v for k, v in plan["profile"].items() if k in D.GraphOptions.__dataclass_fields__}, "compute_type": info["compute_type"], "kv_style": "staged",
                                "staging_rows": info.get("staging_rows", 64)})
        return cls(spec, package / "weights", max_length=L, chunk_sizes=sizes, opt=opt, prefill_fixed_tokens=prefill_fixed_tokens,
                   graphs={"decode": decode, "prefill": prefill, "lm_head": load(table["lm_head"])})

    # ----------------------------------------------------------------------------------------------- weights / host side
    def _weights(self, layer: int, names) -> dict:
        shard = self.reader._layer[layer]
        return {n: self.reader.raw(shard, n) for n in names if self.reader.has(shard, n)}

    def _embed(self, token: int) -> np.ndarray:
        return (self.reader.embedding_rows(token) * np.float32(self.spec.embed_scale)).astype(np.float16)

    def _ple_row(self, token: int) -> np.ndarray:
        return self.reader.ple_rows(token).astype(np.float16)

    def reset(self) -> None:
        s = self.spec
        zeros = lambda i, rows: np.zeros((1, s.decoder_dims(s.layer_types[i]).num_kv_heads, rows, s.decoder_dims(s.layer_types[i]).head_dim), np.float16)  # noqa: E731
        own = [i for i in range(s.num_layers) if self.producer[i] == i]
        self.k = {i: zeros(i, self.P) for i in own}
        self.v = {i: zeros(i, self.P) for i in own}
        self.sk = {i: zeros(i, self.R) for i in own}
        self.sv = {i: zeros(i, self.R) for i in own}
        self.base = self.filled = self.position = 0

    @property
    def cache_full(self) -> bool:
        return self.position >= self.P

    # ----------------------------------------------------------------------------------------------- one forward pass
    def _chain(self, graphs: dict, decode: bool, hidden, ple_token, rope, masks, selectors, last: bool):
        s, N = self.spec, self.spec.num_layers

        def feeds_for(layer: int) -> dict:
            kind, t = self.kinds[layer], self.types[layer]
            cos, sin = rope[t]
            mode = "shared" if G.kind_shared(kind) else "own"
            p = self.producer[layer]
            feeds = {"rope_cos": cos, "rope_sin": sin, "attention_mask": masks[(t, mode)], "past_key": self.k[p], "past_value": self.v[p], **selectors,
                     **self._weights(layer, self.attention_names[kind])}
            if decode:
                feeds.update(staging_key=self.sk[p], staging_value=self.sv[p])
            return feeds

        def store(layer: int, out: dict) -> None:
            if "updated_key" not in out:   # a shared layer writes nothing
                return
            if decode:
                self.sk[layer], self.sv[layer] = out["updated_key"], out["updated_value"]
            else:
                self.k[layer], self.v[layer] = out["updated_key"], out["updated_value"]

        out = run(graphs["head"], {"hidden_states": hidden, "ple_token": ple_token, **{n: self.reader.raw(self.reader._role["head_extra"], n) for n in self.head_names}, **feeds_for(0)})
        store(0, out)
        residual, attention = out["residual"], out["attention_output"]
        ple = [out[f"ple_{i}"] for i in range(N)]
        for i in range(N - 1):
            out = run(graphs["blocks"][self.kinds[i + 1]], {"residual": residual, "attention_output": attention, "per_layer_input": ple[i], **self._weights(i, self.ffn_names), **feeds_for(i + 1)})
            store(i + 1, out)
            residual, attention = out["next_residual"], out["next_attention_output"]
        if not last:
            return None
        tail = run(graphs["tail"], {"residual": residual, "attention_output": attention, "per_layer_input": ple[N - 1], **self._weights(N - 1, self.ffn_names), "final_norm_gamma": self.final_gamma})
        logits = run(self.g["lm_head"], {"final_hidden_states": tail["final_hidden_states"]}, externals=self.externals)["logits"]
        return logits.reshape(-1)[: s.vocab].astype(np.float32)

    # ----------------------------------------------------------------------------------------------- prefill / decode
    def plan(self, token_count: int) -> list:
        return protocol.plan_prefill(self.sizes, token_count, self.fixed_tokens)

    def prefill(self, tokens: list, plan: Optional[list] = None) -> np.ndarray:
        s, n, P = self.spec, len(tokens), self.P
        plan = plan or self.plan(n)
        if sum(plan) < n or sum(plan[:-1]) >= n:
            raise ValueError(f"the chunk plan {plan} does not fit a prompt of {n} tokens")
        if n >= P:
            raise ValueError(f"the prompt ({n} tokens) does not fit into the {P}-row cache")
        self.reset()
        ple_size = s.num_layers * s.ple_dim
        logits, start = None, 0
        for index, C in enumerate(plan):
            valid = min(C, n - start)
            pad = C - valid
            hidden = np.zeros((1, C, s.hidden), np.float16)
            ple_token = np.zeros((1, C, ple_size), np.float16)
            for i in range(valid):
                hidden[0, pad + i] = self._embed(tokens[start + i])
                ple_token[0, pad + i] = self._ple_row(tokens[start + i])
            rope = protocol.rope_tables(s, list(range(start, start + valid)), C, pad)
            own, shared = protocol.prefill_masks(s, P, C, start, valid), protocol.incache_prefill_masks(s, P, C, start, valid)
            masks = {**{(t, "own"): m for t, m in own.items()}, **{(t, "shared"): m for t, m in shared.items()}}
            keep, selector = protocol.prefill_selectors(P, C, start, valid)
            logits = self._chain(self.g["prefill"][C], False, hidden, ple_token, rope, masks, {"rows_to_keep": keep, "row_selector": selector}, index == len(plan) - 1)
            start += valid
        self.base, self.filled, self.position = n, 0, n
        return logits

    def _flush(self) -> None:
        keep, selector = protocol.flush_selectors(self.P, self.R, self.base)
        for i in self.k:
            out = run(self.g["decode"]["flush"][self.types[i]], {"past_key": self.k[i], "past_value": self.v[i], "new_key": self.sk[i], "new_value": self.sv[i],
                                                                  "rows_to_keep": keep, "row_selector": selector})
            self.k[i], self.v[i] = out["updated_key"], out["updated_value"]
        self.base += self.R
        self.filled = 0

    def step(self, token: int) -> np.ndarray:
        s, P, R = self.spec, self.P, self.R
        if self.cache_full:
            raise ValueError("the KV cache is full")
        if self.filled == R:
            self._flush()
        pos = self.position
        rope = protocol.rope_tables(s, [pos], 1, 0)
        decode_masks = protocol.decode_masks(s, P, R, pos, self.base, self.filled)
        masks = {**{(t, "own"): m for t, m in decode_masks.items()}, **{(t, "shared"): m for t, m in decode_masks.items()}}
        keep, selector = protocol.decode_selectors(R, self.filled)
        logits = self._chain(self.g["decode"], True, self._embed(token).reshape(1, 1, -1), self._ple_row(token).reshape(1, 1, -1), rope, masks,
                             {"staging_rows_to_keep": keep, "staging_row_selector": selector}, True)
        self.filled += 1
        self.position += 1
        return logits

    def generate(self, tokens: list, max_new_tokens: int, eos=()) -> tuple:
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

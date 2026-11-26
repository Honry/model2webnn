"""Executes the COMPILED Qwen3.5 graphs (HEAD / BLOCK per kind / TAIL / lm_head / flush) with the numpy interpreter and the packed weights, following the runtime protocol (staged KV for the
full attention layers, a conv state and a recurrent state for the Gated DeltaNet layers, chunked prefill with a left-padded last chunk): the executable specification of how the browser runtime
drives them. `Qwen35PackageReader.hf_state_dict()` gives the dequantized weights under the HF names, to feed the transformers model as the independent reference.

BLOCK i = feed-forward half of layer i + attention half of layer i + 1 (the graph of ITS kind); HEAD = attention half of layer 0; TAIL = feed-forward half of the last layer + final norm.

Images (a package with an image tower, docs/models/qwen35-vision.md): `prefill(tokens, images=[(patches, grid), ...])` with one <|image_pad|> token per image in `tokens`: the pad is expanded to one token
per merge block of the grid, the image tower (qwen35_vision.Qwen35VisionRunner) turns the patches into image tokens whose rows REPLACE the embedding rows of the pad tokens, and the full attention layers
get 3-position interleaved MRoPE tables (protocol.mrope_positions / rope_tables_mrope); decode token i has the rope position i + rope_delta. Without images everything is as before.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from ..frontends.hf_llm.arch import ModelSpec
from ..frontends.hf_llm.builders import decoder as D
from ..frontends.hf_llm.builders import qwen35 as Q
from ..frontends.hf_llm.pack_qwen35 import rope_permutation
from ..quantize import dequantize_int4
from ..ir import serialize
from ..ir.interpreter import run
from . import protocol
from .gemma4_package import Gemma4PackageReader


class Qwen35PackageReader(Gemma4PackageReader):
    """The generic part (raw tensors of a shard, the embedding table) comes from the Gemma reader; this adds the HF-named dequantized view with every package transformation undone."""

    def _gamma(self, shard, name) -> np.ndarray:
        return self.raw(shard, name).astype(np.float32).reshape(-1)

    # ----------------------------------------------------------------------------------------------- the image tower
    def vision_shard(self, role: str, layer=None, grid=None) -> dict:
        """The manifest entry of a vision shard: role vision_head | vision_merger | vision_pos_embed | vision_block (layer) | vision_grid (grid = (h, w))."""
        for f in self.files:
            if f["role"] == role and (layer is None or f.get("layer") == layer) and (grid is None or tuple(f.get("grid", ())) == tuple(grid)):
                return f
        raise KeyError(f"the package has no {role} shard (layer {layer}, grid {grid})")

    def vision_linear(self, shard: dict, name: str) -> np.ndarray:
        """[out, in] float32 of a linear weight of the image tower as the graph sees it: dequantizeLinear gives float16 (scale dtype), int8: q * scale, int4: group-wise."""
        mode = self.manifest["quant"]["vision"]["weights"]
        weight = self.raw(shard, name + "_weight")
        if mode == "int8":
            return (weight.astype(np.float32) * self.raw(shard, name + "_scale").astype(np.float32)).astype(np.float16).astype(np.float32)
        if mode == "int4":
            return dequantize_int4(weight, self.raw(shard, name + "_scale"))
        return weight.astype(np.float32)

    def vision_hf_state_dict(self, prefix: str = "model.visual.") -> dict:
        """The dequantized image tower under the transformers names (Qwen3_5VisionModel): the Conv3d kernel back to [hidden, channels, temporal, patch, patch]."""
        v = self.spec.vision
        flat = lambda shard, name: self.raw(shard, name).astype(np.float32).reshape(-1)  # noqa: E731
        head, merger = self.vision_shard("vision_head"), self.vision_shard("vision_merger")
        out = {prefix + "patch_embed.proj.weight": self.vision_linear(head, "patch_embed").reshape(v["hidden"], v["channels"], v["temporal_patch_size"], v["patch_size"], v["patch_size"]),
               prefix + "patch_embed.proj.bias": flat(head, "patch_embed_bias"),
               prefix + "pos_embed.weight": self.raw(self.vision_shard("vision_pos_embed"), "pos_embed_table").astype(np.float32),
               prefix + "merger.norm.weight": flat(merger, "merger_norm_gamma"), prefix + "merger.norm.bias": flat(merger, "merger_norm_beta")}
        for key in ("fc1", "fc2"):
            out[f"{prefix}merger.linear_{key}.weight"], out[f"{prefix}merger.linear_{key}.bias"] = self.vision_linear(merger, "merger_" + key), flat(merger, f"merger_{key}_bias")
        for i in range(v["depth"]):
            shard, p = self.vision_shard("vision_block", layer=i), f"{prefix}blocks.{i}."
            for norm in ("norm1", "norm2"):
                out[p + norm + ".weight"], out[p + norm + ".bias"] = flat(shard, norm + "_gamma"), flat(shard, norm + "_beta")
            for ours, theirs in (("qkv", "attn.qkv"), ("proj", "attn.proj"), ("fc1", "mlp.linear_fc1"), ("fc2", "mlp.linear_fc2")):
                out[p + theirs + ".weight"], out[p + theirs + ".bias"] = self.vision_linear(shard, ours), flat(shard, ours + "_bias")
        return out

    def hf_state_dict(self, prefix: str = "model.") -> dict:
        spec = self.spec
        d = spec.decoder_dims("full_attention")
        out = {prefix + "embed_tokens.weight": self.embedding_rows(slice(0, spec.vocab)).astype(np.float32),
               prefix + "norm.weight": self._gamma(self._role["final_norm"], "final_norm_gamma") - 1.0}
        out["lm_head.weight"] = out[prefix + "embed_tokens.weight"]
        for i in range(spec.num_layers):
            shard, kind, p = self._layer[i], spec.layer_kind(i), f"{prefix}layers.{i}."
            out[p + "input_layernorm.weight"] = self._gamma(shard, "input_layernorm_gamma") - 1.0
            out[p + "post_attention_layernorm.weight"] = self._gamma(shard, "post_attention_layernorm_gamma") - 1.0
            gate_up = self._dequant(shard, "gate_up_projection")
            out[p + "mlp.gate_proj.weight"], out[p + "mlp.up_proj.weight"] = gate_up[:spec.ffn], gate_up[spec.ffn:]
            out[p + "mlp.down_proj.weight"] = self._dequant(shard, "down_projection")
            if kind == "linear_attention":
                ld = Q.linear_dims(spec)
                qkvz = self._dequant(shard, "qkvz_projection")
                out[p + "linear_attn.in_proj_qkv.weight"], out[p + "linear_attn.in_proj_z.weight"] = qkvz[:ld["conv_width"]], qkvz[ld["conv_width"]:]
                ba = self.raw(shard, "ba_weight").astype(np.float32)
                out[p + "linear_attn.in_proj_b.weight"], out[p + "linear_attn.in_proj_a.weight"] = ba[:ld["v_heads"]], ba[ld["v_heads"]:]
                out[p + "linear_attn.conv1d.weight"] = np.transpose(self.raw(shard, "conv_weight").astype(np.float32)[:, 0, :], (1, 0))[:, None, :]
                out[p + "linear_attn.A_log"] = np.log(self.raw(shard, "decay_rate").astype(np.float32).reshape(-1))
                out[p + "linear_attn.dt_bias"] = self.raw(shard, "dt_bias").astype(np.float32).reshape(-1)
                out[p + "linear_attn.norm.weight"] = self.raw(shard, "norm_gamma").astype(np.float32).reshape(-1)
                out[p + "linear_attn.out_proj.weight"] = self._dequant(shard, "output_projection")
            else:
                rot = int(d.head_dim * float(spec.rope["full_attention"].get("partial_rotary_factor", 1.0)))
                inverse = np.argsort(rope_permutation(d.head_dim, rot))
                aw, kw = d.attention_width, d.kv_width
                merged = self._dequant(shard, "qkvg_projection")
                query = merged[:aw].reshape(d.num_heads, d.head_dim, -1)[:, inverse]
                gate = merged[aw:2 * aw].reshape(d.num_heads, d.head_dim, -1)
                out[p + "self_attn.q_proj.weight"] = np.stack([query, gate], axis=1).reshape(d.num_heads * 2 * d.head_dim, -1)   # per head: [query | gate]
                out[p + "self_attn.k_proj.weight"] = merged[2 * aw:2 * aw + kw].reshape(d.num_kv_heads, d.head_dim, -1)[:, inverse].reshape(kw, -1)
                out[p + "self_attn.v_proj.weight"] = merged[2 * aw + kw:]
                out[p + "self_attn.o_proj.weight"] = self._dequant(shard, "output_projection")
                out[p + "self_attn.q_norm.weight"] = (self._gamma(shard, "query_norm_gamma") - 1.0)[inverse]
                out[p + "self_attn.k_norm.weight"] = (self._gamma(shard, "key_norm_gamma") - 1.0)[inverse]
        return out


class Qwen35Runner:
    def __init__(self, spec: ModelSpec, weights_dir, *, max_length: int = 128, chunk_sizes=(16,), staging_rows: int = 8, opt: Optional[D.GraphOptions] = None,
                 graphs: Optional[dict] = None, prefill_fixed_tokens: int = 200, blocks_per_graph: int = 1):
        self.spec = spec
        self.group = blocks_per_graph   # decode block groups (the prefill chain always runs one dispatch per block)
        self.opt = opt or D.GraphOptions(kv_style="staged", staging_rows=staging_rows)
        self.R, self.P = self.opt.staging_rows, max_length - 1
        self.sizes = sorted(set(chunk_sizes))
        self.fixed_tokens = prefill_fixed_tokens
        self.reader = Qwen35PackageReader(spec, weights_dir)
        self.vision = None   # a qwen35_vision.Qwen35VisionRunner when the package has an image tower (from_package)
        self.g = graphs or self._build_graphs()
        self.kinds = [spec.layer_kind(i) for i in range(spec.num_layers)]
        self.attention_names = {k: [n for n, _, _ in Q.attention_weight_specs(spec, k)] for k in Q.KINDS if k in spec.layer_types}
        self.ffn_names = [n for n, _, _ in Q.feed_forward_weight_specs(spec)]
        self.externals = self._lm_head_externals()
        self.final_gamma = self.reader.raw(self.reader._role["final_norm"], "final_norm_gamma")
        self.reset()

    # ----------------------------------------------------------------------------------------------- graphs
    def _build_graphs(self) -> dict:
        s, opt, P = self.spec, self.opt, self.P
        kinds = sorted(set(s.layer_kind(i) for i in range(1, s.num_layers)))
        decode = {"head": Q.build_head_graph(s, opt, seq=1, past_len=P), "blocks": {k: Q.build_block_graph(s, opt, k, seq=1, past_len=P) for k in kinds}, "tail": Q.build_tail_graph(s, opt, seq=1),
                  "flush": Q.build_flush_graph(s, staging_rows=self.R, past_len=P)}
        if self.group > 1:
            decode["groups"], built = [], {}   # the schedule: (first block, blocks, graph)
            for first, blocks in Q.block_group_schedule(s, self.group):
                kinds_of = Q.block_group_kinds(s, first, blocks)
                key = Q.block_group_key(kinds_of)
                if key not in built:
                    built[key] = Q.build_block_group_graph(s, opt, kinds_of, seq=1, past_len=P)
                decode["groups"].append((first, blocks, built[key]))
        prefill = {C: {"head": Q.build_head_graph(s, opt, seq=C, past_len=P), "tail": Q.build_tail_graph(s, opt, seq=C),
                       "blocks": {k: Q.build_block_graph(s, opt, k, seq=C, past_len=P) for k in kinds}} for C in self.sizes}
        return {"decode": decode, "prefill": prefill, "lm_head": Q.build_lm_head_graph(s, opt)}

    @classmethod
    def from_package(cls, package, spec: ModelSpec, *, variant: Optional[str] = None, max_length: int = 0, chunk_sizes=None, prefill_fixed_tokens: int = 200,
                     decode_group: int = 1) -> "Qwen35Runner":
        package = Path(package)
        plan = json.loads((package / "plan.json").read_text(encoding="utf-8"))
        info = next(v for v in plan["variants"] if v["id"] == (variant or plan["default_variant"]))
        table = plan["graphs"][info["id"]]
        L = max_length or min(plan["max_lengths"])
        sizes = sorted(chunk_sizes or [min(int(c) for c in table["prefill"][str(L)])])
        load = lambda path: serialize.load(package / path)   # noqa: E731
        d = table["decode"][str(L)]
        decode = {"head": load(d["head"]), "tail": load(d["tail"]), "blocks": {k: load(p) for k, p in d["blocks"].items()}, "flush": load(d["flush"])}
        if decode_group > 1:
            if str(decode_group) not in d.get("block_groups", {}):
                raise ValueError(f"the package has no decode block groups of {decode_group} blocks (it has {', '.join(d.get('block_groups', {})) or 'none'})")
            loaded: dict = {}
            decode["groups"] = [(e["first_block"], e["blocks"], loaded.setdefault(e["graph"], load(e["graph"]))) for e in d["block_groups"][str(decode_group)]]
        prefill = {}
        for C in sizes:
            f = table["prefill"][str(L)][str(C)]
            prefill[C] = {"head": load(f["head"]), "tail": load(f["tail"]), "blocks": {k: load(p) for k, p in f["blocks"].items()}}
        opt = D.GraphOptions(**{**{k: v for k, v in plan["profile"].items() if k in D.GraphOptions.__dataclass_fields__}, "compute_type": info["compute_type"], "kv_style": "staged",
                                "staging_rows": info.get("staging_rows", 64)})
        runner = cls(spec, package / "weights", max_length=L, chunk_sizes=sizes, opt=opt, prefill_fixed_tokens=prefill_fixed_tokens, blocks_per_graph=decode_group,
                     graphs={"decode": decode, "prefill": prefill, "lm_head": load(table["lm_head"])})
        if "vision" in plan and spec.vision:
            from .qwen35_vision import Qwen35VisionRunner

            runner.vision = Qwen35VisionRunner(package, spec)
        return runner

    def _lm_head_externals(self) -> dict:
        weight, scale = self.reader.embedding_rows_raw()
        out = {"lm_head_weight": weight, "lm_head_scale": scale}
        parts = Q.lm_head_parts(self.spec, self.opt)
        rows = self.spec.vocab_padded // parts
        for i in range(parts if parts > 1 else 0):
            out[f"lm_head_weight_{i}"], out[f"lm_head_scale_{i}"] = weight[i * rows:(i + 1) * rows], scale[i * rows:(i + 1) * rows]
        return out

    # ----------------------------------------------------------------------------------------------- weights / state
    def _weights(self, layer: int, names) -> dict:
        shard = self.reader._layer[layer]
        return {n: self.reader.raw(shard, n) for n in names if self.reader.has(shard, n)}

    def _embed(self, token: int) -> np.ndarray:
        return (self.reader.embedding_rows(token) * np.float32(self.spec.embed_scale)).astype(np.float16)

    def reset(self) -> None:
        s = self.spec
        d = s.decoder_dims("full_attention")
        full = [i for i in range(s.num_layers) if s.layer_types[i] == "full_attention"]
        zeros = lambda rows: np.zeros((1, d.num_kv_heads, rows, d.head_dim), np.float16)  # noqa: E731
        self.k, self.v = {i: zeros(self.P) for i in full}, {i: zeros(self.P) for i in full}
        self.sk, self.sv = {i: zeros(self.R) for i in full}, {i: zeros(self.R) for i in full}
        self.conv, self.rec = {}, {}
        if s.linear:
            ld = Q.linear_dims(s)
            for i in range(s.num_layers):
                if s.layer_types[i] == "linear_attention":
                    self.conv[i] = np.zeros((1, Q.CONV_TAPS - 1, ld["conv_width"]), np.float16)
                    self.rec[i] = np.zeros((1, ld["v_heads"], ld["k_dim"], ld["v_dim"]), np.float16)
        self.base = self.filled = self.position = 0
        self.rope_delta = 0

    @property
    def cache_full(self) -> bool:
        return self.position >= self.P

    # ----------------------------------------------------------------------------------------------- one forward pass
    def _chain(self, graphs: dict, decode: bool, hidden, rope, masks, selectors, controls, last: bool):
        s, N = self.spec, self.spec.num_layers
        shared_inputs = {"rope_cos", "rope_sin", "attention_mask", "staging_rows_to_keep", "staging_row_selector"}

        def feeds_for(layer: int) -> dict:
            kind = self.kinds[layer]
            weights = self._weights(layer, self.attention_names[kind])
            if kind == "linear_attention":
                return {"conv_state": self.conv[layer], "recurrent_state": self.rec[layer], **controls, **weights}
            cos, sin = rope["full_attention"]
            feeds = {"rope_cos": cos, "rope_sin": sin, "attention_mask": masks["full_attention"], "past_key": self.k[layer], "past_value": self.v[layer], **selectors, **weights}
            if decode:
                feeds.update(staging_key=self.sk[layer], staging_value=self.sv[layer])
            return feeds

        def store(layer: int, out: dict) -> None:
            if self.kinds[layer] == "linear_attention":
                self.conv[layer], self.rec[layer] = out["new_conv_state"], out["new_recurrent_state"]
            elif decode:
                self.sk[layer], self.sv[layer] = out["updated_key"], out["updated_value"]
            else:
                self.k[layer], self.v[layer] = out["updated_key"], out["updated_value"]

        out = run(graphs["head"], {"hidden_states": hidden, **feeds_for(0)})
        store(0, out)
        residual, attention = out["residual"], out["attention_output"]
        self.trace = [residual]   # the hidden state entering layer i (the residual of its BLOCK): what tools/qwen35_trace.py writes for the on-device comparison (bench/raw/qwen35check.html)
        if decode and self.group > 1:
            for first, blocks, graph in graphs["groups"]:
                feeds = {"residual": residual, "attention_output": attention}
                for j in range(blocks):
                    for n, v in self._weights(first + j, self.ffn_names).items():
                        feeds[f"b{j}_{n}"] = v
                    for n, v in feeds_for(first + j + 1).items():   # the controls are shared by the blocks of the group, the weights and the state have the prefix of their block
                        feeds[n if n in shared_inputs else f"b{j}_{n}"] = v
                out = run(graph, {n: v for n, v in feeds.items() if n in graph.inputs})
                for j in range(blocks):
                    store(first + j + 1, {n[len(f"b{j}_"):]: v for n, v in out.items() if n.startswith(f"b{j}_")})
                residual, attention = out["next_residual"], out["next_attention_output"]
                self.trace.append(residual)   # a group has no boundary values inside: one trace entry per group
        else:
            for i in range(N - 1):
                out = run(graphs["blocks"][self.kinds[i + 1]], {"residual": residual, "attention_output": attention, **self._weights(i, self.ffn_names), **feeds_for(i + 1)})
                store(i + 1, out)
                residual, attention = out["next_residual"], out["next_attention_output"]
                self.trace.append(residual)
        if not last:
            return None
        tail = run(graphs["tail"], {"residual": residual, "attention_output": attention, **self._weights(N - 1, self.ffn_names), "final_norm_gamma": self.final_gamma})
        self.trace.append(tail["final_hidden_states"])
        logits = run(self.g["lm_head"], {"final_hidden_states": tail["final_hidden_states"]}, externals=self.externals)["logits"]
        return logits.reshape(-1)[: s.vocab].astype(np.float32)

    # ----------------------------------------------------------------------------------------------- prefill / decode
    def plan(self, token_count: int) -> list:
        return protocol.plan_prefill(self.sizes, token_count, self.fixed_tokens)

    def prefill(self, tokens: list, plan: Optional[list] = None, images: Optional[list] = None) -> np.ndarray:
        """images: [(patches [N, patch_vector], grid)] in the order of the <|image_pad|> tokens of `tokens` (protocol.prepare_image makes them). self.prompt = the expanded token list."""
        s, P = self.spec, self.P
        positions, delta, rows = None, 0, {}
        if images:
            if self.vision is None:
                raise ValueError("this package has no image tower (compile it with a vision checkpoint, or `m2w add-vision`)")
            grids = [tuple(g) for _, g in images]
            tokens = protocol.expand_image_tokens(s, tokens, grids)
            positions, delta = protocol.mrope_positions(s, tokens, grids)
            pad = s.vision["image_token_id"]
            at = [i for i, t in enumerate(tokens) if t == pad]
            embeds = np.concatenate([self.vision.encode(np.asarray(p, np.float16), g) for p, g in images])   # [image tokens, hidden] float16
            rows = {position: embeds[k] for k, position in enumerate(at)}
        self.prompt, n = list(tokens), len(tokens)
        plan = plan or self.plan(n)
        if sum(plan) < n or sum(plan[:-1]) >= n:
            raise ValueError(f"the chunk plan {plan} does not fit a prompt of {n} tokens")
        if n >= P:
            raise ValueError(f"the prompt ({n} tokens) does not fit into the {P}-row cache")
        self.reset()
        logits, start = None, 0
        for index, C in enumerate(plan):
            valid = min(C, n - start)
            pad = C - valid
            hidden = np.zeros((1, C, s.hidden), np.float16)
            for i in range(valid):
                hidden[0, pad + i] = rows[start + i] if start + i in rows else self._embed(tokens[start + i])
            rope = protocol.rope_tables(s, list(range(start, start + valid)), C, pad) if positions is None else protocol.rope_tables_mrope(s, positions[:, start:start + valid], C, pad)
            masks = protocol.prefill_masks(s, P, C, start, valid)
            keep, selector = protocol.prefill_selectors(P, C, start, valid)
            valid_mask, conv_perm = protocol.linear_prefill_controls(C, valid)
            logits = self._chain(self.g["prefill"][C], False, hidden, rope, masks, {"rows_to_keep": keep, "row_selector": selector},
                                 {"valid_mask": valid_mask, "conv_perm": conv_perm}, index == len(plan) - 1)
            start += valid
        self.base, self.filled, self.position, self.rope_delta = n, 0, n, delta
        return logits

    def _flush(self) -> None:
        keep, selector = protocol.flush_selectors(self.P, self.R, self.base)
        for i in self.k:
            out = run(self.g["decode"]["flush"], {"past_key": self.k[i], "past_value": self.v[i], "new_key": self.sk[i], "new_value": self.sv[i], "rows_to_keep": keep, "row_selector": selector})
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
        rope = protocol.rope_tables(s, [pos + self.rope_delta], 1, 0)   # the rope position of a text token after an image is shifted by rope_delta; the cache rows / masks use the sequence index
        masks = protocol.decode_masks(s, P, R, pos, self.base, self.filled)
        keep, selector = protocol.decode_selectors(R, self.filled)
        logits = self._chain(self.g["decode"], True, self._embed(token).reshape(1, 1, -1), rope, masks, {"staging_rows_to_keep": keep, "staging_row_selector": selector}, {}, True)
        self.filled += 1
        self.position += 1
        return logits

    def generate(self, tokens: list, max_new_tokens: int, eos=(), images: Optional[list] = None) -> tuple:
        logits = self.prefill(tokens, images=images)
        out, all_logits = [], [logits]
        for _ in range(max_new_tokens):
            token = int(np.argmax(logits))
            out.append(token)
            if token in eos or len(out) == max_new_tokens or self.cache_full:
                break
            logits = self.step(token)
            all_logits.append(logits)
        return out, all_logits

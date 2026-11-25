# Gemma 4 E4B (google/gemma-4-E4B-it): what it takes

Read from `config.json`, the safetensors header and transformers' `modeling_gemma4.py` / `configuration_gemma4.py`. The numpy reference `python/tools/gemma4_ref.py` implements the text decoder
as described here and is the oracle for the graphs.

**No MoE.** `enable_moe_block: false`, `num_experts: null`, `use_double_wide_mlp: false`. In the modeling code the router / experts are an optional block of a decoder layer
(`enable_moe_block`); another Gemma 4 size uses it, this one does not.

## The model

42 layers, hidden 2560, 8 attention heads, 2 KV heads, FFN 10240, vocabulary 262144 (tied embedding / lm_head), sliding window 512; full attention at layers 5, 11, 17, 23, 29, 35, 41 (every sixth),
sliding attention everywhere else. Only the text decoder matters here (the checkpoint also has vision and audio towers, ignored).

## What differs from the decoders the compiler handles today (Qwen3, Aion)

| | Gemma 4 E4B | consequence |
|---|---|---|
| head dim | 256 in sliding layers, **512** in full layers (`global_head_dim`); both with 8 Q / 2 KV heads | Q / K / V / O projections and KV caches have different shapes per layer type: one BLOCK graph per layer type |
| KV sharing | the last 18 layers (24..41) have **no K / V of their own**: they attend over the K / V of the last non-shared layer of their type (layer 22 sliding, layer 23 full) | 4 kinds of attention half: {sliding, full} x {own KV, shared KV}; 24 KV caches instead of 42; a shared layer reads the producer's cache *including the rows of the current step* |
| per-layer embeddings (PLE) | `embed_tokens_per_layer` [262144, 42 x 256] (2.8 GB at int8), `per_layer_model_projection` [2560 -> 42 x 256], `per_layer_projection_norm`; per layer `per_layer_input_gate` [2560 -> 256], `per_layer_projection` [256 -> 2560], `post_per_layer_input_norm` | a row of the table per prompt token is looked up on the host; the projection runs on the device; each layer gets a [S, 256] input |
| norms | RMSNorm `x * (mean(x^2) + eps)^-0.5 * weight` (weight used as is, not 1 + w); sandwich: `input_layernorm`, `post_attention_layernorm`, `pre_feedforward_layernorm`, `post_feedforward_layernorm`; per-head `q_norm` / `k_norm` (with weight), `v_norm` (**no weight**) | a layer is `x += post_attn_norm(attn(input_norm(x)))`, `x += post_ffn_norm(mlp(pre_ffn_norm(x)))` |
| attention | scaling **1.0** (no 1/sqrt(d); the q / k norms take that role) | no scale multiply |
| activation | `gelu_pytorch_tanh`: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) | needs `tanh` (not in the IR yet) |
| layer scalar | `x *= layer_scalar[i]` at the end of every layer (a trained [1] tensor) | one more weight input per layer |
| RoPE | sliding: default, theta 1e4, rotates all 256 dims; full: `proportional`, theta 1e6, partial rotary 0.25: of the 256 frequencies of a 512 head dim only the first 64 are non-zero (exponent divides by the FULL head dim), the other 192 have frequency 0 (cos 1, sin 0) | a new rope type + per-type table width |
| embedding | `embed_tokens(ids) * sqrt(hidden)` | host side |
| logits | `30 * tanh(logits / 30)` (`final_logit_softcapping`) | three more ops in the lm_head graph |
| chat template | `<|turn>user ... <turn|>`, `<|turn>model\n`, big template with tools / thinking | the template machinery has to cope with it |

PLE per layer, in the order the reference computes it:

    per_layer_inputs = (rms_norm(per_layer_model_projection(embeds) * hidden^-0.5, per_layer_projection_norm) + embed_tokens_per_layer[ids] * sqrt(256)) * 2^-0.5     # [S, 42, 256]
    ... after the MLP of layer i:  x += post_per_layer_input_norm(per_layer_projection(gelu_tanh(per_layer_input_gate(x)) * per_layer_inputs[:, i]))
    x *= layer_scalar[i]

## WebNN tensor limit

The dequantized lm_head matrix (262144 x 2560) is 2.5 GiB in float32 and 1.25 GiB in float16; WebNN rejects tensors above 2^31 - 1 bytes (the browser error was *"The tensor size 2684354560 exceeds the size limit 2147483647"* at the cast),
so the lm_head graph works on row blocks (4 in float32, 2 in float16; `lm_head_parts`). The compiler checks every graph (`check_tensor_limit`); the largest tensor of the package is 640 MiB.

## Size

Text layers about 4 B parameters (int4: about 2 GB), tied embedding / lm_head 262144 x 2560 (int8: 671 MB), PLE table 2.8 B parameters (int8: 2.8 GB; int4: 1.4 GB).
The PLE table is only ever read one row per token: it must not be loaded as a whole; random reads out of an OPFS file (or range requests) are the way.

## Decode block groups (`--blocks-per-graph K`)

Measured per graph on the NPU, a decode BLOCK graph costs about 1.9 ms whatever its size (sliding own 1.89, sliding shared 1.86, full own 2.19, full shared 2.38 ms; 41 blocks = about 80 of the 91 ms of a
decode step): dispatch overhead, not bandwidth. A group fuses K consecutive BLOCKs into one graph (Aion with 5 blocks per graph: 47.6 -> 43.8 ms). Decode only; `prefillBlocksPerGraph > 1` is refused
(prefill grouping gave no TTFT gain).

* **Schedule.** The N - 1 = 41 BLOCKs (BLOCK i = feed-forward half of layer i + attention half of layer i + 1) are cut into runs of K, the last run may be shorter (K = 3: 13 groups of 3 + one of 2 = 14
  dispatches instead of 41). The block kinds differ (sliding / full x own / shared K / V), so a graph exists per distinct SEQUENCE of kinds: key = the kinds joined by `-` (`so` / `fo` = sliding / full attention
  with K / V of its own, `ss` / `fs` = sharing). For K = 3 there are six graphs (`so-so-so`, `so-fo-so`, `so-fo-ss`, `ss-ss-ss`, `ss-fs-ss`, `ss-fs`). Node counts are the sums of the single blocks (K = 3: 573 / 535 / 459 /
  306 nodes), 69 - 85 inputs.
* **Inputs.** `residual`, `attention_output` of the first block; the controls once per layer TYPE (`sliding_attention_rope_cos`, `..._rope_sin`, `..._attention_mask`, `full_attention_...`: the decode masks of own and
  shared layers are the same tensor); per block `b<j>_` + `per_layer_input`, all feed-forward weights of layer i, all attention weights of layer i + 1, and the cache inputs `past_key` / `past_value` (main cache) and
  `staging_key` / `staging_value` of the layer whose K / V the block attends over. The weights stay graph inputs (the runtime binds the weight MLTensors per layer).
* **Shared K / V inside a group.** Cache inputs are declared once per producer layer: by the block with K / V of its own, or by the first block of the group that reads an outside producer (later blocks reuse them: no tensor
  is bound twice). A block that reads the cache of a producer INSIDE the group (K = 3: blocks 21, 22, 23 = layers 22, 23, 24; layer 24 reads layer 22) attends over concat(main cache, the staging buffer after the update of
  that producer computed in the same graph); the runtime gives it no staging input. Across groups the shared layer reads the producer's staging buffer after the producer's group has run.
* **Outputs.** `next_residual`, `next_attention_output` of the last block; `b<j>_updated_key` / `_value` for the blocks with K / V of their own = the spare staging buffers, swapped in after the dispatch (the same main /
  spare swap per layer as without groups: a layer's staging buffer is an input and the spare an output of the same dispatch, never the same tensor).
* **Numerics.** Between two blocks of a group the residual and attention output are rounded to float16 exactly as at the boundary of two dispatches (`to_f16` then back to the compute type), so a group computes the same values
  as K single dispatches (the tests compare logits and the staging caches of the numpy interpreter for K = 2 .. 5 incl. float16 compute) and the residual add never reads the op result that also feeds the next attention half.
* **plan.json.** `plan.blocks_per_graph` lists the K the package has; `graphs.<variant>.decode.<max length>.block_groups[K]` = `[{first_block, blocks, key, graph}]`, the schedule, groups with the same kinds share a graph
  file (`decode_group_<key>_L<max length>.json`); `plan.chain.block_groups` documents the binding. The runtime (`Gemma4LLM`, `blocksPerGraph: K`) and the numpy `Gemma4Runner(decode_group=K)` bind a group by input NAME
  (`b<j>_x` -> layer first + j (feed-forward half) / first + j + 1 (attention half), producer from `kv_producer`), so nothing else model specific is in the plan. A package built without K has no such table
  and a runtime asked for K reports which K it has.
* **Not checked here:** the NPU compiler on these graphs (compile time, whether the output `updated_key` that also feeds a later block's concat is accepted, speed). Re-time with `bench/raw/m2wgraphs.html`.

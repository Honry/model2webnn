# m2w — Hugging Face LLM → static-shape WebNN model package

`m2w` compiles a Hugging Face decoder-only LLM into a **model package**: WebNN-level graphs (`m2w-graph` JSON), int4 / uint8 weight shards, the prompt format and the
tokenizer, ready for a browser runtime (NPUW-style split: HEAD / one shared BLOCK graph with the weights as graph inputs / TAIL / lm_head, chunked prefill, staged KV
cache). Target today: WebNN on the Intel NPU (OpenVINO backend).

```
pip install -e "python[hf,dev]"          # numpy only for the core; [hf] adds tokenizers / jinja2 / safetensors helpers; [dev] pytest + jsonschema

m2w inspect microsoft/Aion-1.0-Instruct-2.3B                       # supported? layers, RoPE, estimated package size (downloads config files only)
m2w compile microsoft/Aion-1.0-Instruct-2.3B --out aion            # download, quantize, write the package
m2w verify  aion --source microsoft/Aion-1.0-Instruct-2.3B         # run the package on the CPU and compare with the original model
```

Useful `compile` options (the graphs are static, so what a runtime can choose between is decided here; weights are shared by all of them):
- `--max-length` / `--chunk 64 128 256`: KV cache capacities and prefill chunk sizes. With several chunk sizes the runtime plans the chunk sequence of each prompt (fewer, bigger dispatches for long prompts, a small one for short prompts).
- `--kv-style staged onehot tensorScatter scatterND`: KV protocols (staged = staging buffer flushed every N steps, onehot = whole-cache update, tensorScatter / scatterND = in-graph scatter of the new rows; `LoadOptions.kvInPlace` writes the scatter result into the same MLTensor that is the input, no spare cache); `--staging-rows 64 32 128`; `--compute-type float32 float16`.
  The one-hot variants keep the value cache transposed (`[1,kvHeads,128,P]`, as the hand-written graphs do: faster on the NPU than the natural layout; the prefill graphs then carry an `attention_guard` output). Every combination becomes a graph **variant** (`staged-r64-float32`, `onehot-float16`, ...) listed in `plan.json`; the first value of each option is the default. Without these options a package has one variant.
- **Recommended session options per model family** live in `python/m2w/frontends/hf_llm/families/<family>.json` (`"recommended": {"blocks_per_graph": 5, "prefill_blocks_per_graph": 1}`: decode blocks per graph that
  the family runs best with; qwen3-like 5, gemma4 3, qwen3.5 1). `m2w compile` builds the groups of that size when `--blocks-per-graph` is not given, writes them to `plan.json` (`recommended`) and the
  runtime (`LLM`, `Gemma4LLM`) and `bench/raw/chat.html` use them whenever the caller does not choose (an explicit `blocksPerGraph` wins; a recommended size the package lacks falls back to 1).
  `m2w recommend <package>` rewrites `plan.recommended` of an existing package from the family file. Change a value after measuring your own device.
- `--blocks-per-graph 1 5` / `--prefill-blocks-per-graph 1 5`: also build graphs with 5 blocks each (fewer dispatches, longer compile; hybrid models such as sliding + full attention get a mask and RoPE table per block as graph inputs). **Gemma 4** takes `--blocks-per-graph` too (decode only: `--prefill-blocks-per-graph` must stay 1): the 41 blocks are cut into runs of K, one graph per distinct sequence of attention kinds (`so-so-fo` ...); see docs/models/gemma4-e4b.md.
- `--sliding-cache full|window` (default `full`): the K / V cache of the sliding-attention layers (Aion, Gemma 4). `window` keeps a ring of window + chunk rows instead of `max_length - 1` rows (staged protocol only, no prefill block groups; `plan.profile.sliding_cache` / `sliding_ring`); much less memory at long max lengths. `m2w rebuild-graphs --sliding-cache window` rewrites an existing package; `python tools/window_sibling.py <package>` makes a `<package>-window` sibling that shares the weights. See docs/superpowers/specs/2026-11-27-sliding-window-kv-design.md.
- `--lm-head-format int8-sym|uint8-asym` (default int8-sym: same accuracy as the OpenVINO-style uint8 table, compiles faster on the NPU).

`m2w repack <package>` converts the weight shards of a package made by an older compiler (raw `*.bin`) to `.safetensors` without quantizing again; `m2w rebuild-graphs <package> --kv-style ...` rewrites the graphs and plan of an existing package with other variants (no quantization); `m2w verify <package> --variant all --chunk 64 128` runs the graph files of
every variant on the CPU (the runtime protocol) against an independent reference.

Behind a network that cannot reach huggingface.co: `--endpoint https://hf-mirror.com` (or `HF_ENDPOINT`). Downloads resume after interruptions and retry on rate limits.

## What is in a package
```
m2w.json            entry point: format version, compiler, target, requires, file index (sizes, sha256)
model.json          architecture: dims, layer types (full / sliding attention), RoPE per layer type (default / linear / YaRN), tokens
plan.json           graph variants (KV protocol x compute type x staging rows) and, per variant, which graph for which phase / max length / chunk;
                    how weights and per-layer-type inputs are bound; the KV protocols
chat_format.json    prompt pieces derived from the chat template (the runtime does not run jinja)
graphs/<variant>/*.json   m2w-graph documents
weights/            manifest.json + layer_000.safetensors ... (standard safetensors files: int4 g128 symmetric, group-major, stored as U8 with the last axis halved) + the tied embedding / lm_head table
                    (int8 per row symmetric by default, or uint8 asymmetric; vocabulary padded to 128)
tokenizer/          tokenizer.json, tokenizer_config.json, generation_config.json, ...
```

The JSON Schemas of all these files are in [`../spec/`](../spec) and are validated against real compiler output in `tests/test_schemas.py`. Compiler outputs are byte-identical on every platform (LF line endings): the package index records the sha256 of every file.

## Supported architectures
Described by data (`m2w/frontends/hf_llm/families/*.json`), not by classes. Today: the **qwen3-like** family — pre-norm decoder with GQA, SwiGLU, RMSNorm, per-head q/k RMSNorm, rotate-half
RoPE, no biases, tied embedding: `Qwen3ForCausalLM`, `AionInstructForCausalLM`. Hybrid layers (sliding window + full attention) and per-layer-type RoPE (incl. YaRN) are supported through the
graph inputs `attention_mask`, `rope_cos`, `rope_sin`, so one BLOCK graph serves every layer. Not yet: attention / MLP biases, untied lm_head, MoE.

The **gemma4** family (`Gemma4ForConditionalGeneration`; text decoder of E2B / E4B): sandwich norms, different head dims per layer type, KV sharing in the last layers, per-layer embeddings (an int8 table read one row per token),
layer scalar, proportional RoPE with partial rotary, gelu-tanh, logit softcap. Staged KV only; details in [`../docs/models/gemma4-e4b.md`](../docs/models/gemma4-e4b.md).

The **qwen3.5** family (`Qwen3_5ForConditionalGeneration`; dense text decoder, e.g. Qwen3.5-4B): 3 Gated DeltaNet (linear attention) layers per full attention layer. Linear attention layers carry a conv
state and a recurrent state instead of a KV cache and are prefilled chunk by chunk without any loop operator; full attention layers have an output gate and partial rotary embeddings. The chunk sizes
must be multiples of 16; staged KV only, no block groups yet. Needs the `exp` and `softplus` operators. A checkpoint with a vision tower also gets the image tower (images only; `--vision-weights int8|int4|float16`, `--vision-grids`, `--no-vision`; `m2w add-vision` adds it to an existing package), see [`../docs/models/qwen35-vision.md`](../docs/models/qwen35-vision.md). Details in [`../docs/models/qwen35.md`](../docs/models/qwen35.md); `tests/test_qwen35.py` needs torch + transformers >= 5.3.

## Limits of the target (WebNN)
- **A tensor may not be bigger than 2^31 - 1 bytes (2 GiB)** in Chromium's WebNN: neither an MLTensor (KV cache, activations) nor an operand inside a graph. The one that bites is the dequantized lm_head matrix:
  `vocab x hidden` in float32 is 1.45 GiB for Qwen3 (151936 x 2560) and 1.53 GiB for Aion, but 2.5 GiB for Gemma 4 (262144 x 2560), where the browser fails with *"The tensor size 2684354560 exceeds the size limit 2147483647"*
  at the cast. The Gemma 4 lm_head graph is therefore built in row blocks (`lm_head_parts`, `lm_head_weight_<i>` / `lm_head_scale_<i>` are contiguous views of the one table) and the logits are concatenated.
- The compiler **checks every tensor of every graph it writes** (`m2w.ir.check_tensor_limit`, constant `WEBNN_MAX_TENSOR_BYTES`) and fails with the offending tensors and sizes; the runtime checks the state tensors it
  creates (`checkTensorSize` in `src/runtime/llm-common.ts`) and names the option to lower (max length, chunk size). A new architecture with a big vocabulary or a very long cache has to split the operation the same way.
- Other things the targets do not accept are listed with the operators they need in `m2w.json` (`requires`).

## Layout of the code
| path | purpose |
|---|---|
| `m2w/ir/` | WebNN-level IR: graph, builder with shape / dtype inference (names follow the WebNN spec), numpy interpreter, JSON (de)serialization, the WebNN tensor size check (`validate.py`) |
| `m2w/frontends/hf_llm/builders/decoder.py` | the NPUW-style subgraphs (HEAD / BLOCK / TAIL / lm_head / KV update), production options only |
| `m2w/frontends/hf_llm/` | `arch.py` (config → ModelSpec), `rope.py`, `safetensors_io.py`, `pack.py`, `chat_format.py`, `hub.py` |
| `m2w/quantize/` | int4 symmetric g128 (signed scale, optional error search), int8 per-row symmetric and uint8 per-row asymmetric (the lm_head table); a scale of 0 is never written (it can make the NPU lm_head dramatically slower) |
| `m2w/reference/` | independent float32 forward (`transformer_ref.py`), the package read back as float32 (`package_reader.py`), the host side of the runtime protocol as pure functions (`protocol.py`: masks, RoPE rows, selectors — the TypeScript runtime is tested against its golden vectors) and `graph_runner.py`: the compiled graphs run by the interpreter with that protocol |
| `m2w/compile.py`, `m2w/verify.py`, `m2w/cli.py` | the commands |

## Tests
`python -m pytest python` — no network, no GPU: IR rules, graph numerics against a float32 reference, three KV styles agreeing, the OpenVINO Qwen3 weights reproduced by the
quantizer (set `QWEN3_WEIGHTS` to the extracted folder to run it; skipped otherwise), YaRN against transformers (skipped without it), chat formats of Qwen3 / Aion, the JSON Schemas against a committed tiny package, and a full compile + verify of a tiny hybrid-layer model.

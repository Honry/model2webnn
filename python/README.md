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
- `--blocks-per-graph 1 5` / `--prefill-blocks-per-graph 1 5`: also build graphs with 5 blocks each (fewer dispatches, longer compile; hybrid models such as sliding + full attention get a mask and RoPE table per block as graph inputs).
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
graph inputs `attention_mask`, `rope_cos`, `rope_sin`, so one BLOCK graph serves every layer. Not yet: attention / MLP biases, other activations, partial rotary, untied lm_head, MoE.

## Layout of the code
| path | purpose |
|---|---|
| `m2w/ir/` | WebNN-level IR: graph, builder with shape / dtype inference (names follow the WebNN spec), numpy interpreter, JSON (de)serialization |
| `m2w/frontends/hf_llm/builders/decoder.py` | the NPUW-style subgraphs (HEAD / BLOCK / TAIL / lm_head / KV update), production options only |
| `m2w/frontends/hf_llm/` | `arch.py` (config → ModelSpec), `rope.py`, `safetensors_io.py`, `pack.py`, `chat_format.py`, `hub.py` |
| `m2w/quantize/` | int4 symmetric g128 (signed scale, optional error search), int8 per-row symmetric and uint8 per-row asymmetric (the lm_head table); a scale of 0 is never written (it can make the NPU lm_head dramatically slower) |
| `m2w/reference/` | independent float32 forward (`transformer_ref.py`), the package read back as float32 (`package_reader.py`), the host side of the runtime protocol as pure functions (`protocol.py`: masks, RoPE rows, selectors — the TypeScript runtime is tested against its golden vectors) and `graph_runner.py`: the compiled graphs run by the interpreter with that protocol |
| `m2w/compile.py`, `m2w/verify.py`, `m2w/cli.py` | the commands |

## Tests
`python -m pytest python` — no network, no GPU: IR rules, graph numerics against a float32 reference, three KV styles agreeing, the OpenVINO Qwen3 weights reproduced by the
quantizer (set `QWEN3_WEIGHTS` to the extracted folder to run it; skipped otherwise), YaRN against transformers (skipped without it), chat formats of Qwen3 / Aion, the JSON Schemas against a committed tiny package, and a full compile + verify of a tiny hybrid-layer model.

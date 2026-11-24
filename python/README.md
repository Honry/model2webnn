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

Useful `compile` options: `--max-length` / `--chunk` (graph variants: the shapes are static), `--blocks-per-graph 1 5` (also build graphs with 5 blocks each: fewer dispatches, longer compile; single layer type models only),
`--lm-head-format int8-sym|uint8-asym` (default int8-sym: same accuracy as the OpenVINO-style uint8 table, compiles faster on the NPU), `--compute-type`, `--staging-rows`.

Behind a network that cannot reach huggingface.co: `--endpoint https://hf-mirror.com` (or `HF_ENDPOINT`). Downloads resume after interruptions and retry on rate limits.

## What is in a package
```
m2w.json            entry point: format version, compiler, target, requires, file index (sizes, sha256)
model.json          architecture: dims, layer types (full / sliding attention), RoPE per layer type (default / linear / YaRN), tokens
plan.json           which graph for which phase / max length / chunk; how weights and per-layer-type inputs are bound; the KV protocol
chat_format.json    prompt pieces derived from the chat template (the runtime does not run jinja)
graphs/*.json       m2w-graph documents
weights/            manifest.json + layer_000.bin ... (int4 g128 symmetric, group-major) + the tied embedding / lm_head table
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

# NPU bench

Browser pages and scripts for running WebNN LLM graphs on the OpenVINO NPU backend: the hand-written Qwen3-4B graphs (`raw/chat.js`, `raw/qwen3_subgraphs.js`),
the package engine of the same page (`raw/chat-package.js`, runs what `m2w compile` writes with `src/runtime`), and the experiment pages (`raw/*.html`).

## Serve

The pages import the runtime from `lib/` (`npm run build:lib`), so serve the **repository root**:

    npm run build:lib
    bash bench/tools/sync-packages.sh          # links bench/packages/<name> to python/_out/<name>, writes models.json (add --ir for raw/irblock.html)
    python bench/server.py                     # http://localhost:8123/bench/raw/chat.html   (COOP/COEP headers, POST /log -> bench/results/)

Any static server rooted at the repository works for the `raw/` pages (they need HTTPS or localhost for WebNN); `server.py` is only needed for `bench.html` / `run_bench.ps1`.

## Local data (not in git)

| folder | content |
|---|---|
| `weights/qwen3_4b`, `weights/tok` | NNCF int4 weights extracted from the OpenVINO Qwen3-4B IR (`tools/extract_ov_weights.py`) + tokenizer files: input of the hand-written path (`?model=js`, the default) |
| `packages/` | links to the model packages built by `m2w compile`; `models.json` is the model list of `chat.html` |
| `ir/` | graph JSON for `raw/irblock.html` |
| `results/` | measurement output of `run_bench.ps1` (`*.jsonl`, `*.mem.csv`) |
| `ort-dist/`, `models/` | onnxruntime-web build and generated ONNX blocks for `bench.html` |
| `data/` | Chrome builds / profiles (`data/chrome`, or set `BENCH_CHROME_DIR`), the OpenVINO GenAI reference model and NPUW dump (`data/OV_GenAI_Qwen3`) |

Measurement notes with hardware-specific numbers are kept in `docs/npu-local/` (also not in git).

## Scripts

- `run_bench.ps1`: starts `server.py`, launches Chrome on a bench page, samples memory, waits for `done` (see the header for options). Env: `BENCH_CHROME_DIR`, `BENCH_DATA_DIR`.
- `bench.html` / `bench.js`, `gen_blocks.py`, `prep.py`, `prep_prompt.py`, `run_*.sh`: ONNX block benchmarks through onnxruntime-web (need `BENCH_QWEN_DIR` for the Qwen3 ONNX / tokenizer files).
- `raw/download.html`: download throughput of a package for different numbers of files in flight and byte ranges per big file (`LoadOptions.parallelDownloads` / `downloadSegments`); run it against the server or CDN the package will be served from.
- `tools/diff-handwritten.mjs`: structural diff between the hand-written graphs and the graphs the Python compiler writes (no browser needed).
- `tools/*.py`: OpenVINO IR helpers (weight extraction, op summary, profiling, zero-point probe).
- `../docs/npu/patches/`: patch for the onnxruntime WebNN EP (static GQA, chunked prefill).

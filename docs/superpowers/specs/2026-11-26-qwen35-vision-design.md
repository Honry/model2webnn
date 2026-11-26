# Qwen3.5 image input (vision tower) on the NPU: design

Status: draft for review. Scope decided with the user: **images only** (no video), vision weights **int8 per-row symmetric** (compile option for f16 / int4), **a few fixed grid presets** chosen at compile time.

## 1. What the model does with an image

`Qwen3_5ForConditionalGeneration` (checkpoint `Qwen/Qwen3.5-4B`, which already contains the vision tower: 297 `model.visual.*` tensors, no separate download):

```
image --smart_resize (multiples of 32)--> pixels [3, H, W] normalised with mean = std = 0.5
      --patchify--> patches [N, 3*2*16*16]   (N = grid_h * grid_w, patch 16, the image is repeated over the temporal patch size 2, order: 2x2 merge blocks)
      --ViT: patch embed (Conv3d == one matmul) + bilinear-interpolated position embedding (48 x 48 table -> grid) --> 24 blocks --> merger (2x2 -> 1 token, MLP) -->
      image tokens [N / 4, 2560]
prompt: <|vision_start|> <|image_pad|> x (N / 4) <|vision_end|>   -> the embedding rows of the pad tokens are replaced by the image tokens
text decoder: the same graphs as today, with 3D positions (t, h, w) for the rope of the 8 full attention layers (interleaved MRoPE, sections 11 / 11 / 10), causal mask
```

* ViT block: LayerNorm (mean subtracted, bias) -> qkv (bias) -> 2D rotary (head_dim 64, half of it row, half column) -> **bidirectional** attention over all N patches of the image (16 heads) -> proj -> residual; LayerNorm -> fc1 -> GELU (tanh) -> fc2 -> residual. hidden 1024, ffn 4096. No deepstack layers in this config.
* MRoPE only changes the numbers in the rope tables: the text graphs already take `rope_cos` / `rope_sin` as host inputs and only the 8 full attention layers use them (the 24 Gated DeltaNet layers have no position input). Text tokens: t = h = w = position. Image tokens: t = start, h = start + row, w = start + column; after the image the position continues at start + max(grid_h, grid_w) / 2. Decode: position = step + rope_delta (all three components equal).

## 2. Architecture

```
python (compiler)                                        package                                 runtime (TS) + page
------------------------------------                     -----------------------------           ---------------------------------------
arch: vision spec from config.vision_config  ----------> model.json  vision{...}, mrope{...}
pack: vision weights (int8 rows, f16 biases)  ---------> weights/vision-*.bin (+ manifest)
builders/qwen35_vision.py                                graphs/<variant>/vision_<grid>_*.json   Qwen35LLM.generate({ text, images })
  head  (patch matmul + bias + pos table)                plan.json vision.presets[]                 preprocess (canvas -> patches, host)
  block (shared by the 24 layers, weights = inputs)      vision tables (pos embed, rotary) / grid   vision graphs: head, 24 x block, merger
  merger                                                                                            embeddings -> rows of the prefill hidden
reference: Qwen35Runner (+ image) vs transformers                                                    mrope positions -> rope tables (prefill, decode)
```

* **Vision graphs follow the LLM split**: HEAD (patch embed + position embedding) / BLOCK (one graph for all 24 layers, weights as inputs) / MERGER (LayerNorm, 2x2 regroup, MLP). One set per grid preset.
* **Presets** = (grid_h, grid_w) in patches, multiples of 2. Default set: 16x16, 24x24, 32x32 (256 / 576 / 1024 patches -> 64 / 144 / 256 image tokens) and the 3:2 / 2:3 rectangles 16x24, 24x16, 24x32, 32x24. The runtime picks the preset with the closest aspect ratio whose size is closest to the original, resizes the image to exactly grid * 16 pixels. Everything that depends on the grid only (interpolated position embedding [N, 1024], rotary cos / sin [N, 64]) is computed at compile time and stored in the package: the graphs have no data-dependent shapes or gathers.
* **Weights**: linears int8 per-row symmetric (`dequantizeLinear` int8, scale [out, 1]: the path the lm_head already runs on the NPU); biases, LayerNorm and the patch-embed bias float16; the Conv3d kernel is reshaped to a [1024, 1536] matrix in the (channel, t, kh, kw) order of the patch vector. About 300 MB. Compile option `--vision-weights {int8,f16,int4}`.
* **Attention**: scores [16, N, N] f16 (N = 1024: 32 MB), full softmax, no mask (one image per graph call: the HF cu_seqlens of a single image is the whole sequence). Several images = several calls.
* **LayerNorm / GELU(tanh)** are decomposed into operators the NPU is known to run (mean / variance via reduceMean, tanh); whether the NPU compiler is happy with them is the first thing to measure (see risks).
* **Text side**: `protocol.rope_tables` gets a 3-position variant (interleaved MRoPE); the prefill chunk writer takes per-row embeddings (a row is either a token embedding or an image-token row); `Qwen35LLM` keeps `ropeDelta` for decode. The chat format / tokenizer produce `<|vision_start|><|image_pad|><|vision_end|>`, the runtime expands the pad token to N / 4 copies.
* **Memory / time order of magnitude** (to be measured): ViT 24 blocks x ~12.6 M params: ~350 GFLOP for 576 patches. The image tokens are prefill rows: 144 tokens = one extra chunk of 128-256.

## 3. Verification (same discipline as the text path)

1. **Python**: the numpy interpreter runs the vision graphs; compared with `Qwen3_5VisionModel` (transformers 5.3, dequantized weights) on random patches and on a real image: final merger output per image token, per-block hidden state.
2. **Full model**: `Qwen35Runner` with an image (embedding injection + MRoPE) vs `Qwen3_5ForConditionalGeneration` on the dequantized weights: prefill logits, a few decode steps (tiny VL config for the unit tests, the real 4B model locally for an end-to-end check).
3. **Runtime (TS)**: vitest with the mock WebNN: dispatch counts, embedding rows written at the image-token positions, rope tables of image rows.
4. **Device (NPU)**: a check page like `qwen35check.html`: per-block ViT hidden state vs the interpreter (the f16 / silent-wrong-result patterns of the text path may reappear), then the chat page with an image.

## 4. Phases

1. Compiler + interpreter: arch / pack / vision builders, verified against transformers (no device).
2. MRoPE + embedding injection in the Python reference runner, verified against `Qwen3_5ForConditionalGeneration`.
3. TS runtime: image preprocessing (canvas), vision graph dispatch, MRoPE tables, `generate` with images, tests on the mock; chat page with an image picker.
4. Device checks and performance on the NPU; recommended presets.

## 5. Risks / open points

* **f16 range**: ViT residual streams have large outliers; `computeType` float32 runs as f16 on the NPU, so overflow / precision has to be checked per block on the device.
* **LayerNorm / softmax over 1024 keys / big batched matmuls** may hit the same silent-wrong-result patterns as the text path (a value feeding both a matmul and an add, tiny-valued squares). Mitigations are known from the text path.
* **Preprocessing parity**: the browser resize (canvas) differs a little from the HF `smart_resize` + PIL bicubic; the verification feeds the same pixels to both, the end-to-end difference is not numerically bit exact.
* **Out of scope**: video, multiple images per prompt in one graph, deepstack (not in this config), other vision models.

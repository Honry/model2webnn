# Qwen3.5 image input (the vision tower)

What was built to let a Qwen3.5 package (`Qwen/Qwen3.5-4B`, which already contains the vision tower: 297 `model.visual.*` tensors) take **images** (no video, no deepstack) next to text. The design and its
reasoning are in [../superpowers/specs/2026-11-26-qwen35-vision-design.md](../superpowers/specs/2026-11-26-qwen35-vision-design.md); this file describes the code. Measurements (accuracy on a real image,
device numbers) are not in this tracked file: they go to the local notes (`docs/npu-local`). Numerics on the NPU are judged by running `bench/raw/qwen35vision.html` there, not by the interpreter.

## What the model does with an image

```
image --resize to a GRID PRESET (grid_h x grid_w patches of 16 px, all presets are multiples of 32 px)--> rescale 1/255, (x - 0.5) / 0.5
      --patchify--> patches [N, 1536]       N = grid_h * grid_w; merge-block order (the 2x2 patches of a merge block are consecutive rows),
                                            a patch is the (channel, temporal 2, kh, kw) vector of the Conv3d kernel; the image is repeated over the temporal patch
      --HEAD   patch matmul + bias + position embedding (48 x 48 learned table, bilinearly resampled to the grid) --> [1, N, 1024]
      --BLOCK  x 24 (ONE graph, weights bound per layer)  LayerNorm -> qkv(+bias) -> 2-D rotary -> bidirectional attention (16 heads, no mask) -> proj;  LayerNorm -> fc1 -> GELU(tanh) -> fc2
      --MERGER LayerNorm -> 4 consecutive rows = one row [N/4, 4096] -> fc1 -> GELU -> fc2 --> image_embeds [1, N/4, 2560]
prompt: <|vision_start|> <|image_pad|> x (N/4) <|vision_end|>  -> the embedding rows of the pad tokens are REPLACED by the image_embeds rows (the host writes them into the prefill chunk)
text:   the same graphs as before; the 8 full attention layers get 3-position (t, h, w) interleaved MRoPE tables (rope_cos / rope_sin are host inputs), decode position = index + rope_delta
```

## Package

| file | content |
|---|---|
| `model.json` | `vision` (dims, patch / merge sizes, image mean / std, token ids, `image_prompt`) and `mrope` (`section` [11, 11, 10], `interleaved`) |
| `plan.json` | `vision`: weight format, compute type, per preset `{grid, patches, tokens, graphs: {head, block, merger}}`, the tensor names bound to the inputs |
| `graphs/vision/<h>x<w>_head\|block\|merger.json` | the graphs of every grid preset (independent of the text variants) |
| `weights/vision_head.safetensors`, `vision_block_NNN.safetensors` (24), `vision_merger.safetensors` | the weights, named like the graph inputs (like the `layer_NNN` files of the text decoder) |
| `weights/vision_grid_<h>x<w>.safetensors` | per preset the tables that depend on the grid only: `pos_embed` [N, 1024], `rope_cos` / `rope_sin` [N, 64] (the external constants of HEAD / BLOCK: no gather and no data-dependent shape in a graph) |
| `weights/vision_pos_embed.safetensors` | the learned 2304 x 1024 table itself (only the Python reference reads it) |

* Weights: `m2w compile ... --vision-weights int8|int4|float16` (default **int8**: per output row symmetric, `dequantizeLinear` with scale [out, 1] and a zero point of 0: the path the lm_head runs on the NPU;
  int4: group 128 like the text layers; float16: no quantization). Biases, LayerNorm and the patch-embed bias are float16, the Conv3d kernel is a [1024, 1536] matrix in the (channel, temporal, kh, kw) order.
  About 300 MB with int8.
* Presets: `--vision-grids 16x16 24x24 ...` (default 16x16 24x24 32x32 16x24 24x16 24x32 32x24; multiples of 2 patches). `--vision-compute-type float32|float16` (graph interior; default float32),
  `--no-vision` leaves the tower out. `m2w add-vision <package> --model <hf dir>` adds the tower to an existing package without quantizing the text again; `m2w rebuild-graphs` keeps it.
* The preset for an image: `choose_grid` (python `frontends/hf_llm/vision.py`, TS `chooseGrid`): the closest aspect ratio first (log distance; within 0.1 of the best counts as the same), then the closest
  pixel count. The image is resized to exactly `grid * 16` pixels (PIL bicubic in the Python reference, a canvas in the browser; the vision graphs are verified on the SAME pixels).

## The vision graphs (`builders/qwen35_vision.py`)

* Same NPUW-style split as the text decoder: HEAD / BLOCK / MERGER, weights as graph inputs. Shapes per preset (N patches): attention scores are [1, 16, N, N] (32 MB in float16 for N = 1024), softmax
  over all N keys, no mask (one image per dispatch chain; several images = several chains).
* **LayerNorm** is decomposed (`reduceMean`, `pow`, `sqrt`, `div`) and computed on the input times a power of two (1/16 in the blocks, 1/64 in the merger): the result does not depend on the scale, the
  float16 range of the mean / squares does. The activations of this ViT leave the comfortable float16 range (a few "register" tokens carry values in the hundreds from block 10 on, the last block raises every
  token into the thousands), so the unscaled squares would overflow where the device runs float32 graphs as float16.
* **GELU**: the tanh form everywhere (transformers uses the exact erf in the merger: they differ by 4.7e-4 at most; this IR has no erf). The polynomial `u + 0.044715 u^3` is formed on `u = clamp(x, -8, 8)`
  (new IR operator `clamp`, WebNN `clamp`): the cube of an fc1 output above 40 overflows float16, and tanh is 1 there anyway; the factor `x` stays unclamped.
* Rotary: `cos` / `sin` [1,1,N,64] = `[row | col | row | col]` angles (16 + 16 + 16 + 16), `apply_rotary` of the text decoder (rotate half). No permutation of the weights is needed.
* NPU pitfalls of the text path that apply here and how they are respected: activation x activation matmuls have all dims >= 16 (head dim 64, N >= 64); the residual feeds the LayerNorm and the add (both
  elementwise), never a matmul and an add; no eps below 6e-5 matters (flushed to 0 on the NPU: only a row without variance would notice).
* `vision_weights=int8` is the NPU-friendly default but the least accurate of the three on this model (see the local notes for numbers: the `fc2` of the blocks suffers most because of the register tokens);
  `float16` keeps the reference quality, `int4` is clearly worse.

## Verification (no device)

* `python/tests/test_qwen35_vision.py`: the compiled graphs (numpy interpreter, packed weights) against `transformers.Qwen3_5VisionModel` loaded with the DEQUANTIZED package weights, for every preset of a
  tiny model: image embeddings and the hidden state after every block; the three weight formats; the float16 compute type; the per-grid tables against the model's own `fast_pos_embed_interpolate` /
  `rot_pos_emb`; `patchify` against `Qwen2VLImageProcessor`; `choose_grid`; `add-vision` / `rebuild-graphs`.
* `python/tools/qwen35_vision_check.py <package> <hf dir> <image>`: the same on the real model and a real image, plus the cost of the quantization against the ORIGINAL weights and the activation ranges;
  `--export` writes the patches and the hidden state after every block into `bench/data/qwen35_vision/` (ignored) for the device check page.

## Image in the prompt: the Python reference runner (`reference/qwen35_runner.py`, `reference/protocol.py`)

`Qwen35Runner.prefill(tokens, images=[(patches, grid), ...])` (executable specification of what the browser runtime does):

1. `protocol.prepare_image(pil_image, spec, grids)` -> `(patches, grid)`: preset by `choose_grid`, bicubic resize to `grid * 16` pixels, rescale + normalise with the mean / std of model.json, `patchify`.
2. `protocol.expand_image_tokens`: the prompt holds ONE `<|image_pad|>` per image (the chat template writes `<|vision_start|><|image_pad|><|vision_end|>`, `model.json vision.image_prompt`);
   it becomes `grid_h * grid_w / 4` pad tokens (a run that already has that length stays).
3. `Qwen35VisionRunner.encode(patches, grid)` -> image tokens [N/4, 2560] float16; their rows REPLACE the embedding rows of the pad tokens in the prefill chunk (they may straddle chunk boundaries).
4. `protocol.mrope_positions` (transformers' `get_rope_index`): text tokens t = h = w = running position; the tokens of an image get `t = start`, `h = start + row`, `w = start + column` of their merge block,
   the position continues at `start + max(rows, columns)` after the image. `rope_delta = max position + 1 - number of tokens`.
5. `protocol.rope_tables_mrope`: the rope tables of the full attention layers with one of the three positions per rotary frequency in the interleaved layout (`mrope_axes`: t, h, w, t, h, w ...; sections
   11 / 11 / 10 of the 32 frequencies; the package's permuted head dims keep the rotating dims in front, the other frequencies are 0). Only the 8 full attention layers see positions at all (the Gated DeltaNet
   layers have no position input), so MRoPE is purely host-side table building.
6. Decode: the token at sequence index `i` uses the rope position `i + rope_delta` (all three components equal); cache rows and masks keep using `i`.

`python/tests/test_qwen35_images.py` checks it on a tiny model against `Qwen3_5ForConditionalGeneration` loaded with the same dequantized weights: positions and `rope_delta` against `get_rope_index`, the
tables against the text model's own rotary embedding, prefill logits and ten decode steps (one image, two images with different grids, an image that straddles a prefill chunk boundary); a prompt without
image is unchanged. `python/tools/qwen35_vl_check.py` does the same on the real 4B model with a real image (slow: the interpreter runs the whole model) and can also print the answer of the original bfloat16
checkpoint.

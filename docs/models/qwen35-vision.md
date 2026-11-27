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

* Weights: `m2w compile ... --vision-weights int8|int8g128|int4|float16` (default **int8g128**; **int8**: per output row symmetric, `dequantizeLinear` with scale [out, 1] and a zero point of 0: the path the lm_head runs on the NPU;
  **int8g128**: int8 data in the layout of the int4 path, `[in/128, out, 128]` with a float16 scale `[in/128, out, 1]` per block of 128 input channels (`dequantizeLinear` with an int8 zero point of 0): the
  same size as int8 plus the block scales (about 0.3 GB for the whole tower, 0.6 GB for float16), a few outlier rows / blocks no longer spoil a whole row; int4: group 128 like the text layers; float16: no quantization).
  **Default: `int8g128`** (the quality of float16 for practical purposes at half the size; the block-scaled `dequantizeLinear` runs on the NPU and the image suite passes with it;
  `float16` is the reference to compare against, `int4` is the smallest but clearly the least faithful: kept as an option only). The numbers behind this are in the local notes (`python/tools/qwen35_vision_quality.py`). Biases, LayerNorm and the patch-embed bias are float16, the Conv3d kernel is a [1024, 1536] matrix in the (channel, temporal, kh, kw) order.
  About 300 MB with int8.
* Presets: `--vision-grids 16x16 24x24 ...` (default 16x16 24x24 32x32 16x24 24x16 24x32 32x24; multiples of 2 patches). `--vision-compute-type float32|float16` (graph interior; default float32),
  `--no-vision` leaves the tower out. `m2w add-vision <package> --model <hf dir>` adds the tower to an existing package without quantizing the text again; `m2w rebuild-graphs` keeps it.
* The preset for an image is chosen AUTOMATICALLY (`choose_grid`, python `frontends/hf_llm/vision.py`, TS `chooseGrid`; see "Choosing the grid" below): nobody has to know the preset names. The image is resized to
  exactly `grid * 16` pixels (PIL bicubic in the Python reference, a canvas in the browser; the vision graphs are verified on the SAME pixels).

## Choosing the grid

The graphs are static, so a picture is resized to one of the package's presets (`--vision-grids`). Which one is decided by the runtime from the width and height of the picture and a **detail level**
(`chooseGrid(grids, width, height, patch, detail)` in `protocol.ts`, `choose_grid` in `vision.py`; `tests/golden/protocol_vectors.json` keeps the two identical):

1. **size**: `target = min(patches of the picture, budget)` with the budget of the detail level: `low` 256 patches (16x16, 64 image tokens, fastest), `auto` 576 (24x24, 144 tokens, the default), `high` 1024 (32x32, 256 tokens: text,
   small details). Presets with more than 1.5 x `target` patches are out (a picture is never blown up far beyond its own size; the smallest preset always stays, so a 200 x 150 thumbnail gets 16x16);
2. **aspect ratio**: of the rest the closest log aspect ratio (presets within 0.1 of the best count as equal): the picture is stretched a little to the preset's ratio, never cropped or padded;
3. of those the one whose patch count is closest to the `target` (ties go to the larger).

The graphs of a preset are built **lazily** on its first use (three graphs, a few seconds on a device; the weights are shared by all presets, so nothing is loaded or compiled for presets that are never used), so the
package should simply offer every preset it can; the page's grid field is gone. `?visionGrids=24x24,16x24` is an advanced URL override that only restricts the candidates (one preset = forced), `?detail=auto|low|high` sets the level.

| preset (h x w) | patches | image tokens | resized pixels (w x h) | aspect (w:h) | vision cost (estimated from the MAC count, 16x16 = 1) | chosen for |
|---|---|---|---|---|---|---|
| 16x16 | 256 | 64 | 256 x 256 | 1.00 | 1.0 | square pictures at `low`; any square picture below ~ 310 x 310 px (a thumbnail is never blown up) |
| 16x24 | 384 | 96 | 384 x 256 | 1.50 | 1.5 | landscape 3:2 and wider (16:9, banners 3:1 are stretched to 3:2); everything wide at `low` |
| 24x16 | 384 | 96 | 256 x 384 | 0.67 | 1.5 | portrait 2:3 and taller |
| 24x24 | 576 | 144 | 384 x 384 | 1.00 | 2.4 | square pictures from ~ 310 x 310 px up at `auto`, and at `high` below ~ 520 x 520 px |
| 24x32 | 768 | 192 | 512 x 384 | 1.33 | 3.2 | 4:3 landscape at `auto` / `high` from ~ 370 x 280 px up (screenshots, scans, documents); smaller ones get 16x24 |
| 32x24 | 768 | 192 | 384 x 512 | 0.75 | 3.2 | 3:4 portrait at `auto` / `high` (phone screenshots, pages) |
| 32x32 | 1024 | 256 | 512 x 512 | 1.00 | 4.5 | square pictures of ~ 520 x 520 px and more at `high` only |
| 16x32 | 512 | 128 | 512 x 256 | 2.00 | 2.0 | 2:1 and wider pictures (banners, wide crops) at `auto` / `high` |
| 32x16 | 512 | 128 | 256 x 512 | 0.50 | 2.0 | 1:2 and taller pictures |
| 18x32 | 576 | 144 | 512 x 288 | 1.78 | 2.4 | 16:9 landscape at `auto`: screens, screenshots, video frames |
| 32x18 | 576 | 144 | 288 x 512 | 0.56 | 2.4 | 9:16 portrait (phone screenshots) at `auto` |
| 24x36 | 864 | 216 | 576 x 384 | 1.50 | 3.8 | 3:2 landscape at `high` (and at `auto` for large 3:2 pictures) |
| 36x24 | 864 | 216 | 384 x 576 | 0.67 | 3.8 | 2:3 portrait at `high` |
| 24x42 | 1008 | 252 | 672 x 384 | 1.75 | 4.5 | 16:9 landscape at `high`: the most detail for a screenshot |
| 42x24 | 1008 | 252 | 384 x 672 | 0.57 | 4.5 | 9:16 portrait at `high` |

The cost class is the order of magnitude of the vision encode (24 blocks over N patches: linear in N plus a small N^2 attention term, about 14 % of the work at N = 1024), not a prediction; the language model then
prefills `prompt + image tokens` (prefill chunks of 64 / 128 tokens: 64 image tokens fill one chunk, 256 fill two to four). Examples of what the rule picks (low / auto / high): 1000 x 1000 -> 16x16 / 24x24 / 32x32;
1500 x 1000 and 1920 x 1080 -> 16x24 at every level; 1280 x 960 -> 16x24 / 24x32 / 24x32; 1824 x 2358 -> 24x16 / 32x24 / 32x24; 400 x 400 -> 16x16 / 24x24 / 24x24; 200 x 150 -> 16x16.

Guidance: a package offers all fifteen default presets (compile default; the cost of an unused preset is its three small graph JSON files and a table of a few MB). Small text needs resolution, not a bigger model: use `high` for OCR. A 16:9 screenshot at `high` gets 24x42 = 672 x 384 pixels (252 image tokens), at `auto` 18x32 = 512 x 288 (144 tokens). `m2w add-vision <package> --model <hf dir> [--vision-grids ...]` adds or changes the presets of an existing package (it repacks the vision weights, the text is untouched). The test suite `bench/raw/vision-suite.html` runs every case at a chosen detail level and reports the grid per case.

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
* `vision_weights=int8` (per row) is the least accurate of the int8 variants on this model (see the local notes for numbers: the `fc2` of the blocks suffers most because of the register tokens);
  `int8g128` removes most of that loss at the same size, `float16` keeps the reference quality, `int4` is clearly worse.

## Verification (no device)

* `python/tests/test_qwen35_vision.py`: the compiled graphs (numpy interpreter, packed weights) against `transformers.Qwen3_5VisionModel` loaded with the DEQUANTIZED package weights, for every preset of a
  tiny model: image embeddings and the hidden state after every block; the three weight formats; the float16 compute type; the per-grid tables against the model's own `fast_pos_embed_interpolate` /
  `rot_pos_emb`; `patchify` against `Qwen2VLImageProcessor`; `choose_grid`; `add-vision` / `rebuild-graphs`.
* `python/tools/qwen35_vision_quality.py` (no device): the image tokens of every weight format against the ORIGINAL bfloat16 vision tower on the same pixels (cosine per token, relative error) and the greedy
  answers / teacher-forced log-likelihood of the full model on the cases of `bench/raw/vision-suite` (transformers on the original checkpoint, transformers on the dequantized package weights of each format, and our numpy
  runner); `--build int8g128 float16 int4` makes the sibling packages (`python/_out/qwen35-4b-vl-<format>-pkg`: text shards hard-linked, own vision shards), `--image <file>` adds a real picture to the comparison.
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

## The browser runtime (`src/runtime/vision.ts`, `llm-qwen35.ts`, `protocol.ts`)

```ts
const llm = await Qwen35LLM.load(url, { vision: true });                             // true = the tower; a picture may get any preset of the package ({ grids: [...] } restricts them, { preload: [...] } builds some now); absent = text only
const image = llm.prepareImage(await createImageBitmap(file), undefined, 'auto');    // preset from the size / aspect ratio / detail (low | auto | high), canvas resize, normalise, patchify -> { patches, grid, tokens }
const ids = tokenizer.encode(buildPrompt(format, [{ user: 'What is this?', images: 1 }], undefined, new Date(), llm.pkg.model.vision.image_prompt));
for await (const token of llm.generate(ids, { images: [image] })) { ... }             // ONE <|image_pad|> in `ids` per image: the runtime expands it
```

* `VisionTower` (vision.ts): uploads `vision_head` / `vision_block_NNN` / `vision_merger` once; HEAD / BLOCK / MERGER of a preset are built on its first use (`ensure(grid)`, called by `encode`; progress
  `build vision 24x16 head | block | merger`; the grid tables are external constants read from the `vision_grid` shard), and per image dispatches HEAD, the same BLOCK graph `depth` times with the weights of layer i bound by name, MERGER; the image tokens are read back as float16 rows. Two hidden-state
  tensors take turns (all of them are kept readable with `VisionTower.debugReadback`, for `bench/raw/qwen35vision.html`).
* `Qwen35LLM.prefill` expands the pad tokens (`protocol.ts expandImageTokens`), computes the positions (`mropePositions`), runs the tower, and writes the image rows into the prefill chunks in place of the
  embedding rows (a run of image tokens may straddle chunk boundaries); the rope tables of the chunks are `ropeTablesMrope`; `step` uses `position + ropeDelta`. The same graphs, controls and states as for text.
  Stats: `imageTokens`, `visionMs`, `visionBuildMs` (both inside `prefillMs`; the first use of a preset builds its graphs, `visionMs` is the encode alone).
* `protocol.ts` mirrors `protocol.py` (`chooseGrid`, `expandImageTokens`, `mropePositions`, `mropeAxes`, `ropeTablesMrope`, `patchify`); `tests/golden/protocol_vectors.json` keeps the two identical.
* Preprocessing in the browser (`prepareImage`): canvas `drawImage` to exactly `grid * 16` pixels with smoothing, RGBA -> `(v / 255 - mean) / std`, `patchify`. The Python reference resizes with PIL bicubic: the two differ a
  little at the pixel level (the vision graphs are verified on the SAME pixels).
* Pages: `bench/raw/chat.html` is a conversation page (session column on the left; message list with a composer, then the numbers of the last turn, in the main area). With a package that has an image tower the composer has
  **Attach image** (file / drop / paste / the suite's samples; a thumbnail chip per picture, a click shows the picture drawn at the pixels of the preset the runtime chose with its image-token blocks and a caption saying why)
  and a Detail select; a message can carry several pictures, each stays with its turn and is encoded again when the history is re-read. The Summary of the last turn gets a group of rows per picture (grid, image tokens, preprocess,
  graph build on the first use of a grid) and the vision encode time. URL: `detail=auto|low|high`, `visionGrids=` (advanced restriction), `vision=0` (no tower), `image=<url>` (attached to the first message), `question=<text>`, with
  `autoRun=1` the message is sent (`tab=vision` is still accepted: with autoRun it then needs `image=`).
  `bench/raw/vision-suite.html` (cases in `bench/raw/vision-suite/`, made by `python/tools/make_vision_suite.py`): every case through the real path, PASS / FAIL against its expected substrings, grid / tokens / timing per case
  and a summary per grid (`model=`, `detail=`, `grids=`, `case=`, `repeat=`, `maxNewTokens=`; greedy decoding; a case may list `xfail_at` detail levels where a FAIL is expected, e.g. `ocr-micro` at `low`: reported as XFAIL and not counted). `bench/raw/qwen35vision.html` (+ `python/tools/qwen35_vision_check.py
  --export`): the device against the interpreter, hidden state after HEAD and after every block, image tokens (relative error, max abs, non-finite values, cosine per token).
  `node scripts/page-smoke.mjs "<page url>" --wait Done` runs a page in headless Chrome against a validating mock of WebNN and the tiny package (no device: checks the page logic only).
* Tests: `src/runtime/llm-qwen35-vision.test.ts` on the validating mock of WebNN (every weight / table / activation of every dispatch is checked by name, shape and dtype; image rows at the right prompt positions
  across a chunk boundary; MRoPE tables of the chunks; decode with `ropeDelta`; the canvas preprocessing with a stubbed `OffscreenCanvas`), `protocol.test.ts` (golden vectors).

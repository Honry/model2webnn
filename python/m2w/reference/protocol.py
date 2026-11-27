"""The host side of the runtime protocol as pure functions: everything a runtime has to write into the graph inputs of one dispatch chain.
The TS runtime (src/runtime/protocol.ts) implements the same functions; tests/golden/protocol_vectors.json (generated from here) keeps the two identical.

Conventions:
  * the KV cache has P = max_length - 1 rows, cache row j holds the key / value of position j; the staging buffer has R rows;
  * masks are additive, float16: 0 = visible, -65504 = hidden; one mask and one rope table per LAYER TYPE (sliding vs full attention);
  * prefill chunk of C rows, the last chunk is LEFT-padded: [pad rows | real rows]; pad rows see only themselves, their RoPE position is the first real position;
    prefill mask columns = [cache rows 0..P-1 | chunk rows 0..C-1];
  * decode (staged KV): mask columns = [main cache rows 0..P-1 | staging rows 0..R-1]; the new token goes into staging row `filled`, which holds position base+filled;
  * decode (onehot KV): no staging buffer, the new token is written into cache row `pos` by the graph; mask columns = [cache rows 0..P-1 | the new row];
  * scatterND KV (prefill and decode): the graph scatters the new rows into the cache at the rows of `kv_indices` and attends over the updated cache (mask columns = cache rows 0..P-1);
    row j holds position j as in the one-hot protocol; the left-padded last chunk parks its pad rows in the cache rows behind the real rows (clamped to P-1), where they are masked;
  * tensorScatter KV (prefill and decode): the graph scatters the new rows into the cache at `kv_write_index` (the rows go in contiguously) and attends over the
    updated cache (mask columns = cache rows 0..P-1). Every prefill chunk is full: the padding of the prompt is in FRONT of it (`pad` rows, cache rows 0..pad-1),
    token i sits in cache row pad + i, and the decode step of position pos writes row pad + pos.
  * sliding window caches (the staged protocol only; plan.profile.sliding_cache = "window"): the sliding layers hold a RING of `ring` rows, position p lives in slot p mod ring, and
    the functions that take `ring` (prefill_masks, prefill_selectors, decode_masks, flush_selectors, incache_prefill_masks) give those layers masks / selectors with `ring` instead of
    P cache rows. P is still max_length - 1; cache_rows() tells which layer type has how many rows. Nothing else changes (RoPE, staging buffer, chunk rows).
"""
from __future__ import annotations

import numpy as np

from ..frontends.hf_llm import rope as rope_lib, vision
from ..frontends.hf_llm.arch import ModelSpec

NEG = np.float16(-65504)


def cache_rows(layer_type: str, P: int, ring: int = 0) -> int:
    """Rows of the cache of a layer type: the ring for sliding layers of a window package (ring < P), else all P."""
    return ring if ring and ring < P and layer_type == "sliding_attention" else P


def slot_positions(rows: int, written: int) -> np.ndarray:
    """The position that every slot of a ring of `rows` rows holds after the positions 0..written-1 were written (position p goes to slot p mod rows): the newest one, -1 = empty.
    With rows = P (the full cache; written <= P) slot j holds position j."""
    slots = np.arange(rows)
    held = slots + ((written - 1 - slots) // rows) * rows
    return np.where(slots < written, held, -1)


def visible(spec: ModelSpec, layer_type: str, q_pos: int, k_pos: int) -> bool:
    if k_pos > q_pos:
        return False
    return layer_type != "sliding_attention" or q_pos - k_pos < spec.sliding_window


def rope_tables(spec: ModelSpec, positions: list, rows: int, pad: int) -> dict:
    """layer type -> (cos, sin) float16 [1,1,rows,head_dim]; the first `pad` rows reuse the first real position."""
    out = {}
    for ltype, entry in spec.rope.items():
        inv, factor = rope_lib.inv_freq_for(entry, spec.decoder_dims(ltype).head_dim)   # the head dim can differ per layer type (Gemma 4)
        cos, sin = rope_lib.cos_sin(positions, inv, factor)
        index = np.maximum(0, np.arange(rows) - pad)
        out[ltype] = (cos[index][None, None].astype(np.float16), sin[index][None, None].astype(np.float16))
    return out


def prefill_masks(spec: ModelSpec, P: int, C: int, start: int, valid: int, ring: int = 0) -> dict:
    """layer type -> mask float16 [1,1,C,rows+C] of the chunk that holds `valid` real tokens (positions start..start+valid-1); rows = cache_rows(). The cache columns are the cache rows
    (row j holds position j) or, for a ring layer, its slots, which hold the positions slot_positions(ring, start)."""
    pad = C - valid
    masks = {}
    for ltype in spec.mask_types():
        rows = cache_rows(ltype, P, ring)
        held = slot_positions(rows, start)
        mask = np.full((C, rows + C), NEG, np.float16)
        for row in range(C):
            if row < pad:
                mask[row, rows + row] = 0        # pad row sees only itself
                continue
            q_pos = start + (row - pad)
            for j in range(rows):                 # cache rows of earlier chunks
                if held[j] >= 0 and visible(spec, ltype, q_pos, int(held[j])):
                    mask[row, j] = 0
            for c in range(pad, row + 1):         # real rows of this chunk up to the row itself
                if visible(spec, ltype, q_pos, start + (c - pad)):
                    mask[row, rows + c] = 0
        masks[ltype] = mask[None, None]
    return masks


def prefill_selectors(P: int, C: int, start: int, valid: int, wrap: bool = False):
    """(rows_to_keep [1,1,P,1], row_selector [1,1,P,C]) float16: chunk row pad+i writes cache row start+i (`wrap`: slot (start+i) mod P of a ring). P = the rows of THIS cache."""
    pad = C - valid
    keep = np.ones((1, 1, P, 1), np.float16)
    selector = np.zeros((1, 1, P, C), np.float16)
    for i in range(valid):
        slot = (start + i) % P if wrap else start + i
        keep[0, 0, slot, 0] = 0
        selector[0, 0, slot, pad + i] = 1
    return keep, selector


def decode_masks(spec: ModelSpec, P: int, R: int, pos: int, base: int, filled: int, ring: int = 0) -> dict:
    """layer type -> mask float16 [1,1,1,rows+R] for the token at position `pos` = base + filled; rows = cache_rows() (the slots of a ring hold slot_positions(ring, base))."""
    masks = {}
    for ltype in spec.mask_types():
        rows = cache_rows(ltype, P, ring)
        held = slot_positions(rows, base)
        mask = np.full((rows + R,), NEG, np.float16)
        for j in range(rows):
            if held[j] >= 0 and visible(spec, ltype, pos, int(held[j])):
                mask[j] = 0
        for i in range(filled + 1):
            if visible(spec, ltype, pos, base + i):
                mask[rows + i] = 0
        masks[ltype] = mask.reshape(1, 1, 1, -1)
    return masks


def decode_selectors(R: int, filled: int):
    """(staging_rows_to_keep [1,1,R,1], staging_row_selector [1,1,R,1]): the new token goes into staging row `filled`."""
    keep = np.ones((1, 1, R, 1), np.float16)
    selector = np.zeros((1, 1, R, 1), np.float16)
    keep[0, 0, filled, 0], selector[0, 0, filled, 0] = 0, 1
    return keep, selector


def flush_selectors(P: int, R: int, base: int, wrap: bool = False):
    """(rows_to_keep [1,1,P,1], row_selector [1,1,P,R]): staging row i -> main cache row base+i (rows beyond the cache are dropped); `wrap`: slot (base+i) mod P of a ring
    (R <= P: the slots of one flush are different). P = the rows of THIS cache."""
    keep = np.ones((1, 1, P, 1), np.float16)
    selector = np.zeros((1, 1, P, R), np.float16)
    for i in range(R):
        slot = (base + i) % P if wrap else base + i
        if slot < P:
            keep[0, 0, slot, 0] = 0
            selector[0, 0, slot, i] = 1
    return keep, selector


# ---------------------------------------------------------------------------------------------------------------------
# prefill planning (several chunk sizes)
# ---------------------------------------------------------------------------------------------------------------------
def plan_prefill(sizes, token_count: int, fixed_tokens: int = 200) -> list:
    """The chunk sizes to use for a prompt of `token_count` tokens: every chunk but the last is full, the last one is left-padded (tensorScatter: the first
    chunk holds the padding). A dispatch costs fixed_tokens + size 'token units' (the int4 weights are dequantized again by every dispatch); minimises the
    sum; ties: fewer chunks, bigger chunks first."""
    ordered = sorted(sizes, reverse=True)
    memo: dict = {}

    def best(remaining: int):
        if remaining in memo:
            return memo[remaining]
        result = None
        for size in ordered:
            cost = fixed_tokens + size
            if size >= remaining:
                candidate = (cost, [size])
            else:
                rest_cost, rest_plan = best(remaining - size)
                candidate = (cost + rest_cost, [size] + rest_plan)
            if result is None or candidate[0] < result[0] or (candidate[0] == result[0] and len(candidate[1]) < len(result[1])):
                result = candidate
        memo[remaining] = result
        return result

    return best(token_count)[1]


# ---------------------------------------------------------------------------------------------------------------------
# onehot KV (decode writes cache row `pos` itself)
# ---------------------------------------------------------------------------------------------------------------------
def decode_masks_onehot(spec: ModelSpec, P: int, pos: int) -> dict:
    """layer type -> mask float16 [1,1,1,P+1]: cache rows 0..pos-1 (that the layer type sees) and the new token (column P)."""
    masks = {}
    for ltype in spec.mask_types():
        mask = np.full((P + 1,), NEG, np.float16)
        for j in range(pos):
            if visible(spec, ltype, pos, j):
                mask[j] = 0
        mask[P] = 0
        masks[ltype] = mask.reshape(1, 1, 1, -1)
    return masks


def decode_selectors_onehot(P: int, pos: int):
    """(rows_to_keep [1,1,P,1], row_selector [1,1,P,1]): the new token goes to cache row `pos`."""
    keep = np.ones((1, 1, P, 1), np.float16)
    selector = np.zeros((1, 1, P, 1), np.float16)
    keep[0, 0, pos, 0], selector[0, 0, pos, 0] = 0, 1
    return keep, selector


# ---------------------------------------------------------------------------------------------------------------------
# tensorScatter KV
# ---------------------------------------------------------------------------------------------------------------------
def scatter_prefill_rope(spec: ModelSpec, C: int, start: int, pad: int) -> dict:
    """RoPE tables of the chunk that covers cache rows start..start+C-1 (token i is in row pad + i): the pad rows reuse the first real position."""
    first_real = max(0, pad - start)
    first_token = max(0, start - pad)
    return rope_tables(spec, list(range(first_token, first_token + C - first_real)), C, first_real)


def scatter_prefill_masks(spec: ModelSpec, P: int, C: int, start: int, pad: int) -> dict:
    """layer type -> mask float16 [1,1,C,P] of the chunk that covers cache rows start..start+C-1: columns are cache rows; a pad row sees only itself."""
    masks = {}
    for ltype in spec.mask_types():
        mask = np.full((C, P), NEG, np.float16)
        for row in range(C):
            cache_row = start + row
            token = cache_row - pad
            if token < 0:
                mask[row, cache_row] = 0
                continue
            for j in range(pad, cache_row + 1):
                if visible(spec, ltype, token, j - pad):
                    mask[row, j] = 0
        masks[ltype] = mask[None, None]
    return masks


def scatter_decode_masks(spec: ModelSpec, P: int, pos: int, pad: int) -> dict:
    """layer type -> mask float16 [1,1,1,P] for the token at position `pos` (cache row pad + pos, written by the graph before attention)."""
    masks = {}
    for ltype in spec.mask_types():
        mask = np.full((P,), NEG, np.float16)
        for j in range(pad, pad + pos + 1):
            if visible(spec, ltype, pos, j - pad):
                mask[j] = 0
        masks[ltype] = mask.reshape(1, 1, 1, -1)
    return masks


# ---------------------------------------------------------------------------------------------------------------------
# scatterND KV
# ---------------------------------------------------------------------------------------------------------------------
def scatternd_prefill_masks(spec: ModelSpec, P: int, C: int, start: int, valid: int) -> dict:
    """layer type -> mask float16 [1,1,C,P] of the chunk that holds `valid` real tokens at positions start..start+valid-1: columns are cache rows (the chunk's rows are scattered in
    before attention); a pad row sees only the cache row it is parked in."""
    pad = C - valid
    masks = {}
    for ltype in spec.mask_types():
        mask = np.full((C, P), NEG, np.float16)
        for row in range(C):
            if row < pad:
                mask[row, min(P - 1, start + valid + row)] = 0
                continue
            q_pos = start + (row - pad)
            for j in range(q_pos + 1):
                if visible(spec, ltype, q_pos, j):
                    mask[row, j] = 0
        masks[ltype] = mask[None, None]
    return masks


def scatternd_prefill_indices(num_kv_heads: int, P: int, C: int, start: int, valid: int) -> np.ndarray:
    """int32 [1,kvHeads,C,3] = (0, head, cache row) of every chunk row: real rows go to start.., pad rows are parked behind them (clamped to P-1)."""
    pad = C - valid
    out = np.zeros((1, num_kv_heads, C, 3), np.int32)
    for h in range(num_kv_heads):
        for row in range(C):
            out[0, h, row] = (0, h, min(P - 1, start + valid + row) if row < pad else start + (row - pad))
    return out


def scatternd_decode_masks(spec: ModelSpec, P: int, pos: int) -> dict:
    """layer type -> mask float16 [1,1,1,P] for the token at position `pos` (cache row pos, written by the graph before attention)."""
    masks = {}
    for ltype in spec.mask_types():
        mask = np.full((P,), NEG, np.float16)
        for j in range(pos + 1):
            if visible(spec, ltype, pos, j):
                mask[j] = 0
        masks[ltype] = mask.reshape(1, 1, 1, -1)
    return masks


def scatternd_decode_indices(num_kv_heads: int, pos: int) -> np.ndarray:
    """int32 [1,kvHeads,1,3]: the new token goes to cache row `pos` of every head."""
    out = np.zeros((1, num_kv_heads, 1, 3), np.int32)
    for h in range(num_kv_heads):
        out[0, h, 0] = (0, h, pos)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# attention that reads another layer's UPDATED cache in a prefill chunk (KV-shared layers, Gemma 4)
# ---------------------------------------------------------------------------------------------------------------------
def incache_prefill_masks(spec: ModelSpec, P: int, C: int, start: int, valid: int, ring: int = 0) -> dict:
    """layer type -> mask float16 [1,1,C,rows]: columns are the rows of the (already updated) cache. A real row sees the rows up to its position (window of its layer type); a pad row sees
    row 0 only (any visible column keeps the softmax finite; its output is never used). A ring layer: the slots hold slot_positions(ring, start + valid) after the update of the chunk."""
    pad = C - valid
    masks = {}
    for ltype in spec.mask_types():
        rows = cache_rows(ltype, P, ring)
        held = slot_positions(rows, start + valid)
        mask = np.full((C, rows), NEG, np.float16)
        for row in range(C):
            if row < pad:
                mask[row, 0] = 0
                continue
            q_pos = start + (row - pad)
            for j in range(rows):
                if held[j] >= 0 and visible(spec, ltype, q_pos, int(held[j])):
                    mask[row, j] = 0
        masks[ltype] = mask[None, None]
    return masks


# ---------------------------------------------------------------------------------------------------------------------
# Gated DeltaNet (linear attention) layers in a prefill chunk (Qwen3.5)
# ---------------------------------------------------------------------------------------------------------------------
def linear_prefill_controls(C: int, valid: int):
    """(valid_mask [1,1,C,1], conv_perm [1,C+3,C+3]) float16 of a chunk of C rows that holds `valid` real rows (the pad rows are in front).
    valid_mask: 1 for a real row, 0 for a pad row (the graph zeroes beta and g there: the pad rows do nothing to the recurrent state).
    conv_perm: a permutation matrix applied to [3 conv-state rows | C chunk rows]: the pad rows move in front of the state rows, so that the causal conv of the first real row sees the
    state rows and not the pad rows (row r of the result = row `source(r)` of the input); the identity when the chunk is full."""
    pad = C - valid
    mask = np.zeros((1, 1, C, 1), np.float16)
    mask[0, 0, pad:, 0] = 1
    perm = np.zeros((1, C + 3, C + 3), np.float16)
    for r in range(C + 3):
        source = 3 + r if r < pad else r - pad if r < pad + 3 else r
        perm[0, r, source] = 1
    return mask, perm


# ---------------------------------------------------------------------------------------------------------------------
# images (Qwen3.5 image tower, docs/models/qwen35-vision.md): preprocessing, the prompt, 3-position interleaved MRoPE
# ---------------------------------------------------------------------------------------------------------------------
def prepare_image(image, spec: ModelSpec, grids, grid=None, detail: str = "auto"):
    """image: a PIL image (RGB) -> (patches float32 [N, patch_vector], grid): the preset whose aspect ratio / size fit the image (`vision.choose_grid` with the `detail` level low | auto | high, or `grid`), the image resized to exactly
    grid * patch pixels (bicubic like the Hugging Face image processor), rescaled by 1 / 255, normalised with the mean / std of model.json and patchified in the order of the Conv3d kernel.
    The browser runtime does the same with a canvas (its resize differs a little from PIL's bicubic: the vision graphs are verified on the SAME pixels)."""
    from PIL import Image

    v = spec.vision
    grid = tuple(grid) if grid else vision.choose_grid(grids, image.width, image.height, v["patch_size"], detail)
    size = (grid[1] * v["patch_size"], grid[0] * v["patch_size"])
    pixels = np.asarray(image.convert("RGB").resize(size, Image.BICUBIC), np.float32).transpose(2, 0, 1)   # [3, H, W]
    mean, std = np.asarray(v["image_mean"], np.float32).reshape(3, 1, 1), np.asarray(v["image_std"], np.float32).reshape(3, 1, 1)
    pixels = (pixels * np.float32(v["rescale_factor"]) - mean) / std
    return vision.patchify(pixels, grid[0], grid[1], v["patch_size"], v["temporal_patch_size"], v["merge_size"]), grid


def image_token_count(spec: ModelSpec, grid) -> int:
    """the image tokens of a grid: one per merge block"""
    return grid[0] * grid[1] // spec.vision["merge_size"] ** 2


def expand_image_tokens(spec: ModelSpec, tokens: list, grids: list) -> list:
    """The prompt as the model sees it: every <|image_pad|> of `tokens` (one per image, in order) becomes image_token_count copies for the grid of its image; a run of pad tokens that already has
    exactly that length stays as it is."""
    pad, out, k, i = spec.vision["image_token_id"], [], 0, 0
    while i < len(tokens):
        if tokens[i] != pad:
            out.append(tokens[i])
            i += 1
            continue
        if k >= len(grids):
            raise ValueError("the prompt has more <|image_pad|> tokens than images")
        run = i
        while run < len(tokens) and tokens[run] == pad:
            run += 1
        want = image_token_count(spec, grids[k])
        if run - i not in (1, want):
            raise ValueError(f"image {k}: {run - i} <|image_pad|> tokens in the prompt, expected 1 or {want}")
        out += [pad] * want
        i, k = run, k + 1
    if k != len(grids):
        raise ValueError(f"{len(grids)} images but {k} <|image_pad|> tokens in the prompt")
    return out


def mrope_positions(spec: ModelSpec, tokens: list, grids: list):
    """(positions int64 [3, n], rope_delta): the (t, h, w) position of every token of the EXPANDED prompt (transformers' get_rope_index): text t = h = w = running position; the image tokens of a
    run of grid (gh, gw) get t = start, h = start + row, w = start + column of their merge block (row-major), and the position continues at start + max(gh, gw) / merge afterwards.
    rope_delta = max position + 1 - n: the decode token at sequence index i has position i + rope_delta (all three components)."""
    v = spec.vision
    pad, merge = v["image_token_id"], v["merge_size"]
    positions, current, i, k = [], 0, 0, 0
    while i < len(tokens):
        if tokens[i] != pad:
            positions.append([current] * 3)
            current += 1
            i += 1
            continue
        h, w = grids[k][0] // merge, grids[k][1] // merge
        if i + h * w > len(tokens) or any(t != pad for t in tokens[i:i + h * w]):
            raise ValueError(f"image {k}: the prompt does not have {h * w} consecutive image tokens")
        positions += [[current, current + r, current + c] for r in range(h) for c in range(w)]
        current += max(h, w)
        i, k = i + h * w, k + 1
    out = np.asarray(positions, np.int64).T.reshape(3, -1)
    return out, int(out.max()) + 1 - len(tokens)


def mrope_axes(section, count: int) -> np.ndarray:
    """which of the 3 positions (0 t, 1 h, 2 w) drives each of the `count` rotary frequencies in the INTERLEAVED layout (transformers' apply_interleaved_mrope): t, h, w, t, h, w ...; h only up to
    3 * section[1], w only up to 3 * section[2], the frequencies after that are t."""
    axes = np.zeros(count, np.int64)
    for axis, offset in ((1, 1), (2, 2)):
        axes[offset:section[axis] * 3:3] = axis
    return axes


def rope_tables_mrope(spec: ModelSpec, positions: np.ndarray, rows: int, pad: int) -> dict:
    """The `rope_tables` of a chunk with 3 positions per token: positions int64 [3, valid] (t, h, w) -> layer type -> (cos, sin) float16 [1,1,rows,head_dim]; the first `pad` rows reuse the first real
    position. The rotating dims of the full attention layers sit at the front of the (permuted) head dim: frequency j of the first sum(section) takes its angle from position axes[j]."""
    out = {}
    for ltype, entry in spec.rope.items():
        inv, factor = rope_lib.inv_freq_for(entry, spec.decoder_dims(ltype).head_dim)
        axes = mrope_axes(spec.mrope["section"], sum(spec.mrope["section"]))
        angles = np.zeros((positions.shape[1], len(inv)), np.float32)
        for j, axis in enumerate(axes):
            angles[:, j] = positions[axis].astype(np.float32) * inv[j]
        emb = np.concatenate([angles, angles], axis=-1)
        cos, sin = np.cos(emb) * factor, np.sin(emb) * factor
        index = np.maximum(0, np.arange(rows) - pad)
        out[ltype] = (cos[index][None, None].astype(np.float16), sin[index][None, None].astype(np.float16))
    return out

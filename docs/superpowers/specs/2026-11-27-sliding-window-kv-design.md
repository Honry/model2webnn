# Sliding-window KV caches (`--sliding-cache window`)

Static graphs give every layer a K / V cache of `max_length - 1` rows. A sliding layer only attends the last `W` (= `sliding_window`) positions, so most of its cache is dead memory
at long max lengths (Gemma 4: 512 of up to 16383 rows, 35 of 42 layers; Aion: 2048, 21 of 28 layers). This is an opt-in compile option; the default stays `full`.

## Idea

RoPE is applied to K before it is cached and softmax does not care about the order of the keys, so the slot a row sits in does not matter, only the mask. A sliding layer keeps a
**ring** of `ring` rows: position `p` lives in slot `p mod ring`. The host knows which position every slot holds (the largest `p < written` with `p mod ring = slot`) and builds the mask
from that. Nothing inside the graphs changes except the length of the cache input of a sliding layer (`past_len = ring` instead of `P = max_length - 1`).

## Ring size

`ring = roundup(max(W - 1 + slack, Cmax, Rmax), 64)`, never above `P` (then the layer has the full cache: identical to `full`, slot = position).

* decode (staged): a query at `pos` needs the main-cache positions in `[pos - W + 1, base - 1]`, at most `W - 1` rows; the rows of the staging buffer are attended from the staging buffer.
  A flush writes `R` rows at slots `(base + i) mod ring` (all different: `R <= ring`); they overwrite positions that no later query can see.
* prefill chunk of `valid <= C` rows: a layer with K / V of its own attends `concat(ring before the chunk, chunk rows)`: the first row needs `W - 1` cached rows. The chunk is written into
  the ring by the one-hot selectors (`C <= ring`: no two rows of a chunk share a slot).
* a layer that SHARES K / V (Gemma 4) attends the producer's cache AFTER the chunk update (mask columns = slots), which must still hold `[q - W + 1, q]` of the first row of the chunk:
  `ring >= W - 1 + valid`. That is the `slack = Cmax` term (only when a sliding layer reads another layer's cache; Aion has no sharing, so `slack = 0`).
* `Cmax` = largest prefill chunk of the package, `Rmax` = largest staging row count of its variants: one ring size per package, whatever chunk / variant the session picks.

## Host side (protocol.py / protocol.ts, `ring` argument, 0 = the old behaviour)

* `slot_positions(rows, written)`: position held by every slot (-1 = empty).
* masks: columns of a sliding layer = `[ring slots | chunk rows]` (prefill), `[ring slots | staging rows]` (decode), `[ring slots]` (shared layer in prefill); a slot is visible to the query at `q`
  when it holds a position `p` with `p <= q` and `q - p < W`. The rows of a chunk / the staging rows are masked as before. Left pad rows behave as before.
* selectors: prefill `chunk row pad + i -> slot (start + i) mod ring`, flush `staging row i -> slot (base + i) mod ring` (`wrap`); both are [rows, .] matrices, so they exist once per DISTINCT
  row count (sliding ring, full cache). Wrap-around inside one write needs nothing special: the slots are distinct.
* RoPE tables, staging selectors, decode selectors: unchanged.

## Graphs and plan

* Graphs differ between a sliding and a full layer only by the cache length, so the shared BLOCK graph splits by layer TYPE: Gemma 4 already has one BLOCK per kind (sliding / full x own / shared),
  only the sliding kinds get `past_len = ring`. The generic path (Aion, tiny hybrid) gets `blocks: {sliding_attention, full_attention}` (instead of `block`), `flush` per type, HEAD built for the
  type of layer 0. Both paths are staged-protocol only in window mode (the other protocols write by absolute row; one-hot / scatter variants are refused).
* Decode block groups of the generic path are cut per sequence of layer types like the Gemma 4 / Qwen3.5 groups (`block_groups[K]` = schedule `[{first_block, blocks, key, graph}]`, key `s-f-f-s-s`);
  Aion: 6 graphs per (variant, max length) instead of 2 (prefill groups are not supported in window mode). Gemma 4's groups are keyed by kind already: no change. Where `ring >= P` the graphs
  of a type are the same file (emitted once).
* `plan.profile.sliding_cache = "window"` and `plan.profile.sliding_ring` (the ring rows) are written only for window packages; their absence means `full`.
* The runtime allocates the cache of a layer with `cacheRows(type, P, ring)`; `memory.ts kvRowsFor` uses the same function.

## Verification

Python: interpreter + reference runners, `window` against `full` and against the independent float reference, prompts of W-1 / W / W+1 / several windows, generations that wrap the ring several times,
chunks 8 / 16 and staging rows 4 / 8 / 16, rings of the minimum size. Golden vectors of the ring protocol for the TS functions. TS: the mock WebNN with window golden packages (shapes of every bound
tensor are validated). Real packages are built as siblings (`*-window`) and validated on the device by the user.

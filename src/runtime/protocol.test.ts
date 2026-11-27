import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { cosSin, invFreqFor } from './rope.js';
import {
  cacheRows, decodeMasks, decodeMasksOnehot, decodeSelectors, decodeSelectorsOnehot, flushSelectors, planPrefill, prefillMasks, prefillSelectors, ringOf, ropeTables, scatterDecodeMasks, scatterPrefillMasks, slotPositions,
  incachePrefillMasks, scatterNDDecodeIndices, scatterNDDecodeMasks, scatterNDPrefillIndices, scatterNDPrefillMasks, scatterPrefillRope, type LayoutSpec,
  chooseGrid, expandImageTokens, mropeAxes, mropePositions, patchify, ropeTablesMrope, type Grid,
} from './protocol.js';

// vectors written by python/tests/test_protocol_golden.py from the Python reference
const g = JSON.parse(readFileSync(new URL('../../tests/golden/protocol_vectors.json', import.meta.url), 'utf8'));
const spec: LayoutSpec = g.spec;
const { P, C, R, PS } = g;
const visibleColumns = (mask: ArrayLike<number>, rows: number) => {
  const columns = (mask.length / rows) | 0;
  return Array.from({ length: rows }, (_, r) => Array.from({ length: columns }, (_, c) => c).filter((c) => mask[r * columns + c] === 0));
};
const onesOf = (array: ArrayLike<number>, columns: number) => Array.from(array as ArrayLike<number>).flatMap((v, i) => (v === 1 ? [[(i / columns) | 0, i % columns]] : []));
const close = (a: ArrayLike<number>, b: number[], tolerance: number) => {
  expect(a.length).toBe(b.length);
  let worst = 0;
  for (let i = 0; i < b.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  expect(worst).toBeLessThan(tolerance);
};

describe('RoPE', () => {
  it('matches the reference for the default and the YaRN entries of Aion', () => {
    for (const [name, v] of Object.entries<any>(g.rope_aion)) {
      const { invFreq, attentionFactor } = invFreqFor(v.entry, v.head_dim);
      expect(attentionFactor).toBeCloseTo(v.attention_factor, 9);
      close(invFreq, v.inv_freq.slice(0, 64), 1e-7);
      const { cos, sin } = cosSin(v.positions, invFreq, attentionFactor);
      close(cos, v.cos, 3e-4);          // float32 angles at position 2047 differ in the 4th digit between libms
      close(sin, v.sin, 3e-4);
      expect(name).toMatch(/attention/);
    }
  });
  it('YaRN scales cos and sin by 0.1 ln(16) + 1', () => {
    expect(g.rope_aion.full_attention.attention_factor).toBeCloseTo(0.1 * Math.log(16) + 1, 9);
  });
});

describe('protocol', () => {
  it('rope rows of a left-padded chunk reuse the first real position', () => {
    const v = g.rope_padded_chunk, tables = ropeTables(spec, v.positions, v.rows, v.pad);
    for (const type of Object.keys(tables)) { close(Array.from(tables[type].cos as ArrayLike<number>), v[type].cos, 2e-3); close(Array.from(tables[type].sin as ArrayLike<number>), v[type].sin, 2e-3); }
  });
  it('prefill masks: full chunk, and a left-padded chunk with a sliding window', () => {
    for (const key of ['prefill_full_chunk', 'prefill_padded_chunk']) {
      const v = g[key], masks = prefillMasks(spec, P, C, v.start, v.valid);
      for (const type of Object.keys(v.masks)) expect(visibleColumns(masks[type], C)).toEqual(v.masks[type]);
    }
  });
  it('prefill selectors', () => {
    const v = g.prefill_padded_selectors, { rowsToKeep, rowSelector } = prefillSelectors(P, C, 16, 5);
    expect(Array.from(rowsToKeep as ArrayLike<number>)).toEqual(v.rows_to_keep);
    expect(onesOf(rowSelector, C)).toEqual(v.row_selector_ones);
  });
  it('decode masks and selectors (staged KV, sliding window crossing the flush boundary)', () => {
    const v = g.decode, masks = decodeMasks(spec, P, R, v.pos, v.base, v.filled);
    for (const type of Object.keys(v.masks)) expect(visibleColumns(masks[type], 1)).toEqual(v.masks[type]);
    const s = decodeSelectors(R, 2);
    expect(Array.from(s.rowsToKeep as ArrayLike<number>)).toEqual(g.decode_selectors.rows_to_keep);
    expect(Array.from(s.rowSelector as ArrayLike<number>)).toEqual(g.decode_selectors.row_selector);
  });
  it('flush selectors, also when the staging rows run past the end of the cache', () => {
    for (const key of ['flush_20', 'flush_22']) {
      const v = g[key], { rowsToKeep, rowSelector } = flushSelectors(P, R, v.base);
      expect(Array.from(rowsToKeep as ArrayLike<number>)).toEqual(v.rows_to_keep);
      expect(onesOf(rowSelector, R)).toEqual(v.row_selector_ones);
    }
  });
  it('onehot KV: decode masks and selectors', () => {
    const v = g.onehot_decode, masks = decodeMasksOnehot(spec, P, v.pos);
    for (const type of Object.keys(v.masks)) expect(visibleColumns(masks[type], 1)).toEqual(v.masks[type]);
    const s = decodeSelectorsOnehot(P, v.pos);
    expect(Array.from(s.rowsToKeep as ArrayLike<number>)).toEqual(g.onehot_decode_selectors.rows_to_keep);
    expect(onesOf(s.rowSelector, 1)).toEqual(g.onehot_decode_selectors.row_selector_ones);
  });
  it('tensorScatter KV: prefill masks and RoPE with the padding in front, decode masks', () => {
    for (const [key, ropeKey] of [['scatter_prefill_pad_chunk', 'scatter_prefill_rope_pad'], ['scatter_prefill_later_chunk', 'scatter_prefill_rope_later']]) {
      const v = g[key], masks = scatterPrefillMasks(spec, PS, C, v.start, v.pad);
      for (const type of Object.keys(v.masks)) expect(visibleColumns(masks[type], C)).toEqual(v.masks[type]);
      const r = g[ropeKey], tables = scatterPrefillRope(spec, C, r.start, r.pad);
      for (const type of Object.keys(tables)) { close(Array.from(tables[type].cos as ArrayLike<number>), r[type].cos, 2e-3); close(Array.from(tables[type].sin as ArrayLike<number>), r[type].sin, 2e-3); }
    }
    const d = g.scatter_decode, masks = scatterDecodeMasks(spec, PS, d.pos, d.pad);
    for (const type of Object.keys(d.masks)) expect(visibleColumns(masks[type], 1)).toEqual(d.masks[type]);
  });
  it('scatterND KV: prefill masks / indices (pad rows parked behind the real rows) and decode masks / indices', () => {
    const p = g.scatternd_prefill, masks = scatterNDPrefillMasks(spec, P, C, p.start, p.valid);
    for (const type of Object.keys(p.masks)) expect(visibleColumns(masks[type], C)).toEqual(p.masks[type]);
    expect(Array.from(scatterNDPrefillIndices(p.kv_heads, P, C, p.start, p.valid))).toEqual(p.indices);
    const d = g.scatternd_decode, decode = scatterNDDecodeMasks(spec, P, d.pos);
    for (const type of Object.keys(d.masks)) expect(visibleColumns(decode[type], 1)).toEqual(d.masks[type]);
    expect(Array.from(scatterNDDecodeIndices(d.kv_heads, d.pos))).toEqual(d.indices);
  });
  it('masks of a layer that reads the updated cache of another layer (KV sharing)', () => {
    const v = g.incache_prefill, masks = incachePrefillMasks(spec, P, C, v.start, v.valid);
    for (const type of Object.keys(v.masks)) expect(visibleColumns(masks[type], C)).toEqual(v.masks[type]);
  });
  it('sliding window caches (ring): slots, masks of prefill chunks / decode steps / shared layers, wrap-around selectors, like the reference', () => {
    const r = g.ring;
    for (const v of r.slots) expect(Array.from(slotPositions(v.rows, v.written))).toEqual(v.held);
    expect([cacheRows('sliding_attention', r.P, r.ring), cacheRows('full_attention', r.P, r.ring), cacheRows('sliding_attention', r.P, 0), cacheRows('sliding_attention', 20, 24)]).toEqual([r.ring, r.P, r.P, 20]);
    expect([ringOf({ profile: { sliding_cache: 'window', sliding_ring: 768 } }), ringOf({ profile: { sliding_cache: 'full', sliding_ring: 768 } }), ringOf({})]).toEqual([768, 0, 0]);
    for (const v of r.prefill) {
      const own = prefillMasks(spec, r.P, C, v.start, v.valid, r.ring), shared = incachePrefillMasks(spec, r.P, C, v.start, v.valid, r.ring);
      for (const type of Object.keys(v.masks)) {
        expect(own[type].length).toBe(C * (cacheRows(type, r.P, r.ring) + C));
        expect(visibleColumns(own[type], C), `prefill ${type} ${v.start}`).toEqual(v.masks[type]);
        expect(visibleColumns(shared[type], C), `incache ${type} ${v.start}`).toEqual(v.incache[type]);
      }
    }
    for (const v of r.decode) {
      const masks = decodeMasks(spec, r.P, r.R, v.pos, v.base, v.filled, r.ring);
      for (const type of Object.keys(v.masks)) expect(visibleColumns(masks[type], 1), `decode ${type} ${v.pos}`).toEqual(v.masks[type]);
    }
    for (const v of r.prefill_selectors) {
      const { rowsToKeep, rowSelector } = prefillSelectors(r.ring, C, v.start, v.valid, true);
      expect(Array.from(rowsToKeep as ArrayLike<number>)).toEqual(v.rows_to_keep);
      expect(onesOf(rowSelector, C)).toEqual(v.row_selector_ones);
    }
    for (const v of r.flush) {
      const { rowsToKeep, rowSelector } = flushSelectors(r.ring, r.R, v.base, true);
      expect(Array.from(rowsToKeep as ArrayLike<number>)).toEqual(v.rows_to_keep);
      expect(onesOf(rowSelector, r.R)).toEqual(v.row_selector_ones);
    }
  });
  it('prefill planning picks the same chunk sequences as the reference', () => {
    for (const v of g.plans) expect(planPrefill(v.sizes, v.tokens, v.fixed)).toEqual(v.plan);
  });
});

describe('images (Qwen3.5 image tower)', () => {
  const v = g.vision;
  it('chooses the grid preset like the reference', () => {
    for (const c of v.choose) expect(chooseGrid(c.grids, c.width, c.height, c.patch, c.detail), JSON.stringify(c)).toEqual(c.grid);
  });
  it('expands the image pad tokens and gives the (t, h, w) positions and rope_delta of get_rope_index', () => {
    const layout = { image_token_id: v.spec.image_token_id, merge_size: v.spec.merge_size }, grids = v.prompt.grids as Grid[];
    const expanded = expandImageTokens(layout, v.prompt.short, grids);
    expect(expanded).toEqual(v.prompt.expanded);
    expect(expandImageTokens(layout, expanded, grids)).toEqual(expanded);   // already expanded
    expect(() => expandImageTokens(layout, v.prompt.short, [grids[0]])).toThrow(/more <\|image_pad\|> tokens than images/);
    expect(() => expandImageTokens(layout, v.prompt.short.filter((t: number) => t !== layout.image_token_id), grids)).toThrow(/images but 0/);
    const { positions, ropeDelta } = mropePositions(layout, expanded, grids);
    expect(positions.map((p) => Array.from(p))).toEqual(v.prompt.positions);
    expect(ropeDelta).toBe(v.prompt.rope_delta);
  });
  it('builds the interleaved MRoPE tables of a left-padded chunk like the reference', () => {
    expect(mropeAxes(v.axes.section, v.axes.count)).toEqual(v.axes.axes);
    const c = v.rope_chunk, layout = { layer_types: ['full_attention'], sliding_window: null, head_dim: v.spec.head_dim, rope: v.spec.rope, mrope: v.spec.mrope } as LayoutSpec & { mrope: { section: number[] } };
    const { positions } = mropePositions({ image_token_id: v.spec.image_token_id, merge_size: v.spec.merge_size }, v.prompt.expanded, v.prompt.grids);
    const tables = ropeTablesMrope(layout, positions.map((p) => p.subarray(c.first, c.first + c.valid)), c.rows, c.pad);
    close(Array.from(tables.full_attention.cos as ArrayLike<number>), c.cos, 2e-3);
    close(Array.from(tables.full_attention.sin as ArrayLike<number>), c.sin, 2e-3);
  });
  it('patchifies like the reference (merge-block order, (channel, temporal, kh, kw) vectors)', () => {
    const p = v.patchify, patches = patchify(Float32Array.from(p.pixels), p.grid, p.patch, p.temporal, p.merge);
    expect(Array.from(patches)).toEqual(p.patches);
    expect(() => patchify(new Float32Array(10), p.grid, p.patch, p.temporal, p.merge)).toThrow(/patchify/);
  });
});

import { checkTensorSize, WEBNN_MAX_TENSOR_BYTES } from './llm-common.js';
describe('WebNN tensor size limit', () => {
  it('rejects a tensor above 2^31 - 1 bytes with a readable error and accepts the rest', () => {
    expect(WEBNN_MAX_TENSOR_BYTES).toBe(2147483647);
    expect(() => checkTensorSize('float32', [262144, 2560], 'dequantized lm_head')).toThrow(/dequantized lm_head float32\[262144,2560\] would be 2\.50 GiB.*2 GiB/);
    expect(() => checkTensorSize('float16', [1, 2, 1151, 512])).not.toThrow();
    expect(() => checkTensorSize('int4', [2 ** 32])).toThrow();          // 2^32 int4 values are 2^31 bytes: one byte over the limit
    expect(() => checkTensorSize('int4', [2 ** 32 - 2])).not.toThrow();
  });
});

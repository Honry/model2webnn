import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { cosSin, invFreqFor } from './rope.js';
import {
  decodeMasks, decodeMasksOnehot, decodeSelectors, decodeSelectorsOnehot, flushSelectors, planPrefill, prefillMasks, prefillSelectors, ropeTables, scatterDecodeMasks, scatterPrefillMasks,
  incachePrefillMasks, scatterNDDecodeIndices, scatterNDDecodeMasks, scatterNDPrefillIndices, scatterNDPrefillMasks, scatterPrefillRope, type LayoutSpec,
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
  it('prefill planning picks the same chunk sequences as the reference', () => {
    for (const v of g.plans) expect(planPrefill(v.sizes, v.tokens, v.fixed)).toEqual(v.plan);
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

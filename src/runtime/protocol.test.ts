import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { cosSin, invFreqFor } from './rope.js';
import { decodeMasks, decodeSelectors, flushSelectors, prefillMasks, prefillSelectors, ropeTables, type LayoutSpec } from './protocol.js';

// vectors written by python/tests/test_protocol_golden.py from the Python reference
const g = JSON.parse(readFileSync(new URL('../../tests/golden/protocol_vectors.json', import.meta.url), 'utf8'));
const spec: LayoutSpec = g.spec;
const { P, C, R } = g;
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
});

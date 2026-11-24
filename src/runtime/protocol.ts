// The host side of the runtime protocol as pure functions (what has to be written into the graph inputs of one dispatch chain).
// Mirrors python/m2w/reference/protocol.py (the executable specification); tests/golden/protocol_vectors.json keeps the two identical. Conventions:
//  * the KV cache has P = maxLength - 1 rows, cache row j holds position j; the staging buffer has R rows;
//  * masks are additive float16 (0 visible, -65504 hidden): one mask and one rope table per LAYER TYPE (sliding vs full attention);
//  * prefill chunk of C rows, the last chunk is LEFT-padded [pad rows | real rows]; pad rows only see themselves, their RoPE position is the first real position;
//    prefill mask columns = [cache rows 0..P-1 | chunk rows 0..C-1];
//  * decode (staged KV): mask columns = [main cache rows 0..P-1 | staging rows 0..R-1]; the new token goes into staging row `filled` = position base + filled.
import { cosSin, invFreqFor, type RopeEntry } from './rope.js';

/* eslint-disable @typescript-eslint/no-explicit-any */
export type Half = any;   // Float16Array (Chrome, Node >= 24); typed loosely because older TypeScript libs do not know it
const Float16: any = (globalThis as any).Float16Array;
export const newHalf = (length: number): Half => {
  if (!Float16) throw new Error('this runtime needs Float16Array (Chrome 135+ / Node 24+)');
  return new Float16(length);
};
export const HIDDEN = -65504;

export interface LayoutSpec {
  layer_types: string[];
  sliding_window: number | null;
  head_dim: number;
  rope: Record<string, RopeEntry>;
}

export const layerTypesOf = (spec: LayoutSpec): string[] => [...new Set(spec.layer_types)].sort();

/** may a query at q_pos attend to the key at k_pos in a layer of this type? */
export function visible(spec: LayoutSpec, layerType: string, qPos: number, kPos: number): boolean {
  if (kPos > qPos) return false;
  return layerType !== 'sliding_attention' || qPos - kPos < (spec.sliding_window as number);
}

/** first position a query at qPos can see in a layer of this type */
const firstVisible = (spec: LayoutSpec, layerType: string, qPos: number) => (layerType === 'sliding_attention' ? Math.max(0, qPos - (spec.sliding_window as number) + 1) : 0);

/** layer type -> { cos, sin } float16 [rows * headDim]; the first `pad` rows reuse the first real position */
export function ropeTables(spec: LayoutSpec, positions: number[], rows: number, pad: number): Record<string, { cos: Half; sin: Half }> {
  const out: Record<string, { cos: Half; sin: Half }> = {};
  for (const type of layerTypesOf(spec)) {
    const { invFreq, attentionFactor } = invFreqFor(spec.rope[type], spec.head_dim);
    const table = cosSin(positions, invFreq, attentionFactor);
    const cos = newHalf(rows * spec.head_dim), sin = newHalf(rows * spec.head_dim);
    for (let row = 0; row < rows; row++) {
      const source = Math.max(0, row - pad) * spec.head_dim;
      cos.set(table.cos.subarray(source, source + spec.head_dim), row * spec.head_dim);
      sin.set(table.sin.subarray(source, source + spec.head_dim), row * spec.head_dim);
    }
    out[type] = { cos, sin };
  }
  return out;
}

/** layer type -> mask float16 [C * (P + C)] of a chunk with `valid` real tokens at positions start .. start+valid-1 */
export function prefillMasks(spec: LayoutSpec, P: number, C: number, start: number, valid: number): Record<string, Half> {
  const pad = C - valid, total = P + C, out: Record<string, Half> = {};
  for (const type of layerTypesOf(spec)) {
    const mask = newHalf(C * total).fill(HIDDEN);
    for (let row = 0; row < C; row++) {
      const base = row * total;
      if (row < pad) { mask[base + P + row] = 0; continue; }   // pad row: sees only itself
      const qPos = start + (row - pad), lowest = firstVisible(spec, type, qPos);
      if (lowest < start) mask.fill(0, base + lowest, base + start);                        // cache rows of earlier chunks (row j holds position j)
      const firstColumn = Math.max(pad, lowest - start + pad);                              // real chunk rows up to the row itself (position start + c - pad)
      if (firstColumn <= row) mask.fill(0, base + P + firstColumn, base + P + row + 1);
    }
    out[type] = mask;
  }
  return out;
}

/** rowsToKeep [P], rowSelector [P * C]: chunk row pad+i writes cache row start+i */
export function prefillSelectors(P: number, C: number, start: number, valid: number): { rowsToKeep: Half; rowSelector: Half } {
  const pad = C - valid, rowsToKeep = newHalf(P).fill(1), rowSelector = newHalf(P * C);
  for (let i = 0; i < valid; i++) { rowsToKeep[start + i] = 0; rowSelector[(start + i) * C + pad + i] = 1; }
  return { rowsToKeep, rowSelector };
}

/** layer type -> mask float16 [P + R] for the token at position pos = base + filled */
export function decodeMasks(spec: LayoutSpec, P: number, R: number, pos: number, base: number, filled: number): Record<string, Half> {
  const out: Record<string, Half> = {};
  for (const type of layerTypesOf(spec)) {
    const mask = newHalf(P + R).fill(HIDDEN), lowest = firstVisible(spec, type, pos);
    if (lowest < base) mask.fill(0, lowest, base);                                          // main cache rows
    const firstStaging = Math.max(0, lowest - base);                                        // staging row i holds position base + i
    if (firstStaging <= filled) mask.fill(0, P + firstStaging, P + filled + 1);
    out[type] = mask;
  }
  return out;
}

/** stagingRowsToKeep [R], stagingRowSelector [R]: the new token goes into staging row `filled` */
export function decodeSelectors(R: number, filled: number): { rowsToKeep: Half; rowSelector: Half } {
  const rowsToKeep = newHalf(R).fill(1), rowSelector = newHalf(R);
  rowsToKeep[filled] = 0; rowSelector[filled] = 1;
  return { rowsToKeep, rowSelector };
}

/** rowsToKeep [P], rowSelector [P * R]: staging row i -> main cache row base+i (rows beyond the cache are dropped) */
export function flushSelectors(P: number, R: number, base: number): { rowsToKeep: Half; rowSelector: Half } {
  const rowsToKeep = newHalf(P).fill(1), rowSelector = newHalf(P * R);
  for (let i = 0; i < R && base + i < P; i++) { rowsToKeep[base + i] = 0; rowSelector[(base + i) * R + i] = 1; }
  return { rowsToKeep, rowSelector };
}

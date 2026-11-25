// The host side of the runtime protocol as pure functions (what has to be written into the graph inputs of one dispatch chain).
// Mirrors python/m2w/reference/protocol.py (the executable specification); tests/golden/protocol_vectors.json keeps the two identical. Conventions:
//  * the KV cache has P = maxLength - 1 rows, cache row j holds position j; the staging buffer has R rows;
//  * masks are additive float16 (0 visible, -65504 hidden): one mask and one rope table per LAYER TYPE (sliding vs full attention);
//  * prefill chunk of C rows, the last chunk is LEFT-padded [pad rows | real rows]; pad rows only see themselves, their RoPE position is the first real position;
//    prefill mask columns = [cache rows 0..P-1 | chunk rows 0..C-1];
//  * decode (staged KV): mask columns = [main cache rows 0..P-1 | staging rows 0..R-1]; the new token goes into staging row `filled` = position base + filled;
//  * decode (onehot KV): no staging buffer, the graph writes the new token into cache row `pos`; mask columns = [cache rows 0..P-1 | the new row];
//  * scatterND KV (prefill and decode): the graph scatters the new rows into the cache at the rows of kv_indices and attends over the updated cache (mask columns = cache rows 0..P-1); the pad rows
//    of the left-padded last chunk are parked behind the real rows (clamped to P-1), where they are masked;
//  * tensorScatter KV (prefill and decode): the graph scatters the new rows into the cache at kv_write_index and attends over the updated cache (mask columns = cache rows 0..P-1);
//    every prefill chunk is full, the padding of the prompt is in FRONT of it (`pad` rows), token i is in cache row pad + i, the decode step of position pos writes row pad + pos.
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
  layer_dims?: Record<string, { head_dim?: number }>;   // layer types whose head dim differs from head_dim (Gemma 4 full attention)
}

/** the head dim of the attention of a layer type */
export const headDimOf = (spec: LayoutSpec, layerType: string): number => spec.layer_dims?.[layerType]?.head_dim ?? spec.head_dim;

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
    const dim = headDimOf(spec, type);
    const { invFreq, attentionFactor } = invFreqFor(spec.rope[type], dim);
    const table = cosSin(positions, invFreq, attentionFactor);
    const cos = newHalf(rows * dim), sin = newHalf(rows * dim);
    for (let row = 0; row < rows; row++) {
      const source = Math.max(0, row - pad) * dim;
      cos.set(table.cos.subarray(source, source + dim), row * dim);
      sin.set(table.sin.subarray(source, source + dim), row * dim);
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

// ---------------------------------------------------------------------------------------------------------------------------------------- prefill planning
/** The chunk sizes for a prompt of `tokenCount` tokens: every chunk but the last is full, the last is left-padded (tensorScatter: the first holds the padding). A dispatch costs
 *  fixedTokens + size units; minimises the sum; ties: fewer chunks, bigger chunks first. */
export function planPrefill(sizes: number[], tokenCount: number, fixedTokens = 200): number[] {
  const ordered = [...sizes].sort((a, b) => b - a), memo = new Map<number, { cost: number; plan: number[] }>();
  const best = (remaining: number): { cost: number; plan: number[] } => {
    const known = memo.get(remaining);
    if (known) return known;
    let result: { cost: number; plan: number[] } | null = null;
    for (const size of ordered) {
      const cost = fixedTokens + size;
      let candidate: { cost: number; plan: number[] };
      if (size >= remaining) candidate = { cost, plan: [size] };
      else { const rest = best(remaining - size); candidate = { cost: cost + rest.cost, plan: [size, ...rest.plan] }; }
      if (!result || candidate.cost < result.cost || (candidate.cost === result.cost && candidate.plan.length < result.plan.length)) result = candidate;
    }
    memo.set(remaining, result as { cost: number; plan: number[] });
    return result as { cost: number; plan: number[] };
  };
  return best(tokenCount).plan;
}

// ---------------------------------------------------------------------------------------------------------------------------------------- onehot KV
/** layer type -> mask float16 [P + 1]: cache rows 0..pos-1 that the layer type sees, and the new token (column P) */
export function decodeMasksOnehot(spec: LayoutSpec, P: number, pos: number): Record<string, Half> {
  const out: Record<string, Half> = {};
  for (const type of layerTypesOf(spec)) {
    const mask = newHalf(P + 1).fill(HIDDEN), lowest = firstVisible(spec, type, pos);
    if (lowest < pos) mask.fill(0, lowest, pos);
    mask[P] = 0;
    out[type] = mask;
  }
  return out;
}

/** rowsToKeep [P], rowSelector [P]: the new token goes to cache row `pos` */
export function decodeSelectorsOnehot(P: number, pos: number): { rowsToKeep: Half; rowSelector: Half } {
  const rowsToKeep = newHalf(P).fill(1), rowSelector = newHalf(P);
  rowsToKeep[pos] = 0; rowSelector[pos] = 1;
  return { rowsToKeep, rowSelector };
}

// ---------------------------------------------------------------------------------------------------------------------------------------- tensorScatter KV
/** RoPE tables of the chunk that covers cache rows start .. start+C-1 (token i is in row pad + i): the pad rows reuse the first real position */
export function scatterPrefillRope(spec: LayoutSpec, C: number, start: number, pad: number): Record<string, { cos: Half; sin: Half }> {
  const firstReal = Math.max(0, pad - start), firstToken = Math.max(0, start - pad);
  return ropeTables(spec, Array.from({ length: C - firstReal }, (_, i) => firstToken + i), C, firstReal);
}

/** layer type -> mask float16 [C * P] of the chunk that covers cache rows start .. start+C-1: columns are cache rows; a pad row sees only itself */
export function scatterPrefillMasks(spec: LayoutSpec, P: number, C: number, start: number, pad: number): Record<string, Half> {
  const out: Record<string, Half> = {};
  for (const type of layerTypesOf(spec)) {
    const mask = newHalf(C * P).fill(HIDDEN);
    for (let row = 0; row < C; row++) {
      const cacheRow = start + row, token = cacheRow - pad, base = row * P;
      if (token < 0) { mask[base + cacheRow] = 0; continue; }
      const first = pad + firstVisible(spec, type, token);                                  // cache row of the first token this layer type sees
      mask.fill(0, base + first, base + cacheRow + 1);
    }
    out[type] = mask;
  }
  return out;
}

/** layer type -> mask float16 [P] for the token at position pos (cache row pad + pos, written by the graph before attention) */
export function scatterDecodeMasks(spec: LayoutSpec, P: number, pos: number, pad: number): Record<string, Half> {
  const out: Record<string, Half> = {};
  for (const type of layerTypesOf(spec)) {
    const mask = newHalf(P).fill(HIDDEN);
    mask.fill(0, pad + firstVisible(spec, type, pos), pad + pos + 1);
    out[type] = mask;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------------------------------- scatterND KV
/** layer type -> mask float16 [C * P] of a chunk with `valid` real tokens at positions start .. start+valid-1: columns are cache rows; a pad row sees only the cache row it is parked in */
export function scatterNDPrefillMasks(spec: LayoutSpec, P: number, C: number, start: number, valid: number): Record<string, Half> {
  const pad = C - valid, out: Record<string, Half> = {};
  for (const type of layerTypesOf(spec)) {
    const mask = newHalf(C * P).fill(HIDDEN);
    for (let row = 0; row < C; row++) {
      const base = row * P;
      if (row < pad) { mask[base + Math.min(P - 1, start + valid + row)] = 0; continue; }
      const qPos = start + (row - pad);
      mask.fill(0, base + firstVisible(spec, type, qPos), base + qPos + 1);
    }
    out[type] = mask;
  }
  return out;
}

/** int32 [kvHeads * C * 3] = (0, head, cache row) of every chunk row: real rows go to start .., pad rows are parked behind them (clamped to P-1) */
export function scatterNDPrefillIndices(kvHeads: number, P: number, C: number, start: number, valid: number): Int32Array {
  const pad = C - valid, out = new Int32Array(kvHeads * C * 3);
  for (let h = 0; h < kvHeads; h++) for (let row = 0; row < C; row++) {
    const at = (h * C + row) * 3;
    out[at] = 0; out[at + 1] = h; out[at + 2] = row < pad ? Math.min(P - 1, start + valid + row) : start + (row - pad);
  }
  return out;
}

/** layer type -> mask float16 [P] for the token at position pos (cache row pos, written by the graph before attention) */
export function scatterNDDecodeMasks(spec: LayoutSpec, P: number, pos: number): Record<string, Half> {
  const out: Record<string, Half> = {};
  for (const type of layerTypesOf(spec)) out[type] = newHalf(P).fill(HIDDEN).fill(0, firstVisible(spec, type, pos), pos + 1);
  return out;
}

/** int32 [kvHeads * 3]: the new token goes to cache row `pos` of every head */
export function scatterNDDecodeIndices(kvHeads: number, pos: number): Int32Array {
  const out = new Int32Array(kvHeads * 3);
  for (let h = 0; h < kvHeads; h++) { out[h * 3] = 0; out[h * 3 + 1] = h; out[h * 3 + 2] = pos; }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------------------------------- attention over another layer's updated cache
/** layer type -> mask float16 [C * P] for a layer that reads the (already updated) cache of another layer in a prefill chunk (Gemma 4 KV sharing): columns are cache rows; a real row sees the rows up
 *  to its position (window of its layer type), a pad row sees row 0 only (any visible column keeps the softmax finite; its output is never used) */
export function incachePrefillMasks(spec: LayoutSpec, P: number, C: number, start: number, valid: number): Record<string, Half> {
  const pad = C - valid, out: Record<string, Half> = {};
  for (const type of layerTypesOf(spec)) {
    const mask = newHalf(C * P).fill(HIDDEN);
    for (let row = 0; row < C; row++) {
      const base = row * P;
      if (row < pad) { mask[base] = 0; continue; }
      const qPos = start + (row - pad);
      mask.fill(0, base + firstVisible(spec, type, qPos), base + qPos + 1);
    }
    out[type] = mask;
  }
  return out;
}

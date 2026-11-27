import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { deviceWeightBytes, expectedMemory, FIXED_OVERHEAD_BYTES, formatBytes, kvBytes, kvRowsFor, looksLikeMemoryError, nextSmallerLength } from './memory.js';
import type { ModelSpecJson, Plan, WeightsManifest } from './package.js';

const MB = 2 ** 20, GB = 2 ** 30;
const model = (extra: Partial<ModelSpecJson> = {}): ModelSpecJson => ({ architecture: 'X', hidden: 64, num_layers: 4, num_heads: 4, num_kv_heads: 2, head_dim: 128, ffn: 128, vocab: 10, vocab_padded: 10, eps: 1e-6, qk_norm: true, tie_embeddings: true,
  max_position_embeddings: 1000, layer_types: ['full_attention', 'full_attention', 'full_attention', 'full_attention'], sliding_window: null, rope: {}, eos_token_ids: [1], bos_token_id: null, group_size: 128, ...extra });
const plan = (): Plan => ({ max_lengths: [1024, 2048, 4096, 8192, 16384] } as unknown as Plan);
const manifest = (...files: [string, number][]): WeightsManifest => ({ quant: { group_size: 128, merge_projections: true }, files: files.map(([role, bytes]) => ({ file: role + bytes, role, bytes, tensors: [] }) as never) });

describe('KV bytes', () => {
  it('K + V, f16, double buffered, for the layers that own a cache (rows = max length - 1)', () => {
    expect(kvRowsFor('full_attention', 1024, plan())).toBe(1023);
    expect(kvBytes(model(), plan(), 1024)).toBe(4 * (2 * 2 * 2 * 1023 * 128 * 2));
  });
  it('skips layers that share the cache of an earlier layer and Gated DeltaNet layers, and uses the dims of a layer type', () => {
    const hybrid = model({ layer_types: ['linear_attention', 'sliding_attention', 'full_attention', 'sliding_attention'], kv_producer: [-1, -1, -1, 1], layer_dims: { full_attention: { head_dim: 256, num_heads: 4, num_kv_heads: 1 } } });
    const perLayer = (heads: number, dim: number) => 2 * 2 * heads * 1023 * dim * 2;
    expect(kvBytes(hybrid, plan(), 1024)).toBe(perLayer(2, 128) /* layer 1 */ + perLayer(1, 256) /* layer 2 */);
  });
});

describe('device weight bytes', () => {
  it('leaves out the host-side per-layer table, and the image tower unless it is loaded', () => {
    const files = manifest(['layer', 100], ['embedding', 50], ['ple_table', 1000], ['vision_block', 30], ['vision_head', 5]);
    expect(deviceWeightBytes(files)).toBe(150);
    expect(deviceWeightBytes(files, { vision: true })).toBe(185);
  });
});

describe('expectedMemory', () => {
  const big = model({ num_layers: 36, layer_types: Array(36).fill('full_attention'), num_kv_heads: 8 });   // 1024 -> 288 MB, 16384 -> 4.5 GB
  const weights = manifest(['layer', 2 * GB]);
  it('weights + the KV cache of the length + a fixed overhead', () => {
    const e = expectedMemory({ model: big, plan: plan(), manifest: weights, maxLength: 1024 });
    expect(Math.round(e.kvBytes / MB)).toBe(288);
    expect(e.weightBytes).toBe(2 * GB);
    expect(e.overheadBytes).toBe(FIXED_OVERHEAD_BYTES);
    expect(e.totalBytes).toBe(2 * GB + e.kvBytes + FIXED_OVERHEAD_BYTES);
  });
  it('grows with the max length and counts the image tower only when it is loaded', () => {
    const totals = [1024, 4096, 16384].map((maxLength) => expectedMemory({ model: big, plan: plan(), manifest: weights, maxLength }).totalBytes);
    expect(totals[0]).toBeLessThan(totals[1]); expect(totals[1]).toBeLessThan(totals[2]);
    const withTower = manifest(['layer', 2 * GB], ['vision_block', GB]);
    expect(expectedMemory({ model: big, plan: plan(), manifest: withTower, maxLength: 1024, vision: true }).totalBytes).toBeGreaterThan(expectedMemory({ model: big, plan: plan(), manifest: withTower, maxLength: 1024 }).totalBytes);
  });
  it('formats GB with one decimal below 10, rounded above, MB below 1 GB', () => {
    expect(formatBytes(5.14 * GB)).toBe('5.1 GB'); expect(formatBytes(12.4 * GB)).toBe('12 GB'); expect(formatBytes(640 * MB)).toBe('640 MB');
  });
});

describe('load failures', () => {
  it('recognises memory / size errors and suggests the next smaller length', () => {
    expect(looksLikeMemoryError(new Error('Failed to execute createTensor: out of memory'))).toBe(true);
    expect(looksLikeMemoryError('tensor size 4294967296 exceeds the limit')).toBe(true);
    expect(looksLikeMemoryError(new Error('failed to fetch weights/layer_017.safetensors: 404'))).toBe(false);
    expect(nextSmallerLength([1024, 2048, 4096], 4096)).toBe(2048);
    expect(nextSmallerLength([1024, 2048, 4096], 1024)).toBeNull();
  });
});

describe('the committed tiny packages', () => {
  it('give numbers that are finite and grow with the max length', () => {
    const root = new URL('../../tests/golden/tiny-package/', import.meta.url);
    const m = JSON.parse(readFileSync(new URL('model.json', root), 'utf8')), p = JSON.parse(readFileSync(new URL('plan.json', root), 'utf8')), w = JSON.parse(readFileSync(new URL('weights/manifest.json', root), 'utf8'));
    expect(deviceWeightBytes(w)).toBeGreaterThan(0);
    expect(kvBytes(m, p, 128)).toBeGreaterThan(kvBytes(m, p, 64));
  });
});

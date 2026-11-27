// A rough memory figure per max length: what the weights and the KV cache of a package are expected to need on the device. Browsers hide the real RAM (navigator.deviceMemory stops at 8 GB), so the page
// shows this number as a hint on every max length and leaves the choice to the user. It is an ESTIMATE, not a measurement.
import type { ModelSpecJson, Plan, WeightsManifest } from './package.js';
import { cacheRows, ringOf } from './protocol.js';

/** A fixed overhead added to weights + KV cache: the browser, the graph compiler and the activations. Calibrated on ONE NPU measurement (a 4B text model at the largest max length); refit when more points exist. */
export const FIXED_OVERHEAD_BYTES = 2 * 2 ** 30;

/** The files that are read on the host and never become device tensors (Gemma 4's per-layer embedding table is read one row per token). */
const HOST_ROLES = new Set(['ple_table']);

/** Bytes of the weights that live on the device: every manifest file except the host-side tables; the image tower only when it is loaded. */
export function deviceWeightBytes(manifest: WeightsManifest, { vision = false }: { vision?: boolean } = {}): number {
  let bytes = 0;
  for (const file of manifest.files) {
    if (HOST_ROLES.has(file.role)) continue;
    if (file.role.startsWith('vision_') && !vision) continue;
    bytes += file.bytes;
  }
  return bytes;
}

/**
 * Rows of the KV cache of a layer of this type at this max length. ONE place for it: a cache that is smaller than the whole context for some layers (a sliding window) changes only this function.
 * Every layer holds all but one row of the max length (the row of the token being decoded sits in the staging buffer / is computed in the graph), except the sliding-attention layers of a package
 * built with --sliding-cache window: they hold a ring of plan.profile.sliding_ring rows (never more than the others; protocol.ts cacheRows is the function the runtime allocates with).
 */
export function kvRowsFor(layerType: string, maxLength: number, plan?: Plan): number {
  return cacheRows(layerType, maxLength - 1, plan ? ringOf(plan) : 0);
}

/** Bytes of the KV cache at this max length: K + V, f16, double buffered, of the layers that own a cache (not the layers that share the cache of an earlier one, not Gated DeltaNet layers, which carry a small state). */
export function kvBytes(model: ModelSpecJson, plan: Plan | undefined, maxLength: number): number {
  let bytes = 0;
  for (let i = 0; i < model.num_layers; i++) {
    if (model.kv_producer && model.kv_producer[i] >= 0) continue;
    const type = model.layer_types[i];
    if (type === 'linear_attention') continue;
    const dims = model.layer_dims?.[type];
    bytes += 2 /* K + V */ * 2 /* double buffer */ * (dims?.num_kv_heads ?? model.num_kv_heads) * kvRowsFor(type, maxLength, plan) * (dims?.head_dim ?? model.head_dim) * 2 /* f16 */;
  }
  return bytes;
}

export interface MemoryNeed { maxLength: number; weightBytes: number; kvBytes: number; overheadBytes: number; /** weights + KV cache + FIXED_OVERHEAD_BYTES */ totalBytes: number }
/** What a max length is expected to need: the device weights + the KV cache of that length + a fixed overhead. `vision`: the image tower is loaded. */
export function expectedMemory(input: { model: ModelSpecJson; plan: Plan; manifest: WeightsManifest; maxLength: number; vision?: boolean }): MemoryNeed {
  const weightBytes = deviceWeightBytes(input.manifest, { vision: input.vision }), kv = kvBytes(input.model, input.plan, input.maxLength);
  return { maxLength: input.maxLength, weightBytes, kvBytes: kv, overheadBytes: FIXED_OVERHEAD_BYTES, totalBytes: weightBytes + kv + FIXED_OVERHEAD_BYTES };
}

/** "5.1 GB" below 10 GB, "12 GB" above; "640 MB" below 1 GB. */
export function formatBytes(bytes: number): string {
  const gb = bytes / 2 ** 30;
  if (gb >= 10) return `${Math.round(gb)} GB`;
  if (gb >= 1) return `${gb.toFixed(1)} GB`;
  return `${Math.round(bytes / 2 ** 20)} MB`;
}

/** Does this error look like a memory / size problem of a load (an allocation, a tensor that is too big, out of memory, a graph that cannot be compiled)? */
export const looksLikeMemoryError = (error: unknown): boolean =>
  /out of memory|not enough memory|memory|alloc|exceeds?|too (big|large)|tensor size|maximum|buffer size|device (was )?lost|compile|compilation|failed to build|build failed|RangeError/i.test(String((error as { message?: string })?.message ?? error));

/** The next smaller max length of the list than `current`, or null. */
export const nextSmallerLength = (lengths: number[], current: number): number | null => {
  const smaller = lengths.filter((l) => l < current).sort((a, b) => b - a);
  return smaller.length ? smaller[0] : null;
};

import { readFileSync } from 'node:fs';
import { beforeAll, describe, expect, it } from 'vitest';
import { Gemma4LLM } from './llm-gemma4.js';
import { LLM, type LoadOptions } from './llm.js';
import { kvBytes, kvRowsFor } from './memory.js';
import type { PackageSource } from './package.js';
import { flushSelectors, prefillSelectors } from './protocol.js';
import { installMockBuilder, MockContext } from './mock-ml.js';

// The committed tiny packages with sliding window caches (tests/golden/tiny-window-package and tiny-gemma4-window-package, scripts/make-tiny-package.py): window 16, ring of 64 rows, max lengths 64
// (the ring is not smaller than the cache) and 128 (cache rows 127: the sliding layers hold 64), staged protocol with 8 staging rows, chunks 8 and 16. The mock validates every binding (names, dtypes,
// shapes), so a layer bound to a cache / mask / selector of the wrong size fails the dispatch.
const sourceOf = (name: string, id: string): PackageSource => {
  const root = new URL(`../../tests/golden/${name}/`, import.meta.url);
  return {
    id,
    text: async (path) => readFileSync(new URL(path, root), 'utf8'),
    bytes: async (path) => { const b = readFileSync(new URL(path, root)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); },
    slice: async (path, start, end) => { const b = readFileSync(new URL(path, root)); return b.buffer.slice(b.byteOffset + start, b.byteOffset + end); },
  };
};
const prompt: number[] = JSON.parse(readFileSync(new URL('../../tests/golden/tiny-window-package/verification.json', import.meta.url), 'utf8')).prompt_ids;   // 45 tokens
const RING = 64, R = 8;

async function generic(name: string, options: LoadOptions = {}) {
  installMockBuilder();
  const context = new MockContext();
  const llm = await LLM.load(sourceOf(name, name), { context, cache: false, blocksPerGraph: 1, chunkSize: 16, ...options });
  context.onRead = (_t, destination) => { destination.fill(0); destination[7] = 5; };
  return { llm, context, x: llm as any };
}
async function gemma(options: LoadOptions = {}) {
  installMockBuilder();
  const context = new MockContext();
  const llm = await Gemma4LLM.load(sourceOf('tiny-gemma4-window-package', 'tiny-gemma4-window'), { context, cache: false, blocksPerGraph: 1, chunkSize: 16, ...options });
  context.onRead = (_t, destination) => { destination.fill(0); destination[7] = 5; };
  return { llm, context, x: llm as any };
}
async function run(llm: { generate: LLM['generate'] }, tokens: number[], maxNewTokens: number) {
  const out: number[] = [];
  for await (const token of llm.generate(tokens, { maxNewTokens, stopTokens: [], temperature: 0 })) out.push(token);
  return out;
}
const halves = (data: ArrayLike<number>) => Array.from(data as ArrayLike<number>);

describe('sliding window caches: Qwen3-like package (layers sliding, full, sliding)', () => {
  beforeAll(() => { if (!(globalThis as any).Float16Array) throw new Error('needs Node 24+'); });

  it('allocates a ring for the sliding layers and the whole cache for the full layer; both caches of a type have the same rows', async () => {
    const { llm, x } = await generic('tiny-window-package', { maxLength: 128 });
    expect(llm.pkg.plan.profile).toMatchObject({ sliding_cache: 'window', sliding_ring: RING });
    expect(x.kv.map((c: any) => c.key.shape[2])).toEqual([RING, 127, RING]);
    expect(x.kvSpare.map((c: any) => c.value.shape[2])).toEqual([RING, 127, RING]);
    expect(x.stg.map((c: any) => c.key.shape[2])).toEqual([R, R, R]);
    expect(x.dec.mask.sliding_attention.shape).toEqual([1, 1, 1, RING + R]);   // mask columns: [ring slots | staging rows]
    expect(x.dec.mask.full_attention.shape).toEqual([1, 1, 1, 127 + R]);
    expect(x.prefillSets.get(16).act.mask.sliding_attention.shape).toEqual([1, 1, 16, RING + 16]);
    expect([...x.prefillSets.get(16).sel.keys()].sort((a, b) => a - b)).toEqual([RING, 127]);   // one selector pair per distinct cache size
    expect([...x.flushSel.keys()].sort((a, b) => a - b)).toEqual([RING, 127]);
    expect([...x.decodeSet.flushes.keys()].sort()).toEqual(['full_attention', 'sliding_attention']);
    expect([...x.decodeSet.blocks.keys()].sort()).toEqual(['full_attention', 'sliding_attention']);
    expect(x.decodeSet.blocks.get('sliding_attention')).not.toBe(x.decodeSet.blocks.get('full_attention'));
  });

  it('a ring that is not smaller than the cache is the full cache (max length 64): one graph for both types', async () => {
    const { x } = await generic('tiny-window-package', { maxLength: 64 });
    expect(x.kv.map((c: any) => c.key.shape[2])).toEqual([63, 63, 63]);
    expect(x.decodeSet.blocks.get('sliding_attention')).toBe(x.decodeSet.blocks.get('full_attention'));
    expect(x.decodeSet.flushes.get('sliding_attention')).toBe(x.decodeSet.flushes.get('full_attention'));
    expect([...x.flushSel.keys()]).toEqual([63]);
  });

  it('runs the whole protocol through several ring wraps with valid bindings, like the full cache package (same dispatch counts)', async () => {
    const { llm, context, x } = await generic('tiny-window-package', { maxLength: 128 });
    const generated = await run(llm, prompt, 64);   // positions 45 .. 108: the ring (64 rows) of the sliding layers wraps
    expect(generated).toEqual(Array(64).fill(7));
    // the package with full caches of the same shape runs the same dispatch chain
    installMockBuilder();
    const reference = new MockContext();
    const full = await LLM.load(sourceOf('tiny-package', 'tiny'), { context: reference, cache: false, blocksPerGraph: 1, chunkSize: 16, maxLength: 128 });
    reference.onRead = (_t, destination) => { destination.fill(0); destination[7] = 5; };
    await run(full, prompt, 64);
    expect(context.dispatches.length).toBe(reference.dispatches.length);
    expect(llm.stats.flushes).toBe(full.stats.flushes);
    expect(llm.stats.flushes).toBe(7);
    // every layer's flush dispatch got the selectors of ITS cache: the ring selectors (wrapping slots) for the sliding layers, the plain ones for the full layer
    const flushGraph = (type: string) => x.decodeSet.flushes.get(type).graph;
    const flushes = context.dispatches.filter((d) => d.graph === flushGraph('sliding_attention') || d.graph === flushGraph('full_attention'));
    expect(flushes.length).toBe(7 * 3);
    const last = flushes.slice(-3), base = 45 + 8 * 6;   // the seventh flush moves the staging rows of positions 93 .. 100
    expect(last.map((d) => d.inputs.rows_to_keep.shape[2])).toEqual([RING, 127, RING]);
    const written = (tensor: any) => context.writes.filter((w) => w.tensor === tensor).pop()?.data as unknown as ArrayLike<number>;
    const ring = flushSelectors(RING, R, base, true), plain = flushSelectors(127, R, base);
    expect(halves(written(last[0].inputs.rows_to_keep))).toEqual(halves(ring.rowsToKeep));
    expect(halves(written(last[0].inputs.row_selector))).toEqual(halves(ring.rowSelector));
    expect(halves(written(last[1].inputs.rows_to_keep))).toEqual(halves(plain.rowsToKeep));
    expect([...halves(ring.rowsToKeep).keys()].filter((i) => halves(ring.rowsToKeep)[i] === 0)).toEqual([29, 30, 31, 32, 33, 34, 35, 36]);
    // the prefill selectors of the first chunk of a prompt are per cache size too
    const firstPrefill = context.dispatches[0];
    expect(firstPrefill.inputs.rows_to_keep.shape).toEqual([1, 1, RING, 1]);   // HEAD = the attention half of layer 0 (sliding)
    const lastChunk = context.writes.filter((w) => w.tensor === x.prefillSets.get(16).sel.get(RING).keep).pop()?.data as unknown as ArrayLike<number>;
    expect(halves(lastChunk)).toEqual(halves(prefillSelectors(RING, 16, 32, 13, true).rowsToKeep));   // the third chunk: 13 real rows at positions 32 .. 44, left-padded
  });

  it('decode block groups: the groups are cut per sequence of layer types and bound with the caches of their layers', async () => {
    const { llm, context, x } = await generic('tiny-window-package', { maxLength: 128, blocksPerGraph: 2, prefillBlocksPerGraph: 1 });
    expect(x.decodeSet.schedule.map((e: any) => [e.first, e.blocks, e.built.name])).toEqual([[0, 2, 'decode_group_f-s_L128']]);
    const generated = await run(llm, prompt, 30);
    expect(generated).toEqual(Array(30).fill(7));
    const prefillDispatches = 2 * 3 + (3 + 2), decodeSteps = 29;   // chunks of 16: two without tail, the last one with; every step: HEAD, one group, TAIL, lm_head
    expect(context.dispatches.length).toBe(prefillDispatches + decodeSteps * 4 + llm.stats.flushes * 3);
    const group = context.dispatches.find((d) => d.graph === x.decodeSet.schedule[0].built.graph) as any;
    expect(group.inputs.b0_past_key.shape[2]).toBe(127);   // the first block of the group runs the attention half of layer 1 (full)
    expect(group.inputs.b1_past_key.shape[2]).toBe(RING);  // the second one that of layer 2 (sliding)
    expect(group.inputs.b0_attention_mask.shape[3]).toBe(127 + R);
    expect(group.inputs.b1_attention_mask.shape[3]).toBe(RING + R);
  });

  it('the memory estimate and the allocated caches agree: kvRowsFor / kvBytes of the plan', async () => {
    const { llm, x } = await generic('tiny-window-package', { maxLength: 128 });
    const plan = llm.pkg.plan, model = llm.pkg.model;
    expect(['sliding_attention', 'full_attention'].map((t) => kvRowsFor(t, 128, plan))).toEqual([RING, 127]);
    expect(kvRowsFor('sliding_attention', 64, plan)).toBe(63);
    expect(kvRowsFor('sliding_attention', 128, { ...plan, profile: { ...plan.profile, sliding_cache: 'full' } })).toBe(127);
    expect(kvRowsFor('sliding_attention', 128)).toBe(127);
    const allocated = x.kv.reduce((n: number, c: any) => n + c.key.shape.reduce((a: number, b: number) => a * b, 1) + c.value.shape.reduce((a: number, b: number) => a * b, 1), 0) * 2 * 2;   // f16, double buffered
    expect(kvBytes(model, plan, 128)).toBe(allocated);
    expect(kvBytes(model, plan, 128)).toBeLessThan(kvBytes(model, { ...plan, profile: { ...plan.profile, sliding_cache: 'full' } }, 128));
  });

  it('refuses a variant that is not staged (none exists in a window package)', async () => {
    await expect(generic('tiny-window-package', { kvStyle: 'onehot' })).rejects.toThrow(/no graphs for kv style onehot|staged/);
  });
});

describe('sliding window caches: Gemma 4 package (KV sharing, different head dims)', () => {
  it('allocates rings for the sliding layers that own a cache; the shared layers read their producers', async () => {
    const { llm, x } = await gemma({ maxLength: 128 });
    expect(llm.pkg.plan.profile).toMatchObject({ sliding_cache: 'window', sliding_ring: RING });
    expect([...x.kv.keys()]).toEqual([0, 1, 2, 3]);
    expect([0, 1, 2, 3].map((l) => x.kv.get(l).key.shape)).toEqual([[1, 2, RING, 32], [1, 2, RING, 32], [1, 2, 127, 64], [1, 2, RING, 32]]);
    expect(x.dec.mask.sliding_attention.shape).toEqual([1, 1, 1, RING + R]);
    expect(x.dec.maskShared.sliding_attention).toBe(x.dec.mask.sliding_attention);   // decode: one mask for own and shared layers
    const act = x.prefillSets.get(16).act;
    expect(act.mask.sliding_attention.shape).toEqual([1, 1, 16, RING + 16]);       // own: [ring | chunk rows]
    expect(act.maskShared.sliding_attention.shape).toEqual([1, 1, 16, RING]);      // shared: the producer's ring after the update of the chunk
    expect(act.maskShared.full_attention.shape).toEqual([1, 1, 16, 127]);
    expect([...x.flushSel.keys()].sort((a: number, b: number) => a - b)).toEqual([RING, 127]);
  });

  it('runs the protocol through several ring wraps with valid bindings (single blocks and block groups of 3)', async () => {
    for (const blocksPerGraph of [1, 3]) {
      const { llm, context, x } = await gemma({ maxLength: 128, blocksPerGraph });
      const generated = await run(llm, prompt, 50);   // positions 45 .. 93 wrap the 64 row ring of the sliding layers
      expect(generated).toEqual(Array(50).fill(7));
      expect(llm.stats.flushes).toBe(6);
      const ownLayers = 4, flushDispatches = context.dispatches.filter((d) => [...x.decodeSet.flush.values()].some((g: any) => g.graph === d.graph));
      expect(flushDispatches.length).toBe(6 * ownLayers);
      expect(flushDispatches.slice(-4).map((d) => d.inputs.rows_to_keep.shape[2])).toEqual([RING, RING, 127, RING]);
      // the shared sliding layer (5) reads the ring of its producer (3): in the first decode step its block is bound with that cache
      if (blocksPerGraph === 1) {
        const prefillDispatches = 2 * 6 + (6 + 2), step = context.dispatches.slice(prefillDispatches, prefillDispatches + 8);
        expect(step[5].inputs.past_key).toBe(step[3].inputs.past_key);
        expect(step[5].inputs.past_key.shape[2]).toBe(RING);
        expect(step[4].inputs.past_key.shape[2]).toBe(127);
      }
    }
  });
});

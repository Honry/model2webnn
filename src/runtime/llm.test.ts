import { readFileSync } from 'node:fs';
import { beforeAll, describe, expect, it } from 'vitest';
import { LLM, type LoadOptions } from './llm.js';
import type { PackageSource } from './package.js';
import { planPrefill } from './protocol.js';
import { installMockBuilder, MockContext } from './mock-ml.js';

// The committed tiny hybrid-layer package (tests/golden/tiny-package, made by scripts/make-tiny-package.py): 3 layers [sliding, full, sliding], max length 128, chunks 8 and 16,
// variants: staged (8 staging rows) / onehot / tensorScatter, each with float32 and float16 compute.
const root = new URL('../../tests/golden/tiny-package/', import.meta.url);
const fsSource: PackageSource = {
  id: 'tiny',
  text: async (path) => readFileSync(new URL(path, root), 'utf8'),
  bytes: async (path) => { const b = readFileSync(new URL(path, root)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); },
};
const verification = JSON.parse(readFileSync(new URL('verification.json', root), 'utf8'));
const prompt: number[] = verification.prompt_ids;   // 45 tokens

async function loaded(options: LoadOptions = {}) {
  installMockBuilder();
  const context = new MockContext();
  const llm = await LLM.load(fsSource, { context, cache: false, chunkSize: 16, ...options });
  context.onRead = (_t, destination) => { destination.fill(0); destination[7] = 5; };   // argmax = token 7
  return { llm, context };
}
async function run(llm: LLM, tokens: number[], maxNewTokens: number) {
  const out: number[] = [];
  for await (const token of llm.generate(tokens, { maxNewTokens, stopTokens: [] })) out.push(token);
  return out;
}
const names = (graph: any) => graph.graph.inputs;
const N = 3;   // layers: a full pass = head + (N-1) blocks + tail + lm_head = N + 2 dispatches; a prefill chunk that is not the last one = N dispatches

describe('LLM scheduler on a validating mock of WebNN', () => {
  beforeAll(() => { if (!(globalThis as any).Float16Array) throw new Error('needs Node 24+'); });

  it('loads: graphs for decode / prefill, weights as writable tensors, state tensors', async () => {
    const { llm, context } = await loaded();
    expect(llm.maxLength).toBe(128);
    expect(llm.chunkSize).toBe(16);
    expect(llm.variant.id).toBe('staged-r8-float32');
    const spec = llm.pkg.model;
    expect(spec.layer_types).toEqual(['sliding_attention', 'full_attention', 'sliding_attention']);
    // every weight shard tensor was uploaded: 3 layers x 12 tensors (merged projections, qk-norm), the final norm, nothing left unwritten
    const writtenIds = new Set(context.writes.map((w) => w.tensor.id));
    const weightTensors = context.tensors.filter((t) => t.writable && ['int4', 'uint8'].includes(t.dataType) || (t.dataType === 'float16' && t.shape.length === 3 && t.shape[2] === 1 && t.writable));
    expect(weightTensors.length).toBeGreaterThan(0);
    for (const t of weightTensors) expect(writtenIds.has(t.id)).toBe(true);
    // the lm_head graph keeps the embedding as constants; the decode / prefill graphs take no embedding
    expect(Object.keys(names((llm as any).lmHead))).toEqual(['final_hidden_states']);
  });

  it('runs the whole protocol: chunked prefill, staged decode, flushes; every dispatch has valid bindings', async () => {
    const { llm, context } = await loaded();
    const generated = await run(llm, prompt, 20);                                       // 45 tokens = 3 chunks of 16 (the last one is left-padded)
    expect(generated).toEqual(Array(20).fill(7));
    const steps = 19, flushes = 2;   // staging has 8 rows: it is full before step 9 and before step 17
    const prefillDispatches = 2 * N + (N + 2);   // two chunks without tail / lm_head, the last one with
    const decodeDispatches = steps * (N + 2);
    expect(context.dispatches.length).toBe(prefillDispatches + decodeDispatches + flushes * N);
    const flushGraph = (llm as any).decodeSet.flush.graph;
    expect(context.dispatches.filter((d) => d.graph === flushGraph).length).toBe(flushes * N);
    expect(llm.stats.promptTokens).toBe(45);
    expect(llm.stats.decodeSteps).toBe(19);
    expect(llm.stats.prefillPlan).toEqual([16, 16, 16]);
  });

  it('binds the rope / mask tensors of the layer type of the layer whose attention runs (HEAD: layer 0, BLOCK i: layer i+1)', async () => {
    const { llm, context } = await loaded();
    await run(llm, prompt.slice(0, 5), 1);   // one prefill chunk, no decode
    const internals = llm as any, types: string[] = llm.pkg.model.layer_types, pre = internals.prefillSets.get(16).act;
    const dispatches = context.dispatches;
    const head = dispatches[0], blocks = dispatches.slice(1, 3);
    expect(head.inputs.attention_mask).toBe(pre.mask[types[0]]);
    expect(head.inputs.rope_cos).toBe(pre.ropeCos[types[0]]);
    blocks.forEach((d, i) => {
      expect(d.inputs.attention_mask).toBe(pre.mask[types[i + 1]]);
      expect(d.inputs.rope_sin).toBe(pre.ropeSin[types[i + 1]]);
    });
    expect(pre.mask.sliding_attention).not.toBe(pre.mask.full_attention);
    expect(pre.ropeCos.sliding_attention).not.toBe(pre.ropeCos.full_attention);
  });

  it('double-buffers the KV cache: the updated rows go to the spare tensors, which become the cache', async () => {
    const { llm } = await loaded();
    const before = (llm as any).kv.map((c: any) => c.key.id);
    await run(llm, prompt.slice(0, 5), 1);
    const after = (llm as any).kv.map((c: any) => c.key.id), spare = (llm as any).kvSpare.map((c: any) => c.key.id);
    expect(after).not.toEqual(before);   // one chunk = one update per layer: every cache was swapped with its spare ...
    expect(spare).toEqual(before);       // ... so the spare is now exactly the buffer that was the cache before
  });

  it('rejects prompts that do not fit and reports missing graph variants', async () => {
    const { llm } = await loaded();
    await expect(async () => { for await (const _ of llm.generate(Array(127).fill(5))) void _; }).rejects.toThrow(/KV cache holds 127/);
    installMockBuilder();
    const load = (o: LoadOptions) => LLM.load(fsSource, { context: new MockContext(), cache: false, ...o });
    await expect(load({ maxLength: 4096 })).rejects.toThrow(/no graphs for max length 4096/);
    await expect(load({ blocksPerGraph: 2 })).rejects.toThrow(/one layer type/);
    await expect(load({ variant: 'nope' })).rejects.toThrow(/no variant 'nope'/);
    await expect(load({ chunkSize: 32 })).rejects.toThrow(/no prefill graphs for chunk 32/);
    await expect(load({ stagingRows: 5 })).rejects.toThrow(/no graphs for kv style staged, 5 staging rows/);
  });
});

describe('every variant of the package runs through the protocol with valid bindings', () => {
  const variants = ['staged-r8-float32', 'staged-r8-float16', 'onehot-float32', 'onehot-float16', 'tensorScatter-float32', 'tensorScatter-float16'];

  it.each(variants)('%s: prefill + decode dispatch counts, flush graph only for staged, write index only for tensorScatter', async (variant) => {
    const { llm, context } = await loaded({ variant });
    expect(llm.variant.id).toBe(variant);
    const generated = await run(llm, prompt, 12);   // 45 tokens, 3 chunks of 16, 11 decode steps (one flush for staged: 8 staging rows)
    expect(generated.length).toBe(12);
    const staged = variant.startsWith('staged'), steps = 11, flushes = staged ? 1 : 0;
    const prefillDispatches = 2 * N + (N + 2), decodeDispatches = steps * (N + 2);
    expect(context.dispatches.length).toBe(prefillDispatches + decodeDispatches + flushes * N);
    const hasInput = (name: string) => context.dispatches.some((d) => name in d.inputs);
    expect(hasInput('kv_write_index')).toBe(variant.startsWith('tensorScatter'));
    expect(hasInput('staging_rows_to_keep')).toBe(staged);
    expect(hasInput('rows_to_keep')).toBe(!variant.startsWith('tensorScatter'));
    expect((llm as any).decodeSet.flush !== null).toBe(staged);
  });

  it('tensorScatter writes the chunk start rows, then the row of every decode step behind the front padding', async () => {
    const { llm, context } = await loaded({ variant: 'tensorScatter-float32' });
    await run(llm, prompt, 5);                                  // 45 tokens in 3 chunks of 16 = 48 rows: 3 pad rows in front
    const index = (llm as any).writeIndexTensor, written = context.writes.filter((w) => w.tensor === index).map((w) => (w.data as Int32Array)[0]);
    expect(written).toEqual([0, 16, 32, 48, 49, 50, 51]);       // chunk starts, then pad + position of the tokens 45, 46, 47, 48
    expect(index.dataType).toBe('int32');
  });

  it('onehot decode selects the cache row of the position', async () => {
    const { llm, context } = await loaded({ variant: 'onehot-float32' });
    await run(llm, prompt, 3);
    const keep = (llm as any).decodeSel.keep, rows = context.writes.filter((w) => w.tensor === keep).map((w) => Array.from(w.data as unknown as ArrayLike<number>).indexOf(0));
    expect(rows).toEqual([45, 46]);                              // positions of the two decode steps; no flush graph exists
  });

  it('picks the variant from kvStyle / computeType / stagingRows', async () => {
    expect((await loaded({ kvStyle: 'onehot' })).llm.variant.id).toBe('onehot-float32');
    expect((await loaded({ kvStyle: 'tensorScatter', computeType: 'float16' })).llm.variant.id).toBe('tensorScatter-float16');
    expect((await loaded({ computeType: 'float16', stagingRows: 8 })).llm.variant.id).toBe('staged-r8-float16');
  });
});

describe('prefill with several chunk sizes', () => {
  it('plans the chunk sequence per prompt and uses one graph set per size', async () => {
    const { llm, context } = await loaded({ chunkSizes: [8, 16], prefillFixedTokens: 2 });
    expect(llm.chunkSizes).toEqual([8, 16]);
    for (const tokens of [5, 21, 45]) {
      context.dispatches.length = 0;
      await run(llm, prompt.slice(0, tokens), 1);
      const plan = planPrefill([8, 16], tokens, 2);
      expect(llm.stats.prefillPlan).toEqual(plan);
      expect(context.dispatches.length).toBe((plan.length - 1) * N + (N + 2));
      const used = new Set(context.dispatches.map((d) => d.inputs.hidden_states?.shape?.[1]).filter(Boolean));
      expect([...used].sort((a, b) => (a as number) - (b as number))).toEqual([...new Set(plan)].sort((a, b) => a - b));
    }
    expect(llm.stats.prefillChunks).toBe(llm.stats.prefillPlan.length);
  });

  it('works with the tensorScatter protocol too (front padding spans the chunks)', async () => {
    const { llm } = await loaded({ chunkSizes: [8, 16], prefillFixedTokens: 2, variant: 'tensorScatter-float32' });
    expect((await run(llm, prompt.slice(0, 21), 4)).length).toBe(4);
    expect(llm.stats.prefillPlan).toEqual(planPrefill([8, 16], 21, 2));
  });
});

describe('reconfigure: other session options on a loaded model', () => {
  it('keeps the weights on the device, rebuilds graphs and state, and runs the new variant', async () => {
    const { llm, context } = await loaded();
    await run(llm, prompt, 2);
    const weightWrites = () => context.writes.filter((w) => ['int4', 'uint8'].includes(w.tensor.dataType)).length;
    const writesBefore = weightWrites(), oldGraphs = [...(llm as any).builtGraphs], oldCache = (llm as any).kv[0].key;
    await llm.reconfigure({ variant: 'tensorScatter-float16', chunkSizes: [8, 16] });
    expect(llm.variant.id).toBe('tensorScatter-float16');
    expect(llm.chunkSizes).toEqual([8, 16]);
    expect(weightWrites()).toBe(writesBefore);                       // no weight was uploaded again
    expect(oldGraphs.every((g: any) => g.destroyed)).toBe(true);     // old graphs and state are released
    expect(oldCache.destroyed).toBe(true);
    context.dispatches.length = 0;
    expect((await run(llm, prompt.slice(0, 21), 3)).length).toBe(3);
    expect(context.dispatches.some((d) => 'kv_write_index' in d.inputs)).toBe(true);
    await llm.reconfigure({});                                       // back to the defaults
    expect(llm.variant.id).toBe('staged-r8-float32');
    expect(llm.chunkSizes).toEqual([8]);
    expect((await run(llm, prompt.slice(0, 21), 3)).length).toBe(3);
  });

  it('keeps the current session when the new options are invalid', async () => {
    const { llm } = await loaded();
    await expect(llm.reconfigure({ variant: 'nope' })).rejects.toThrow(/no variant/);
    expect(llm.variant.id).toBe('staged-r8-float32');
    expect((await run(llm, prompt.slice(0, 21), 3)).length).toBe(3);   // still usable
  });
});

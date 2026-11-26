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
  context.allowAlias = !!options.kvInPlace;   // the mock rejects an output that is also an input of the dispatch, unless the test asks for it
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
    await expect(load({ blocksPerGraph: 3 })).rejects.toThrow(/no decode graph with 3 blocks/);
    await expect(load({ variant: 'nope' })).rejects.toThrow(/no variant 'nope'/);
    await expect(load({ chunkSize: 32 })).rejects.toThrow(/no prefill graphs for chunk 32/);
    await expect(load({ stagingRows: 5 })).rejects.toThrow(/no graphs for kv style staged, 5 staging rows/);
  });
});

describe('every variant of the package runs through the protocol with valid bindings', () => {
  const variants = ['staged-r8-float32', 'staged-r8-float16', 'onehot-float32', 'onehot-float16', 'tensorScatter-float32', 'tensorScatter-float16', 'scatterND-float32', 'scatterND-float16'];

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
    expect(hasInput('kv_indices')).toBe(variant.startsWith('scatterND'));
    expect(hasInput('rows_to_keep')).toBe(!variant.startsWith('tensorScatter') && !variant.startsWith('scatterND'));
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

describe('block groups of a hybrid model', () => {
  it('bind the rope / mask of the layer type of EVERY block of the group (b<j>_ inputs), in decode and prefill', async () => {
    const { llm, context } = await loaded({ blocksPerGraph: 2, prefillBlocksPerGraph: 2, chunkSize: 16 });
    await run(llm, prompt.slice(0, 5), 2);   // one prefill chunk and one decode step
    const types: string[] = llm.pkg.model.layer_types, internals = llm as any, dec = internals.dec, pre = internals.prefillSets.get(16).act;
    const groups = context.dispatches.filter((d) => 'b0_attention_mask' in d.inputs);
    expect(groups.length).toBe(2);   // N - 1 = 2 blocks = one group per phase: [prefill group, decode group]
    [[groups[0], pre], [groups[1], dec]].forEach(([d, act]: any) => {
      for (const j of [0, 1]) {
        expect(d.inputs[`b${j}_attention_mask`]).toBe(act.mask[types[j + 1]]);
        expect(d.inputs[`b${j}_rope_cos`]).toBe(act.ropeCos[types[j + 1]]);
        expect(d.inputs[`b${j}_rope_sin`]).toBe(act.ropeSin[types[j + 1]]);
      }
      expect(d.inputs.b0_attention_mask).not.toBe(d.inputs.b1_attention_mask);   // layers 1 (full) and 2 (sliding): two different masks
    });
    expect(context.dispatches.length).toBe((1 + 1 + 2) + (1 + 1 + 2));   // head + group + tail + lm_head, in prefill and in decode
  });
});

describe('one-hot variants store the value cache transposed', () => {
  it('value cache [kvHeads, 128, P]; the prefill graphs write an attention_guard per block (single blocks and groups), decode has none', async () => {
    for (const groups of [1, 2]) {
      const { llm, context } = await loaded({ variant: 'onehot-float32', blocksPerGraph: groups, prefillBlocksPerGraph: groups });
      const spec = llm.pkg.model, internals = llm as any;
      expect(llm.variant.value_layout).toBe('transposed');
      expect(internals.kv[0].value.shape).toEqual([1, spec.num_kv_heads, spec.head_dim, 127]);
      expect(internals.kv[0].key.shape).toEqual([1, spec.num_kv_heads, 127, spec.head_dim]);
      await run(llm, prompt.slice(0, 5), 2);
      const withGuard = context.dispatches.filter((d) => Object.keys(d.outputs).some((n) => n.endsWith('attention_guard')));
      expect(withGuard.length).toBe(groups === 1 ? 3 : 2);   // prefill: head + 2 blocks, or head + 1 group; decode graphs have no guard
      const guards = new Set(withGuard.flatMap((d) => Object.entries(d.outputs).filter(([n]) => n.endsWith('attention_guard')).map(([, t]) => t.id)));
      expect(guards.size).toBe(groups === 1 ? 1 : 2);        // one scratch tensor for single blocks; one per block of a group (the head reuses the first)
    }
  });
  it('staged and tensorScatter variants keep the natural layout', async () => {
    for (const variant of ['staged-r8-float32', 'tensorScatter-float32']) {
      const { llm } = await loaded({ variant });
      expect((llm as any).kv[0].value.shape).toEqual([1, llm.pkg.model.num_kv_heads, 127, llm.pkg.model.head_dim]);
    }
  });
});

describe('scatterND and in-place KV updates', () => {
  it('scatterND writes the indices of the cache rows (real rows at their position, pad rows parked behind them) for every kv head', async () => {
    const { llm, context } = await loaded({ variant: 'scatterND-float32' });
    await run(llm, prompt.slice(0, 21), 2);                      // chunk 16: the last chunk holds 5 real rows (position 16..20) and 11 pad rows
    const set = (llm as any).prefillSets.get(16), heads = llm.pkg.model.num_kv_heads;
    const prefill = context.writes.filter((w) => w.tensor === set.indices).map((w) => Array.from(w.data as unknown as ArrayLike<number>));
    expect(prefill.length).toBe(2);
    expect(prefill[0].filter((_, i) => i % 3 === 2).slice(0, 16)).toEqual(Array.from({ length: 16 }, (_, i) => i));                    // the first chunk is full: rows 0..15 of head 0
    const second = prefill[1].filter((_, i) => i % 3 === 2).slice(0, 16);
    expect(second.slice(11)).toEqual([16, 17, 18, 19, 20]);                                                                           // real rows
    expect(second.slice(0, 11)).toEqual(Array.from({ length: 11 }, (_, i) => 21 + i));                                                // pad rows parked behind them
    expect(prefill[1].filter((_, i) => i % 3 === 1).slice(16, 17)).toEqual([1 % heads]);                                              // head index of the second head's first row
    const decode = context.writes.filter((w) => w.tensor === (llm as any).decodeIndices).map((w) => (w.data as unknown as number[])[2]);
    expect(decode).toEqual([21]);                                // one decode step after the prompt of 21 tokens: row 21
  });

  it.each(['tensorScatter-float32', 'scatterND-float32'])('kvInPlace: %s writes the cache into the same tensor that is its input; no spare cache', async (variant) => {
    const { llm, context } = await loaded({ variant, kvInPlace: true });
    const internals = llm as any;
    expect(internals.kvSpare.every((spare: any, l: number) => spare === internals.kv[l])).toBe(true);
    const tensors = context.tensors.length;
    await run(llm, prompt.slice(0, 21), 3);
    const head = context.dispatches[0];
    expect(head.outputs.updated_key).toBe(head.inputs.past_key);
    expect(head.outputs.updated_value).toBe(head.inputs.past_value);
    expect(context.dispatches.every((d) => !('updated_key' in d.outputs) || d.outputs.updated_key.id === d.inputs.past_key.id)).toBe(true);
    const normal = await loaded({ variant });
    expect(context.tensors.length).toBe(tensors);                 // nothing else was allocated while running
    expect(normal.context.tensors.length).toBeGreaterThan(tensors);   // the double-buffered cache needs a second set of KV tensors
  });

  it('kvInPlace is ignored by the protocols that rewrite the whole cache, and the mock rejects an aliased dispatch unless asked for', async () => {
    const { llm } = await loaded({ variant: 'onehot-float32', kvInPlace: true });
    expect((llm as any).kvSpare[0]).not.toBe((llm as any).kv[0]);
    const strict = await loaded({ variant: 'tensorScatter-float32', kvInPlace: true });
    strict.context.allowAlias = false;
    await expect(run(strict.llm, prompt.slice(0, 5), 1)).rejects.toThrow(/also an input tensor/);
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

describe('per-graph build times', () => {
  it('the stats list every graph of the session with its build time, in build order', async () => {
    const { llm } = await loaded({ chunkSizes: [8, 16] });
    const labels = llm.stats.graphBuilds.map((g) => g.label);
    expect(labels).toEqual(['lm_head', 'decode head', 'decode block', 'decode tail', 'kv flush', 'prefill c8 head', 'prefill c8 block', 'prefill c8 tail', 'prefill c16 head', 'prefill c16 block', 'prefill c16 tail']);
    expect(llm.stats.graphBuilds.every((g) => g.ms >= 0)).toBe(true);
    await run(llm, prompt.slice(0, 5), 1);
    expect(llm.stats.graphBuilds.length).toBe(11);   // kept when a generation resets the other stats
    await llm.reconfigure({ variant: 'onehot-float32', chunkSizes: [16] });
    expect(llm.stats.graphBuilds.map((g) => g.label)).toEqual(['lm_head', 'decode head', 'decode block', 'decode tail', 'prefill c16 head', 'prefill c16 block', 'prefill c16 tail']);
  });
});

describe('recommended session options of the plan', () => {
  const withRecommended = (recommended: object): PackageSource => ({
    ...fsSource, id: 'tiny-recommended',
    text: async (path) => (path === 'plan.json' ? JSON.stringify({ ...JSON.parse(await fsSource.text(path)), recommended }) : fsSource.text(path)),
  });

  it('are used when the caller does not choose; an explicit choice wins; a group size the variant lacks falls back to single blocks', async () => {
    installMockBuilder();
    const options = { context: new MockContext(), cache: false, chunkSize: 16 };
    const group = async (source: PackageSource, extra: LoadOptions = {}) => { const llm = await LLM.load(source, { ...options, context: new MockContext(), ...extra }); return [(llm as any).groupSize, (llm as any).prefillGroupSize]; };
    expect(await group(fsSource)).toEqual([1, 1]);                                                                       // a plan without recommendation: single blocks
    expect(await group(withRecommended({ blocks_per_graph: 2, prefill_blocks_per_graph: 1 }))).toEqual([2, 1]);          // decode groups of 2, single prefill blocks
    expect(await group(withRecommended({ blocks_per_graph: 2, prefill_blocks_per_graph: 1 }), { blocksPerGraph: 1 })).toEqual([1, 1]);   // the caller chose
    expect(await group(withRecommended({ blocks_per_graph: 5, prefill_blocks_per_graph: 1 }))).toEqual([1, 1]);          // the package has no groups of 5: no error for a default
    await expect(LLM.load(withRecommended({ blocks_per_graph: 5 }), { ...options, context: new MockContext(), blocksPerGraph: 5 })).rejects.toThrow(/no decode graph with 5 blocks/);   // but one the caller asks for is an error
  });
});

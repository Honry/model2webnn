import { readFileSync } from 'node:fs';
import { beforeAll, describe, expect, it } from 'vitest';
import { Gemma4LLM } from './llm-gemma4.js';
import { loadModel } from './model.js';
import { LLM, type LoadOptions } from './llm.js';
import type { PackageSource } from './package.js';
import { planPrefill } from './protocol.js';
import { installMockBuilder, MockContext } from './mock-ml.js';

// The committed tiny Gemma 4 style package (tests/golden/tiny-gemma4-package, scripts/make-tiny-package.py): 6 layers; layers 0..3 have K / V of their own (sliding, sliding, full, sliding),
// layer 4 (full) and 5 (sliding) read the caches of layers 2 and 3; per-layer embeddings; max length 128, chunks 8 and 16, 8 staging rows, float32 and float16 compute.
const root = new URL('../../tests/golden/tiny-gemma4-package/', import.meta.url);
const sliceCalls: { path: string; start: number; end: number }[] = [];
const fsSource: PackageSource = {
  id: 'tiny-gemma4',
  text: async (path) => readFileSync(new URL(path, root), 'utf8'),
  bytes: async (path) => { const b = readFileSync(new URL(path, root)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); },
  slice: async (path, start, end) => { sliceCalls.push({ path, start, end }); const b = readFileSync(new URL(path, root)); return b.buffer.slice(b.byteOffset + start, b.byteOffset + end); },
};
const verification = JSON.parse(readFileSync(new URL('verification.json', root), 'utf8'));
const manifest = JSON.parse(readFileSync(new URL('weights/manifest.json', root), 'utf8'));
const prompt: number[] = verification.prompt_ids;   // 45 tokens
const N = 6;

async function loaded(options: LoadOptions = {}) {
  installMockBuilder();
  const context = new MockContext();
  const llm = await Gemma4LLM.load(fsSource, { context, cache: false, ...options });
  context.onRead = (_t, destination) => { destination.fill(0); destination[7] = 5; };   // argmax = token 7
  sliceCalls.length = 0;
  return { llm, context };
}
async function run(llm: Gemma4LLM, tokens: number[], maxNewTokens: number) {
  const out: number[] = [];
  for await (const token of llm.generate(tokens, { maxNewTokens, stopTokens: [] })) out.push(token);
  return out;
}
const internals = (llm: Gemma4LLM) => llm as any;

describe('Gemma 4 runtime on a validating mock of WebNN', () => {
  beforeAll(() => { if (!(globalThis as any).Float16Array) throw new Error('needs Node 24+'); });

  it('loads: kinds, KV caches only for the layers with K / V of their own, one block graph per kind, flush graphs per layer type', async () => {
    const { llm } = await loaded();
    const x = internals(llm);
    expect(llm.variant.id).toBe('staged-r8-float32');
    expect(x.kinds).toEqual(['sliding_attention_own', 'sliding_attention_own', 'full_attention_own', 'sliding_attention_own', 'full_attention_shared', 'sliding_attention_shared']);
    expect([...x.kv.keys()]).toEqual([0, 1, 2, 3]);
    expect(x.producer).toEqual([0, 1, 2, 3, 2, 3]);
    expect([...x.decodeSet.blocks.keys()].sort()).toEqual(['full_attention_own', 'full_attention_shared', 'sliding_attention_own', 'sliding_attention_shared']);
    expect([...x.decodeSet.flush.keys()].sort()).toEqual(['full_attention', 'sliding_attention']);
    expect(x.kv.get(2).key.shape).toEqual([1, 2, 127, 64]);    // full attention layers have the bigger head dim
    expect(x.kv.get(3).key.shape).toEqual([1, 2, 127, 32]);
  });

  it('runs the protocol: chunked prefill, staged decode with flushes, every dispatch has valid bindings; shared layers read the cache of their producer', async () => {
    const { llm, context } = await loaded({ chunkSizes: [8, 16] });
    const generated = await run(llm, prompt, 12);
    expect(generated).toEqual(Array(12).fill(7));
    const plan = planPrefill([8, 16], 45, 200), steps = 11, flushes = 1;   // 8 staging rows: full before the 9th step
    expect(llm.stats.prefillPlan).toEqual(plan);
    const prefillDispatches = (plan.length - 1) * N + (N + 2), decodeDispatches = steps * (N + 2), ownLayers = 4;
    expect(context.dispatches.length).toBe(prefillDispatches + decodeDispatches + flushes * ownLayers);
    // the dispatches of the first decode step (after the prefill chain): HEAD, 5 blocks, TAIL, lm_head
    const first = context.dispatches[prefillDispatches], step = context.dispatches.slice(prefillDispatches, prefillDispatches + N + 2);
    expect(first.inputs.hidden_states.shape).toEqual([1, 1, 128]);
    const x = internals(llm);
    // the dispatch of index L runs the attention half of layer L: step[0] = HEAD (layer 0), step[L] = BLOCK L - 1; layer 4 (full) and 5 (sliding) share the K / V of layers 2 and 3
    for (const [layer, producer] of [[4, 2], [5, 3]]) {
      const d = step[layer];
      expect(d.inputs.past_key).toBe(step[producer].inputs.past_key);                 // the main cache the producer attends over
      expect(d.inputs.staging_key).toBe(step[producer].outputs.updated_key);          // the staging buffer AFTER this step's update of the producer (the output of the producer's dispatch)
      expect(d.inputs.staging_value).toBe(step[producer].outputs.updated_value);
      expect('updated_key' in d.outputs).toBe(false);
      expect(d.inputs.attention_mask).toBe(x.dec.maskShared[x.types[layer]]);
      expect(d.inputs.per_layer_input).toBe(x.dec.ples[layer - 1]);
    }
    // blocks of layers with their own K / V write the spare staging buffers, which become the staging buffers
    expect('updated_key' in step[2].outputs).toBe(true);
  });

  it('prefill: a shared layer attends over the updated cache of its producer with its own mask (columns = cache rows), the others over concat(cache, chunk)', async () => {
    const { llm, context } = await loaded({ chunkSize: 16 });
    await run(llm, prompt.slice(0, 5), 1);   // one prefill chunk
    const x = internals(llm), pre = x.prefillSets.get(16).act;
    const chain = context.dispatches;
    expect(pre.mask.full_attention.shape).toEqual([1, 1, 16, 127 + 16]);
    expect(pre.maskShared.full_attention.shape).toEqual([1, 1, 16, 127]);
    expect(chain[4].inputs.attention_mask).toBe(pre.maskShared.full_attention);   // chain[L] runs the attention half of layer L: layer 4
    expect(chain[4].inputs.past_key).toBe(x.kv.get(2).key);                        // the producer's cache after its update in this chunk (swapped)
    expect(chain[2].inputs.attention_mask).toBe(pre.mask.full_attention);          // layer 2 has its own K / V
  });

  it('reads the per-layer embedding rows as byte ranges (one per token) and scales the embedding on the host', async () => {
    const { llm, context } = await loaded({ chunkSize: 16 });
    await run(llm, prompt.slice(0, 5), 2);   // 5 prompt tokens + 1 decode step
    const file = manifest.files.find((f: any) => f.role === 'ple_table'), weight = file.tensors.find((t: any) => t.name === 'ple_weight'), width = weight.shape[1];
    const rows = sliceCalls.filter((c) => c.path === `weights/${file.file}`);
    expect(rows.length).toBe(6);
    for (const [i, c] of rows.slice(0, 5).entries()) expect(c).toEqual({ path: c.path, start: weight.offset + prompt[i] * width, end: weight.offset + (prompt[i] + 1) * width });
    // the row of the first prompt token as the host wrote it into the HEAD input (left padded chunk: 11 pad rows first)
    const bytes = readFileSync(new URL(`weights/${file.file}`, root)), scaleEntry = file.tensors.find((t: any) => t.name === 'ple_scale');
    const q = new Int8Array(bytes.buffer, bytes.byteOffset + weight.offset + prompt[0] * width, width);
    const scale = new (globalThis as any).Float16Array(bytes.buffer.slice(bytes.byteOffset + scaleEntry.offset, bytes.byteOffset + scaleEntry.offset + scaleEntry.bytes))[prompt[0]];
    const pleToken = internals(llm).prefillSets.get(16).act.pleToken, write = context.writes.find((w) => w.tensor === pleToken) as any;
    const got = new (globalThis as any).Float16Array(write.data.buffer, write.data.byteOffset, write.data.byteLength / 2);
    for (let j = 0; j < width; j += 37) expect(got[11 * width + j]).toBeCloseTo(q[j] * scale, 2);
    expect(got[0]).toBe(0);   // pad rows are zero
  });

  it('plans several chunk sizes and reconfigures to another variant while the weights stay on the device', async () => {
    const { llm, context } = await loaded({ chunkSizes: [8, 16], prefillFixedTokens: 2 });
    for (const tokens of [5, 21]) {
      await run(llm, prompt.slice(0, tokens), 1);
      expect(llm.stats.prefillPlan).toEqual(planPrefill([8, 16], tokens, 2));
    }
    const weightWrites = () => context.writes.filter((w) => ['int4', 'int8', 'uint8'].includes(w.tensor.dataType)).length, before = weightWrites();
    await llm.reconfigure({ computeType: 'float16', chunkSize: 8 });
    expect(llm.variant.id).toBe('staged-r8-float16');
    expect(weightWrites()).toBe(before);
    expect((await run(llm, prompt.slice(0, 21), 3)).length).toBe(3);
    await expect(llm.reconfigure({ kvStyle: 'onehot' })).rejects.toThrow(/staged/);
    await expect(llm.reconfigure({ blocksPerGraph: 6 })).rejects.toThrow(/block groups/);
  });

  it('loadModel picks the runtime from the plan; the standard runtime refuses a Gemma 4 package', async () => {
    installMockBuilder();
    const model = await loadModel(fsSource, { context: new MockContext(), cache: false });
    expect(model).toBeInstanceOf(Gemma4LLM);
    await expect(LLM.load(fsSource, { context: new MockContext(), cache: false })).rejects.toThrow(/gemma4 runtime/);
  });

  it('rejects prompts that do not fit', async () => {
    const { llm } = await loaded();
    await expect(async () => { for await (const _ of llm.generate(Array(127).fill(5))) void _; }).rejects.toThrow(/KV cache holds 127/);
  });
});

describe('Gemma 4 decode block groups', () => {
  // the tiny model has 5 BLOCKs (the attention halves of layers 1..5): sliding own, full own, sliding own, full shared (reads layer 2), sliding shared (reads layer 3)
  const stateIndex = (llm: Gemma4LLM) => {
    const x = internals(llm), at = (t: any) => x.stateTensors.indexOf(t);
    return { kv: [...x.kv.entries()].map(([l, c]: any) => [l, at(c.key), at(c.value)]), stg: [...x.stg.entries()].map(([l, c]: any) => [l, at(c.key), at(c.value)]) };
  };

  it.each([2, 3, 4, 5])('%i blocks per graph: one dispatch per group, the same tokens and the same main / staging buffers as without groups', async (group) => {
    const plain = await loaded({ chunkSizes: [8, 16] }), grouped = await loaded({ chunkSizes: [8, 16], blocksPerGraph: group });
    const a = await run(plain.llm, prompt, 12), b = await run(grouped.llm, prompt, 12);
    expect(b).toEqual(a);
    const steps = 11, groups = Math.ceil(5 / group), plan = planPrefill([8, 16], 45, 200), prefill = (plan.length - 1) * N + (N + 2);
    expect(plain.context.dispatches.length).toBe(prefill + steps * (N + 2) + 4);
    expect(grouped.context.dispatches.length).toBe(prefill + steps * (groups + 3) + 4);   // HEAD + groups + TAIL + lm_head per decode step; the prefill chain and the flushes are as before
    expect(stateIndex(grouped.llm)).toEqual(stateIndex(plain.llm));   // every layer's main / spare staging buffers were swapped exactly as often
    const labels = grouped.llm.stats.graphBuilds.map((g) => g.label);
    expect(labels.some((l) => l.startsWith('decode block'))).toBe(false);
    expect(labels.filter((l) => l.startsWith('decode group')).length).toBe(group === 2 ? 3 : group === 3 ? 2 : group === 4 ? 2 : 1);
  });

  it('binds a group by input name: controls per layer type, caches of the producer layers, the staging buffer of a producer in the group is its own output of the group', async () => {
    const { llm, context } = await loaded({ chunkSizes: [8, 16], blocksPerGraph: 4 });
    await run(llm, prompt, 2);
    const x = internals(llm), prefill = (planPrefill([8, 16], 45, 200).length - 1) * N + (N + 2);
    const [head, g0, g1, tail] = context.dispatches.slice(prefill, prefill + 4);   // the decode step: HEAD, blocks 0..3, block 4, TAIL
    expect(head.inputs.hidden_states.shape).toEqual([1, 1, 128]);
    expect(g0.inputs.residual).toBe(head.outputs.residual);
    expect(g0.inputs.attention_output).toBe(head.outputs.attention_output);
    expect(g0.inputs.sliding_attention_attention_mask).toBe(x.dec.mask.sliding_attention);
    expect(g0.inputs.full_attention_rope_sin).toBe(x.dec.ropeSin.full_attention);
    expect(g0.inputs.b3_per_layer_input).toBe(x.dec.ples[3]);
    // block 3 (layer 4, full, shared) reads the caches of layer 2 = block 1 inside the group: no inputs of its own for them
    expect(Object.keys(g0.inputs).filter((n) => n.startsWith('b3_') && /past|staging/.test(n))).toEqual([]);
    expect(Object.keys(g0.inputs).filter((n) => n.endsWith('past_key'))).toEqual(['b0_past_key', 'b1_past_key', 'b2_past_key']);
    expect(g0.inputs.b1_past_key.shape).toEqual([1, 2, 127, 64]);
    expect(g0.inputs.b3_query_projection_weight).toBe(x.weights[4].query_projection_weight);
    expect(g0.inputs.b3_gate_up_projection_weight).toBe(x.weights[3].gate_up_projection_weight);   // the feed-forward half of layer 3, the attention half of layer 4
    expect(Object.keys(g0.outputs)).toEqual(['b0_updated_key', 'b0_updated_value', 'b1_updated_key', 'b1_updated_value', 'b2_updated_key', 'b2_updated_value', 'next_residual', 'next_attention_output']);
    // block 4 (layer 5, sliding, shared) reads layer 3 = block 2 of the previous group: its staging input is the staging buffer after that group's update
    expect(g1.inputs.residual).toBe(g0.outputs.next_residual);
    expect(g1.inputs.b0_staging_key).toBe(g0.outputs.b2_updated_key);
    expect(g1.inputs.b0_past_key).toBe(g0.inputs.b2_past_key);
    expect('b0_updated_key' in g1.outputs).toBe(false);
    expect(tail.inputs.residual).toBe(g1.outputs.next_residual);
  });

  it('groups that read the caches of another group: the staging buffers are the updated ones of the dispatch that wrote them', async () => {
    const { llm, context } = await loaded({ chunkSizes: [8, 16], blocksPerGraph: 2 });
    await run(llm, prompt, 2);
    const prefill = (planPrefill([8, 16], 45, 200).length - 1) * N + (N + 2);
    const [, g0, g1, g2] = context.dispatches.slice(prefill, prefill + 5);   // blocks 0..1 | 2..3 | 4
    expect(g1.inputs.b1_staging_key).toBe(g0.outputs.b1_updated_key);   // layer 4 reads layer 2 (block 1)
    expect(g1.inputs.b1_past_key).toBe(g0.inputs.b1_past_key);
    expect(g2.inputs.b0_staging_key).toBe(g1.outputs.b0_updated_key);   // layer 5 reads layer 3 (block 2)
  });

  it('reconfigures from and to groups; refuses prefill groups and sizes the package does not have', async () => {
    const { llm, context } = await loaded({ chunkSizes: [8, 16] });
    await expect(llm.reconfigure({ prefillBlocksPerGraph: 2 })).rejects.toThrow(/decode only/);
    await expect(llm.reconfigure({ blocksPerGraph: 6 })).rejects.toThrow(/no decode block groups of 6 blocks \(it has 2, 3, 4, 5\)/);
    await llm.reconfigure({ blocksPerGraph: 2, chunkSizes: [8, 16] });
    context.onRead = (_t, destination) => { destination.fill(0); destination[7] = 5; };
    expect(await run(llm, prompt.slice(0, 21), 4)).toEqual([7, 7, 7, 7]);
    await llm.reconfigure({ blocksPerGraph: 1, chunkSizes: [8, 16] });
    expect((internals(llm).decodeSet.blocks as Map<string, unknown>).size).toBe(4);
  });
});

describe('Gemma 4 per-graph build times', () => {
  it('lists every graph, one block graph per attention kind', async () => {
    const { llm } = await loaded({ chunkSize: 16 });
    const labels = llm.stats.graphBuilds.map((g) => g.label);
    expect(labels.filter((l) => l.startsWith('decode block'))).toHaveLength(4);
    expect(labels).toContain('decode block full_attention_shared');
    expect(labels).toContain('kv flush sliding_attention');
    expect(labels).toContain('prefill c16 tail');
    expect(labels[0]).toBe('lm_head');
  });
});

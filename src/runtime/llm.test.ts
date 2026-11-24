import { readFileSync } from 'node:fs';
import { beforeAll, describe, expect, it } from 'vitest';
import { LLM } from './llm.js';
import type { PackageSource } from './package.js';
import { installMockBuilder, MockContext } from './mock-ml.js';

// The committed tiny hybrid-layer package (tests/golden/tiny-package, made by scripts/make-tiny-package.py): 3 layers [sliding, full, sliding], max length 128, chunk 16, staging 8.
const root = new URL('../../tests/golden/tiny-package/', import.meta.url);
const fsSource: PackageSource = {
  id: 'tiny',
  text: async (path) => readFileSync(new URL(path, root), 'utf8'),
  bytes: async (path) => { const b = readFileSync(new URL(path, root)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); },
};
const verification = JSON.parse(readFileSync(new URL('verification.json', root), 'utf8'));

async function loaded() {
  installMockBuilder();
  const context = new MockContext();
  const llm = await LLM.load(fsSource, { context, cache: false });
  return { llm, context };
}
const names = (graph: any) => graph.graph.inputs;

describe('LLM scheduler on a validating mock of WebNN', () => {
  beforeAll(() => { if (!(globalThis as any).Float16Array) throw new Error('needs Node 24+'); });

  it('loads: graphs for decode / prefill, weights as writable tensors, state tensors', async () => {
    const { llm, context } = await loaded();
    expect(llm.maxLength).toBe(128);
    expect(llm.chunkSize).toBe(16);
    const spec = llm.pkg.model;
    expect(spec.layer_types).toEqual(['sliding_attention', 'full_attention', 'sliding_attention']);
    // every weight shard tensor was uploaded: 3 layers x 12 tensors (merged projections, qk-norm), the final norm, nothing left unwritten
    const writtenIds = new Set(context.writes.map((w) => w.tensor.id));
    const weightTensors = context.tensors.filter((t) => t.writable && ['int4', 'uint8'].includes(t.dataType) || (t.dataType === 'float16' && t.shape.length === 3 && t.shape[2] === 1 && t.writable));
    expect(weightTensors.length).toBeGreaterThan(0);
    for (const t of weightTensors) expect(writtenIds.has(t.id)).toBe(true);
    // the lm_head graph keeps the embedding as constants; the decode / prefill graphs take no embedding
    expect(Object.keys(names((llm as any).graphs.lmHead))).toEqual(['final_hidden_states']);
  });

  it('runs the whole protocol: chunked prefill, staged decode, flushes; every dispatch has valid bindings', async () => {
    const { llm, context } = await loaded();
    context.onRead = (_t, destination) => { destination.fill(0); destination[7] = 5; };   // argmax = token 7
    const prompt: number[] = verification.prompt_ids;                                    // 45 tokens = 3 chunks of 16 (the last one is left-padded)
    const generated: number[] = [];
    for await (const token of llm.generate(prompt, { maxNewTokens: 20, stopTokens: [] })) generated.push(token);
    expect(generated).toEqual(Array(20).fill(7));
    const N = 3, steps = 19, flushes = 2;   // staging has 8 rows: it is full before step 9 and before step 17
    const prefillDispatches = 2 * (1 + (N - 1)) + (1 + (N - 1) + 1 + 1);   // two chunks without tail / lm_head, the last one with
    const decodeDispatches = steps * (1 + (N - 1) + 1 + 1);
    expect(context.dispatches.length).toBe(prefillDispatches + decodeDispatches + flushes * N);
    const flushGraph = (llm as any).graphs.decode.flush.graph;
    expect(context.dispatches.filter((d) => d.graph === flushGraph).length).toBe(flushes * N);
    expect(llm.stats.promptTokens).toBe(45);
    expect(llm.stats.decodeSteps).toBe(19);
  });

  it('binds the rope / mask tensors of the layer type of the layer whose attention runs (HEAD: layer 0, BLOCK i: layer i+1)', async () => {
    const { llm, context } = await loaded();
    context.onRead = (_t, d) => d.fill(0);
    for await (const _ of llm.generate(verification.prompt_ids.slice(0, 5), { maxNewTokens: 1, stopTokens: [] })) void _;   // one prefill chunk, no decode
    const internals = llm as any, types: string[] = llm.pkg.model.layer_types, pre = internals.pre;
    const chain = context.dispatches;
    const head = chain[0], blocks = chain.slice(1, 3);
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
    const { llm, context } = await loaded();
    context.onRead = (_t, d) => d.fill(0);
    const before = (llm as any).kv.map((c: any) => c.key.id);
    for await (const _ of llm.generate(verification.prompt_ids.slice(0, 5), { maxNewTokens: 1, stopTokens: [] })) void _;
    const after = (llm as any).kv.map((c: any) => c.key.id), spare = (llm as any).kvSpare.map((c: any) => c.key.id);
    expect(after).not.toEqual(before);   // one chunk = one update per layer: every cache was swapped with its spare ...
    expect(spare).toEqual(before);       // ... so the spare is now exactly the buffer that was the cache before
  });

  it('rejects prompts that do not fit and reports missing graph variants', async () => {
    const { llm } = await loaded();
    await expect(async () => { for await (const _ of llm.generate(Array(127).fill(5))) void _; }).rejects.toThrow(/KV cache holds 127/);
    installMockBuilder();
    await expect(LLM.load(fsSource, { context: new MockContext(), cache: false, maxLength: 4096 })).rejects.toThrow(/no graphs for max length 4096/);
    await expect(LLM.load(fsSource, { context: new MockContext(), cache: false, blocksPerGraph: 2 })).rejects.toThrow(/one layer type/);
  });
});

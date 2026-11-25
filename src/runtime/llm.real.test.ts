// Optional: the same validating-mock run as llm.test.ts, but on a REAL package (all its graphs, the real layer count, weight shards, hybrid layer types):
//   M2W_PACKAGE=python/_out/aion-2.3b npx vitest run src/runtime/llm.real.test.ts      (also: M2W_VARIANT=onehot-float32 M2W_CHUNKS=64,256 M2W_MAX_LENGTH=1152 M2W_BPG=5 M2W_PBPG=5)
// It finds binding mistakes (names, dtypes, shapes, aliasing) before the package ever meets an NPU. Skipped when M2W_PACKAGE is not set.
import { existsSync, openSync, readSync, closeSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { describe, expect, it } from 'vitest';
import { LLM } from './llm.js';
import type { PackageSource } from './package.js';
import { installMockBuilder, MockContext } from './mock-ml.js';

const dir = process.env.M2W_PACKAGE ? path.resolve(process.env.M2W_PACKAGE) : '';
const fileSource: PackageSource = {
  id: dir,
  text: async (p) => readFileSync(path.join(dir, p), 'utf8'),
  bytes: async (p) => {
    const file = path.join(dir, p), size = statSync(file).size, buffer = new Uint8Array(size), fd = openSync(file, 'r');
    try { for (let done = 0; done < size;) done += readSync(fd, buffer, done, Math.min(size - done, 1 << 28), done); } finally { closeSync(fd); }
    return buffer.buffer;
  },
};

describe.skipIf(!dir || !existsSync(path.join(dir, 'm2w.json')))('real package on the validating mock', () => {
  it('loads, prefills, decodes through a flush and validates every dispatch', async () => {
    installMockBuilder();
    const context = new MockContext();
    context.allowAlias = process.env.M2W_INPLACE === '1';
    const chunkSizes = process.env.M2W_CHUNKS ? process.env.M2W_CHUNKS.split(',').map(Number) : undefined;
    const llm = await LLM.load(fileSource, {
      context, cache: false, maxLength: process.env.M2W_MAX_LENGTH ? +process.env.M2W_MAX_LENGTH : undefined, variant: process.env.M2W_VARIANT, chunkSizes, kvInPlace: process.env.M2W_INPLACE === '1',
      blocksPerGraph: process.env.M2W_BPG ? +process.env.M2W_BPG : 1, prefillBlocksPerGraph: process.env.M2W_PBPG ? +process.env.M2W_PBPG : 0,
    });
    const m = llm.pkg.model, N = m.num_layers, R = llm.variant.staging_rows ?? 0, staged = llm.variant.kv_style === 'staged';
    context.onRead = (_t, d) => { d.fill(0); d[3] = 5; };
    const prompt = Array.from({ length: 46 }, (_, i) => 3 + (i % 50)), steps = (staged ? R : 0) + 3;   // staged: enough decode steps to need one flush
    const out: number[] = [];
    for await (const token of llm.generate(prompt, { maxNewTokens: steps + 1, stopTokens: [] })) out.push(token);
    expect(out.length).toBe(steps + 1);
    if (staged) {
      const flush = (llm as any).decodeSet.flush.graph;
      expect(context.dispatches.filter((d) => d.graph === flush).length).toBe(N);   // exactly one flush
    }
    const bpg = process.env.M2W_BPG ? +process.env.M2W_BPG : 1, pbpg = process.env.M2W_PBPG ? +process.env.M2W_PBPG : bpg;
    const decodeChain = 1 + Math.ceil((N - 1) / bpg) + 2;
    const plan = llm.stats.prefillPlan, blockDispatches = 1 + Math.ceil((N - 1) / pbpg);
    const prefillChain = (plan.length - 1) * blockDispatches + (blockDispatches + 2);   // head + blocks per chunk, plus tail + lm_head on the last
    expect(context.dispatches.length).toBe(prefillChain + steps * decodeChain + (staged ? N : 0));
    console.log(`${m.architecture}: ${N} layers, ${context.dispatches.length} dispatches validated, ${context.tensors.length} tensors, graphs ${new Set(context.dispatches.map((d) => d.graph.name ?? '')).size}, variant ${llm.variant.id}, prefill plan ${plan.join('+')}`);
  }, 300000);
});

import { readFileSync } from 'node:fs';
import { beforeAll, describe, expect, it } from 'vitest';
import { Qwen35LLM } from './llm-qwen35.js';
import { loadModel } from './model.js';
import { LLM, type LoadOptions } from './llm.js';
import type { PackageSource } from './package.js';
import { layerTypesOf, linearPrefillControls, planPrefill, ropeTables } from './protocol.js';
import { invFreqFor } from './rope.js';
import { installMockBuilder, MockContext } from './mock-ml.js';

// The committed tiny Qwen3.5 package (tests/golden/tiny-qwen35-package, scripts/make-tiny-package.py): 6 layers = linear, linear, full, linear, linear, full (Gated DeltaNet layers carry a conv state
// and a recurrent state, only the full attention layers 2 and 5 have KV caches); max length 128, chunks 16 and 32, 8 staging rows, float32 and float16 compute.
const root = new URL('../../tests/golden/tiny-qwen35-package/', import.meta.url);
const fsSource: PackageSource = {
  id: 'tiny-qwen35',
  text: async (path) => readFileSync(new URL(path, root), 'utf8'),
  bytes: async (path) => { const b = readFileSync(new URL(path, root)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); },
};
const verification = JSON.parse(readFileSync(new URL('verification.json', root), 'utf8'));
const prompt: number[] = verification.prompt_ids;   // 45 tokens
const N = 6, LINEAR = [0, 1, 3, 4], FULL = [2, 5];

async function loaded(options: LoadOptions = {}) {
  installMockBuilder();
  const context = new MockContext();
  const llm = await Qwen35LLM.load(fsSource, { context, cache: false, blocksPerGraph: 1, ...options });   // the plan may recommend groups: the tests of the single-block protocol ask for 1
  context.onRead = (_t, destination) => { destination.fill(0); destination[7] = 5; };   // argmax = token 7
  return { llm, context };
}
async function run(llm: Qwen35LLM, tokens: number[], maxNewTokens: number) {
  const out: number[] = [];
  for await (const token of llm.generate(tokens, { maxNewTokens, stopTokens: [] })) out.push(token);
  return out;
}
const internals = (llm: Qwen35LLM) => llm as any;
const f16 = (t: any) => new (globalThis as any).Float16Array(t.data.buffer, t.data.byteOffset, t.data.byteLength / 2);

describe('Qwen3.5 runtime on a validating mock of WebNN', () => {
  beforeAll(() => { if (!(globalThis as any).Float16Array) throw new Error('needs Node 24+'); });

  it('loads: KV caches only for the full attention layers, a conv + recurrent state per linear layer, one block graph per kind', async () => {
    const { llm } = await loaded();
    const x = internals(llm);
    expect(llm.variant.id).toBe('staged-r8-float32');
    expect([...x.kv.keys()]).toEqual(FULL);
    expect([...x.state.keys()]).toEqual(LINEAR);
    expect([...x.decodeSet.blocks.keys()].sort()).toEqual(['full_attention', 'linear_attention']);
    expect(x.kv.get(2).key.shape).toEqual([1, 2, 127, 32]);
    expect(x.state.get(0).conv.shape).toEqual([1, 3, 1024]);       // conv channels = 2 * 2 * 128 + 4 * 128
    expect(x.state.get(0).rec.shape).toEqual([1, 4, 128, 128]);    // [value heads, k dim, v dim]
    expect(layerTypesOf(llm.pkg.model)).toEqual(['full_attention']);   // linear attention layers have no masks / rope tables
  });

  it('runs the protocol: chunked prefill, staged decode with flushes; states are zeroed per prompt, swapped with their spares, and only prefill graphs take the pad controls', async () => {
    const { llm, context } = await loaded({ chunkSizes: [16, 32] });
    const generated = await run(llm, prompt, 12);
    expect(generated).toEqual(Array(12).fill(7));
    const plan = planPrefill([16, 32], 45, 200), steps = 11, flushes = 1;   // 8 staging rows: full before the 9th step
    expect(llm.stats.prefillPlan).toEqual(plan);
    const prefillDispatches = (plan.length - 1) * N + (N + 2), decodeDispatches = steps * (N + 2);
    expect(context.dispatches.length).toBe(prefillDispatches + decodeDispatches + flushes * FULL.length);
    const x = internals(llm);
    // prefill: every linear layer's graph has valid_mask and conv_perm, the full attention layers do not
    const chunk = context.dispatches.slice(0, N);   // HEAD + 5 BLOCKs of the first chunk; chain[L] = the attention half of layer L
    for (const layer of LINEAR) { expect('valid_mask' in chunk[layer].inputs).toBe(true); expect('conv_perm' in chunk[layer].inputs).toBe(true); }
    for (const layer of FULL) { expect('valid_mask' in chunk[layer].inputs).toBe(false); expect(chunk[layer].inputs.attention_mask.shape).toEqual([1, 1, plan[0], 127 + plan[0]]); }
    // decode: no pad controls; the output states of one step are the input states of the next
    const step1 = context.dispatches.slice(prefillDispatches, prefillDispatches + N + 2), step2 = context.dispatches.slice(prefillDispatches + N + 2, prefillDispatches + 2 * (N + 2));
    for (const layer of LINEAR) {
      expect('valid_mask' in step1[layer].inputs).toBe(false);
      expect(step2[layer].inputs.conv_state).toBe(step1[layer].outputs.new_conv_state);
      expect(step2[layer].inputs.recurrent_state).toBe(step1[layer].outputs.new_recurrent_state);
      expect(step1[layer].outputs.new_conv_state).not.toBe(step1[layer].inputs.conv_state);
    }
    // the staging buffers of the full attention layers alternate with their spares too
    expect(step2[2].inputs.staging_key).toBe(step1[2].outputs.updated_key);
    expect(x.state.size).toBe(4);
  });

  it('zeroes every state when a prompt starts', async () => {
    const { llm, context } = await loaded({ chunkSize: 16 });
    const x = internals(llm);
    await run(llm, prompt.slice(0, 5), 1);
    const states = [...x.state.values()].flatMap((s: any) => [s.conv, s.rec]);
    const mark = context.writes.length;
    await run(llm, prompt.slice(0, 5), 1);
    const zeroed = new Set(context.writes.slice(mark).filter((w) => states.includes(w.tensor) && f16(w).every((v: number) => v === 0)).map((w) => w.tensor));
    expect(zeroed.size).toBe(states.length);
  });

  it('pads the last chunk: valid_mask zeroes the pad rows, conv_perm moves them in front of the state rows', async () => {
    const { llm, context } = await loaded({ chunkSize: 16 });
    await run(llm, prompt.slice(0, 5), 1);   // one chunk of 16 with 5 real rows
    const set = internals(llm).prefillSets.get(16);
    const mask = f16(context.writes.find((w) => w.tensor === set.validMask) as any), perm = f16(context.writes.find((w) => w.tensor === set.convPerm) as any);
    expect([...mask]).toEqual([...Array(11).fill(0), ...Array(5).fill(1)]);
    const size = 19, source = (r: number) => [...perm.slice(r * size, (r + 1) * size)].indexOf(1);
    expect(source(0)).toBe(3);      // pad row 0 (chunk row 0 = state rows 0..2 come first in the input, so the chunk rows start at 3)
    expect(source(11)).toBe(0);     // the first state row now sits behind the pad rows
    expect(source(13)).toBe(2);
    expect(source(14)).toBe(14);    // the real rows stay where they are
    expect(source(18)).toBe(18);
  });

  it('the protocol controls match the Python protocol and the partial rotary frequencies match rope.partial_inv_freq', () => {
    const { validMask, convPerm } = linearPrefillControls(6, 4);
    expect([...validMask]).toEqual([0, 0, 1, 1, 1, 1]);
    const rows = [...Array(9).keys()].map((r) => [...convPerm.slice(r * 9, r * 9 + 9)].indexOf(1));
    expect(rows).toEqual([3, 4, 0, 1, 2, 5, 6, 7, 8]);
    expect([...linearPrefillControls(6, 6).convPerm.slice(0, 9)]).toEqual([1, 0, 0, 0, 0, 0, 0, 0, 0]);
    const { invFreq } = invFreqFor({ type: 'default', theta: 1e7, partial_rotary_factor: 0.25 }, 256);
    expect(invFreq).toHaveLength(128);
    expect(invFreq[0]).toBe(1);
    expect(invFreq[31]).toBeCloseTo(1 / 1e7 ** (62 / 64), 12);
    expect(invFreq.slice(32).every((f) => f === 0)).toBe(true);
    const tables = ropeTables({ layer_types: ['linear_attention', 'full_attention'], sliding_window: null, head_dim: 32, rope: { full_attention: { type: 'default', theta: 1e7, partial_rotary_factor: 0.25 } } }, [3], 1, 0);
    expect(Object.keys(tables)).toEqual(['full_attention']);
    expect(tables.full_attention.cos[20]).toBe(1);   // dims that do not rotate: cos 1, sin 0
    expect(tables.full_attention.sin[20]).toBe(0);
  });

  it('plans several chunk sizes and reconfigures to another variant while the weights stay on the device', async () => {
    const { llm, context } = await loaded({ chunkSizes: [16, 32], prefillFixedTokens: 2 });
    for (const tokens of [5, 21]) {
      await run(llm, prompt.slice(0, tokens), 1);
      expect(llm.stats.prefillPlan).toEqual(planPrefill([16, 32], tokens, 2));
    }
    const weightWrites = () => context.writes.filter((w) => ['int4', 'int8', 'uint8'].includes(w.tensor.dataType)).length, before = weightWrites();
    await llm.reconfigure({ computeType: 'float16', chunkSize: 16 });
    expect(llm.variant.id).toBe('staged-r8-float16');
    expect(weightWrites()).toBe(before);
    expect((await run(llm, prompt.slice(0, 21), 3)).length).toBe(3);
    await expect(llm.reconfigure({ kvStyle: 'onehot' })).rejects.toThrow(/staged/);
    await expect(llm.reconfigure({ blocksPerGraph: 7 })).rejects.toThrow(/block groups/);
    await expect(llm.reconfigure({ prefillBlocksPerGraph: 2 })).rejects.toThrow(/decode only/);
  });

  it('loadModel picks the runtime from the plan; the standard runtime refuses a Qwen3.5 package', async () => {
    installMockBuilder();
    const model = await loadModel(fsSource, { context: new MockContext(), cache: false });
    expect(model).toBeInstanceOf(Qwen35LLM);
    await expect(LLM.load(fsSource, { context: new MockContext(), cache: false })).rejects.toThrow();
  });

  it('rejects prompts that do not fit; lists the graph build times', async () => {
    const { llm } = await loaded({ chunkSize: 16 });
    await expect(async () => { for await (const _ of llm.generate(Array(127).fill(5))) void _; }).rejects.toThrow(/KV cache holds 127/);
    const labels = llm.stats.graphBuilds.map((g) => g.label);
    expect(labels.filter((l) => l.startsWith('decode block'))).toHaveLength(2);
    expect(labels).toContain('decode block linear_attention');
    expect(labels).toContain('prefill c16 block full_attention');
    expect(labels[0]).toBe('lm_head');
  });
});

describe('Qwen3.5 decode block groups', () => {
  // the tiny model has 5 BLOCKs (the attention halves of layers 1..5): linear, full, linear, linear, full
  const stateIndex = (llm: Qwen35LLM) => {
    const x = internals(llm), at = (t: any) => x.stateTensors.indexOf(t);
    return {
      kv: [...x.kv.entries()].map(([l, c]: any) => [l, at(c.key), at(c.value)]), stg: [...x.stg.entries()].map(([l, c]: any) => [l, at(c.key), at(c.value)]),
      state: [...x.state.entries()].map(([l, s]: any) => [l, at(s.conv), at(s.rec)]),
    };
  };

  it.each([2, 3, 5])('%i blocks per graph: one dispatch per group, the same tokens and the same buffers (caches, staging, states) as without groups', async (group) => {
    const plain = await loaded({ chunkSizes: [16, 32] }), grouped = await loaded({ chunkSizes: [16, 32], blocksPerGraph: group });
    const a = await run(plain.llm, prompt, 12), b = await run(grouped.llm, prompt, 12);
    expect(b).toEqual(a);
    const steps = 11, groups = Math.ceil(5 / group), plan = planPrefill([16, 32], 45, 200), prefill = (plan.length - 1) * N + (N + 2);
    expect(grouped.context.dispatches.length - plain.context.dispatches.length).toBe(steps * (groups - 5));   // HEAD + groups + TAIL + lm_head per decode step; prefill and flushes are as before
    expect(plain.context.dispatches.length).toBeGreaterThan(prefill);
    expect(stateIndex(grouped.llm)).toEqual(stateIndex(plain.llm));   // every state / cache and its spare were swapped exactly as often
    const labels = grouped.llm.stats.graphBuilds.map((g) => g.label);
    expect(labels.some((l) => l.startsWith('decode block'))).toBe(false);
    expect(labels.filter((l) => l.startsWith('decode group'))).toHaveLength(group === 2 ? 3 : group === 3 ? 2 : 1);
  });

  it('binds a group by input name: shared controls once, weights / states / caches with the prefix of their block, the spare tensors as outputs', async () => {
    const { llm, context } = await loaded({ chunkSizes: [16, 32], blocksPerGraph: 5 });
    await run(llm, prompt, 2);
    const x = internals(llm), prefill = (planPrefill([16, 32], 45, 200).length - 1) * N + (N + 2);
    const [head, g0] = context.dispatches.slice(prefill, prefill + 2);   // the decode step: HEAD, the one group of 5 blocks, TAIL
    expect(g0.inputs.residual).toBe(head.outputs.residual);
    expect(g0.inputs.attention_output).toBe(head.outputs.attention_output);
    expect(g0.inputs.rope_cos).toBe(x.dec.ropeCos);
    expect(g0.inputs.attention_mask).toBe(x.dec.mask);
    expect(Object.keys(g0.inputs).filter((n) => n.endsWith('rope_cos'))).toEqual(['rope_cos']);
    expect(g0.inputs.b0_conv_state).toBe(x.stateSpare.get(1).conv);              // block 0 = the attention half of layer 1 (linear); the step has swapped its state with the spare since
    expect(g0.inputs.b1_past_key).toBe(x.kv.get(2).key);                         // block 1 = layer 2 (full)
    expect(g0.inputs.b1_staging_value).toBe(x.stgSpare.get(2).value);
    expect(g0.inputs.b2_qkvz_projection_weight).toBe(x.weights[3].qkvz_projection_weight);
    expect(g0.inputs.b2_gate_up_projection_weight).toBe(x.weights[2].gate_up_projection_weight);   // the feed-forward half of layer 2, the attention half of layer 3
    expect(g0.outputs.b1_updated_key).toBe(x.stg.get(2).key);                    // the outputs are now the current tensors
    expect(g0.outputs.b0_new_conv_state).toBe(x.state.get(1).conv);
    expect(g0.outputs.b0_new_recurrent_state.shape).toEqual([1, 4, 128, 128]);
  });
});

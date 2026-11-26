import { readFileSync } from 'node:fs';
import { beforeAll, describe, expect, it } from 'vitest';
import { Qwen35LLM } from './llm-qwen35.js';
import type { LoadOptions } from './llm.js';
import type { PackageSource } from './package.js';
import { chooseGrid, expandImageTokens, mropePositions, planPrefill, ropeTables, ropeTablesMrope, type LayoutSpec } from './protocol.js';
import { installMockBuilder, MockContext, type MockTensor } from './mock-ml.js';
import { pixelsFromRGBA, prepareImage, patchesOf, VisionTower, type PreparedImage } from './vision.js';

// The committed tiny Qwen3.5 package with its tiny image tower (patch 8, 2 blocks, hidden 128, grid presets 4x4 / 4x6 / 6x4; python/tests/tiny_qwen35.py): the vision graphs, weights, tables, the
// embedding injection and the MRoPE tables are checked against a validating mock of WebNN (names, shapes and dtypes of every dispatch), the numbers against python/tests/test_qwen35_images.py.
const root = new URL('../../tests/golden/tiny-qwen35-package/', import.meta.url);
const fsSource: PackageSource = {
  id: 'tiny-qwen35-vision',
  text: async (path) => readFileSync(new URL(path, root), 'utf8'),
  bytes: async (path) => { const b = readFileSync(new URL(path, root)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); },
};
const PAD = 290, START = 288, END = 289;

// the same package with its image tower in the int8g128 weight format (scripts/make-tiny-vision-overlay.py): the plan's vision section, the vision_* manifest entries and the vision files come from the overlay
const overlayRoot = new URL('../../tests/golden/tiny-qwen35-vision-int8g128/', import.meta.url);
const overlay = JSON.parse(readFileSync(new URL('vision.json', overlayRoot), 'utf8'));
const g128Source: PackageSource = {
  id: 'tiny-qwen35-vision-g128',
  text: async (path) => {
    if (path.startsWith('graphs/vision/')) return readFileSync(new URL(path, overlayRoot), 'utf8');
    const text = await fsSource.text(path);
    if (path === 'plan.json') return JSON.stringify({ ...JSON.parse(text), vision: overlay.plan });
    if (path === 'weights/manifest.json') {
      const manifest = JSON.parse(text);
      return JSON.stringify({ ...manifest, quant: { ...manifest.quant, vision: overlay.quant }, files: [...manifest.files.filter((f: { role: string }) => !f.role.startsWith('vision_')), ...overlay.files] });
    }
    return text;
  },
  bytes: async (path) => {
    if (!path.startsWith('weights/vision_')) return fsSource.bytes(path);
    const b = readFileSync(new URL(path, overlayRoot));
    return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
  },
};

async function loaded(options: LoadOptions = {}) {
  installMockBuilder();
  const context = new MockContext();
  const llm = await Qwen35LLM.load(fsSource, { context, cache: false, blocksPerGraph: 1, chunkSizes: [16], ...options });
  return { llm, context };
}
const f16 = (t: { data: ArrayBufferView }) => new (globalThis as any).Float16Array(t.data.buffer, t.data.byteOffset, t.data.byteLength / 2);
const fakeImage = (grid: [number, number], seed = 1): PreparedImage => {
  const patches = new (globalThis as any).Float16Array(grid[0] * grid[1] * 384);
  for (let i = 0; i < patches.length; i++) patches[i] = Math.sin(i * seed) * 0.5;
  return { patches, grid, tokens: (grid[0] * grid[1]) / 4, width: grid[1] * 8, height: grid[0] * 8 };
};
async function run(llm: Qwen35LLM, tokens: number[], images: PreparedImage[], maxNewTokens: number) {
  const out: number[] = [];
  for await (const token of llm.generate(tokens, { maxNewTokens, stopTokens: [], images, temperature: 0 })) out.push(token);
  return out;
}

describe('Qwen3.5 image tower on a validating mock of WebNN', () => {
  beforeAll(() => { if (!(globalThis as any).Float16Array) throw new Error('needs Node 24+'); });

  it('is not loaded unless asked for: a text-only model has no vision tensors, and images are refused', async () => {
    const { llm, context } = await loaded();
    expect(llm.vision).toBeNull();
    const withVision = await loaded({ vision: true });
    expect(withVision.context.tensors.length).toBeGreaterThan(context.tensors.length + 40);   // 2 x 18 block tensors, head, merger, patches, hidden states ...
    context.onRead = (_t, d) => { d.fill(0); d[7] = 5; };
    await expect(run(llm, [5, PAD, 6], [fakeImage([4, 4])], 1)).rejects.toThrow(/without the image tower/);
    expect(() => llm.prepareImage({ width: 10, height: 10 })).toThrow(/without the image tower/);
  });

  it('loads the weights of every block, builds the graphs of a preset on first use, and dispatches HEAD, the blocks and MERGER with every input bound by name', async () => {
    const { llm, context } = await loaded({ vision: { grids: [[4, 4], [4, 6]] } });
    const tower = llm.vision!;
    expect(tower.grids).toEqual([[4, 4], [4, 6]]);   // the candidates a picture may get
    expect(tower.loadedGrids).toEqual([]);   // nothing is built yet
    const messages: string[] = [];
    tower.progress = (p) => { if (p.stage === 'build') messages.push(p.message ?? ''); };
    const rowsOf = (shape: number[]) => shape.join('x');
    // weights: 2 blocks x 18 tensors + head (patch embed: int8 weight + scale + bias) + merger, as the manifest lists them
    expect(context.tensors.filter((t) => rowsOf(t.shape) === '384x128' && t.dataType === 'int8').length).toBe(2);   // qkv of the two blocks
    context.onRead = (tensor: MockTensor, d: any) => { d.fill(0.25); void tensor; };
    const image = fakeImage([4, 6]);
    const before = context.dispatches.length;
    const rows = await tower.encode(image);
    const dispatched = context.dispatches.slice(before);
    expect(dispatched.map((d) => Object.keys(d.outputs)[0])).toEqual(['hidden_states', 'next_hidden_states', 'next_hidden_states', 'image_embeds']);
    expect(rows.length).toBe(6 * 128);
    expect(dispatched[0].inputs.patches.shape).toEqual([24, 384]);
    expect(dispatched[3].outputs.image_embeds.shape).toEqual([1, 6, 128]);
    // the same block graph for both blocks, with the weights of layer 0 and 1 bound; the hidden state chains output -> input
    expect(dispatched[1].graph).toBe(dispatched[2].graph);
    expect(dispatched[1].inputs.qkv_weight).not.toBe(dispatched[2].inputs.qkv_weight);
    expect(dispatched[2].inputs.hidden_states).toBe(dispatched[1].outputs.next_hidden_states);
    expect(dispatched[3].inputs.hidden_states).toBe(dispatched[2].outputs.next_hidden_states);
    expect(tower.loadedGrids).toEqual([[4, 6]]);
    expect(messages).toEqual(['build vision 4x6 head', 'build vision 4x6 block', 'build vision 4x6 merger']);   // built on first use, once
    await tower.encode(image);
    expect(messages.length).toBe(3);
    expect(tower.lastEncode.buildMs).toBe(0);
    await tower.encode(fakeImage([6, 4]));   // not a candidate, but a preset of the package: built on demand
    expect(tower.loadedGrids).toEqual([[4, 6], [6, 4]]);
    await expect(tower.encode(fakeImage([8, 8]))).rejects.toThrow(/no image preset 8x8/);
    await expect(tower.ensure([4, 4])).resolves.toBeGreaterThanOrEqual(0);
    expect(tower.buildMs).toBeGreaterThanOrEqual(0);
  });

  it('int8g128 weights: int8 blocks [in/128, out, 128] with a block scale are uploaded and bound by name to every dispatch (same loader as the other formats)', async () => {
    installMockBuilder();
    const context = new MockContext();
    const llm = await Qwen35LLM.load(g128Source, { context, cache: false, blocksPerGraph: 1, chunkSizes: [16], vision: { grids: [[4, 4]] } });
    expect(llm.pkg.plan.vision!.weights).toBe('int8g128');
    const shapes = (dataType: string) => context.tensors.filter((t) => t.dataType === dataType).map((t) => t.shape.join('x'));
    expect(shapes('int8').filter((s) => s === '1x384x128').length).toBe(2);   // qkv of the two blocks: one block of 128 input channels, 384 output rows
    expect(shapes('int8').filter((s) => s === '2x128x128').length).toBe(2);   // fc2: ffn 256 = 2 groups
    context.onRead = (_t: MockTensor, d: any) => { d.fill(0.25); };
    const before = context.dispatches.length;
    expect((await llm.vision!.encode(fakeImage([4, 4]))).length).toBe(4 * 128);
    const dispatched = context.dispatches.slice(before);
    expect(dispatched.length).toBe(4);
    expect(dispatched[1].inputs.qkv_weight.dataType).toBe('int8');
    expect(dispatched[1].inputs.qkv_scale.shape).toEqual([1, 384, 1]);
  });

  it('debug mode: the hidden state after HEAD and after every block is read back, every dispatch is named first and awaited', async () => {
    VisionTower.debugReadback = true;
    try {
      const { llm, context } = await loaded({ vision: { grids: [[4, 4]] } });
      context.onRead = (_t: MockTensor, d: any) => { d.fill(0.5); };
      const labels: string[] = [];
      llm.vision!.onDispatch = (label) => { labels.push(label); };
      await llm.vision!.encode(fakeImage([4, 4]));
      expect(labels).toEqual(['1/4 4x4_head', '2/4 4x4_block layer 0', '3/4 4x4_block layer 1', '4/4 4x4_merger']);
      expect(llm.vision!.trace.length).toBe(3);   // HEAD + 2 blocks
      expect(llm.vision!.trace[0].length).toBe(16 * 128);
    } finally { VisionTower.debugReadback = false; }
  });

  it('preload builds presets at load; concurrent first uses share one build', async () => {
    const { llm, context } = await loaded({ vision: { preload: [[4, 4]] } });
    expect(llm.vision!.loadedGrids).toEqual([[4, 4]]);
    expect(llm.vision!.grids).toEqual([[4, 4], [4, 6], [6, 4]]);
    const graphs = context.dispatches.length;
    context.onRead = (_t: MockTensor, d: any) => { d.fill(0.25); };
    await Promise.all([llm.vision!.encode(fakeImage([4, 6])), llm.vision!.encode(fakeImage([4, 6]))]);
    expect(llm.vision!.loadedGrids).toEqual([[4, 4], [4, 6]]);
    expect(context.dispatches.length - graphs).toBe(8);
  });

  it('chooses the preset automatically from the size of the picture and the detail level (prepareImage, llm.prepareImage)', async () => {
    const { llm } = await loaded({ vision: true });
    class FakeCanvas {
      constructor(public width: number, public height: number) {}
      getContext() { return { drawImage() {}, getImageData: (_x: number, _y: number, w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4).fill(128) }) }; }
    }
    (globalThis as any).OffscreenCanvas = FakeCanvas;
    try {
      // patch 8: the tiny package's presets are 4x4 / 4x6 / 6x4 (16 / 24 / 24 patches)
      expect(llm.prepareImage({ width: 1000, height: 1000 }).grid).toEqual([4, 4]);
      expect(llm.prepareImage({ width: 1500, height: 1000 }).grid).toEqual([4, 6]);
      expect(llm.prepareImage({ width: 1000, height: 1500 }, undefined, 'low').grid).toEqual([6, 4]);
      const explicit = llm.prepareImage({ width: 10, height: 10 }, [6, 4]);
      expect(explicit.grid).toEqual([6, 4]); expect(explicit.detail).toBeUndefined();
      expect(llm.prepareImage({ width: 50, height: 50 }, undefined, 'high').detail).toBe('high');
    } finally { delete (globalThis as any).OffscreenCanvas; }
  });

  it('rejects a preset the package does not have', async () => {
    await expect(loaded({ vision: { grids: [[8, 8]] } })).rejects.toThrow(/no image preset 8x8/);
  });

  it('writes the image tokens into the prefill chunks (straddling a chunk boundary) and the interleaved MRoPE tables; decode uses index + rope_delta', async () => {
    const { llm, context } = await loaded({ vision: { grids: [[4, 6]] }, chunkSizes: [16] });
    const x = llm as any, model = llm.pkg.model, hidden = model.hidden;
    // the image tower returns row k = k + 1 in every channel; the logits are a pattern (argmax 7)
    context.onRead = (tensor: MockTensor, d: any) => {
      if (tensor.shape.length === 3 && tensor.shape[1] === 6 && tensor.shape[2] === 128) for (let k = 0; k < 6; k++) d.fill(k + 1, k * 128, (k + 1) * 128);
      else if (tensor.shape[tensor.shape.length - 1] === model.vocab_padded) { d.fill(0); d[7] = 5; } else d.fill(0);
    };
    const short = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, START, PAD, END, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];   // 12 text, start, pad, end, 10 text
    const image = fakeImage([4, 6]);
    const first = context.writes.length;
    const generated = await run(llm, short, [image], 3);
    const writes = context.writes.slice(first);   // the host writes of this generation (the weights were written at load)
    expect(generated).toEqual([7, 7, 7]);
    const expanded = expandImageTokens({ image_token_id: PAD, merge_size: 2 }, short, [image.grid]);
    expect(expanded.length).toBe(short.length - 1 + 6);
    expect(llm.stats.promptTokens).toBe(expanded.length);
    expect(llm.stats.imageTokens).toBe(6);
    expect(llm.stats.prefillPlan).toEqual(planPrefill([16], expanded.length, 200));
    expect(llm.stats.prefillPlan.length).toBe(2);   // 30 tokens: two chunks of 16; the image tokens (rows 13..18) straddle the boundary at 16
    // the chunk tensors written (hidden = [1, 16, hidden]) in order
    const chunks = writes.filter((w) => w.tensor.shape.join() === `1,16,${hidden}`).map((w) => f16(w));
    expect(chunks.length).toBe(2);
    const imageAt = expanded.flatMap((t, i) => (t === PAD ? [i] : []));   // 13..18
    expect(imageAt).toEqual([13, 14, 15, 16, 17, 18]);
    const rowOf = (position: number) => { const chunk = position < 16 ? 0 : 1, r = position < 16 ? position : position - 16 + (32 - expanded.length); return chunks[chunk].subarray(r * hidden, (r + 1) * hidden); };
    // the second chunk holds 14 real rows (positions 16..29), left padded by 2
    imageAt.forEach((position, k) => expect(Array.from(rowOf(position)).every((v) => v === k + 1)).toBe(true));
    expect(Array.from(rowOf(12)).some((v) => v !== 0 && Number.isInteger(v))).toBe(false);   // a text row is an embedding, not an image row
    // rope tables of the two chunks == ropeTablesMrope of the positions of their rows
    const spec = model as unknown as LayoutSpec & { mrope: { section: number[] } };
    const { positions, ropeDelta } = mropePositions({ image_token_id: PAD, merge_size: 2 }, expanded, [image.grid]);
    const cosWrites = writes.filter((w) => w.tensor.shape.join() === `1,1,16,${model.head_dim}`).map((w) => f16(w));   // cos, sin of chunk 0, cos, sin of chunk 1
    const want0 = ropeTablesMrope(spec, positions.map((p) => p.subarray(0, 16)), 16, 0).full_attention, want1 = ropeTablesMrope(spec, positions.map((p) => p.subarray(16, 30)), 16, 2).full_attention;
    expect(Array.from(cosWrites[0])).toEqual(Array.from(want0.cos)); expect(Array.from(cosWrites[1])).toEqual(Array.from(want0.sin));
    expect(Array.from(cosWrites[2])).toEqual(Array.from(want1.cos)); expect(Array.from(cosWrites[3])).toEqual(Array.from(want1.sin));
    // decode: the rope position of the token at sequence index 30 is 30 + ropeDelta (a 1-row table)
    const decodeRope = writes.filter((w) => w.tensor.shape.join() === `1,1,1,${model.head_dim}`).map((w) => f16(w));
    const wantStep = ropeTables(spec, [expanded.length + ropeDelta], 1, 0).full_attention;
    expect(Array.from(decodeRope[0])).toEqual(Array.from(wantStep.cos)); expect(Array.from(decodeRope[1])).toEqual(Array.from(wantStep.sin));
    expect(ropeDelta).not.toBe(0);
    expect(x.ropeDelta).toBe(ropeDelta);
  });

  it('a text-only prompt after an image prompt starts with ropeDelta 0 again', async () => {
    const { llm, context } = await loaded({ vision: { grids: [[4, 4]] } });
    context.onRead = (_t: MockTensor, d: any) => { d.fill(0); d[7] = 5; };
    await run(llm, [5, START, PAD, END, 6], [fakeImage([4, 4])], 1);
    expect((llm as any).ropeDelta).not.toBe(0);
    await run(llm, [5, 6, 7], [], 1);
    expect((llm as any).ropeDelta).toBe(0);
  });
});

describe('image prompt', () => {
  it('writes the image prompt in front of the text of a turn with pictures (once per picture)', async () => {
    const { buildPrompt } = await import('./chat.js');
    const format = { template_kwargs: {}, first_prefix: '<|im_start|>user\n', first_prefix_with_system: '', user_suffix_generation_prompt: '<|im_end|>\n<|im_start|>assistant\n', assistant_suffix: '', next_user_prefix: '', user_suffix_assistant_prefix: '' };
    const marker = '<|vision_start|><|image_pad|><|vision_end|>';
    expect(buildPrompt(format, [{ user: 'What is this?', images: 1 }], undefined, new Date(0), marker)).toBe(`<|im_start|>user\n${marker}What is this?<|im_end|>\n<|im_start|>assistant\n`);
    expect(buildPrompt(format, [{ user: 'Compare', images: 2 }], undefined, new Date(0), marker)).toContain(marker + marker + 'Compare');
    expect(buildPrompt(format, [{ user: 'plain' }], undefined, new Date(0), marker)).toBe('<|im_start|>user\nplain<|im_end|>\n<|im_start|>assistant\n');
  });
});

describe('image preprocessing', () => {
  const vision = { image_mean: [0.5, 0.5, 0.5], image_std: [0.5, 0.5, 0.5], rescale_factor: 1 / 255, patch_size: 8, temporal_patch_size: 2, merge_size: 2, channels: 3 } as any;
  it('rescales and normalises RGBA bytes into [3, H, W]', () => {
    const rgba = new Uint8ClampedArray([0, 255, 51, 255, 255, 0, 102, 255]);   // 2 pixels
    const pixels = pixelsFromRGBA(rgba, 2, 1, vision);
    expect(Array.from(pixels).map((v) => +v.toFixed(4))).toEqual([-1, 1, 1, -1, -0.6, -0.2]);
    expect(() => pixelsFromRGBA(rgba, 3, 1, vision)).toThrow(/bytes/);
  });

  it('resizes a picture on a canvas to the grid of the best preset and patchifies it', () => {
    const calls: { size: number[]; draw?: number[] } = { size: [] };
    class FakeCanvas {
      constructor(public width: number, public height: number) { calls.size = [width, height]; }
      getContext() {
        return { drawImage: (_i: unknown, x: number, y: number, w: number, h: number) => { calls.draw = [x, y, w, h]; },
          getImageData: (_x: number, _y: number, w: number, h: number) => ({ data: new Uint8ClampedArray(w * h * 4).fill(255) }) };
      }
    }
    (globalThis as any).OffscreenCanvas = FakeCanvas;
    try {
      const prepared = prepareImage({ width: 1500, height: 1000 }, vision, [[4, 4], [4, 6], [6, 4]]);   // 3:2 landscape -> 4x6 patches
      expect(prepared.grid).toEqual([4, 6]);
      expect(calls.size).toEqual([48, 32]);   // width 6 * 8, height 4 * 8
      expect(calls.draw).toEqual([0, 0, 48, 32]);
      expect(prepared.patches.length).toBe(24 * 384);
      expect(prepared.tokens).toBe(6);
      expect(Array.from(prepared.patches).every((v) => v === 1)).toBe(true);   // all-white image: (1 - 0.5) / 0.5
      expect(prepareImage({ width: 100, height: 100 }, vision, [[4, 4], [4, 6]], [4, 6]).grid).toEqual([4, 6]);   // an explicit grid wins
      expect(() => prepareImage({ width: 10, height: 10 }, vision, [])).toThrow(/no image grid/);
    } finally { delete (globalThis as any).OffscreenCanvas; }
    expect(patchesOf(new Float32Array(3 * 32 * 48), [4, 6], vision).length).toBe(24 * 384);
    expect(chooseGrid([[4, 4], [4, 6], [6, 4]], 1000, 1500, 8)).toEqual([6, 4]);
  });
});

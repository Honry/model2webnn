// The image tower of a Qwen3.5 package on WebNN (docs/models/qwen35-vision.md): preprocessing of an image in the browser (canvas resize to a grid preset, rescale / normalise, patchify) and the vision
// graphs of a preset: HEAD (patch matmul + position embedding), BLOCK x depth (ONE graph, the weights of layer i bound per dispatch), MERGER -> the image tokens as float16 rows.
// Python counterpart (the executable specification): python/m2w/reference/qwen35_vision.py + protocol.prepare_image.
import { loadGraph, type GraphDoc } from './ir-loader.js';
import { checkTensorSize } from './llm-common.js';
import type { ManifestFile, ModelPackage, Progress, VisionPlan, VisionSpecJson } from './package.js';
import { chooseGrid, imageTokenCount, newHalf, patchify, type Detail, type Grid, type Half } from './protocol.js';

/* eslint-disable @typescript-eslint/no-explicit-any */
type ML = any;

/** An image ready for the image tower: its grid preset and the patches (float16, [grid_h * grid_w * patch_vector]); `tokens` = the image tokens it becomes. */
export interface PreparedImage { patches: Half; grid: Grid; tokens: number; width: number; height: number; detail?: Detail }

/** rescale + normalise RGBA bytes (canvas ImageData) into [3, height, width] float32: (value * rescale_factor - mean) / std */
export function pixelsFromRGBA(rgba: ArrayLike<number>, width: number, height: number, vision: Pick<VisionSpecJson, 'image_mean' | 'image_std' | 'rescale_factor'>): Float32Array {
  const plane = width * height, out = new Float32Array(3 * plane);
  if (rgba.length !== 4 * plane) throw new Error(`pixelsFromRGBA: ${rgba.length} bytes for ${width} x ${height}`);
  for (let c = 0; c < 3; c++) {
    const mean = vision.image_mean[c], inverse = 1 / vision.image_std[c], scale = vision.rescale_factor;
    for (let i = 0; i < plane; i++) out[c * plane + i] = (rgba[4 * i + c] * scale - mean) * inverse;
  }
  return out;
}

/** patches of already normalised pixels [3, H, W] of the exact size of `grid` as float16 */
export function patchesOf(pixels: Float32Array, grid: Grid, vision: VisionSpecJson): Half {
  const patches = patchify(pixels, grid, vision.patch_size, vision.temporal_patch_size, vision.merge_size, vision.channels), half = newHalf(patches.length);
  half.set(patches);
  return half;
}

/** The browser's preprocessing: the preset for the picture (chooseGrid among `grids`: size, aspect ratio, `detail`), the picture resized to exactly grid * patch pixels on a canvas, normalised, patchified.
 *  `image`: anything drawImage takes (ImageBitmap, HTMLImageElement, a canvas ...). The resize is the canvas's (smooth), the Python reference uses PIL's bicubic: the vision graphs are verified on the SAME pixels. */
export function prepareImage(image: { width: number; height: number }, vision: VisionSpecJson, grids: Grid[], grid?: Grid, detail: Detail = 'auto'): PreparedImage {
  if (!grids.length) throw new Error('no image grid preset to choose from');
  const width = (image as any).naturalWidth ?? image.width, height = (image as any).naturalHeight ?? image.height;
  const chosen = grid ?? chooseGrid(grids, width, height, vision.patch_size, detail), w = chosen[1] * vision.patch_size, h = chosen[0] * vision.patch_size;
  const canvas: any = typeof OffscreenCanvas !== 'undefined' ? new OffscreenCanvas(w, h) : Object.assign(document.createElement('canvas'), { width: w, height: h });
  const context = canvas.getContext('2d', { willReadFrequently: true });
  context.imageSmoothingEnabled = true; context.imageSmoothingQuality = 'high';
  context.drawImage(image as any, 0, 0, w, h);
  const pixels = pixelsFromRGBA(context.getImageData(0, 0, w, h).data, w, h, vision);
  return { patches: patchesOf(pixels, chosen, vision), grid: chosen, tokens: imageTokenCount(chosen, vision.merge_size), width, height, ...(grid ? {} : { detail }) };
}

interface Preset { grid: Grid; tokens: number; head: Built; block: Built; merger: Built; patches: ML; hidden: ML[]; embeds: ML; tables: ManifestFile }
interface Built { name: string; graph: ML; inputs: string[]; outputs: string[] }
const key = (grid: Grid) => `${grid[0]}x${grid[1]}`;

export class VisionTower {
  /** debugging (bench/raw/qwen35vision.html): keep the hidden state after HEAD and after every block readable so that debugTrace() can read them back; set BEFORE the model is loaded */
  static debugReadback = false;
  /** the presets a picture may be given (chooseGrid picks among them): all the package has, or the restriction passed to create() */
  readonly grids: Grid[];
  /** weightsMs: the time to load the weights; buildMs: the total time spent building graphs (a preset is built on its first use: ensure / encode) */
  buildMs = 0; weightsMs = 0;
  lastEncode = { hostMs: 0, deviceMs: 0, buildMs: 0 };
  /** called with { stage: 'build', message: 'build vision 24x16' } while the graphs of a preset are built; replaceable (a page shows it as a status) */
  progress: Progress;
  private presets = new Map<string, Preset>();
  private building = new Map<string, Promise<Preset>>();
  private head: Record<string, ML> = {}; private blocks: Record<string, ML>[] = []; private merger: Record<string, ML> = {};
  private tensors: ML[] = []; private graphs: ML[] = [];
  private busy: Promise<unknown> = Promise.resolve();

  private constructor(private ctx: ML, private pkg: ModelPackage, private plan: VisionPlan, readonly vision: VisionSpecJson, grids: Grid[], progress: Progress) { this.grids = grids; this.progress = progress; }

  /** Loads the weights of the image tower to the device. The graphs of a grid preset are built LAZILY on its first use (about two seconds per preset on a device, the weights are shared); `grids` restricts
   *  the presets a picture may get (default: all the package has), `preload` builds some now. */
  static async create(ctx: ML, pkg: ModelPackage, options: { grids?: Grid[]; preload?: Grid[] } = {}, progress: Progress = () => {}): Promise<VisionTower> {
    const plan = pkg.plan.vision, vision = pkg.model.vision;
    if (!plan || !vision) throw new Error('this package has no image tower (compile it from a checkpoint with a vision tower, or `m2w add-vision`)');
    const have = plan.presets.map((p) => p.grid), wanted = options.grids?.length ? options.grids : have;
    for (const grid of [...wanted, ...(options.preload ?? [])]) if (!have.some((g) => g[0] === grid[0] && g[1] === grid[1])) throw new Error(`the package has no image preset ${key(grid)} (it has ${have.map(key).join(', ')})`);
    const tower = new VisionTower(ctx, pkg, plan, vision, wanted.map((g) => [g[0], g[1]] as Grid), progress);
    await tower.load(progress);
    for (const grid of options.preload ?? []) await tower.ensure(grid);
    return tower;
  }

  /** the presets whose graphs are built */
  get loadedGrids(): Grid[] { return [...this.presets.values()].map((p) => p.grid); }

  private async tensor(dataType: string, shape: number[], flags: { writable?: boolean; readable?: boolean } = {}) {
    checkTensorSize(dataType, shape);
    const tensor = await this.ctx.createTensor({ dataType, shape, writable: flags.writable ?? false, readable: flags.readable ?? false });
    this.tensors.push(tensor);
    return tensor;
  }

  private async upload(shard: ManifestFile): Promise<Record<string, ML>> {
    const buffer = await this.pkg.source.bytes(`weights/${shard.file}`, undefined, shard.bytes), out: Record<string, ML> = {};
    for (const entry of shard.tensors) {
      const tensor = await this.tensor(entry.dtype, entry.shape, { writable: true });
      this.ctx.writeTensor(tensor, new Uint8Array(buffer, entry.offset, entry.bytes));
      out[entry.name] = tensor;
    }
    return out;
  }

  private async load(progress: Progress) {
    const t0 = performance.now(), files = this.pkg.manifest.files, role = (r: ManifestFile['role']) => files.filter((f) => f.role === r);
    progress({ stage: 'vision', message: 'loading the image tower weights' });
    this.head = await this.upload(role('vision_head')[0]); this.merger = await this.upload(role('vision_merger')[0]);
    const blocks = role('vision_block').sort((a, b) => (a.layer as number) - (b.layer as number));
    if (blocks.length !== this.plan.depth) throw new Error(`the package lists ${blocks.length} vision blocks, the plan says ${this.plan.depth}`);
    for (const shard of blocks) { progress({ stage: 'vision', message: `vision block ${(shard.layer as number) + 1}/${blocks.length}`, loaded: shard.layer as number, total: blocks.length }); this.blocks.push(await this.upload(shard)); }
    this.weightsMs = performance.now() - t0;
  }

  /** builds the graphs of a preset if that has not happened yet (concurrent callers share one build); resolves with the milliseconds this call waited for a build (0 when it was there) */
  async ensure(grid: Grid): Promise<number> {
    const k = key(grid);
    if (this.presets.has(k)) return 0;
    const t0 = performance.now();
    let pending = this.building.get(k);
    if (!pending) {
      pending = this.buildPreset(grid, this.progress).then((p) => { this.presets.set(k, p); return p; }).finally(() => this.building.delete(k));
      this.building.set(k, pending);
    }
    await pending;
    const ms = performance.now() - t0;
    this.buildMs += ms;
    return ms;
  }

  private async buildPreset(grid: Grid, progress: Progress): Promise<Preset> {
    const entry = this.plan.presets.find((p) => p.grid[0] === grid[0] && p.grid[1] === grid[1]), v = this.vision;
    if (!entry) throw new Error(`the package has no image preset ${key(grid)} (it has ${this.plan.presets.map((p) => key(p.grid)).join(', ')})`);
    const tables = this.pkg.manifest.files.find((f) => f.role === 'vision_grid' && f.grid?.[0] === grid[0] && f.grid?.[1] === grid[1]);
    if (!tables) throw new Error(`the package has no vision_grid tables for ${key(grid)}`);
    const tableBytes = await this.pkg.source.bytes(`weights/${tables.file}`, undefined, tables.bytes);
    const external = (c: { external?: string }) => {
      const t = tables.tensors.find((x) => x.name === c.external);
      if (!t) throw new Error(`the vision graph wants the external tensor '${c.external}' which the tables of ${key(grid)} do not have`);
      return new Uint8Array(tableBytes, t.offset, t.bytes);
    };
    const build = async (path: string, label: string): Promise<Built> => {
      progress({ stage: 'build', message: `build ${label}` });
      const doc: GraphDoc = JSON.parse(await this.pkg.source.text(path)), builder = new (globalThis as any).MLGraphBuilder(this.ctx);
      const { inputs, outputs } = loadGraph(builder, doc, external);
      const graph = await builder.build(outputs);
      this.graphs.push(graph);
      return { name: doc.name, graph, inputs: Object.keys(inputs), outputs: Object.keys(outputs) };
    };
    const head = await build(entry.graphs.head, `vision ${key(grid)} head`), block = await build(entry.graphs.block, `vision ${key(grid)} block`), merger = await build(entry.graphs.merger, `vision ${key(grid)} merger`);
    const n = grid[0] * grid[1], readable = { readable: VisionTower.debugReadback };
    const patches = await this.tensor('float16', [n, v.channels * v.temporal_patch_size * v.patch_size ** 2], { writable: true });
    // the hidden state after HEAD and after every block: all of them when debugging, else two tensors that take turns
    const count = VisionTower.debugReadback ? this.plan.depth + 1 : 2, hidden: ML[] = [];
    for (let i = 0; i < count; i++) hidden.push(await this.tensor('float16', [1, n, v.hidden], readable));
    const embeds = await this.tensor('float16', [1, entry.tokens, v.out_hidden], { readable: true });
    return { grid, tokens: entry.tokens, head, block, merger, patches, hidden, embeds, tables };
  }

  private bind(graph: Built, values: Record<string, ML>, outputs = false) {
    const out: Record<string, ML> = {};
    for (const name of outputs ? graph.outputs : graph.inputs) {
      if (values[name] === undefined) throw new Error(`graph ${graph.name}: no tensor is bound to ${outputs ? 'output' : 'input'} '${name}'`);
      out[name] = values[name];
    }
    return out;
  }

  /** the preset for a picture of this size (chooseGrid among the candidates: size, aspect ratio, detail) */
  chooseGrid(width: number, height: number, detail: Detail = 'auto'): Grid { return chooseGrid(this.grids, width, height, this.vision.patch_size, detail); }

  /** preprocess a picture (canvas) for this tower: the preset is chosen automatically among the candidates (or given) */
  prepare(image: { width: number; height: number }, grid?: Grid, detail: Detail = 'auto'): PreparedImage { return prepareImage(image, this.vision, this.grids, grid, detail); }

  /** The image tokens of a prepared image: HEAD, the blocks, MERGER on the device, the rows read back. -> float16 [tokens * out_hidden]. With debugReadback the hidden states are in `trace` afterwards. */
  encode(image: PreparedImage): Promise<Half> {
    const run = this.busy.then(() => this.encodeNow(image));
    this.busy = run.catch(() => {});
    return run;
  }

  /** debugging: the hidden state after HEAD and after every block of the last encode (needs VisionTower.debugReadback) */
  trace: Half[] = [];
  /** debugging (needs debugReadback): called with the name of every dispatch BEFORE it is queued; the dispatches then run one by one, each followed by a readback, so that after a crash / lost context the
   *  last name is the graph that caused it */
  onDispatch: ((label: string) => void | Promise<void>) | null = null;

  private async encodeNow(image: PreparedImage): Promise<Half> {
    const buildMs = await this.ensure(image.grid), p = this.presets.get(key(image.grid)) as Preset;
    const t0 = performance.now(), depth = this.plan.depth, names = this.plan.tensors;
    const pick = (weights: Record<string, ML>, list: string[]) => Object.fromEntries(list.map((n) => [n, weights[n]]));
    this.ctx.writeTensor(p.patches, image.patches);
    const hiddenAt = (i: number) => (VisionTower.debugReadback ? p.hidden[i] : p.hidden[i % 2]);
    const stepwise = VisionTower.debugReadback && this.onDispatch !== null;
    let count = 0;
    const dispatch = async (g: Built, inputs: Record<string, ML>, outputs: Record<string, ML>, label: string) => {
      if (stepwise) await (this.onDispatch as NonNullable<VisionTower['onDispatch']>)(`${++count}/${depth + 2} ${label}`);
      this.ctx.dispatch(g.graph, this.bind(g, inputs), this.bind(g, outputs, true));
      if (stepwise) { const t = Object.values(outputs)[0], one = newHalf(t.shape.reduce((x: number, y: number) => x * y, 1)); await this.ctx.readTensor(t, one); }
    };
    await dispatch(p.head, { patches: p.patches, ...pick(this.head, names.head) }, { hidden_states: hiddenAt(0) }, `${p.head.name}`);
    for (let layer = 0; layer < depth; layer++) await dispatch(p.block, { hidden_states: hiddenAt(layer), ...pick(this.blocks[layer], names.block) }, { next_hidden_states: hiddenAt(layer + 1) }, `${p.block.name} layer ${layer}`);
    await dispatch(p.merger, { hidden_states: hiddenAt(depth), ...pick(this.merger, names.merger) }, { image_embeds: p.embeds }, `${p.merger.name}`);
    const t1 = performance.now(), rows = newHalf(p.tokens * this.vision.out_hidden);
    await this.ctx.readTensor(p.embeds, rows);
    if (VisionTower.debugReadback) {
      this.trace = [];
      for (const t of p.hidden) { const h = newHalf(t.shape.reduce((x: number, y: number) => x * y, 1)); await this.ctx.readTensor(t, h); this.trace.push(h); }
    }
    this.lastEncode = { hostMs: t1 - t0, deviceMs: performance.now() - t1, buildMs };
    return rows;
  }

  destroy() {
    for (const t of this.tensors) t.destroy?.();
    for (const g of this.graphs) g.destroy?.();
    this.tensors = []; this.graphs = []; this.presets.clear(); this.building.clear(); this.blocks = [];
  }
}

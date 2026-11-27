// LLM runtime for the packages written by `m2w compile`: loads the weights into MLTensors, builds the graphs of one variant of the plan, and runs chunked prefill and decode
// with one of the KV protocols on a WebNN context (the NPUW-style split: HEAD / BLOCK x (N-1) / TAIL / lm_head). A port of a hand-written scheduler, driven by the
// package (model.json, plan.json), with per-layer-type masks and RoPE tables so that hybrid (sliding / full attention) models run on the same BLOCK graph.
// The protocol itself (what is written into the inputs) is in protocol.ts and checked against the Python reference.
//
// Session options (variant = KV protocol x compute type x staging rows, chunk sizes, blocks per graph, max length) can be changed with reconfigure(): the weights stay
// on the device, the graphs are rebuilt and the state (KV cache, activations) is reallocated.
import type { PreparedImage } from './vision.js';
import { loadGraph, type GraphDoc } from './ir-loader.js';
import { checkTensorSize, emptyStats, Prefetcher, type Built, type Stats } from './llm-common.js';
import { pickToken, samplingOf } from './sampling.js';
import { openPackage, type GraphTable, type ModelPackage, type PackageSource, type Progress, type Variant } from './package.js';
import {
  decodeMasks, decodeMasksOnehot, decodeSelectors, decodeSelectorsOnehot, flushSelectors, layerTypesOf, newHalf, planPrefill, prefillMasks, prefillSelectors, ropeTables,
  scatterDecodeMasks, scatterNDDecodeIndices, scatterNDDecodeMasks, scatterNDPrefillIndices, scatterNDPrefillMasks, scatterPrefillMasks, scatterPrefillRope, type Half,
} from './protocol.js';

/* eslint-disable @typescript-eslint/no-explicit-any */
type ML = any;

/** What can change between sessions of one loaded model (each one is applied by reconfigure()). */
export interface SessionOptions {
  maxLength?: number;              // KV cache capacity (one of the package's max_lengths); default: the smallest
  variant?: string;                // id of a graph variant of the package; or pick one with kvStyle / computeType / stagingRows (default: the package's default variant)
  kvStyle?: 'staged' | 'onehot' | 'tensorScatter' | 'scatterND';
  kvInPlace?: boolean;             // scatterND / tensorScatter: the updated cache is written into the SAME MLTensor that is the input (no spare cache; the WebNN implementation has to allow it); ignored for the other protocols
  computeType?: 'float32' | 'float16';
  stagingRows?: number;            // staged only
  chunkSizes?: number[];           // prefill chunk sizes (the package's prefill chunks); several: the chunk sequence of a prompt is planned (planPrefill); default: the smallest
  chunkSize?: number;              // shorthand for chunkSizes: [chunkSize]
  prefillFixedTokens?: number;     // planning cost of one prefill dispatch in tokens (default 200: the int4 weights are dequantized again by every dispatch)
  blocksPerGraph?: number;         // decode: K consecutive blocks per dispatch (needs a package built with --blocks-per-graph K)
  prefillBlocksPerGraph?: number;  // the same for the prefill graphs (default 1)
}
export interface LoadOptions extends SessionOptions {
  device?: 'npu' | 'gpu' | 'cpu';
  context?: ML;                    // an existing MLContext instead of navigator.ml.createContext({ deviceType: device })
  cache?: boolean;                 // OPFS cache of the downloaded files (default true)
  parallelDownloads?: number;      // files downloaded at the same time (a window ahead of the consumer), default 4
  downloadSegments?: number;       // big files are fetched as this many parallel byte ranges (default 4; 1 = one request per file)
  onProgress?: Progress;
  /** Qwen3.5 packages with an image tower: load it (weights + graphs of the grid presets; true = every preset of the package, or { grids: [[24, 24], ...] } the ones you want). Default: not loaded (text only). */
  vision?: boolean | { grids?: [number, number][]; preload?: [number, number][] };   // grids: the presets a picture may get (default all of the package); their graphs are built on first use, preload builds some at load
}
export interface GenerateOptions {
  maxNewTokens?: number;
  /** Qwen3.5 with an image tower: the images of the prompt, in the order of its <|image_pad|> tokens (one per image; the runtime expands each to one token per image patch group): Qwen35LLM.prepareImage() makes them */
  images?: PreparedImage[];
  stopTokens?: number[];           // default: the model's eos tokens
  temperature?: number;            // 0 = greedy; undefined: the package's recommended sampling (plan.recommended.sampling: the model card's values) or greedy
  topK?: number;
  topP?: number;                   // nucleus: the smallest set of the top-k candidates whose probability adds up to topP
  presencePenalty?: number;        // subtracted from the logit of every token already generated
  signal?: AbortSignal;
  /** prefill progress: { stage: 'prefill', loaded: chunks done, total: chunks } after every chunk (with a signal the device has finished the chunk) */
  onProgress?: Progress;
}
export type { StepTiming, Stats } from './llm-common.js';
export type { PreparedImage } from './vision.js';

interface Activations {
  hidden: ML; ropeCos: Record<string, ML>; ropeSin: Record<string, ML>; mask: Record<string, ML>;
  residuals: ML[]; attentionOutputs: ML[]; finalHidden: ML;
}
interface Selectors { keep: ML; select: ML }
interface PrefillSet { head: Built; block: Built; tail: Built; groups: Map<number, Built>; act: Activations; sel: Selectors | null; guards: ML[]; indices: ML | null }
interface DecodeSet { head: Built; block: Built; tail: Built; flush: Built | null; groups: Map<number, Built> }

/** The variant of the package that matches the options (the default variant when none is asked for). */
export function resolveVariant(plan: ModelPackage['plan'], o: SessionOptions): Variant {
  if (o.variant) {
    const found = plan.variants.find((v) => v.id === o.variant);
    if (!found) throw new Error(`the package has no variant '${o.variant}' (it has ${plan.variants.map((v) => v.id).join(', ')})`);
    return found;
  }
  const base = plan.variants.find((v) => v.id === plan.default_variant) as Variant;
  const kv = o.kvStyle ?? base.kv_style, compute = o.computeType ?? base.compute_type;
  const rows = o.stagingRows ?? (kv === base.kv_style ? base.staging_rows : plan.variants.find((v) => v.kv_style === kv)?.staging_rows);
  const found = plan.variants.find((v) => v.kv_style === kv && v.compute_type === compute && (kv !== 'staged' || v.staging_rows === rows));
  if (!found) throw new Error(`the package has no graphs for kv style ${kv}${kv === 'staged' ? `, ${rows} staging rows` : ''}, compute type ${compute} (variants: ${plan.variants.map((v) => v.id).join(', ')})`);
  return found;
}

export class LLM {
  readonly pkg: ModelPackage;
  maxLength = 0;
  variant!: Variant;
  chunkSizes: number[] = [];
  stats: Stats = emptyStats();

  private ctx!: ML;
  private P = 0; private R = 0; private N: number;
  private style: Variant['kv_style'] = 'staged';
  private types: string[];
  private groupSize = 1; private prefillGroupSize = 1; private fixedTokens = 200; private inPlace = false;
  private embed!: { weight: Uint8Array | Int8Array; zeroPoint: Uint8Array | null; scale: Half };   // int8-sym tables have no zero point
  private embedBytes!: { weight: Uint8Array; scale: Uint8Array; zeroPoint: Uint8Array | null };
  private weights: Record<string, ML>[] = [];
  private finalGamma!: ML;
  private lmHead!: Built;
  private decodeSet!: DecodeSet;
  private prefillSets = new Map<number, PrefillSet>();
  private kv: { key: ML; value: ML }[] = []; private kvSpare: { key: ML; value: ML }[] = [];
  private stg: { key: ML; value: ML }[] = []; private stgSpare: { key: ML; value: ML }[] = [];
  private dec!: Activations;
  private decodeSel: Selectors | null = null; private flushSel: Selectors | null = null; private decodeIndices: ML = null;   // scatterND: kv_indices of the decode step
  private writeIndexTensor: ML = null;   // tensorScatter: the first cache row of the dispatch (prefill chunk / decode step)
  private logitsTensor!: ML; private logits!: Half;
  private weightTensors: ML[] = []; private stateTensors: ML[] = []; private builtGraphs: ML[] = [];
  private base = 0; private filled = 0; private position = 0; private pad = 0;
  private busy = false;
  private lastStep = { hostMs: 0, enqueueMs: 0, waitMs: 0 };

  /** the first (smallest) chunk size, for callers that think in one chunk size */
  get chunkSize(): number { return this.chunkSizes[0]; }

  private constructor(pkg: ModelPackage) {
    this.pkg = pkg;
    this.N = pkg.model.num_layers; this.types = layerTypesOf(pkg.model);
  }

  // ------------------------------------------------------------------------------------------------------------------------------- loading
  static async load(src: string | PackageSource, options: LoadOptions = {}): Promise<LLM> {
    const progress = options.onProgress ?? (() => {});
    progress({ stage: 'package', message: 'reading the package index' });
    return LLM.loadFrom(await openPackage(src, options.cache ?? true, { segments: options.downloadSegments }), options);
  }

  static async loadFrom(pkg: ModelPackage, options: LoadOptions = {}): Promise<LLM> {
    const t0 = performance.now();
    const progress = options.onProgress ?? (() => {});
    if (pkg.plan.architecture_kind) throw new Error(`this package needs the ${pkg.plan.architecture_kind} runtime: use loadModel() instead of LLM.load()`);
    const llm = new LLM(pkg);
    llm.adopt(llm.resolve(options));   // fails early (before anything is downloaded) when the package lacks what is asked for
    await llm.init(options, progress);
    llm.stats.loadMs = performance.now() - t0;
    return llm;
  }

  /** Check the session options against the plan (no side effects); returns what to adopt. */
  private resolve(o: SessionOptions) {
    const plan = this.pkg.plan, key = String(o.maxLength ?? Math.min(...plan.max_lengths));
    const variant = resolveVariant(plan, o), table = plan.graphs[variant.id];
    if (!table.decode[key]) throw new Error(`the package has no graphs for max length ${key} (it has ${plan.max_lengths.join(', ')})`);
    const available = Object.keys(table.prefill[key]).map(Number).sort((a, b) => a - b);
    const sizes = [...new Set(o.chunkSizes?.length ? o.chunkSizes : [o.chunkSize ?? available[0]])].sort((a, b) => a - b);
    for (const size of sizes) if (!available.includes(size)) throw new Error(`the package has no prefill graphs for chunk ${size} at max length ${key} (it has ${available.join(', ')})`);
    // not chosen by the caller: the recommended group sizes of the package (when this variant / max length / chunk has them)
    const rec = plan.recommended ?? {}, groupsOfTable = (t: GraphTable) => Object.keys(t.block_groups ?? {}).map(Number);
    let group = o.blocksPerGraph ?? rec.blocks_per_graph ?? 1, prefillGroup = o.prefillBlocksPerGraph || (o.blocksPerGraph !== undefined ? group : rec.prefill_blocks_per_graph ?? group);
    if (o.blocksPerGraph === undefined && group > 1 && !groupsOfTable(table.decode[key]).includes(group)) group = 1;
    if (!o.prefillBlocksPerGraph && prefillGroup > 1 && sizes.some((size) => !groupsOfTable(table.prefill[key][String(size)]).includes(prefillGroup)) && o.blocksPerGraph === undefined) prefillGroup = 1;
    if ((group > 1 || prefillGroup > 1) && this.types.length > 1 && plan.block_group_controls !== 'per_block') {
      throw new Error('blocksPerGraph > 1 needs a model with one layer type, or a package whose block groups take a mask and RoPE table per block (m2w compile of a newer version)');
    }
    const groupsOf = (t: GraphTable) => Object.keys(t.block_groups ?? {}).map(Number);
    if (group > 1 && !groupsOf(table.decode[key]).includes(group)) throw new Error(`the package has no decode graph with ${group} blocks (it has ${groupsOf(table.decode[key]).join(', ') || 'none'})`);
    for (const size of sizes) {
      if (prefillGroup > 1 && !groupsOf(table.prefill[key][String(size)]).includes(prefillGroup)) {
        throw new Error(`the package has no prefill graph with ${prefillGroup} blocks at chunk ${size} (it has ${groupsOf(table.prefill[key][String(size)]).join(', ') || 'none'})`);
      }
    }
    const inPlace = !!o.kvInPlace && (variant.kv_style === 'tensorScatter' || variant.kv_style === 'scatterND');
    return { maxLength: +key, variant, sizes, group, prefillGroup, fixedTokens: o.prefillFixedTokens ?? 200, inPlace };
  }

  private adopt(c: ReturnType<LLM['resolve']>) {
    this.maxLength = c.maxLength; this.variant = c.variant; this.style = c.variant.kv_style; this.chunkSizes = c.sizes;
    this.P = this.maxLength - 1; this.R = c.variant.staging_rows ?? 0;
    this.groupSize = c.group; this.prefillGroupSize = c.prefillGroup; this.fixedTokens = c.fixedTokens; this.inPlace = c.inPlace;
  }

  private async createTensor(dataType: string, shape: number[], flags: { writable?: boolean; readable?: boolean } = {}, into = this.stateTensors) {
    checkTensorSize(dataType, shape);
    const tensor = await this.ctx.createTensor({ dataType, shape, writable: flags.writable ?? false, readable: flags.readable ?? false });
    into.push(tensor);
    return tensor;
  }
  private writable = (dataType: string, shape: number[]) => this.createTensor(dataType, shape, { writable: true });

  private async init(options: LoadOptions, progress: Progress) {
    const { pkg } = this, plan = pkg.plan, source = pkg.source;
    this.ctx = options.context ?? await (navigator as any).ml.createContext({ deviceType: options.device ?? 'npu' });

    // weight files: prefetched in the order they are needed, a few at a time (embedding, final norm, layer 0, 1, ...)
    const byRole = (role: string) => pkg.manifest.files.filter((f) => f.role === role);
    const ordered = [...byRole('embedding'), ...byRole('final_norm'), ...byRole('layer').sort((a, b) => (a.layer as number) - (b.layer as number))];
    const total = ordered.reduce((s, f) => s + f.bytes, 0), loaded = new Map<string, number>(), cachedBytes = new Map<string, number>(), sum = (m: Map<string, number>) => [...m.values()].reduce((a, b) => a + b, 0);
    const prefetch = new Prefetcher(ordered.map((f) => f.file), (file, i) => source.bytes(`weights/${file}`, (l, _total, cached) => {
      loaded.set(file, l); if (cached) cachedBytes.set(file, l);
      progress({ stage: 'download', loaded: sum(loaded), total, fromCache: sum(cachedBytes), message: ordered[i].file });
    }, ordered[i].bytes), options.parallelDownloads ?? 4);

    const weightsStart = performance.now();
    progress({ stage: 'embedding', message: 'loading the embedding table' });
    const part = async (name: string) => {
      const file = byRole('embedding').find((f) => f.tensors[0].name === name);
      if (!file) throw new Error(`the package has no embedding tensor '${name}' (plan.weights.embedding: ${plan.weights.embedding.join(', ')})`);
      const tensor = file.tensors.find((t) => t.name === name) as (typeof file.tensors)[number];
      return new Uint8Array(await prefetch.get(ordered.indexOf(file)), tensor.offset, tensor.bytes);   // a view of the tensor inside its .safetensors shard (no copy)
    };
    // the tied embedding / lm_head table: int8 per-row symmetric (scale only) or uint8 per-row asymmetric (scale + zero point); the plan lists which tensors exist
    const hasZeroPoint = plan.weights.embedding.includes('lm_head_zero_point');
    const weightBytes = await part('lm_head_weight'), scaleBytes = (await part('lm_head_scale')).slice();   // the scales are copied: Float16Array needs an aligned ArrayBuffer of its own
    const zeroPoint = hasZeroPoint ? await part('lm_head_zero_point') : null;
    const signed = byRole('embedding').find((f) => f.tensors[0].name === 'lm_head_weight')?.tensors[0].dtype === 'int8';
    this.embed = { weight: signed ? new Int8Array(weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength) : weightBytes, zeroPoint, scale: new (globalThis as any).Float16Array(scaleBytes.buffer) };
    this.embedBytes = { weight: weightBytes, scale: scaleBytes, zeroPoint };

    // graphs (compiled while the layers download)
    await this.buildGraphs(progress);

    // weights -> MLTensors, layer by layer as the files arrive
    const finalFile = byRole('final_norm')[0];
    const uploadShard = async (shard: (typeof ordered)[number], buffer: ArrayBuffer) => {
      const tensors: Record<string, ML> = {};
      for (const entry of shard.tensors) {
        const tensor = await this.createTensor(entry.dtype, entry.shape, { writable: true }, this.weightTensors);
        this.ctx.writeTensor(tensor, new Uint8Array(buffer, entry.offset, entry.bytes));
        tensors[entry.name] = tensor;
      }
      return tensors;
    };
    this.finalGamma = Object.values(await uploadShard(finalFile, await prefetch.get(ordered.indexOf(finalFile))))[0];
    const layers = byRole('layer').sort((a, b) => (a.layer as number) - (b.layer as number));
    for (const shard of layers) {
      progress({ stage: 'weights', message: `layer ${(shard.layer as number) + 1}/${layers.length}`, loaded: shard.layer as number, total: layers.length });
      this.weights.push(await uploadShard(shard, await prefetch.get(ordered.indexOf(shard))));
      prefetch.release(ordered.indexOf(shard));
    }

    this.stats.weightsMs = performance.now() - weightsStart - this.stats.buildMs;   // download + upload, graph compilation excluded
    progress({ stage: 'state', message: 'allocating the KV cache and activations' });
    await this.allocateState();
    progress({ stage: 'ready' });
  }

  /** Build the graphs of the resolved session (variant, max length, chunk sizes, groups). */
  private async buildGraphs(progress: Progress) {
    const builds: { label: string; ms: number }[] = [], t = performance.now(), plan = this.pkg.plan, source = this.pkg.source, key = String(this.maxLength);
    const table = plan.graphs[this.variant.id], decode = table.decode[key];
    const bytes = this.embedBytes;
    const external = (c: { external?: string }) => {
      if (c.external === 'lm_head_weight') return bytes.weight;
      if (c.external === 'lm_head_scale') return new Uint8Array(bytes.scale);
      if (c.external === 'lm_head_zero_point' && bytes.zeroPoint) return bytes.zeroPoint;
      throw new Error(`the lm_head graph wants the external tensor '${c.external}' which this package does not have`);
    };
    // how many graphs this session builds (for the progress: n of m)
    const groupCount = (files: GraphTable, size: number) => (size > 1 ? Object.keys(files.block_groups ?? {}).length : 0);
    const expected = 1 + 3 + (this.style === 'staged' ? 1 : 0) + groupCount(decode, this.groupSize) + this.chunkSizes.reduce((n, size) => n + 3 + groupCount(table.prefill[key][String(size)], this.prefillGroupSize), 0);
    const build = async (path: string, label: string, provider?: (c: any) => ArrayBufferView): Promise<Built> => {
      progress({ stage: 'build', message: label, loaded: builds.length, total: expected });
      const doc: GraphDoc = JSON.parse(await source.text(path));
      const builder = new (globalThis as any).MLGraphBuilder(this.ctx);
      const { inputs, outputs } = loadGraph(builder, doc, provider);
      const started = performance.now();
      const graph = await builder.build(outputs);
      builds.push({ label, ms: performance.now() - started });
      this.builtGraphs.push(graph);
      return { name: doc.name, graph, inputs: Object.keys(inputs), outputs: Object.keys(outputs) };
    };
    const groupsOf = async (files: GraphTable, size: number, label: string) => {
      const groups = new Map<number, Built>();
      if (size > 1) for (const [count, path] of Object.entries(files.block_groups ?? {})) groups.set(+count, await build(path, `${label} block x${count}`));
      return groups;
    };
    this.lmHead = await build(table.lm_head, 'lm_head', external);
    this.decodeSet = {
      head: await build(decode.head, 'decode head'), block: await build(decode.block, 'decode block'), tail: await build(decode.tail, 'decode tail'),
      flush: this.style === 'staged' ? await build(decode.flush as string, 'kv flush') : null, groups: await groupsOf(decode, this.groupSize, 'decode'),
    };
    this.prefillSets = new Map();
    for (const size of this.chunkSizes) {
      const files = table.prefill[key][String(size)], label = `prefill c${size}`;
      this.prefillSets.set(size, {
        head: await build(files.head, `${label} head`), block: await build(files.block, `${label} block`), tail: await build(files.tail, `${label} tail`),
        groups: await groupsOf(files, this.prefillGroupSize, label), act: null as any, sel: null, guards: [], indices: null,
      });
    }
    this.stats.buildMs = performance.now() - t;
    this.stats.graphBuilds = builds;
  }

  /** Change the session options of a loaded model: graphs are rebuilt and the state reallocated, the weights stay on the device. */
  async reconfigure(options: SessionOptions) {
    if (this.busy) throw new Error('a generation is running on this model');
    const progress = (options as LoadOptions).onProgress ?? (() => {});
    const config = this.resolve(options);   // throws before anything is released
    this.releaseSession();
    this.adopt(config);
    await this.buildGraphs(progress);
    progress({ stage: 'state', message: 'allocating the KV cache and activations' });
    await this.allocateState();
    progress({ stage: 'ready' });
  }

  private releaseSession() {
    for (const t of this.stateTensors) t.destroy?.();
    for (const g of this.builtGraphs) g.destroy?.();
    this.stateTensors = []; this.builtGraphs = [];
    this.kv = []; this.kvSpare = []; this.stg = []; this.stgSpare = [];
    this.base = this.filled = this.position = this.pad = 0;
  }

  private async allocateState() {
    const s = this.pkg.model, { P, R, N } = this;
    const zero = (n: number) => newHalf(n);
    const kvShape = (rows: number) => [1, s.num_kv_heads, rows, s.head_dim];
    const transposed = this.variant.value_layout === 'transposed';   // value cache [1, kvHeads, 128, rows] (one-hot variants)
    const valueShape = (rows: number) => (transposed ? [1, s.num_kv_heads, s.head_dim, rows] : kvShape(rows));
    const makeKv = async (rows: number) => {
      const t = { key: await this.writable('float16', kvShape(rows)), value: await this.writable('float16', valueShape(rows)) };
      const zeros = zero(s.num_kv_heads * rows * s.head_dim);
      this.ctx.writeTensor(t.key, zeros); this.ctx.writeTensor(t.value, zeros);
      return t;
    };
    const staged = this.style === 'staged';
    for (let l = 0; l < N; l++) {
      const cache = await makeKv(P);
      this.kv.push(cache); this.kvSpare.push(this.inPlace ? cache : await makeKv(P));   // in place: the "spare" IS the cache, the swaps change nothing
      if (staged) { this.stg.push(await makeKv(R)); this.stgSpare.push(await makeKv(R)); }
    }
    const activations = async (S: number, maskColumns: number): Promise<Activations> => {
      const a: Activations = { hidden: await this.writable('float16', [1, S, s.hidden]), ropeCos: {}, ropeSin: {}, mask: {}, residuals: [], attentionOutputs: [], finalHidden: null };
      for (const type of this.types) {
        a.ropeCos[type] = await this.writable('float16', [1, 1, S, s.head_dim]);
        a.ropeSin[type] = await this.writable('float16', [1, 1, S, s.head_dim]);
        a.mask[type] = await this.writable('float16', [1, 1, S, maskColumns]);
      }
      for (let l = 0; l < N; l++) {
        a.residuals.push(await this.createTensor('float16', [1, S, s.hidden], l === N - 1 ? { readable: true } : {}));   // the last one is read between prefill chunks of an abortable generation
        a.attentionOutputs.push(await this.createTensor('float16', [1, S, s.num_heads * s.head_dim]));
      }
      a.finalHidden = await this.createTensor('float16', [1, 1, s.hidden]);
      return a;
    };
    const pair = async (rows: number, columns: number): Promise<Selectors> => ({ keep: await this.writable('float16', [1, 1, rows, 1]), select: await this.writable('float16', [1, 1, rows, columns]) });
    const scatter = this.style === 'tensorScatter', scatterND = this.style === 'scatterND', inCache = scatter || scatterND;   // attention reads the updated cache (P columns)
    for (const [size, set] of this.prefillSets) {
      set.act = await activations(size, inCache ? P : P + size);
      set.sel = inCache ? null : await pair(P, size);
      if (scatterND) set.indices = await this.writable('int32', [1, s.num_kv_heads, size, 3]);
      // transposed value cache: the prefill graphs have an attention_guard output per block (see decoder.py); the values are not needed, the tensors only receive them
      if (transposed) for (let i = 0; i <= this.prefillGroupSize; i++) set.guards.push(await this.createTensor('float16', [1, s.num_heads, P + size, 1]));
    }
    this.dec = await activations(1, this.style === 'staged' ? P + R : inCache ? P : P + 1);
    this.decodeIndices = scatterND ? await this.writable('int32', [1, s.num_kv_heads, 1, 3]) : null;
    this.decodeSel = this.style === 'staged' ? await pair(R, 1) : this.style === 'onehot' ? await pair(P, 1) : null;
    this.flushSel = staged ? await pair(P, R) : null;
    this.writeIndexTensor = scatter ? await this.writable(this.variant.scatter_index_type ?? 'int32', [1]) : null;
    this.logitsTensor = await this.createTensor('float16', [1, 1, s.vocab_padded], { readable: true });
    this.logits = newHalf(s.vocab_padded);
  }

  // ------------------------------------------------------------------------------------------------------------------------------- dispatch
  private bind(graph: Built, values: Record<string, ML>, outputs = false) {
    const out: Record<string, ML> = {};
    for (const name of outputs ? graph.outputs : graph.inputs) {
      if (values[name] === undefined) throw new Error(`graph ${graph.name}: no tensor is bound to ${outputs ? 'output' : 'input'} '${name}'`);
      out[name] = values[name];
    }
    return out;
  }
  private pick(layer: number, names: string[], prefix = '') {
    const out: Record<string, ML> = {};
    for (const n of names) out[prefix + n] = this.weights[layer][n];
    return out;
  }
  private swap(list: { key: ML; value: ML }[], spare: { key: ML; value: ML }[], layer: number) { [list[layer], spare[layer]] = [spare[layer], list[layer]]; }
  private writeIndex(row: number) {
    this.ctx.writeTensor(this.writeIndexTensor, this.variant.scatter_index_type === 'int64' ? BigInt64Array.of(BigInt(row)) : Int32Array.of(row));
  }

  /** One forward pass: HEAD, the BLOCKs and (last chunk / decode) TAIL + lm_head. All host inputs of the phase are already written. */
  private enqueue(size: number, lastChunk: boolean) {   // size 0 = decode, else the chunk size of the prefill set
    const decode = size === 0, set = (decode ? this.decodeSet : this.prefillSets.get(size)) as DecodeSet | PrefillSet, a = decode ? this.dec : (set as PrefillSet).act;
    const w = this.pkg.plan.weights as { attention: string[]; ffn: string[] }, types = this.pkg.model.layer_types, N = this.N, stagedDecode = decode && this.style === 'staged';
    const caches = stagedDecode ? this.stg : this.kv, spare = stagedDecode ? this.stgSpare : this.kvSpare;
    const selectors: Record<string, ML> = this.style === 'tensorScatter' ? { kv_write_index: this.writeIndexTensor }
      : this.style === 'scatterND' ? { kv_indices: decode ? this.decodeIndices : (set as PrefillSet).indices }
        : stagedDecode ? { staging_rows_to_keep: (this.decodeSel as Selectors).keep, staging_row_selector: (this.decodeSel as Selectors).select }
        : decode ? { rows_to_keep: (this.decodeSel as Selectors).keep, row_selector: (this.decodeSel as Selectors).select }
          : { rows_to_keep: (set as PrefillSet).sel!.keep, row_selector: (set as PrefillSet).sel!.select };
    const shared = (layer: number) => ({ rope_cos: a.ropeCos[types[layer]], rope_sin: a.ropeSin[types[layer]], attention_mask: a.mask[types[layer]], ...selectors });
    const cacheInputs = (layer: number, prefix = '') => ({
      [prefix + 'past_key']: this.kv[layer].key, [prefix + 'past_value']: this.kv[layer].value,
      ...(stagedDecode ? { [prefix + 'staging_key']: this.stg[layer].key, [prefix + 'staging_value']: this.stg[layer].value } : {}),
    });
    const dispatch = (graph: Built, inputs: Record<string, ML>, outputs: Record<string, ML>) => this.ctx.dispatch(graph.graph, this.bind(graph, inputs), this.bind(graph, outputs, true));

    const guards = decode ? [] : (set as PrefillSet).guards, guardOut = (i: number, prefix = '') => (guards.length ? { [prefix + 'attention_guard']: guards[i] } : {});

    // HEAD: attention half of layer 0
    dispatch(set.head, { hidden_states: a.hidden, ...shared(0), ...cacheInputs(0), ...this.pick(0, w.attention) },
      { residual: a.residuals[0], attention_output: a.attentionOutputs[0], updated_key: spare[0].key, updated_value: spare[0].value, ...guardOut(0) });
    this.swap(caches, spare, 0);

    // BLOCK i: feed-forward half of layer i + attention half of layer i+1
    const groupSize = decode ? this.groupSize : this.prefillGroupSize, groups = groupSize > 1 ? set.groups : null;
    if (groups) {
      for (let start = 0; start < N - 1; start += groupSize) {
        const count = Math.min(groupSize, N - 1 - start), graph = groups.get(count) as Built;
        const perBlock = this.pkg.plan.block_group_controls === 'per_block';
        const inputs: Record<string, ML> = { residual: a.residuals[start], attention_output: a.attentionOutputs[start], ...(perBlock ? selectors : shared(start + 1)) };
        const outputs: Record<string, ML> = { next_residual: a.residuals[start + count], next_attention_output: a.attentionOutputs[start + count] };
        for (let j = 0; j < count; j++) {
          const layer = start + j, prefix = `b${j}_`;
          Object.assign(inputs, this.pick(layer, w.ffn, prefix), this.pick(layer + 1, w.attention, prefix), cacheInputs(layer + 1, prefix));
          if (perBlock) {   // the attention half of layer + 1 has the rope tables / mask of ITS layer type
            Object.assign(inputs, { [prefix + 'rope_cos']: a.ropeCos[types[layer + 1]], [prefix + 'rope_sin']: a.ropeSin[types[layer + 1]], [prefix + 'attention_mask']: a.mask[types[layer + 1]] });
          }
          outputs[prefix + 'updated_key'] = spare[layer + 1].key; outputs[prefix + 'updated_value'] = spare[layer + 1].value;
          Object.assign(outputs, guardOut(j, prefix));
        }
        dispatch(graph, inputs, outputs);
        for (let j = 0; j < count; j++) this.swap(caches, spare, start + j + 1);
      }
    } else {
      for (let layer = 0; layer < N - 1; layer++) {
        dispatch(set.block, { residual: a.residuals[layer], attention_output: a.attentionOutputs[layer], ...shared(layer + 1), ...cacheInputs(layer + 1),
          ...this.pick(layer, w.ffn), ...this.pick(layer + 1, w.attention) },
        { next_residual: a.residuals[layer + 1], next_attention_output: a.attentionOutputs[layer + 1], updated_key: spare[layer + 1].key, updated_value: spare[layer + 1].value, ...guardOut(0) });
        this.swap(caches, spare, layer + 1);
      }
    }
    if (!lastChunk) return;
    // TAIL: feed-forward half of the last layer + final norm; lm_head
    dispatch(set.tail, { residual: a.residuals[N - 1], attention_output: a.attentionOutputs[N - 1], ...this.pick(N - 1, w.ffn), final_norm_gamma: this.finalGamma },
      { final_hidden_states: a.finalHidden });
    dispatch(this.lmHead, { final_hidden_states: a.finalHidden }, { logits: this.logitsTensor });
  }

  private embedRow(token: number, out: Half, offset: number) {
    const { weight, zeroPoint, scale } = this.embed, hidden = this.pkg.model.hidden, start = token * hidden, zp = zeroPoint ? zeroPoint[token] : 0, s = scale[token];
    for (let i = 0; i < hidden; i++) out[offset + i] = (weight[start + i] - zp) * s;
  }

  private writeRope(tables: ReturnType<typeof ropeTables>, a: Activations) {
    for (const type of this.types) { this.ctx.writeTensor(a.ropeCos[type], tables[type].cos); this.ctx.writeTensor(a.ropeSin[type], tables[type].sin); }
  }

  private async readLogits(): Promise<Half> {
    await this.ctx.readTensor(this.logitsTensor, this.logits);
    return this.logits;
  }

  /** The chunk sizes the next prompt of `tokenCount` tokens would use. */
  planFor(tokenCount: number): number[] { return planPrefill(this.chunkSizes, tokenCount, this.fixedTokens); }

  /** No row left for the next decode step (tensorScatter: the padding rows in front of the prompt are used too). */
  private get cacheFull() { return this.position + (this.style === 'tensorScatter' ? this.pad : 0) >= this.P; }

  // ------------------------------------------------------------------------------------------------------------------------------- generation
  /** Chunked prefill of the prompt; returns the logits of the last prompt token (null: `signal` was aborted between two chunks; the next prefill starts from scratch anyway). */
  private async prefill(tokens: number[], signal?: AbortSignal, onProgress?: Progress): Promise<Half | null> {
    const s = this.pkg.model, { P } = this, n = tokens.length, scatter = this.style === 'tensorScatter';
    if (n === 0) throw new Error('empty prompt');
    const plan = this.planFor(n), rows = plan.reduce((a, b) => a + b, 0);
    if (n >= P || (scatter && rows > P)) throw new Error(`the prompt has ${n} tokens (${rows} rows of chunks) but the KV cache holds ${P}: load the package with a larger max length`);
    this.stats.prefillPlan = plan;
    const padTotal = scatter ? rows - n : 0;
    let start = 0;
    for (let index = 0; index < plan.length; index++) {
      if (signal?.aborted) return null;
      const C = plan[index], last = index === plan.length - 1, set = this.prefillSets.get(C) as PrefillSet, a = set.act, hidden = newHalf(C * s.hidden);
      if (scatter) {   // the chunk covers cache rows start .. start+C-1, token i is in row padTotal + i
        for (let r = 0; r < C; r++) if (start + r - padTotal >= 0) this.embedRow(tokens[start + r - padTotal], hidden, r * s.hidden);
        this.ctx.writeTensor(a.hidden, hidden);
        this.writeRope(scatterPrefillRope(s, C, start, padTotal), a);
        const masks = scatterPrefillMasks(s, P, C, start, padTotal);
        for (const type of this.types) this.ctx.writeTensor(a.mask[type], masks[type]);
        this.writeIndex(start);
        start += C;
      } else {
        const valid = Math.min(C, n - start), pad = C - valid;
        for (let i = 0; i < valid; i++) this.embedRow(tokens[start + i], hidden, (pad + i) * s.hidden);
        this.ctx.writeTensor(a.hidden, hidden);
        this.writeRope(ropeTables(s, Array.from({ length: valid }, (_, i) => start + i), C, pad), a);
        if (this.style === 'scatterND') {
          const masks = scatterNDPrefillMasks(s, P, C, start, valid);
          for (const type of this.types) this.ctx.writeTensor(a.mask[type], masks[type]);
          this.ctx.writeTensor(set.indices, scatterNDPrefillIndices(s.num_kv_heads, P, C, start, valid));
        } else {
          const masks = prefillMasks(s, P, C, start, valid);
          for (const type of this.types) this.ctx.writeTensor(a.mask[type], masks[type]);
          const { rowsToKeep, rowSelector } = prefillSelectors(P, C, start, valid);
          this.ctx.writeTensor((set.sel as Selectors).keep, rowsToKeep); this.ctx.writeTensor((set.sel as Selectors).select, rowSelector);
        }
        start += valid;
      }
      this.enqueue(C, last);
      if (signal && !last) {   // abortable: wait for the device to finish this chunk (read back its last residual), then look at the signal; without a signal the chunks stay queued back to back
        await this.ctx.readTensor(a.residuals[this.N - 1], newHalf(C * s.hidden));
        if (signal.aborted) return null;
      }
      onProgress?.({ stage: 'prefill', loaded: index + 1, total: plan.length });
    }
    this.base = n; this.filled = 0; this.pad = padTotal; this.position = n;
    return this.readLogits();
  }

  private flush() {
    const sel = this.flushSel as Selectors, { rowsToKeep, rowSelector } = flushSelectors(this.P, this.R, this.base);
    this.ctx.writeTensor(sel.keep, rowsToKeep); this.ctx.writeTensor(sel.select, rowSelector);
    const g = this.decodeSet.flush as Built;
    for (let l = 0; l < this.N; l++) {
      this.ctx.dispatch(g.graph, this.bind(g, { past_key: this.kv[l].key, past_value: this.kv[l].value, new_key: this.stg[l].key, new_value: this.stg[l].value,
        rows_to_keep: sel.keep, row_selector: sel.select }),
      this.bind(g, { updated_key: this.kvSpare[l].key, updated_value: this.kvSpare[l].value }, true));
      this.swap(this.kv, this.kvSpare, l);
    }
    this.base += this.R; this.filled = 0;
  }

  /** Feed one token at the current position; returns the logits for the next one. */
  private async step(token: number): Promise<Half> {
    const s = this.pkg.model, { P, R } = this;
    if (this.cacheFull) throw new Error('the KV cache is full');
    let flushMs = 0;
    if (this.style === 'staged' && this.filled === R) { const f = performance.now(); this.flush(); flushMs = performance.now() - f; this.stats.flushes++; }
    const t0 = performance.now();
    const pos = this.position, a = this.dec;
    const hidden = newHalf(s.hidden);
    this.embedRow(token, hidden, 0);
    this.ctx.writeTensor(a.hidden, hidden);
    this.writeRope(ropeTables(s, [pos], 1, 0), a);
    let masks: Record<string, Half>;
    if (this.style === 'staged') {
      masks = decodeMasks(s, P, R, pos, this.base, this.filled);
      const { rowsToKeep, rowSelector } = decodeSelectors(R, this.filled);
      this.ctx.writeTensor((this.decodeSel as Selectors).keep, rowsToKeep); this.ctx.writeTensor((this.decodeSel as Selectors).select, rowSelector);
    } else if (this.style === 'onehot') {
      masks = decodeMasksOnehot(s, P, pos);
      const { rowsToKeep, rowSelector } = decodeSelectorsOnehot(P, pos);
      this.ctx.writeTensor((this.decodeSel as Selectors).keep, rowsToKeep); this.ctx.writeTensor((this.decodeSel as Selectors).select, rowSelector);
    } else if (this.style === 'scatterND') {
      masks = scatterNDDecodeMasks(s, P, pos);
      this.ctx.writeTensor(this.decodeIndices, scatterNDDecodeIndices(s.num_kv_heads, pos));
    } else {
      masks = scatterDecodeMasks(s, P, pos, this.pad);
      this.writeIndex(this.pad + pos);
    }
    for (const type of this.types) this.ctx.writeTensor(a.mask[type], masks[type]);
    const t1 = performance.now();
    this.enqueue(0, true);
    const t2 = performance.now();
    this.filled++; this.position++;
    const logits = await this.readLogits();
    this.lastStep = { hostMs: t1 - t0, enqueueMs: t2 - t1 + flushMs, waitMs: performance.now() - t2 };
    return logits;
  }

  /** Greedy (or sampled) generation from prompt token ids; yields each new token id (the stop token is yielded too, then the loop ends). */
  async *generate(tokens: number[], options: GenerateOptions = {}): AsyncGenerator<number> {
    if (this.busy) throw new Error('a generation is already running on this model');
    this.busy = true;
    try {
      const stop = new Set(options.stopTokens ?? this.pkg.model.eos_token_ids), max = options.maxNewTokens ?? 128;
      let t = performance.now();
      let logits = await this.prefill(tokens, options.signal, options.onProgress);
      if (!logits) return;   // aborted during the prefill: nothing was yielded
      this.stats = { ...emptyStats(), loadMs: this.stats.loadMs, buildMs: this.stats.buildMs, weightsMs: this.stats.weightsMs, prefillMs: performance.now() - t,
        prefillChunks: this.stats.prefillPlan.length, prefillPlan: this.stats.prefillPlan, promptTokens: tokens.length, graphBuilds: this.stats.graphBuilds };
      t = performance.now();
      const seen = new Set<number>(), sampling = samplingOf(options, this.pkg.plan.recommended?.sampling);   // seen: the tokens generated so far (presence penalty)
      for (let i = 0; i < max; i++) {
        const sampleStart = performance.now();
        const token = pickToken(logits, this.pkg.model.vocab, sampling, seen);
        seen.add(token);
        const samplingMs = performance.now() - sampleStart;
        yield token;
        if (stop.has(token) || i === max - 1 || this.cacheFull || options.signal?.aborted) break;
        const stepStart = performance.now();
        logits = await this.step(token);
        const stepMs = performance.now() - stepStart;
        this.stats.stepMs.push(stepMs);   // host input writes + the whole dispatch chain + reading the logits
        this.stats.steps.push({ ...this.lastStep, samplingMs, totalMs: stepMs + samplingMs });
        this.stats.decodeSteps++;
        this.stats.decodeMs = performance.now() - t;
        this.stats.tpotMs = this.stats.decodeMs / this.stats.decodeSteps;
      }
    } finally { this.busy = false; }
  }

  destroy() {
    this.releaseSession();
    for (const t of this.weightTensors) t.destroy?.();
    this.weightTensors = []; this.weights = [];
  }
}

// LLM runtime for Gemma 4 packages (plan.architecture_kind === 'gemma4'): the same NPUW-style chain as llm.ts (HEAD / BLOCK x (N-1) / TAIL / lm_head, chunked prefill, staged KV), with
//  * one BLOCK graph per attention KIND (layer type x own / shared K / V): BLOCK i = feed-forward half of layer i + attention half of layer i + 1 of the kind of layer i + 1;
//  * KV caches only for the layers that have K / V of their own; a layer that shares K / V reads the cache of its producer (decode: main cache + the staging buffer AFTER this step's update,
//    prefill: the cache after the update of the chunk, mask columns = cache rows);
//  * per-layer embeddings: a row of the int8 table per token is read on the host (a byte range of the file: the table is gigabytes), HEAD turns it into the per-layer inputs ple_0 .. ple_{N-1};
//  * the embedding is multiplied by sqrt(hidden) on the host.
//  * decode block groups (blocksPerGraph = K > 1, a package built with --blocks-per-graph K): the N - 1 BLOCKs in runs of K, one dispatch per run (plan.chain.block_groups); the inputs of a group graph
//    are bound by NAME: residual / attention_output of its first block, <layer type>_rope_cos / _rope_sin / _attention_mask, and b<j>_ = the weights, per-layer input and caches of block j. Prefill has no groups.
// Python counterpart (the executable specification): python/m2w/reference/gemma4_runner.py. Only the staged KV protocol exists.
// Sliding window caches (a package built with --sliding-cache window, plan.profile.sliding_cache): the caches of the sliding layers are rings of plan.profile.sliding_ring rows (protocol.ts cacheRows);
// their masks and the chunk / flush selectors are built for those rows, one selector pair per distinct cache size; the graphs already differ per kind, only the cache length of the sliding kinds changes.
import { loadGraph, type GraphDoc } from './ir-loader.js';
import { checkTensorSize, emptyStats, Prefetcher, type Built, type Stats } from './llm-common.js';
import { pickToken, samplingOf } from './sampling.js';
import type { GenerateOptions, LoadOptions, SessionOptions } from './llm.js';
import { openPackage, type GemmaBlockGroup, type GemmaDecodeTable, type GraphTable, type ManifestFile, type ModelPackage, type PackageSource, type Progress, type Variant } from './package.js';
import {
  cacheRows, decodeMasks, decodeSelectors, flushSelectors, incachePrefillMasks, layerTypesOf, newHalf, planPrefill, prefillMasks, prefillSelectors, ringOf, ropeTables, type Half,
} from './protocol.js';

/* eslint-disable @typescript-eslint/no-explicit-any */
type ML = any;

interface Activations {
  hidden: ML; pleToken: ML; ropeCos: Record<string, ML>; ropeSin: Record<string, ML>; mask: Record<string, ML>; maskShared: Record<string, ML>;
  residuals: ML[]; attentionOutputs: ML[]; ples: ML[]; finalHidden: ML;
}
interface Selectors { keep: ML; select: ML }
interface PrefillSet { head: Built; blocks: Map<string, Built>; tail: Built; act: Activations; sel: Map<number, Selectors> }   // sel: by the rows of the cache the chunk is written into
type Cache = { key: ML; value: ML };

export class Gemma4LLM {
  readonly pkg: ModelPackage;
  maxLength = 0;
  variant!: Variant;
  chunkSizes: number[] = [];
  stats: Stats = emptyStats();

  private ctx!: ML;
  private P = 0; private R = 0; private N: number; private ring = 0;   // ring: rows of the caches of the sliding layers of a window package (0: all P)
  private fixedTokens = 200; private groupSize = 1;
  private types: string[]; private kinds: string[]; private producer: number[];
  private weights: Record<string, ML>[] = [];
  private headExtra: Record<string, ML> = {};
  private finalGamma!: ML;
  private lmHead!: Built;
  private decodeSet!: { head: Built; blocks: Map<string, Built>; tail: Built; flush: Map<string, Built>; groups: { first: number; blocks: number; built: Built }[] };
  private prefillSets = new Map<number, PrefillSet>();
  private kv = new Map<number, Cache>(); private kvSpare = new Map<number, Cache>(); private stg = new Map<number, Cache>(); private stgSpare = new Map<number, Cache>();
  private dec!: Activations;
  private decodeSel!: Selectors; private flushSel!: Map<number, Selectors>;   // flushSel: by the rows of the cache
  private logitsTensor!: ML; private logits!: Half;
  private embed!: { weight: Int8Array; scale: Half };
  private ple!: { path: string; weightOffset: number; width: number; scale: Half };
  private weightTensors: ML[] = []; private stateTensors: ML[] = []; private builtGraphs: ML[] = [];
  private base = 0; private filled = 0; private position = 0;
  private busy = false;
  private lastStep = { hostMs: 0, enqueueMs: 0, waitMs: 0 };

  get chunkSize(): number { return this.chunkSizes[0]; }

  private constructor(pkg: ModelPackage) {
    const s = pkg.model;
    this.pkg = pkg; this.N = s.num_layers; this.types = s.layer_types;
    this.producer = s.layer_types.map((_, i) => (s.kv_producer && s.kv_producer[i] >= 0 ? s.kv_producer[i] : i));
    this.kinds = s.layer_types.map((t, i) => `${t}_${this.producer[i] === i ? 'own' : 'shared'}`);
  }

  // ------------------------------------------------------------------------------------------------------------------------------- loading
  static async load(src: string | PackageSource, options: LoadOptions = {}): Promise<Gemma4LLM> {
    const progress = options.onProgress ?? (() => {});
    progress({ stage: 'package', message: 'reading the package index' });
    return Gemma4LLM.loadFrom(await openPackage(src, options.cache ?? true, { segments: options.downloadSegments }), options);
  }

  static async loadFrom(pkg: ModelPackage, options: LoadOptions = {}): Promise<Gemma4LLM> {
    const t0 = performance.now();
    const progress = options.onProgress ?? (() => {});
    if (pkg.plan.architecture_kind !== 'gemma4') throw new Error('not a Gemma 4 package');
    const llm = new Gemma4LLM(pkg);
    llm.adopt(llm.resolve(options));
    await llm.init(options, progress);
    llm.stats.loadMs = performance.now() - t0;
    return llm;
  }

  private resolve(o: SessionOptions) {
    const plan = this.pkg.plan, key = String(o.maxLength ?? Math.min(...plan.max_lengths));
    if (o.kvStyle && o.kvStyle !== 'staged') throw new Error('Gemma 4 packages support the staged KV protocol only');
    const base = plan.variants.find((v) => v.id === plan.default_variant) as Variant;
    const compute = o.computeType ?? base.compute_type, rows = o.stagingRows ?? base.staging_rows;
    const variant = o.variant ? plan.variants.find((v) => v.id === o.variant) : plan.variants.find((v) => v.compute_type === compute && v.staging_rows === rows);
    if (!variant) throw new Error(`the package has no variant ${o.variant ?? `with compute type ${compute} and ${rows} staging rows`} (it has ${plan.variants.map((v) => v.id).join(', ')})`);
    const table = plan.graphs[variant.id];
    if (!table.decode[key]) throw new Error(`the package has no graphs for max length ${key} (it has ${plan.max_lengths.join(', ')})`);
    const available = Object.keys(table.prefill[key]).map(Number).sort((a, b) => a - b);
    const sizes = [...new Set(o.chunkSizes?.length ? o.chunkSizes : [o.chunkSize ?? available[0]])].sort((a, b) => a - b);
    for (const size of sizes) if (!available.includes(size)) throw new Error(`the package has no prefill graphs for chunk ${size} at max length ${key} (it has ${available.join(', ')})`);
    if ((o.prefillBlocksPerGraph ?? 0) > 1) throw new Error('Gemma 4 packages have block groups for decode only: prefillBlocksPerGraph must be 1');
    let group = o.blocksPerGraph ?? plan.recommended?.blocks_per_graph ?? 1;   // not chosen by the caller: the recommended group size, when this variant has it
    if (o.blocksPerGraph === undefined && group > 1 && !(table.decode[key] as unknown as GemmaDecodeTable).block_groups?.[String(group)]) group = 1;
    const groupSizes = Object.keys((table.decode[key] as unknown as GemmaDecodeTable).block_groups ?? {}).join(', ') || 'none';
    if (group > 1 && !(table.decode[key] as unknown as GemmaDecodeTable).block_groups?.[String(group)]) throw new Error(`the package has no decode block groups of ${group} blocks (it has ${groupSizes})`);
    return { maxLength: +key, variant, sizes, group, fixedTokens: o.prefillFixedTokens ?? 200 };
  }

  private adopt(c: ReturnType<Gemma4LLM['resolve']>) {
    this.maxLength = c.maxLength; this.variant = c.variant; this.chunkSizes = c.sizes; this.P = c.maxLength - 1; this.R = c.variant.staging_rows ?? 0; this.ring = ringOf(this.pkg.plan); this.fixedTokens = c.fixedTokens; this.groupSize = c.group;
  }

  /** rows of the K / V cache of a layer type (the ring for the sliding layers of a window package) */
  private rowsOfType(type: string) { return cacheRows(type, this.P, this.ring); }
  /** the distinct row counts of the caches of this session */
  private cacheRowCounts() { return [...new Set(layerTypesOf(this.pkg.model).map((t) => this.rowsOfType(t)))]; }

  private async createTensor(dataType: string, shape: number[], flags: { writable?: boolean; readable?: boolean } = {}, into = this.stateTensors) {
    checkTensorSize(dataType, shape);
    const tensor = await this.ctx.createTensor({ dataType, shape, writable: flags.writable ?? false, readable: flags.readable ?? false });
    into.push(tensor);
    return tensor;
  }
  private writable = (dataType: string, shape: number[]) => this.createTensor(dataType, shape, { writable: true });
  private dims(type: string) {
    const s = this.pkg.model, o = s.layer_dims?.[type];
    return { heads: o?.num_heads ?? s.num_heads, kvHeads: o?.num_kv_heads ?? s.num_kv_heads, headDim: o?.head_dim ?? s.head_dim };
  }

  private async init(options: LoadOptions, progress: Progress) {
    const { pkg } = this, source = pkg.source;
    this.ctx = options.context ?? await (navigator as any).ml.createContext({ deviceType: options.device ?? 'npu' });
    const role = (r: ManifestFile['role']) => pkg.manifest.files.filter((f) => f.role === r);
    // weight files in the order they are needed; the per-layer embedding table is NOT downloaded: its rows are read as byte ranges
    const ordered = [...role('embedding'), ...role('final_norm'), ...role('head_extra'), ...role('layer').sort((a, b) => (a.layer as number) - (b.layer as number))];
    const total = ordered.reduce((s, f) => s + f.bytes, 0), loaded = new Map<string, number>(), cachedBytes = new Map<string, number>(), sum = (m: Map<string, number>) => [...m.values()].reduce((a, b) => a + b, 0);
    const prefetch = new Prefetcher(ordered.map((f) => f.file), (file, i) => source.bytes(`weights/${file}`, (l, _total, cached) => {
      loaded.set(file, l); if (cached) cachedBytes.set(file, l);
      progress({ stage: 'download', loaded: sum(loaded), total, fromCache: sum(cachedBytes), message: ordered[i].file });
    }, ordered[i].bytes), options.parallelDownloads ?? 4);

    const weightsStart = performance.now();
    progress({ stage: 'embedding', message: 'loading the embedding table' });
    const part = async (name: string) => {
      const file = role('embedding').find((f) => f.tensors[0].name === name) as ManifestFile, tensor = file.tensors[0];
      return new Uint8Array(await prefetch.get(ordered.indexOf(file)), tensor.offset, tensor.bytes);
    };
    const weightBytes = await part('lm_head_weight'), scaleBytes = (await part('lm_head_scale')).slice();
    this.embed = { weight: new Int8Array(weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength), scale: new (globalThis as any).Float16Array(scaleBytes.buffer) };
    // the per-layer embedding table: the scales are read once, the weights row by row
    const pleFile = role('ple_table')[0], pleWeight = pleFile.tensors.find((t) => t.name === 'ple_weight') as ManifestFile['tensors'][number], pleScale = pleFile.tensors.find((t) => t.name === 'ple_scale') as ManifestFile['tensors'][number];
    if (!source.slice) throw new Error('this package source cannot read byte ranges, which Gemma 4 needs for its per-layer embedding table');
    const scales = new Uint8Array(await source.slice(`weights/${pleFile.file}`, pleScale.offset, pleScale.offset + pleScale.bytes));
    this.ple = { path: `weights/${pleFile.file}`, weightOffset: pleWeight.offset, width: pleWeight.shape[1], scale: new (globalThis as any).Float16Array(scales.buffer) };

    await this.buildGraphs(progress, { weight: weightBytes, scale: new Uint8Array(scaleBytes) });

    const uploadShard = async (shard: ManifestFile, buffer: ArrayBuffer) => {
      const tensors: Record<string, ML> = {};
      for (const entry of shard.tensors) {
        const tensor = await this.createTensor(entry.dtype, entry.shape, { writable: true }, this.weightTensors);
        this.ctx.writeTensor(tensor, new Uint8Array(buffer, entry.offset, entry.bytes));
        tensors[entry.name] = tensor;
      }
      return tensors;
    };
    const finalFile = role('final_norm')[0], headFile = role('head_extra')[0];
    this.finalGamma = Object.values(await uploadShard(finalFile, await prefetch.get(ordered.indexOf(finalFile))))[0];
    this.headExtra = await uploadShard(headFile, await prefetch.get(ordered.indexOf(headFile)));
    const layers = role('layer').sort((a, b) => (a.layer as number) - (b.layer as number));
    for (const shard of layers) {
      progress({ stage: 'weights', message: `layer ${(shard.layer as number) + 1}/${layers.length}`, loaded: shard.layer as number, total: layers.length });
      this.weights.push(await uploadShard(shard, await prefetch.get(ordered.indexOf(shard))));
      prefetch.release(ordered.indexOf(shard));
    }
    this.stats.weightsMs = performance.now() - weightsStart - this.stats.buildMs;
    progress({ stage: 'state', message: 'allocating the KV caches and activations' });
    await this.allocateState();
    progress({ stage: 'ready' });
  }

  private async buildGraphs(progress: Progress, lmHead: { weight: Uint8Array; scale: Uint8Array }) {
    const builds: { label: string; ms: number }[] = [], t = performance.now(), plan = this.pkg.plan, source = this.pkg.source, key = String(this.maxLength), table = plan.graphs[this.variant.id];
    const decode = table.decode[key] as unknown as GemmaDecodeTable;
    // the lm_head graph binds the table as row blocks lm_head_weight_<i> / lm_head_scale_<i> when it is too big for one WebNN tensor: contiguous views of the table
    const external = (c: { external?: string; shape?: number[] }) => {
      const part = /^lm_head_(weight|scale)(?:_(\d+))?$/.exec(c.external ?? '');
      if (!part) throw new Error(`the lm_head graph wants the external tensor '${c.external}' which this package does not have`);
      const [, kind, index] = part, rows = (c.shape as number[])[0], first = index === undefined ? 0 : +index * rows, count = index === undefined ? undefined : rows;
      const bytes = kind === 'weight' ? lmHead.weight : lmHead.scale, width = kind === 'weight' ? this.pkg.model.hidden : 2;
      return count === undefined ? (kind === 'weight' ? bytes : new Uint8Array(bytes)) : new Uint8Array(bytes.buffer, bytes.byteOffset + first * width, count * width);
    };
    let expected = 0;   // how many graphs this session builds (for the progress: n of m), set below
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
    const kindsNeeded = [...new Set(this.kinds.slice(1))];
    expected = (this.groupSize > 1 ? new Set((decode.block_groups as Record<string, GemmaBlockGroup[]>)[String(this.groupSize)].map((e) => e.graph)).size : 0) + 1 + Object.keys(decode.flush).length + 2
      + (this.groupSize > 1 ? 0 : kindsNeeded.length) + this.chunkSizes.length * (2 + kindsNeeded.length);
    const blocks = async (paths: Record<string, string>, label: string) => {
      const map = new Map<string, Built>();
      for (const kind of kindsNeeded) map.set(kind, await build(paths[kind], `${label} block ${kind}`));
      return map;
    };
    // decode block groups replace the single decode blocks; groups with the same sequence of kinds share one graph
    const groups: { first: number; blocks: number; built: Built }[] = [], byGraph = new Map<string, Built>();
    if (this.groupSize > 1) {
      for (const e of (decode.block_groups as Record<string, GemmaBlockGroup[]>)[String(this.groupSize)]) {
        if (!byGraph.has(e.graph)) byGraph.set(e.graph, await build(e.graph, `decode group ${e.key}`));
        groups.push({ first: e.first_block, blocks: e.blocks, built: byGraph.get(e.graph) as Built });
      }
    }
    this.lmHead = await build(table.lm_head, 'lm_head', external);
    const flush = new Map<string, Built>();
    for (const [type, path] of Object.entries(decode.flush)) flush.set(type, await build(path, `kv flush ${type}`));
    this.decodeSet = { head: await build(decode.head, 'decode head'), blocks: this.groupSize > 1 ? new Map() : await blocks(decode.blocks, 'decode'), tail: await build(decode.tail, 'decode tail'), flush, groups };
    this.prefillSets = new Map();
    for (const size of this.chunkSizes) {
      const files = table.prefill[key][String(size)] as unknown as GraphTable & { blocks: Record<string, string> }, label = `prefill c${size}`;
      this.prefillSets.set(size, { head: await build(files.head, `${label} head`), blocks: await blocks(files.blocks, label), tail: await build(files.tail, `${label} tail`), act: null as any, sel: null as any });
    }
    this.stats.buildMs = performance.now() - t;
    this.stats.graphBuilds = builds;
  }

  /** Change the session options of a loaded model: graphs are rebuilt and the state reallocated, the weights stay on the device. */
  async reconfigure(options: SessionOptions) {
    if (this.busy) throw new Error('a generation is running on this model');
    const progress = (options as LoadOptions).onProgress ?? (() => {});
    const config = this.resolve(options);
    this.releaseSession();
    this.adopt(config);
    const scale = this.embed;
    await this.buildGraphs(progress, { weight: new Uint8Array(scale.weight.buffer, scale.weight.byteOffset, scale.weight.byteLength), scale: new Uint8Array(scale.scale.buffer.slice(0)) });
    await this.allocateState();
    progress({ stage: 'ready' });
  }

  private releaseSession() {
    for (const t of this.stateTensors) t.destroy?.();
    for (const g of this.builtGraphs) g.destroy?.();
    this.stateTensors = []; this.builtGraphs = [];
    this.kv.clear(); this.kvSpare.clear(); this.stg.clear(); this.stgSpare.clear();
    this.base = this.filled = this.position = 0;
  }

  private async allocateState() {
    const s = this.pkg.model, { R, N } = this, ple = s.ple_dim as number;
    const makeCache = async (layer: number, rows: number): Promise<Cache> => {
      const d = this.dims(this.types[layer]), shape = [1, d.kvHeads, rows, d.headDim], zeros = newHalf(d.kvHeads * rows * d.headDim);
      const c = { key: await this.writable('float16', shape), value: await this.writable('float16', shape) };
      this.ctx.writeTensor(c.key, zeros); this.ctx.writeTensor(c.value, zeros);
      return c;
    };
    for (let l = 0; l < N; l++) {
      if (this.producer[l] !== l) continue;   // a layer that shares K / V has no cache
      const rows = this.rowsOfType(this.types[l]);
      this.kv.set(l, await makeCache(l, rows)); this.kvSpare.set(l, await makeCache(l, rows));
      this.stg.set(l, await makeCache(l, R)); this.stgSpare.set(l, await makeCache(l, R));
    }
    const activations = async (S: number, ownColumns: (type: string) => number, sharedColumns: (type: string) => number): Promise<Activations> => {
      const a: Activations = { hidden: await this.writable('float16', [1, S, s.hidden]), pleToken: await this.writable('float16', [1, S, N * ple]), ropeCos: {}, ropeSin: {}, mask: {}, maskShared: {},
        residuals: [], attentionOutputs: [], ples: [], finalHidden: null };
      for (const type of layerTypesOf(s)) {
        const d = this.dims(type);
        a.ropeCos[type] = await this.writable('float16', [1, 1, S, d.headDim]);
        a.ropeSin[type] = await this.writable('float16', [1, 1, S, d.headDim]);
        a.mask[type] = await this.writable('float16', [1, 1, S, ownColumns(type)]);
        a.maskShared[type] = sharedColumns(type) === ownColumns(type) ? a.mask[type] : await this.writable('float16', [1, 1, S, sharedColumns(type)]);
      }
      for (let l = 0; l < N; l++) {
        a.residuals.push(await this.createTensor('float16', [1, S, s.hidden], l === N - 1 ? { readable: true } : {}));   // the last one is read between prefill chunks of an abortable generation
        a.attentionOutputs.push(await this.createTensor('float16', [1, S, s.hidden]));
        a.ples.push(await this.createTensor('float16', [1, S, ple]));
      }
      a.finalHidden = await this.createTensor('float16', [1, 1, s.hidden]);
      return a;
    };
    const pair = async (rows: number, columns: number): Promise<Selectors> => ({ keep: await this.writable('float16', [1, 1, rows, 1]), select: await this.writable('float16', [1, 1, rows, columns]) });
    for (const [size, set] of this.prefillSets) {
      set.act = await activations(size, (type) => this.rowsOfType(type) + size, (type) => this.rowsOfType(type));   // a layer with K / V of its own attends over concat(cache, chunk rows); a shared layer over the updated cache of its producer
      set.sel = new Map();
      for (const rows of this.cacheRowCounts()) set.sel.set(rows, await pair(rows, size));
    }
    this.dec = await activations(1, (type) => this.rowsOfType(type) + R, (type) => this.rowsOfType(type) + R);
    this.decodeSel = await pair(R, 1);
    this.flushSel = new Map();
    for (const rows of this.cacheRowCounts()) this.flushSel.set(rows, await pair(rows, R));
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
  private pick(names: string[], layer: number) {
    const out: Record<string, ML> = {};
    for (const n of names) if (this.weights[layer][n] !== undefined) out[n] = this.weights[layer][n];
    return out;
  }
  private swap(list: Map<number, Cache>, spare: Map<number, Cache>, layer: number) { const a = list.get(layer) as Cache; list.set(layer, spare.get(layer) as Cache); spare.set(layer, a); }

  /** One forward pass: HEAD, the BLOCKs and (last chunk / decode) TAIL + lm_head. size 0 = decode, else the chunk size of the prefill set. All host inputs of the phase are already written. */
  private enqueue(size: number, lastChunk: boolean) {
    const decode = size === 0, set = decode ? this.decodeSet : (this.prefillSets.get(size) as PrefillSet), a = decode ? this.dec : (set as PrefillSet).act;
    const plan = this.pkg.plan, N = this.N, attentionNames = plan.weights.attention as Record<string, string[]>, ffnNames = plan.weights.ffn;
    // decode: the staging selectors; a prefill chunk: the one-hot selectors of the cache of the layer type (a ring and the full cache have different rows)
    const stagingSelectors = { staging_rows_to_keep: this.decodeSel.keep, staging_row_selector: this.decodeSel.select };
    const selectorsOf = (type: string) => {
      if (decode) return stagingSelectors;
      const sel = (set as PrefillSet).sel.get(this.rowsOfType(type)) as Selectors;
      return { rows_to_keep: sel.keep, row_selector: sel.select };
    };
    const caches = decode ? this.stg : this.kv, spare = decode ? this.stgSpare : this.kvSpare;
    const dispatch = (graph: Built, inputs: Record<string, ML>, outputs: Record<string, ML>) => this.ctx.dispatch(graph.graph, this.bind(graph, inputs), this.bind(graph, outputs, true));
    // the inputs of the attention half of `layer`: rope / mask of its type, its weights, the cache it reads (its own or its producer's)
    const attentionInputs = (layer: number) => {
      const type = this.types[layer], p = this.producer[layer], shared = p !== layer;
      return {
        rope_cos: a.ropeCos[type], rope_sin: a.ropeSin[type], attention_mask: shared ? a.maskShared[type] : a.mask[type], ...selectorsOf(type),
        past_key: (this.kv.get(p) as Cache).key, past_value: (this.kv.get(p) as Cache).value,
        ...(decode ? { staging_key: (this.stg.get(p) as Cache).key, staging_value: (this.stg.get(p) as Cache).value } : {}),
        ...this.pick(attentionNames[this.kinds[layer]], layer),
      };
    };
    const kvOutputs = (layer: number) => (this.producer[layer] === layer ? { updated_key: (spare.get(layer) as Cache).key, updated_value: (spare.get(layer) as Cache).value } : {});
    const swapIfOwn = (layer: number) => { if (this.producer[layer] === layer) this.swap(caches, spare, layer); };

    // HEAD: the per-layer inputs + the attention half of layer 0
    const pleOutputs: Record<string, ML> = {};
    a.ples.forEach((t, i) => { pleOutputs[`ple_${i}`] = t; });
    dispatch(set.head, { hidden_states: a.hidden, ple_token: a.pleToken, ...this.headExtra, ...attentionInputs(0) },
      { residual: a.residuals[0], attention_output: a.attentionOutputs[0], ...pleOutputs, ...kvOutputs(0) });
    swapIfOwn(0);
    // BLOCK i: feed-forward half of layer i + attention half of layer i + 1
    if (decode && this.groupSize > 1) {
      // a block group (decode only): BLOCKs first .. first + blocks - 1 in one dispatch; the inputs are bound by name
      const bindGroup = (first: number, name: string): ML => {
        if (name === 'residual') return a.residuals[first];
        if (name === 'attention_output') return a.attentionOutputs[first];
        if (name in stagingSelectors) return (stagingSelectors as Record<string, ML>)[name];
        const block = /^b(\d+)_(.+)$/.exec(name);
        if (block) {
          const ffnLayer = first + +block[1], layer = ffnLayer + 1, rest = block[2], p = this.producer[layer];
          if (rest === 'per_layer_input') return a.ples[ffnLayer];
          if (rest === 'past_key' || rest === 'past_value') return (this.kv.get(p) as Cache)[rest === 'past_key' ? 'key' : 'value'];
          if (rest === 'staging_key' || rest === 'staging_value') return (this.stg.get(p) as Cache)[rest === 'staging_key' ? 'key' : 'value'];
          return ffnNames.includes(rest) ? this.weights[ffnLayer][rest] : attentionNames[this.kinds[layer]].includes(rest) ? this.weights[layer][rest] : undefined;
        }
        const control = /^(.+)_(rope_cos|rope_sin|attention_mask)$/.exec(name);
        return control ? (control[2] === 'rope_cos' ? a.ropeCos : control[2] === 'rope_sin' ? a.ropeSin : a.mask)[control[1]] : undefined;
      };
      for (const { first, blocks, built } of this.decodeSet.groups) {
        const inputs: Record<string, ML> = {}, outputs: Record<string, ML> = { next_residual: a.residuals[first + blocks], next_attention_output: a.attentionOutputs[first + blocks] };
        for (const name of built.inputs) inputs[name] = bindGroup(first, name);
        for (let j = 0; j < blocks; j++) {
          const layer = first + j + 1;   // a block with K / V of its own writes the spare staging buffer of its layer
          if (this.producer[layer] === layer) { outputs[`b${j}_updated_key`] = (spare.get(layer) as Cache).key; outputs[`b${j}_updated_value`] = (spare.get(layer) as Cache).value; }
        }
        dispatch(built, inputs, outputs);
        for (let j = 0; j < blocks; j++) swapIfOwn(first + j + 1);
      }
    } else {
      for (let layer = 0; layer < N - 1; layer++) {
        dispatch(set.blocks.get(this.kinds[layer + 1]) as Built,
          { residual: a.residuals[layer], attention_output: a.attentionOutputs[layer], per_layer_input: a.ples[layer], ...this.pick(ffnNames, layer), ...attentionInputs(layer + 1) },
          { next_residual: a.residuals[layer + 1], next_attention_output: a.attentionOutputs[layer + 1], ...kvOutputs(layer + 1) });
        swapIfOwn(layer + 1);
      }
    }
    if (!lastChunk) return;
    dispatch(set.tail, { residual: a.residuals[N - 1], attention_output: a.attentionOutputs[N - 1], per_layer_input: a.ples[N - 1], ...this.pick(ffnNames, N - 1), final_norm_gamma: this.finalGamma },
      { final_hidden_states: a.finalHidden });
    dispatch(this.lmHead, { final_hidden_states: a.finalHidden }, { logits: this.logitsTensor });
  }

  // ------------------------------------------------------------------------------------------------------------------------------- host side
  private embedRow(token: number, out: Half, offset: number) {
    const hidden = this.pkg.model.hidden, start = token * hidden, scale = this.embed.scale[token] * (this.pkg.model.embed_scale as number);
    for (let i = 0; i < hidden; i++) out[offset + i] = this.embed.weight[start + i] * scale;
  }

  /** the per-layer embedding rows of the tokens (int8 row * row scale) into `out` at `offsets[i]`; a few byte-range reads at a time */
  private async pleRows(tokens: number[], out: Half, offsets: number[]) {
    const { path, weightOffset, width, scale } = this.ple, source = this.pkg.source as PackageSource & { slice: NonNullable<PackageSource['slice']> };
    let next = 0;
    const worker = async () => {
      while (next < tokens.length) {
        const i = next++, start = weightOffset + tokens[i] * width;
        const row = new Int8Array(await source.slice(path, start, start + width)), s = scale[tokens[i]];
        for (let j = 0; j < width; j++) out[offsets[i] + j] = row[j] * s;
      }
    };
    await Promise.all(Array.from({ length: Math.min(8, tokens.length) }, worker));
  }

  private writeRope(tables: ReturnType<typeof ropeTables>, a: Activations) {
    for (const type of layerTypesOf(this.pkg.model)) { this.ctx.writeTensor(a.ropeCos[type], tables[type].cos); this.ctx.writeTensor(a.ropeSin[type], tables[type].sin); }
  }
  private async readLogits(): Promise<Half> {
    await this.ctx.readTensor(this.logitsTensor, this.logits);
    return this.logits;
  }

  planFor(tokenCount: number): number[] { return planPrefill(this.chunkSizes, tokenCount, this.fixedTokens); }

  private get cacheFull() { return this.position >= this.P; }

  // ------------------------------------------------------------------------------------------------------------------------------- generation
  /** returns the logits of the last prompt token (null: `signal` was aborted between two chunks) */
  private async prefill(tokens: number[], signal?: AbortSignal, onProgress?: Progress): Promise<Half | null> {
    const s = this.pkg.model, { P } = this, n = tokens.length, pleSize = this.N * (s.ple_dim as number);
    if (n === 0) throw new Error('empty prompt');
    const plan = this.planFor(n);
    if (n >= P) throw new Error(`the prompt has ${n} tokens but the KV cache holds ${P}: load the package with a larger max length`);
    this.stats.prefillPlan = plan;
    let start = 0;
    for (let index = 0; index < plan.length; index++) {
      if (signal?.aborted) return null;
      const C = plan[index], last = index === plan.length - 1, set = this.prefillSets.get(C) as PrefillSet, a = set.act;
      const valid = Math.min(C, n - start), pad = C - valid, hidden = newHalf(C * s.hidden), pleToken = newHalf(C * pleSize);
      for (let i = 0; i < valid; i++) this.embedRow(tokens[start + i], hidden, (pad + i) * s.hidden);
      await this.pleRows(tokens.slice(start, start + valid), pleToken, Array.from({ length: valid }, (_, i) => (pad + i) * pleSize));
      this.ctx.writeTensor(a.hidden, hidden); this.ctx.writeTensor(a.pleToken, pleToken);
      this.writeRope(ropeTables(s, Array.from({ length: valid }, (_, i) => start + i), C, pad), a);
      const own = prefillMasks(s, P, C, start, valid, this.ring), shared = incachePrefillMasks(s, P, C, start, valid, this.ring);
      for (const type of layerTypesOf(s)) { this.ctx.writeTensor(a.mask[type], own[type]); this.ctx.writeTensor(a.maskShared[type], shared[type]); }
      for (const [rows, sel] of set.sel) {   // a ring writes slot (start + i) mod ring
        const { rowsToKeep, rowSelector } = prefillSelectors(rows, C, start, valid, rows < P);
        this.ctx.writeTensor(sel.keep, rowsToKeep); this.ctx.writeTensor(sel.select, rowSelector);
      }
      this.enqueue(C, last);
      if (signal && !last) {   // abortable: wait for the device to finish this chunk (read back its last residual), then look at the signal; without a signal the chunks stay queued back to back
        await this.ctx.readTensor(a.residuals[this.N - 1], newHalf(C * s.hidden));
        if (signal.aborted) return null;
      }
      onProgress?.({ stage: 'prefill', loaded: index + 1, total: plan.length });
      start += valid;
    }
    this.base = n; this.filled = 0; this.position = n;
    return this.readLogits();
  }

  private flush() {
    for (const [rows, sel] of this.flushSel) {   // one selector pair per cache size; a ring takes the staging rows at slots (base + i) mod ring
      const { rowsToKeep, rowSelector } = flushSelectors(rows, this.R, this.base, rows < this.P);
      this.ctx.writeTensor(sel.keep, rowsToKeep); this.ctx.writeTensor(sel.select, rowSelector);
    }
    for (const layer of this.kv.keys()) {
      const g = this.decodeSet.flush.get(this.types[layer]) as Built, main = this.kv.get(layer) as Cache, staging = this.stg.get(layer) as Cache;
      const sel = this.flushSel.get(this.rowsOfType(this.types[layer])) as Selectors;
      this.ctx.dispatch(g.graph, this.bind(g, { past_key: main.key, past_value: main.value, new_key: staging.key, new_value: staging.value, rows_to_keep: sel.keep, row_selector: sel.select }),
        this.bind(g, { updated_key: (this.kvSpare.get(layer) as Cache).key, updated_value: (this.kvSpare.get(layer) as Cache).value }, true));
      this.swap(this.kv, this.kvSpare, layer);
    }
    this.base += this.R; this.filled = 0;
  }

  private async step(token: number): Promise<Half> {
    const s = this.pkg.model, { P, R } = this, pleSize = this.N * (s.ple_dim as number);
    if (this.cacheFull) throw new Error('the KV cache is full');
    let flushMs = 0;
    if (this.filled === R) { const f = performance.now(); this.flush(); flushMs = performance.now() - f; this.stats.flushes++; }
    const t0 = performance.now();
    const pos = this.position, a = this.dec, hidden = newHalf(s.hidden), pleToken = newHalf(pleSize);
    this.embedRow(token, hidden, 0);
    await this.pleRows([token], pleToken, [0]);
    this.ctx.writeTensor(a.hidden, hidden); this.ctx.writeTensor(a.pleToken, pleToken);
    this.writeRope(ropeTables(s, [pos], 1, 0), a);
    const masks = decodeMasks(s, P, R, pos, this.base, this.filled, this.ring);
    for (const type of layerTypesOf(s)) this.ctx.writeTensor(a.mask[type], masks[type]);   // maskShared is the same tensor in decode
    const { rowsToKeep, rowSelector } = decodeSelectors(R, this.filled);
    this.ctx.writeTensor(this.decodeSel.keep, rowsToKeep); this.ctx.writeTensor(this.decodeSel.select, rowSelector);
    const t1 = performance.now();
    this.enqueue(0, true);
    const t2 = performance.now();
    this.filled++; this.position++;
    const logits = await this.readLogits();
    this.lastStep = { hostMs: t1 - t0, enqueueMs: t2 - t1 + flushMs, waitMs: performance.now() - t2 };
    return logits;
  }

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
        this.stats.stepMs.push(stepMs);
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


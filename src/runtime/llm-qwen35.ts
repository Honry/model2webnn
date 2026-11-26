// LLM runtime for Qwen3.5 packages (plan.architecture_kind === 'qwen35'): the NPUW-style chain of llm.ts (HEAD / BLOCK x (N-1) / TAIL / lm_head, chunked prefill, staged KV) for a HYBRID model:
//  * two kinds of attention half = two BLOCK graphs: `full_attention` (KV caches + staging buffers, only for these layers) and `linear_attention` (Gated DeltaNet, no KV cache: a conv state
//    [1,3,channels] and a recurrent state [1,vHeads,kDim,vDim], float16, that the layer's graph reads and writes: every state tensor has a spare and the two are swapped after the dispatch);
//  * the states are zeroed at the start of every prompt; prefill graphs of linear layers take two controls built by protocol.linearPrefillControls (valid_mask, conv_perm) for the left-padded
//    last chunk;
//  * the embedding is not scaled; the lm_head is built in row blocks of the table (it is too big for one WebNN tensor).
//  * images (a package with an image tower, LoadOptions.vision): `generate(tokens, { images })` with one <|image_pad|> per image in `tokens`: the pad is expanded to one token per merge block of the
//    grid, the image tower (vision.ts) turns the patches into image tokens whose rows REPLACE the embedding rows of the pad tokens in the prefill chunks, the full attention layers get 3-position
//    interleaved MRoPE tables (protocol.ts ropeTablesMrope) and decode token i has the rope position i + ropeDelta. Without images everything is as before.
//  * decode block groups (blocksPerGraph = K > 1, a package built with --blocks-per-graph K): the N - 1 BLOCKs in runs of K, one dispatch per run (plan.chain.block_groups); the inputs of a group graph
//    are bound by NAME: residual / attention_output of its first block, the shared controls of the full attention layers, and b<j>_ = the weights and the state / caches of block j. Prefill has no groups.
// Python counterpart (the executable specification): python/m2w/reference/qwen35_runner.py. Only the staged KV protocol exists.
import { loadGraph, type GraphDoc } from './ir-loader.js';
import { VisionTower, type PreparedImage } from './vision.js';
import { checkTensorSize, emptyStats, Prefetcher, type Built, type Stats } from './llm-common.js';
import type { GenerateOptions, LoadOptions, SessionOptions } from './llm.js';
import { openPackage, type GemmaBlockGroup, type GraphTable, type ManifestFile, type ModelPackage, type PackageSource, type Progress, type QwenDecodeTable, type Variant } from './package.js';
import {
  decodeMasks, decodeSelectors, expandImageTokens, flushSelectors, layerTypesOf, linearPrefillControls, mropePositions, newHalf, planPrefill, prefillMasks, prefillSelectors,
  ropeTables, ropeTablesMrope, type Detail, type Grid, type Half, type LayoutSpec,
} from './protocol.js';

/* eslint-disable @typescript-eslint/no-explicit-any */
type ML = any;

interface Activations { hidden: ML; ropeCos: ML; ropeSin: ML; mask: ML; residuals: ML[]; attentionOutputs: ML[]; finalHidden: ML }
interface Selectors { keep: ML; select: ML }
interface PrefillSet { head: Built; blocks: Map<string, Built>; tail: Built; act: Activations; sel: Selectors; validMask: ML; convPerm: ML }
type Cache = { key: ML; value: ML };
type LinearState = { conv: ML; rec: ML };

export class Qwen35LLM {
  /** debugging (bench/raw/qwen35check.html): create the per-layer residual tensors readable so that debugTrace() can read them back; set BEFORE load */
  static debugReadback = false;
  readonly pkg: ModelPackage;
  maxLength = 0;
  variant!: Variant;
  chunkSizes: number[] = [];
  stats: Stats = emptyStats();

  private ctx!: ML;
  private P = 0; private R = 0; private N: number;
  private fixedTokens = 200; private groupSize = 1;
  private kinds: string[];
  private weights: Record<string, ML>[] = [];
  private finalGamma!: ML;
  private lmHead!: Built;
  private decodeSet!: { head: Built; blocks: Map<string, Built>; tail: Built; flush: Built; groups: { first: number; blocks: number; built: Built }[] };
  private prefillSets = new Map<number, PrefillSet>();
  private kv = new Map<number, Cache>(); private kvSpare = new Map<number, Cache>(); private stg = new Map<number, Cache>(); private stgSpare = new Map<number, Cache>();
  private state = new Map<number, LinearState>(); private stateSpare = new Map<number, LinearState>();
  private dec!: Activations;
  private decodeSel!: Selectors; private flushSel!: Selectors;
  private logitsTensor!: ML; private logits!: Half;
  private embed!: { weight: Int8Array; scale: Half };
  private weightTensors: ML[] = []; private stateTensors: ML[] = []; private builtGraphs: ML[] = [];
  private base = 0; private filled = 0; private position = 0; private ropeDelta = 0;
  /** the image tower when the model was loaded with LoadOptions.vision */
  vision: VisionTower | null = null;
  private busy = false;
  private lastStep = { hostMs: 0, enqueueMs: 0, waitMs: 0 };
  private queue: { label: string; run: () => void; sync?: ML }[] | null = null;   // debugTrace(stepwise): dispatches are collected and run one by one, each followed by a readback
  private onDispatch: ((label: string) => void | Promise<void>) | null = null;

  get chunkSize(): number { return this.chunkSizes[0]; }

  private constructor(pkg: ModelPackage) {
    this.pkg = pkg; this.N = pkg.model.num_layers; this.kinds = pkg.model.layer_types;
  }

  // ------------------------------------------------------------------------------------------------------------------------------- loading
  static async load(src: string | PackageSource, options: LoadOptions = {}): Promise<Qwen35LLM> {
    const progress = options.onProgress ?? (() => {});
    progress({ stage: 'package', message: 'reading the package index' });
    return Qwen35LLM.loadFrom(await openPackage(src, options.cache ?? true, { segments: options.downloadSegments }), options);
  }

  static async loadFrom(pkg: ModelPackage, options: LoadOptions = {}): Promise<Qwen35LLM> {
    const t0 = performance.now();
    const progress = options.onProgress ?? (() => {});
    if (pkg.plan.architecture_kind !== 'qwen35') throw new Error('not a Qwen3.5 package');
    const llm = new Qwen35LLM(pkg);
    llm.adopt(llm.resolve(options));
    await llm.init(options, progress);
    if (options.vision) {
      const vision = typeof options.vision === 'object' ? options.vision : {};
      llm.vision = await VisionTower.create(llm.ctx, pkg, { grids: vision.grids as Grid[] | undefined, preload: vision.preload as Grid[] | undefined }, progress);
    }
    llm.stats.loadMs = performance.now() - t0;
    return llm;
  }

  private resolve(o: SessionOptions) {
    const plan = this.pkg.plan, key = String(o.maxLength ?? Math.min(...plan.max_lengths));
    if (o.kvStyle && o.kvStyle !== 'staged') throw new Error('Qwen3.5 packages support the staged KV protocol only');
    const base = plan.variants.find((v) => v.id === plan.default_variant) as Variant;
    const compute = o.computeType ?? base.compute_type, rows = o.stagingRows ?? base.staging_rows;
    const variant = o.variant ? plan.variants.find((v) => v.id === o.variant) : plan.variants.find((v) => v.compute_type === compute && v.staging_rows === rows);
    if (!variant) throw new Error(`the package has no variant ${o.variant ?? `with compute type ${compute} and ${rows} staging rows`} (it has ${plan.variants.map((v) => v.id).join(', ')})`);
    const table = plan.graphs[variant.id];
    if (!table.decode[key]) throw new Error(`the package has no graphs for max length ${key} (it has ${plan.max_lengths.join(', ')})`);
    const available = Object.keys(table.prefill[key]).map(Number).sort((a, b) => a - b);
    const recommendedChunks = (plan.recommended?.prefill_chunks ?? []).filter((c) => available.includes(c));   // not chosen by the caller: the recommended chunk sizes of the package
    const sizes = [...new Set(o.chunkSizes?.length ? o.chunkSizes : o.chunkSize !== undefined ? [o.chunkSize] : recommendedChunks.length ? recommendedChunks : [available[0]])].sort((a, b) => a - b);
    for (const size of sizes) if (!available.includes(size)) throw new Error(`the package has no prefill graphs for chunk ${size} at max length ${key} (it has ${available.join(', ')})`);
    if ((o.prefillBlocksPerGraph ?? 0) > 1) throw new Error('Qwen3.5 packages have block groups for decode only: prefillBlocksPerGraph must be 1');
    const groupsOf = (table.decode[key] as unknown as QwenDecodeTable).block_groups ?? {};
    let group = o.blocksPerGraph ?? plan.recommended?.blocks_per_graph ?? 1;   // not chosen by the caller: the recommended group size, when this variant has it
    if (o.blocksPerGraph === undefined && group > 1 && !groupsOf[String(group)]) group = 1;
    if (group > 1 && !groupsOf[String(group)]) throw new Error(`the package has no decode block groups of ${group} blocks (it has ${Object.keys(groupsOf).join(', ') || 'none'})`);
    return { maxLength: +key, variant, sizes, group, fixedTokens: o.prefillFixedTokens ?? 200 };
  }

  private adopt(c: ReturnType<Qwen35LLM['resolve']>) {
    this.maxLength = c.maxLength; this.variant = c.variant; this.chunkSizes = c.sizes; this.P = c.maxLength - 1; this.R = c.variant.staging_rows ?? 0; this.fixedTokens = c.fixedTokens; this.groupSize = c.group;
  }

  private async createTensor(dataType: string, shape: number[], flags: { writable?: boolean; readable?: boolean } = {}, into = this.stateTensors) {
    checkTensorSize(dataType, shape);
    const tensor = await this.ctx.createTensor({ dataType, shape, writable: flags.writable ?? false, readable: flags.readable ?? false });
    into.push(tensor);
    return tensor;
  }
  private writable = (dataType: string, shape: number[]) => this.createTensor(dataType, shape, { writable: true });

  private async init(options: LoadOptions, progress: Progress) {
    const { pkg } = this, source = pkg.source;
    this.ctx = options.context ?? await (navigator as any).ml.createContext({ deviceType: options.device ?? 'npu' });
    const role = (r: ManifestFile['role']) => pkg.manifest.files.filter((f) => f.role === r);
    const ordered = [...role('embedding'), ...role('final_norm'), ...role('layer').sort((a, b) => (a.layer as number) - (b.layer as number))];
    const total = ordered.reduce((s, f) => s + f.bytes, 0), loaded = new Map<string, number>();
    const prefetch = new Prefetcher(ordered.map((f) => f.file), (file, i) => source.bytes(`weights/${file}`, (l) => {
      loaded.set(file, l);
      progress({ stage: 'download', loaded: [...loaded.values()].reduce((a, b) => a + b, 0), total, message: ordered[i].file });
    }, ordered[i].bytes), options.parallelDownloads ?? 4);

    const weightsStart = performance.now();
    progress({ stage: 'embedding', message: 'loading the embedding table' });
    const part = async (name: string) => {
      const file = role('embedding').find((f) => f.tensors[0].name === name) as ManifestFile, tensor = file.tensors[0];
      return new Uint8Array(await prefetch.get(ordered.indexOf(file)), tensor.offset, tensor.bytes);
    };
    const weightBytes = await part('lm_head_weight'), scaleBytes = (await part('lm_head_scale')).slice();
    this.embed = { weight: new Int8Array(weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength), scale: new (globalThis as any).Float16Array(scaleBytes.buffer) };

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
    const finalFile = role('final_norm')[0];
    this.finalGamma = Object.values(await uploadShard(finalFile, await prefetch.get(ordered.indexOf(finalFile))))[0];
    const layers = role('layer').sort((a, b) => (a.layer as number) - (b.layer as number));
    for (const shard of layers) {
      progress({ stage: 'weights', message: `layer ${(shard.layer as number) + 1}/${layers.length}`, loaded: shard.layer as number, total: layers.length });
      this.weights.push(await uploadShard(shard, await prefetch.get(ordered.indexOf(shard))));
      prefetch.release(ordered.indexOf(shard));
    }
    this.stats.weightsMs = performance.now() - weightsStart - this.stats.buildMs;
    progress({ stage: 'state', message: 'allocating the KV caches, the linear attention states and the activations' });
    await this.allocateState();
    progress({ stage: 'ready' });
  }

  private async buildGraphs(progress: Progress, lmHead: { weight: Uint8Array; scale: Uint8Array }) {
    const builds: { label: string; ms: number }[] = [], t = performance.now(), plan = this.pkg.plan, source = this.pkg.source, key = String(this.maxLength), table = plan.graphs[this.variant.id];
    const decode = table.decode[key] as unknown as QwenDecodeTable;
    // the lm_head graph binds the table as row blocks lm_head_weight_<i> / lm_head_scale_<i> when it is too big for one WebNN tensor: contiguous views of the table
    const external = (c: { external?: string; shape?: number[] }) => {
      const part = /^lm_head_(weight|scale)(?:_(\d+))?$/.exec(c.external ?? '');
      if (!part) throw new Error(`the lm_head graph wants the external tensor '${c.external}' which this package does not have`);
      const [, kind, index] = part, rows = (c.shape as number[])[0], first = index === undefined ? 0 : +index * rows, count = index === undefined ? undefined : rows;
      const bytes = kind === 'weight' ? lmHead.weight : lmHead.scale, width = kind === 'weight' ? this.pkg.model.hidden : 2;
      return count === undefined ? (kind === 'weight' ? bytes : new Uint8Array(bytes)) : new Uint8Array(bytes.buffer, bytes.byteOffset + first * width, count * width);
    };
    const build = async (path: string, label: string, provider?: (c: any) => ArrayBufferView): Promise<Built> => {
      progress({ stage: 'build', message: label });
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
    const flush = await build(decode.flush, 'kv flush');
    this.decodeSet = { head: await build(decode.head, 'decode head'), blocks: this.groupSize > 1 ? new Map() : await blocks(decode.blocks, 'decode'), tail: await build(decode.tail, 'decode tail'), flush, groups };
    this.prefillSets = new Map();
    for (const size of this.chunkSizes) {
      const files = table.prefill[key][String(size)] as unknown as GraphTable & { blocks: Record<string, string> }, label = `prefill c${size}`;
      this.prefillSets.set(size, { head: await build(files.head, `${label} head`), blocks: await blocks(files.blocks, label), tail: await build(files.tail, `${label} tail`),
        act: null as any, sel: null as any, validMask: null, convPerm: null });
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
    const table = this.embed;
    await this.buildGraphs(progress, { weight: new Uint8Array(table.weight.buffer, table.weight.byteOffset, table.weight.byteLength), scale: new Uint8Array(table.scale.buffer.slice(0)) });
    await this.allocateState();
    progress({ stage: 'ready' });
  }

  private releaseSession() {
    for (const t of this.stateTensors) t.destroy?.();
    for (const g of this.builtGraphs) g.destroy?.();
    this.stateTensors = []; this.builtGraphs = [];
    this.kv.clear(); this.kvSpare.clear(); this.stg.clear(); this.stgSpare.clear(); this.state.clear(); this.stateSpare.clear();
    this.base = this.filled = this.position = 0;
  }

  private async allocateState() {
    const s = this.pkg.model, { P, R, N } = this, ld = this.pkg.plan.state;
    const makeCache = async (rows: number): Promise<Cache> => {
      const shape = [1, s.num_kv_heads, rows, s.head_dim], zeros = newHalf(s.num_kv_heads * rows * s.head_dim);
      const c = { key: await this.writable('float16', shape), value: await this.writable('float16', shape) };
      this.ctx.writeTensor(c.key, zeros); this.ctx.writeTensor(c.value, zeros);
      return c;
    };
    const makeState = async (): Promise<LinearState> => ({ conv: await this.writable('float16', [1, 3, (ld as any).conv_channels]), rec: await this.writable('float16', [1, (ld as any).v_heads, (ld as any).k_dim, (ld as any).v_dim]) });
    for (let l = 0; l < N; l++) {
      if (this.kinds[l] === 'linear_attention') { this.state.set(l, await makeState()); this.stateSpare.set(l, await makeState()); continue; }
      this.kv.set(l, await makeCache(P)); this.kvSpare.set(l, await makeCache(P));
      this.stg.set(l, await makeCache(R)); this.stgSpare.set(l, await makeCache(R));
    }
    const activations = async (S: number, columns: number): Promise<Activations> => {
      const a: Activations = { hidden: await this.writable('float16', [1, S, s.hidden]), ropeCos: await this.writable('float16', [1, 1, S, s.head_dim]), ropeSin: await this.writable('float16', [1, 1, S, s.head_dim]),
        mask: await this.writable('float16', [1, 1, S, columns]), residuals: [], attentionOutputs: [], finalHidden: null };
      const readable = { readable: Qwen35LLM.debugReadback };
      for (let l = 0; l < N; l++) {
        a.residuals.push(await this.createTensor('float16', [1, S, s.hidden], readable));
        a.attentionOutputs.push(await this.createTensor('float16', [1, S, s.hidden]));
      }
      a.finalHidden = await this.createTensor('float16', [1, 1, s.hidden], readable);
      return a;
    };
    const pair = async (rows: number, columns: number): Promise<Selectors> => ({ keep: await this.writable('float16', [1, 1, rows, 1]), select: await this.writable('float16', [1, 1, rows, columns]) });
    for (const [size, set] of this.prefillSets) {
      set.act = await activations(size, P + size);
      set.sel = await pair(P, size);
      set.validMask = await this.writable('float16', [1, 1, size, 1]);
      set.convPerm = await this.writable('float16', [1, size + 3, size + 3]);
    }
    this.dec = await activations(1, P + R);
    this.decodeSel = await pair(R, 1);
    this.flushSel = await pair(P, R);
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
  private swap<T>(list: Map<number, T>, spare: Map<number, T>, layer: number) { const a = list.get(layer) as T; list.set(layer, spare.get(layer) as T); spare.set(layer, a); }

  /** One forward pass: HEAD, the BLOCKs and (last chunk / decode) TAIL + lm_head. size 0 = decode, else the chunk size of the prefill set. All host inputs of the phase are already written. */
  private enqueue(size: number, lastChunk: boolean) {
    const decode = size === 0, set = decode ? this.decodeSet : (this.prefillSets.get(size) as PrefillSet), a = decode ? this.dec : (set as PrefillSet).act;
    const plan = this.pkg.plan, N = this.N, attentionNames = plan.weights.attention as Record<string, string[]>, ffnNames = plan.weights.ffn;
    const sel = decode ? this.decodeSel : (set as PrefillSet).sel;
    const selectors = decode ? { staging_rows_to_keep: sel.keep, staging_row_selector: sel.select } : { rows_to_keep: sel.keep, row_selector: sel.select };
    const controls = decode ? {} : { valid_mask: (set as PrefillSet).validMask, conv_perm: (set as PrefillSet).convPerm };
    const caches = decode ? this.stg : this.kv, spare = decode ? this.stgSpare : this.kvSpare;
    const dispatch = (graph: Built, inputs: Record<string, ML>, outputs: Record<string, ML>) => {
      const ins = this.bind(graph, inputs), outs = this.bind(graph, outputs, true);
      if (this.queue) this.queue.push({ label: graph.name, run: () => this.ctx.dispatch(graph.graph, ins, outs), sync: Object.values(outs).find((t: ML) => t.readable) });
      else this.ctx.dispatch(graph.graph, ins, outs);
    };
    // the inputs / outputs of the attention half of `layer`
    const attentionInputs = (layer: number) => {
      const kind = this.kinds[layer], names = attentionNames[kind];
      if (kind === 'linear_attention') {
        const st = this.state.get(layer) as LinearState;
        return { conv_state: st.conv, recurrent_state: st.rec, ...controls, ...this.pick(names, layer) };
      }
      return {
        rope_cos: a.ropeCos, rope_sin: a.ropeSin, attention_mask: a.mask, ...selectors,
        past_key: (this.kv.get(layer) as Cache).key, past_value: (this.kv.get(layer) as Cache).value,
        ...(decode ? { staging_key: (this.stg.get(layer) as Cache).key, staging_value: (this.stg.get(layer) as Cache).value } : {}),
        ...this.pick(names, layer),
      };
    };
    const stateOutputs = (layer: number) => {
      if (this.kinds[layer] === 'linear_attention') { const sp = this.stateSpare.get(layer) as LinearState; return { new_conv_state: sp.conv, new_recurrent_state: sp.rec }; }
      return { updated_key: (spare.get(layer) as Cache).key, updated_value: (spare.get(layer) as Cache).value };
    };
    const swapState = (layer: number) => { if (this.kinds[layer] === 'linear_attention') this.swap(this.state, this.stateSpare, layer); else this.swap(caches, spare, layer); };

    dispatch(set.head, { hidden_states: a.hidden, ...attentionInputs(0) }, { residual: a.residuals[0], attention_output: a.attentionOutputs[0], ...stateOutputs(0) });
    swapState(0);
    // BLOCK i: feed-forward half of layer i + attention half of layer i + 1
    if (decode && this.groupSize > 1) {
      // a block group (decode only): BLOCKs first .. first + blocks - 1 in one dispatch; the inputs are bound by name: b<j>_<name> = block j's weight / state / cache, the rest is shared
      const shared: Record<string, ML> = { rope_cos: a.ropeCos, rope_sin: a.ropeSin, attention_mask: a.mask, ...selectors };
      for (const { first, blocks, built } of this.decodeSet.groups) {
        const inputs: Record<string, ML> = { residual: a.residuals[first], attention_output: a.attentionOutputs[first] };
        const outputs: Record<string, ML> = { next_residual: a.residuals[first + blocks], next_attention_output: a.attentionOutputs[first + blocks] };
        for (let j = 0; j < blocks; j++) {
          const layer = first + j + 1, own = { ...this.pick(ffnNames, first + j), ...attentionInputs(layer) };
          for (const [name, tensor] of Object.entries(own)) { if (!(name in shared)) inputs[`b${j}_${name}`] = tensor; }
          for (const [name, tensor] of Object.entries(stateOutputs(layer))) outputs[`b${j}_${name}`] = tensor;
        }
        for (const name of built.inputs) if (name in shared) inputs[name] = shared[name];
        dispatch(built, inputs, outputs);
        for (let j = 0; j < blocks; j++) swapState(first + j + 1);
      }
    } else {
      for (let layer = 0; layer < N - 1; layer++) {
        dispatch(set.blocks.get(this.kinds[layer + 1]) as Built, { residual: a.residuals[layer], attention_output: a.attentionOutputs[layer], ...this.pick(ffnNames, layer), ...attentionInputs(layer + 1) },
          { next_residual: a.residuals[layer + 1], next_attention_output: a.attentionOutputs[layer + 1], ...stateOutputs(layer + 1) });
        swapState(layer + 1);
      }
    }
    if (!lastChunk) return;
    dispatch(set.tail, { residual: a.residuals[N - 1], attention_output: a.attentionOutputs[N - 1], ...this.pick(ffnNames, N - 1), final_norm_gamma: this.finalGamma }, { final_hidden_states: a.finalHidden });
    dispatch(this.lmHead, { final_hidden_states: a.finalHidden }, { logits: this.logitsTensor });
  }

  // ------------------------------------------------------------------------------------------------------------------------------- host side
  /** a picture -> the patches of the image tower (the grid preset is chosen automatically from its size and `detail`, canvas resize, normalise, patchify); needs LoadOptions.vision */
  prepareImage(image: { width: number; height: number }, grid?: Grid, detail?: Detail): PreparedImage {
    if (!this.vision) throw new Error('this model was loaded without the image tower: pass { vision: true } to load()');
    return this.vision.prepare(image, grid, detail);
  }

  /** the prompt with its <|image_pad|> tokens expanded for these images */
  expandPrompt(tokens: number[], images: PreparedImage[]): number[] {
    return expandImageTokens(this.visionLayout(), tokens, images.map((i) => i.grid));
  }

  private visionLayout() {
    const v = this.pkg.model.vision;
    if (!v) throw new Error('this package has no image tower');
    return { image_token_id: v.image_token_id, merge_size: v.merge_size };
  }

  private embedRow(token: number, out: Half, offset: number) {
    const hidden = this.pkg.model.hidden, start = token * hidden, scale = this.embed.scale[token] * (this.pkg.model.embed_scale ?? 1);
    for (let i = 0; i < hidden; i++) out[offset + i] = this.embed.weight[start + i] * scale;
  }

  private writeRope(tables: ReturnType<typeof ropeTables>, a: Activations) {
    const type = layerTypesOf(this.pkg.model)[0];
    this.ctx.writeTensor(a.ropeCos, tables[type].cos); this.ctx.writeTensor(a.ropeSin, tables[type].sin);
  }
  private async readLogits(): Promise<Half> {
    await this.ctx.readTensor(this.logitsTensor, this.logits);
    return this.logits;
  }

  /** a new prompt starts with zero conv / recurrent states in every linear attention layer */
  private resetStates() {
    const ld = this.pkg.plan.state as NonNullable<typeof this.pkg.plan.state>, conv = newHalf(3 * ld.conv_channels), rec = newHalf(ld.v_heads * ld.k_dim * ld.v_dim);
    for (const st of this.state.values()) { this.ctx.writeTensor(st.conv, conv); this.ctx.writeTensor(st.rec, rec); }
  }

  planFor(tokenCount: number): number[] { return planPrefill(this.chunkSizes, tokenCount, this.fixedTokens); }

  /** stepwise debugging: run the collected dispatches one by one, waiting for each (so that a crash / lost context names the graph that caused it) */
  private async flushQueue() {
    const queue = this.queue as NonNullable<typeof this.queue>;
    this.queue = [];
    const scratch = new Map<ML, Half>();
    for (const [i, entry] of queue.entries()) {
      await this.onDispatch?.(`${i + 1}/${queue.length} ${entry.label}`);
      entry.run();
      const t = entry.sync;
      if (t) { if (!scratch.has(t)) scratch.set(t, newHalf(t.shape.reduce((x: number, y: number) => x * y, 1))); await this.ctx.readTensor(t, scratch.get(t)); }
    }
  }

  /** Debugging: prefill `tokens`, then one decode step with `decodeToken` (not sampled), reading back after each of the two the hidden state that enters every layer (and the final hidden state).
   *  -> per phase an array of N + 1 float16 arrays ([rows * hidden]: all rows of the last prefill chunk, left-padded, | the one row of the decode step) and the logits. Needs Qwen35LLM.debugReadback. */
  async debugTrace(tokens: number[], decodeToken: number, options: { stepwise?: boolean; onDispatch?: (label: string) => void | Promise<void> } = {}) {
    if (!Qwen35LLM.debugReadback) throw new Error('set Qwen35LLM.debugReadback = true before loading the model');
    if (options.stepwise) { this.queue = []; this.onDispatch = options.onDispatch ?? null; }
    try {
      return await this.debugTraceRun(tokens, decodeToken);
    } finally { this.queue = null; this.onDispatch = null; }
  }

  private async debugTraceRun(tokens: number[], decodeToken: number) {
    const readAll = async (a: Activations) => {
      const out: Half[] = [];
      for (const t of [...a.residuals, a.finalHidden]) { const h = newHalf(t.shape.reduce((x: number, y: number) => x * y, 1)); await this.ctx.readTensor(t, h); out.push(h); }
      return out;
    };
    const plan = this.planFor(tokens.length), last = this.prefillSets.get(plan[plan.length - 1]) as PrefillSet;
    const prefillLogits = (await this.prefill(tokens)).slice();
    const prefill = await readAll(last.act);
    const decodeLogits = (await this.step(decodeToken)).slice();
    return { plan, prefill, prefillLogits, decode: await readAll(this.dec), decodeLogits };
  }

  private get cacheFull() { return this.position >= this.P; }

  // ------------------------------------------------------------------------------------------------------------------------------- generation
  private async prefill(tokens: number[], images: PreparedImage[] = []): Promise<Half> {
    const s = this.pkg.model, { P } = this;
    let positions: [Int32Array, Int32Array, Int32Array] | null = null, ropeDelta = 0, imageRows: Map<number, Half> | null = null, visionMs = 0;
    if (images.length) {
      if (!this.vision) throw new Error('this model was loaded without the image tower: pass { vision: true } to load()');
      const layout = this.visionLayout(), grids = images.map((i) => i.grid);
      tokens = expandImageTokens(layout, tokens, grids);
      ({ positions, ropeDelta } = mropePositions(layout, tokens, grids));
      const started = performance.now(), hidden = s.hidden, rows = new Map<number, Half>();
      let at = 0;
      const pads = tokens.flatMap((t, i) => (t === layout.image_token_id ? [i] : []));
      let buildMs = 0;
      for (const image of images) {
        const embeds = await this.vision.encode(image);
        buildMs += this.vision.lastEncode.buildMs;
        for (let k = 0; k < image.tokens; k++) rows.set(pads[at++], embeds.subarray(k * hidden, (k + 1) * hidden));
      }
      imageRows = rows; visionMs = performance.now() - started - buildMs;
      this.stats.imageTokens = rows.size; this.stats.visionMs = visionMs; this.stats.visionBuildMs = buildMs;
    }
    const n = tokens.length;
    if (n === 0) throw new Error('empty prompt');
    this.stats.promptTokens = n;
    const plan = this.planFor(n);
    if (n >= P) throw new Error(`the prompt has ${n} tokens but the KV cache holds ${P}: load the package with a larger max length`);
    this.stats.prefillPlan = plan;
    this.resetStates();
    let start = 0;
    for (let index = 0; index < plan.length; index++) {
      const C = plan[index], last = index === plan.length - 1, set = this.prefillSets.get(C) as PrefillSet, a = set.act;
      const valid = Math.min(C, n - start), pad = C - valid, hidden = newHalf(C * s.hidden);
      for (let i = 0; i < valid; i++) {
        const row = imageRows?.get(start + i);
        if (row) hidden.set(row, (pad + i) * s.hidden); else this.embedRow(tokens[start + i], hidden, (pad + i) * s.hidden);
      }
      this.ctx.writeTensor(a.hidden, hidden);
      this.writeRope(positions ? ropeTablesMrope(s as LayoutSpec & { mrope: { section: number[] } }, positions.map((p) => p.subarray(start, start + valid)), C, pad)
        : ropeTables(s, Array.from({ length: valid }, (_, i) => start + i), C, pad), a);
      this.ctx.writeTensor(a.mask, prefillMasks(s, P, C, start, valid)[layerTypesOf(s)[0]]);
      const { rowsToKeep, rowSelector } = prefillSelectors(P, C, start, valid);
      this.ctx.writeTensor(set.sel.keep, rowsToKeep); this.ctx.writeTensor(set.sel.select, rowSelector);
      const { validMask, convPerm } = linearPrefillControls(C, valid);
      this.ctx.writeTensor(set.validMask, validMask); this.ctx.writeTensor(set.convPerm, convPerm);
      this.enqueue(C, last);
      if (this.queue) await this.flushQueue();
      start += valid;
    }
    this.base = n; this.filled = 0; this.position = n; this.ropeDelta = ropeDelta;
    return this.readLogits();
  }

  private flush() {
    const { rowsToKeep, rowSelector } = flushSelectors(this.P, this.R, this.base);
    this.ctx.writeTensor(this.flushSel.keep, rowsToKeep); this.ctx.writeTensor(this.flushSel.select, rowSelector);
    const g = this.decodeSet.flush;
    for (const layer of this.kv.keys()) {
      const main = this.kv.get(layer) as Cache, staging = this.stg.get(layer) as Cache;
      this.ctx.dispatch(g.graph, this.bind(g, { past_key: main.key, past_value: main.value, new_key: staging.key, new_value: staging.value, rows_to_keep: this.flushSel.keep, row_selector: this.flushSel.select }),
        this.bind(g, { updated_key: (this.kvSpare.get(layer) as Cache).key, updated_value: (this.kvSpare.get(layer) as Cache).value }, true));
      this.swap(this.kv, this.kvSpare, layer);
    }
    this.base += this.R; this.filled = 0;
  }

  private async step(token: number): Promise<Half> {
    const s = this.pkg.model, { P, R } = this;
    if (this.cacheFull) throw new Error('the KV cache is full');
    let flushMs = 0;
    if (this.filled === R) { const f = performance.now(); this.flush(); flushMs = performance.now() - f; this.stats.flushes++; }
    const t0 = performance.now();
    const pos = this.position, a = this.dec, hidden = newHalf(s.hidden);
    this.embedRow(token, hidden, 0);
    this.ctx.writeTensor(a.hidden, hidden);
    this.writeRope(ropeTables(s, [pos + this.ropeDelta], 1, 0), a);   // after an image the rope position is shifted by ropeDelta; the cache rows and masks use the sequence index
    this.ctx.writeTensor(a.mask, decodeMasks(s, P, R, pos, this.base, this.filled)[layerTypesOf(s)[0]]);
    const { rowsToKeep, rowSelector } = decodeSelectors(R, this.filled);
    this.ctx.writeTensor(this.decodeSel.keep, rowsToKeep); this.ctx.writeTensor(this.decodeSel.select, rowSelector);
    const t1 = performance.now();
    this.enqueue(0, true);
    if (this.queue) await this.flushQueue();
    const t2 = performance.now();
    this.filled++; this.position++;
    const logits = await this.readLogits();
    this.lastStep = { hostMs: t1 - t0, enqueueMs: t2 - t1 + flushMs, waitMs: performance.now() - t2 };
    return logits;
  }

  /** The sampling of a generation: what the caller passed; without a temperature the model card's parameters of the package (plan.recommended.sampling: greedy decoding of these models falls into repetition
   *  loops); temperature 0 = greedy. */
  private samplingOf(o: GenerateOptions) {
    const rec = o.temperature === undefined ? this.pkg.plan.recommended?.sampling : undefined;
    return { temperature: o.temperature ?? rec?.temperature ?? 0, topK: o.topK ?? rec?.top_k ?? 40, topP: o.topP ?? (o.temperature === undefined ? rec?.top_p : undefined) ?? 1,
      presencePenalty: o.presencePenalty ?? (o.temperature === undefined ? rec?.presence_penalty : undefined) ?? 0 };
  }

  private pickToken(logits: Half, o: GenerateOptions, seen: Set<number> = new Set()): number {
    const vocab = this.pkg.model.vocab, s = this.samplingOf(o);
    if (s.presencePenalty) for (const t of seen) logits[t] -= s.presencePenalty;   // presence penalty: a token that was generated before is less likely (in place: the logits are read anew every step)
    if (!s.temperature) {
      let best = 0, bestValue = -Infinity;
      for (let i = 0; i < vocab; i++) if (logits[i] > bestValue) { bestValue = logits[i]; best = i; }
      return best;
    }
    const k = Math.min(s.topK, vocab), top: number[] = [];
    for (let i = 0; i < vocab; i++) {
      if (top.length < k || logits[i] > logits[top[top.length - 1]]) {
        top.push(i);
        top.sort((x, y) => logits[y] - logits[x]);
        if (top.length > k) top.pop();
      }
    }
    const max = logits[top[0]], weights = top.map((i) => Math.exp((logits[i] - max) / s.temperature)), total = weights.reduce((x, y) => x + y, 0);
    let keep = top.length;
    if (s.topP < 1) {   // nucleus: the shortest prefix of the (sorted) candidates with the probability mass >= topP
      let mass = 0;
      for (keep = 0; keep < top.length && mass < s.topP; keep++) mass += weights[keep] / total;
    }
    let r = Math.random() * weights.slice(0, keep).reduce((x, y) => x + y, 0);
    for (let j = 0; j < keep; j++) { r -= weights[j]; if (r <= 0) return top[j]; }
    return top[0];
  }

  async *generate(tokens: number[], options: GenerateOptions = {}): AsyncGenerator<number> {
    if (this.busy) throw new Error('a generation is already running on this model');
    this.busy = true;
    try {
      const stop = new Set(options.stopTokens ?? this.pkg.model.eos_token_ids), max = options.maxNewTokens ?? 128;
      let t = performance.now();
      this.stats.imageTokens = undefined; this.stats.visionMs = undefined; this.stats.visionBuildMs = undefined;
      let logits = await this.prefill(tokens, options.images ?? []);
      this.stats = { ...emptyStats(), loadMs: this.stats.loadMs, buildMs: this.stats.buildMs, weightsMs: this.stats.weightsMs, prefillMs: performance.now() - t,
        prefillChunks: this.stats.prefillPlan.length, prefillPlan: this.stats.prefillPlan, promptTokens: this.stats.promptTokens, graphBuilds: this.stats.graphBuilds,
        ...(options.images?.length ? { imageTokens: this.stats.imageTokens, visionMs: this.stats.visionMs, visionBuildMs: this.stats.visionBuildMs } : {}) };
      t = performance.now();
      const seen = new Set<number>();   // the tokens generated so far (presence penalty)
      for (let i = 0; i < max; i++) {
        const sampleStart = performance.now();
        const token = this.pickToken(logits, options, seen);
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
    this.vision?.destroy(); this.vision = null;
    this.releaseSession();
    for (const t of this.weightTensors) t.destroy?.();
    this.weightTensors = []; this.weights = [];
  }
}

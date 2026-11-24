// LLM runtime for the packages written by `m2w compile`: loads the weights into MLTensors, builds the graphs of the plan, and runs chunked prefill and staged-KV
// decode on a WebNN context (the NPUW-style split: HEAD / BLOCK x (N-1) / TAIL / lm_head). A port of a hand-written scheduler, driven by the
// package (model.json, plan.json), with per-layer-type masks and RoPE tables so that hybrid (sliding / full attention) models run on the same BLOCK graph.
// The protocol itself (what is written into the inputs) is in protocol.ts and checked against the Python reference.
import { loadGraph, type GraphDoc } from './ir-loader.js';
import { openPackage, type ModelPackage, type PackageSource, type Progress } from './package.js';
import {
  decodeMasks, decodeSelectors, flushSelectors, layerTypesOf, newHalf, prefillMasks, prefillSelectors, ropeTables, type Half,
} from './protocol.js';

/* eslint-disable @typescript-eslint/no-explicit-any */
type ML = any;

export interface LoadOptions {
  device?: 'npu' | 'gpu' | 'cpu';
  context?: ML;                    // an existing MLContext instead of navigator.ml.createContext({ deviceType: device })
  maxLength?: number;              // KV cache capacity (must be one of the package's max_lengths); default: the smallest
  chunkSize?: number;              // prefill chunk (one of the package's prefill chunks); default: the smallest
  blocksPerGraph?: number;         // decode: K consecutive blocks per dispatch (needs a package built with --blocks-per-graph K and a single layer type)
  cache?: boolean;                 // OPFS cache of the downloaded files (default true)
  parallelDownloads?: number;      // default 4
  onProgress?: Progress;
}
export interface GenerateOptions {
  maxNewTokens?: number;
  stopTokens?: number[];           // default: the model's eos tokens
  temperature?: number;            // 0 / undefined = greedy
  topK?: number;
  signal?: AbortSignal;
}
/** Per decode step: host = writing the step inputs, enqueue = queueing the dispatches (incl. a flush when due), wait = reading the logits (the device finishing the chain), sampling = picking the token. */
export interface StepTiming { hostMs: number; enqueueMs: number; waitMs: number; samplingMs: number; totalMs: number }
export interface Stats {
  loadMs: number; buildMs: number; weightsMs: number; prefillMs: number; prefillChunks: number; promptTokens: number; decodeSteps: number; decodeMs: number; tpotMs: number;
  stepMs: number[]; steps: StepTiming[]; flushes: number;
}
const emptyStats = (): Stats => ({ loadMs: 0, buildMs: 0, weightsMs: 0, prefillMs: 0, prefillChunks: 0, promptTokens: 0, decodeSteps: 0, decodeMs: 0, tpotMs: 0, stepMs: [], steps: [], flushes: 0 });

interface Built { name: string; graph: ML; inputs: string[]; outputs: string[] }
interface Activations {
  hidden: ML; ropeCos: Record<string, ML>; ropeSin: Record<string, ML>; mask: Record<string, ML>;
  residuals: ML[]; attentionOutputs: ML[]; finalHidden: ML;
}

export class LLM {
  readonly pkg: ModelPackage;
  readonly maxLength: number;
  readonly chunkSize: number;
  stats: Stats = emptyStats();

  private ctx!: ML;
  private P: number; private R: number; private C: number; private N: number;
  private types: string[];
  private groupSize = 1;
  private embed!: { weight: Uint8Array | Int8Array; zeroPoint: Uint8Array | null; scale: Half };   // int8-sym tables have no zero point
  private weights: Record<string, ML>[] = [];
  private finalGamma!: ML;
  private graphs!: {
    lmHead: Built; decode: { head: Built; block: Built; tail: Built; flush: Built; groups: Map<number, Built> }; prefill: { head: Built; block: Built; tail: Built };
  };
  private kv: { key: ML; value: ML }[] = []; private kvSpare: { key: ML; value: ML }[] = [];
  private stg: { key: ML; value: ML }[] = []; private stgSpare: { key: ML; value: ML }[] = [];
  private pre!: Activations; private dec!: Activations;
  private sel!: { prefill: { keep: ML; select: ML }; decode: { keep: ML; select: ML }; flush: { keep: ML; select: ML } };
  private logitsTensor!: ML; private logits!: Half;
  private tensors: ML[] = []; private builtGraphs: ML[] = [];
  private base = 0; private filled = 0; private position = 0;
  private busy = false;
  private lastStep = { hostMs: 0, enqueueMs: 0, waitMs: 0 };

  private constructor(pkg: ModelPackage, maxLength: number, chunkSize: number) {
    this.pkg = pkg; this.maxLength = maxLength; this.chunkSize = chunkSize;
    const spec = pkg.model;
    this.P = maxLength - 1; this.R = pkg.plan.staging_rows; this.C = chunkSize; this.N = spec.num_layers; this.types = layerTypesOf(spec);
  }

  // ------------------------------------------------------------------------------------------------------------------------------- loading
  static async load(src: string | PackageSource, options: LoadOptions = {}): Promise<LLM> {
    const t0 = performance.now();
    const progress = options.onProgress ?? (() => {});
    progress({ stage: 'package', message: 'reading the package index' });
    const pkg = await openPackage(src, options.cache ?? true);
    const plan = pkg.plan, key = String(options.maxLength ?? Math.min(...plan.max_lengths));
    if (!plan.graphs.decode[key]) throw new Error(`the package has no graphs for max length ${key} (it has ${plan.max_lengths.join(', ')})`);
    const chunks = Object.keys(plan.graphs.prefill[key]).map(Number).sort((a, b) => a - b);
    const chunk = options.chunkSize ?? chunks[0];
    if (!chunks.includes(chunk)) throw new Error(`the package has no prefill graphs for chunk ${chunk} at max length ${key} (it has ${chunks.join(', ')})`);
    const llm = new LLM(pkg, +key, chunk);
    llm.groupSize = options.blocksPerGraph ?? 1;
    if (llm.groupSize > 1 && llm.types.length > 1) throw new Error('blocksPerGraph > 1 needs a model with one layer type (blocks of a group share one mask and one RoPE table)');
    await llm.init(options, progress);
    llm.stats.loadMs = performance.now() - t0;
    return llm;
  }

  private async createTensor(dataType: string, shape: number[], flags: { writable?: boolean; readable?: boolean } = {}) {
    const tensor = await this.ctx.createTensor({ dataType, shape, writable: flags.writable ?? false, readable: flags.readable ?? false });
    this.tensors.push(tensor);
    return tensor;
  }
  private writable = (dataType: string, shape: number[]) => this.createTensor(dataType, shape, { writable: true });

  private async init(options: LoadOptions, progress: Progress) {
    const { pkg } = this, plan = pkg.plan, source = pkg.source;
    this.ctx = options.context ?? await (navigator as any).ml.createContext({ deviceType: options.device ?? 'npu' });

    // weight files: prefetched in the order they are needed, a few at a time (embedding, final norm, layer 0, 1, ...)
    const byRole = (role: string) => pkg.manifest.files.filter((f) => f.role === role);
    const ordered = [...byRole('embedding'), ...byRole('final_norm'), ...byRole('layer').sort((a, b) => (a.layer as number) - (b.layer as number))];
    const total = ordered.reduce((s, f) => s + f.bytes, 0), loaded = new Map<string, number>();
    const prefetch = new Prefetcher(ordered.map((f) => f.file), (file, i) => source.bytes(`weights/${file}`, (l) => {
      loaded.set(file, l);
      progress({ stage: 'download', loaded: [...loaded.values()].reduce((a, b) => a + b, 0), total, message: ordered[i].file });
    }), options.parallelDownloads ?? 4);

    const weightsStart = performance.now();
    progress({ stage: 'embedding', message: 'loading the embedding table' });
    const part = async (name: string) => {
      const file = byRole('embedding').find((f) => f.tensors[0].name === name);
      if (!file) throw new Error(`the package has no embedding tensor '${name}' (plan.weights.embedding: ${plan.weights.embedding.join(', ')})`);
      return prefetch.get(ordered.indexOf(file));
    };
    // the tied embedding / lm_head table: int8 per-row symmetric (scale only) or uint8 per-row asymmetric (scale + zero point); the plan lists which tensors exist
    const hasZeroPoint = plan.weights.embedding.includes('lm_head_zero_point');
    const weightBytes = new Uint8Array(await part('lm_head_weight')), scaleBytes = await part('lm_head_scale');
    const zeroPoint = hasZeroPoint ? new Uint8Array(await part('lm_head_zero_point')) : null;
    const signed = byRole('embedding').find((f) => f.tensors[0].name === 'lm_head_weight')?.tensors[0].dtype === 'int8';
    this.embed = { weight: signed ? new Int8Array(weightBytes.buffer, weightBytes.byteOffset, weightBytes.byteLength) : weightBytes, zeroPoint, scale: new (globalThis as any).Float16Array(scaleBytes) };

    // graphs (compiled while the layers download)
    const t = performance.now();
    const key = String(this.maxLength), decode = plan.graphs.decode[key], prefill = plan.graphs.prefill[key][String(this.C)];
    const external = (c: { external?: string }) => {
      if (c.external === 'lm_head_weight') return weightBytes;
      if (c.external === 'lm_head_scale') return new Uint8Array(scaleBytes);
      if (c.external === 'lm_head_zero_point' && zeroPoint) return zeroPoint;
      throw new Error(`the lm_head graph wants the external tensor '${c.external}' which this package does not have`);
    };
    const build = async (path: string, label: string, provider?: (c: any) => ArrayBufferView): Promise<Built> => {
      progress({ stage: 'build', message: label });
      const doc: GraphDoc = JSON.parse(await source.text(path));
      const builder = new (globalThis as any).MLGraphBuilder(this.ctx);
      const { inputs, outputs } = loadGraph(builder, doc, provider);
      const graph = await builder.build(outputs);
      this.builtGraphs.push(graph);
      return { name: doc.name, graph, inputs: Object.keys(inputs), outputs: Object.keys(outputs) };
    };
    const groups = new Map<number, Built>();
    if (this.groupSize > 1) {
      for (const [count, path] of Object.entries(decode.block_groups ?? {})) groups.set(+count, await build(path, `decode block x${count}`));
      if (!groups.has(this.groupSize)) throw new Error(`the package has no decode graph with ${this.groupSize} blocks (it has ${Object.keys(decode.block_groups ?? {}).join(', ') || 'none'})`);
    }
    this.graphs = {
      lmHead: await build(plan.graphs.lm_head, 'lm_head', external),
      decode: { head: await build(decode.head, 'decode head'), block: await build(decode.block, 'decode block'), tail: await build(decode.tail, 'decode tail'),
                flush: await build(decode.flush as string, 'kv flush'), groups },
      prefill: { head: await build(prefill.head, 'prefill head'), block: await build(prefill.block, 'prefill block'), tail: await build(prefill.tail, 'prefill tail') },
    };
    this.stats.buildMs = performance.now() - t;

    // weights -> MLTensors, layer by layer as the files arrive
    const finalFile = byRole('final_norm')[0];
    const uploadShard = async (shard: (typeof ordered)[number], buffer: ArrayBuffer) => {
      const tensors: Record<string, ML> = {};
      for (const entry of shard.tensors) {
        const tensor = await this.writable(entry.dtype, entry.shape);
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

  private async allocateState() {
    const s = this.pkg.model, { P, R, C, N } = this;
    const zero = (n: number) => newHalf(n);
    const kvShape = (rows: number) => [1, s.num_kv_heads, rows, s.head_dim];
    const makeKv = async (rows: number) => {
      const t = { key: await this.writable('float16', kvShape(rows)), value: await this.writable('float16', kvShape(rows)) };
      const zeros = zero(s.num_kv_heads * rows * s.head_dim);
      this.ctx.writeTensor(t.key, zeros); this.ctx.writeTensor(t.value, zeros);
      return t;
    };
    for (let l = 0; l < N; l++) {
      this.kv.push(await makeKv(P)); this.kvSpare.push(await makeKv(P));
      this.stg.push(await makeKv(R)); this.stgSpare.push(await makeKv(R));
    }
    const activations = async (S: number, maskColumns: number): Promise<Activations> => {
      const a: Activations = { hidden: await this.writable('float16', [1, S, s.hidden]), ropeCos: {}, ropeSin: {}, mask: {}, residuals: [], attentionOutputs: [], finalHidden: null };
      for (const type of this.types) {
        a.ropeCos[type] = await this.writable('float16', [1, 1, S, s.head_dim]);
        a.ropeSin[type] = await this.writable('float16', [1, 1, S, s.head_dim]);
        a.mask[type] = await this.writable('float16', [1, 1, S, maskColumns]);
      }
      for (let l = 0; l < N; l++) {
        a.residuals.push(await this.createTensor('float16', [1, S, s.hidden]));
        a.attentionOutputs.push(await this.createTensor('float16', [1, S, s.num_heads * s.head_dim]));
      }
      a.finalHidden = await this.createTensor('float16', [1, 1, s.hidden]);
      return a;
    };
    this.pre = await activations(C, P + C);
    this.dec = await activations(1, P + R);
    const pair = async (rows: number, columns: number) => ({ keep: await this.writable('float16', [1, 1, rows, 1]), select: await this.writable('float16', [1, 1, rows, columns]) });
    this.sel = { prefill: await pair(P, C), decode: await pair(R, 1), flush: await pair(P, R) };
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

  /** One forward pass: HEAD, the BLOCKs and (last chunk / decode) TAIL + lm_head. All host inputs of the phase are already written. */
  private enqueue(phase: 'prefill' | 'decode', lastChunk: boolean) {
    const decode = phase === 'decode', a = decode ? this.dec : this.pre, g = decode ? this.graphs.decode : this.graphs.prefill;
    const w = this.pkg.plan.weights, types = this.pkg.model.layer_types, N = this.N;
    const caches = decode ? this.stg : this.kv, spare = decode ? this.stgSpare : this.kvSpare;
    const selectors = decode
      ? { staging_rows_to_keep: this.sel.decode.keep, staging_row_selector: this.sel.decode.select }
      : { rows_to_keep: this.sel.prefill.keep, row_selector: this.sel.prefill.select };
    const shared = (layer: number) => ({ rope_cos: a.ropeCos[types[layer]], rope_sin: a.ropeSin[types[layer]], attention_mask: a.mask[types[layer]], ...selectors });
    const cacheInputs = (layer: number, prefix = '') => ({
      [prefix + 'past_key']: this.kv[layer].key, [prefix + 'past_value']: this.kv[layer].value,
      ...(decode ? { [prefix + 'staging_key']: this.stg[layer].key, [prefix + 'staging_value']: this.stg[layer].value } : {}),
    });
    const dispatch = (graph: Built, inputs: Record<string, ML>, outputs: Record<string, ML>) => this.ctx.dispatch(graph.graph, this.bind(graph, inputs), this.bind(graph, outputs, true));

    // HEAD: attention half of layer 0
    dispatch(g.head, { hidden_states: a.hidden, ...shared(0), ...cacheInputs(0), ...this.pick(0, w.attention) },
      { residual: a.residuals[0], attention_output: a.attentionOutputs[0], updated_key: spare[0].key, updated_value: spare[0].value });
    this.swap(caches, spare, 0);

    // BLOCK i: feed-forward half of layer i + attention half of layer i+1
    const groups = decode && this.groupSize > 1 ? this.graphs.decode.groups : null;
    if (groups) {
      for (let start = 0; start < N - 1; start += this.groupSize) {
        const count = Math.min(this.groupSize, N - 1 - start), graph = groups.get(count) as Built;
        const inputs: Record<string, ML> = { residual: a.residuals[start], attention_output: a.attentionOutputs[start], ...shared(start + 1) };
        const outputs: Record<string, ML> = { next_residual: a.residuals[start + count], next_attention_output: a.attentionOutputs[start + count] };
        for (let j = 0; j < count; j++) {
          const layer = start + j, prefix = `b${j}_`;
          Object.assign(inputs, this.pick(layer, w.ffn, prefix), this.pick(layer + 1, w.attention, prefix), cacheInputs(layer + 1, prefix));
          outputs[prefix + 'updated_key'] = spare[layer + 1].key; outputs[prefix + 'updated_value'] = spare[layer + 1].value;
        }
        dispatch(graph, inputs, outputs);
        for (let j = 0; j < count; j++) this.swap(caches, spare, start + j + 1);
      }
    } else {
      for (let layer = 0; layer < N - 1; layer++) {
        dispatch(g.block, { residual: a.residuals[layer], attention_output: a.attentionOutputs[layer], ...shared(layer + 1), ...cacheInputs(layer + 1),
          ...this.pick(layer, w.ffn), ...this.pick(layer + 1, w.attention) },
        { next_residual: a.residuals[layer + 1], next_attention_output: a.attentionOutputs[layer + 1], updated_key: spare[layer + 1].key, updated_value: spare[layer + 1].value });
        this.swap(caches, spare, layer + 1);
      }
    }
    if (!lastChunk) return;
    // TAIL: feed-forward half of the last layer + final norm; lm_head
    dispatch(g.tail, { residual: a.residuals[N - 1], attention_output: a.attentionOutputs[N - 1], ...this.pick(N - 1, w.ffn), final_norm_gamma: this.finalGamma },
      { final_hidden_states: a.finalHidden });
    dispatch(this.graphs.lmHead, { final_hidden_states: a.finalHidden }, { logits: this.logitsTensor });
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

  // ------------------------------------------------------------------------------------------------------------------------------- generation
  /** Chunked prefill of the prompt; returns the logits of the last prompt token. */
  private async prefill(tokens: number[]): Promise<Half> {
    const s = this.pkg.model, { P, C } = this, n = tokens.length;
    if (n === 0) throw new Error('empty prompt');
    if (n >= P) throw new Error(`the prompt has ${n} tokens but the KV cache holds ${P}: load the package with a larger max length`);
    const a = this.pre;
    for (let start = 0; start < n; start += C) {
      const valid = Math.min(C, n - start), pad = C - valid, last = start + C >= n;
      const hidden = newHalf(C * s.hidden);
      for (let i = 0; i < valid; i++) this.embedRow(tokens[start + i], hidden, (pad + i) * s.hidden);
      this.ctx.writeTensor(a.hidden, hidden);
      this.writeRope(ropeTables(s, Array.from({ length: valid }, (_, i) => start + i), C, pad), a);
      const masks = prefillMasks(s, P, C, start, valid);
      for (const type of this.types) this.ctx.writeTensor(a.mask[type], masks[type]);
      const { rowsToKeep, rowSelector } = prefillSelectors(P, C, start, valid);
      this.ctx.writeTensor(this.sel.prefill.keep, rowsToKeep); this.ctx.writeTensor(this.sel.prefill.select, rowSelector);
      this.enqueue('prefill', last);
    }
    this.base = n; this.filled = 0; this.position = n;
    return this.readLogits();
  }

  private flush() {
    const { rowsToKeep, rowSelector } = flushSelectors(this.P, this.R, this.base);
    this.ctx.writeTensor(this.sel.flush.keep, rowsToKeep); this.ctx.writeTensor(this.sel.flush.select, rowSelector);
    const g = this.graphs.decode.flush;
    for (let l = 0; l < this.N; l++) {
      this.ctx.dispatch(g.graph, this.bind(g, { past_key: this.kv[l].key, past_value: this.kv[l].value, new_key: this.stg[l].key, new_value: this.stg[l].value,
        rows_to_keep: this.sel.flush.keep, row_selector: this.sel.flush.select }),
      this.bind(g, { updated_key: this.kvSpare[l].key, updated_value: this.kvSpare[l].value }, true));
      this.swap(this.kv, this.kvSpare, l);
    }
    this.base += this.R; this.filled = 0;
  }

  /** Feed one token at the current position; returns the logits for the next one. */
  private async step(token: number): Promise<Half> {
    const s = this.pkg.model, { P, R } = this;
    if (this.position >= P) throw new Error('the KV cache is full');
    let flushMs = 0;
    if (this.filled === R) { const f = performance.now(); this.flush(); flushMs = performance.now() - f; this.stats.flushes++; }
    const t0 = performance.now();
    const pos = this.position, a = this.dec;
    const hidden = newHalf(s.hidden);
    this.embedRow(token, hidden, 0);
    this.ctx.writeTensor(a.hidden, hidden);
    this.writeRope(ropeTables(s, [pos], 1, 0), a);
    const masks = decodeMasks(s, P, R, pos, this.base, this.filled);
    for (const type of this.types) this.ctx.writeTensor(a.mask[type], masks[type]);
    const { rowsToKeep, rowSelector } = decodeSelectors(R, this.filled);
    this.ctx.writeTensor(this.sel.decode.keep, rowsToKeep); this.ctx.writeTensor(this.sel.decode.select, rowSelector);
    const t1 = performance.now();
    this.enqueue('decode', true);
    const t2 = performance.now();
    this.filled++; this.position++;
    const logits = await this.readLogits();
    this.lastStep = { hostMs: t1 - t0, enqueueMs: t2 - t1 + flushMs, waitMs: performance.now() - t2 };
    return logits;
  }

  private pickToken(logits: Half, o: GenerateOptions): number {
    const vocab = this.pkg.model.vocab;
    if (!o.temperature) {
      let best = 0, bestValue = -Infinity;
      for (let i = 0; i < vocab; i++) if (logits[i] > bestValue) { bestValue = logits[i]; best = i; }
      return best;
    }
    const k = Math.min(o.topK ?? 40, vocab), top: number[] = [];
    for (let i = 0; i < vocab; i++) {
      if (top.length < k || logits[i] > logits[top[top.length - 1]]) {
        top.push(i);
        top.sort((x, y) => logits[y] - logits[x]);
        if (top.length > k) top.pop();
      }
    }
    const max = logits[top[0]], weights = top.map((i) => Math.exp((logits[i] - max) / (o.temperature as number)));
    let r = Math.random() * weights.reduce((x, y) => x + y, 0);
    for (let j = 0; j < top.length; j++) { r -= weights[j]; if (r <= 0) return top[j]; }
    return top[0];
  }

  /** Greedy (or sampled) generation from prompt token ids; yields each new token id (the stop token is yielded too, then the loop ends). */
  async *generate(tokens: number[], options: GenerateOptions = {}): AsyncGenerator<number> {
    if (this.busy) throw new Error('a generation is already running on this model');
    this.busy = true;
    try {
      const stop = new Set(options.stopTokens ?? this.pkg.model.eos_token_ids), max = options.maxNewTokens ?? 128;
      let t = performance.now();
      let logits = await this.prefill(tokens);
      this.stats = { ...emptyStats(), loadMs: this.stats.loadMs, buildMs: this.stats.buildMs, weightsMs: this.stats.weightsMs, prefillMs: performance.now() - t, prefillChunks: Math.ceil(tokens.length / this.C), promptTokens: tokens.length };
      t = performance.now();
      for (let i = 0; i < max; i++) {
        const sampleStart = performance.now();
        const token = this.pickToken(logits, options);
        const samplingMs = performance.now() - sampleStart;
        yield token;
        if (stop.has(token) || i === max - 1 || this.position >= this.P || options.signal?.aborted) break;
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
    for (const t of this.tensors) t.destroy?.();
    for (const g of this.builtGraphs) g.destroy?.();
    this.tensors = []; this.builtGraphs = [];
  }
}

/** Starts the downloads in order, at most `window` ahead of the consumer (so the whole model is never held in memory at once). */
class Prefetcher {
  private tasks = new Map<number, Promise<ArrayBuffer>>();
  private started = new Set<number>();
  constructor(private files: string[], private fetchOne: (file: string, index: number) => Promise<ArrayBuffer>, private window: number) {}
  get(index: number): Promise<ArrayBuffer> {
    for (let i = 0; i < Math.min(this.files.length, index + this.window); i++) {
      if (this.started.has(i)) continue;
      this.started.add(i);
      const task = this.fetchOne(this.files[i], i);
      task.catch(() => {});   // reported when consumed
      this.tasks.set(i, task);
    }
    return this.tasks.get(index) as Promise<ArrayBuffer>;
  }
  release(index: number) { this.tasks.delete(index); }
}

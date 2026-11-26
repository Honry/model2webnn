// Model packages: where the files come from (URL / Hugging Face / a local folder), an OPFS cache so that the second visit needs no network, and the
// typed view of m2w.json / model.json / plan.json / chat_format.json / the weights manifest written by the Python compiler (python/m2w).
/* eslint-disable @typescript-eslint/no-explicit-any */
import type { RopeEntry } from './rope.js';

export interface ModelSpecJson {
  architecture: string; hidden: number; num_layers: number; num_heads: number; num_kv_heads: number; head_dim: number; ffn: number;
  vocab: number; vocab_padded: number; eps: number; qk_norm: boolean; tie_embeddings: boolean; max_position_embeddings: number;
  layer_types: string[]; sliding_window: number | null; rope: Record<string, RopeEntry>; eos_token_ids: number[]; bos_token_id: number | null; group_size: number;
  builder?: string;                                                                   // "gemma4" for Gemma-style models
  layer_dims?: Record<string, { head_dim: number; num_heads: number; num_kv_heads: number }>;   // layer types whose attention dims differ from the defaults above
  kv_producer?: number[];                                                             // per layer: the layer whose K / V it attends over, -1 = its own (empty / absent: nothing shared)
  ple_dim?: number; embed_scale?: number; attention_scale?: number | null; logit_softcap?: number;
  linear?: { num_k_heads: number; num_v_heads: number; k_dim: number; v_dim: number; conv_kernel: number };   // Qwen3.5: the Gated DeltaNet (linear_attention) layers
  attn_output_gate?: boolean;
}
export interface GraphTable {
  head: string; block: string; tail: string; flush?: string; block_groups?: Record<string, string>;
  blocks?: Record<string, string>;                // Gemma 4: one block graph per attention kind (instead of `block`)
}
/** One decode block group of a Gemma 4 package: BLOCKs first_block .. first_block + blocks - 1 in one graph (plan.chain.block_groups) */
export interface GemmaBlockGroup { first_block: number; blocks: number; key: string; graph: string }
export interface GemmaDecodeTable { head: string; blocks: Record<string, string>; tail: string; flush: Record<string, string>; block_groups?: Record<string, GemmaBlockGroup[]> }
export interface QwenDecodeTable { head: string; blocks: Record<string, string>; tail: string; flush: string }
/** One set of graphs of the package: a KV protocol x a compute type (x the staging rows of the staged protocol). Weights are shared by all variants. */
export interface Variant { id: string; kv_style: 'staged' | 'onehot' | 'tensorScatter' | 'scatterND'; compute_type: 'float32' | 'float16'; staging_rows?: number; scatter_index_type?: 'int32' | 'int64'; value_layout?: 'natural' | 'transposed' }
export interface VariantGraphs { lm_head: string; decode: Record<string, GraphTable>; prefill: Record<string, Record<string, GraphTable>> }
export interface Plan {
  max_lengths: number[]; prefill_chunks: number[]; blocks_per_graph: number[]; prefill_blocks_per_graph: number[];
  variants: Variant[]; default_variant: string;
  block_group_controls?: 'shared' | 'per_block';   // block groups: one rope / mask for all blocks (single layer type), or b<j>_rope_cos / b<j>_attention_mask per block (hybrid models)
  graphs: Record<string, VariantGraphs>;
  weights: { attention: string[] | Record<string, string[]>; ffn: string[]; final_norm: string[]; embedding: string[]; head_extra?: string[]; ple_table?: string[] };
  /** the session options the model family runs best with (families/*.json `recommended`): used when the caller does not choose; a group size the package does not contain is already 1 here */
  recommended?: { blocks_per_graph?: number; prefill_blocks_per_graph?: number };
  architecture_kind?: 'gemma4' | 'qwen35';
  state?: { conv_channels: number; v_heads: number; k_dim: number; v_dim: number; layers: number[] };   // qwen35: the layers that carry a Gated DeltaNet state instead of a KV cache
  profile: { merge_projections: boolean; lm_head_format: string };
}
export interface ManifestTensor { name: string; dtype: string; shape: number[]; offset: number; bytes: number }
export interface ManifestFile { file: string; role: 'layer' | 'final_norm' | 'embedding' | 'head_extra' | 'ple_table'; layer?: number; kind?: string; bytes: number; tensors: ManifestTensor[] }
export interface WeightsManifest { quant: { group_size: number; merge_projections: boolean }; files: ManifestFile[] }
export interface ChatFormat {
  template_kwargs: Record<string, unknown>; first_prefix: string; first_prefix_with_system: string; user_suffix_generation_prompt: string;
  assistant_suffix: string; next_user_prefix: string; user_suffix_assistant_prefix: string;
}
export interface PackageIndex { format: string; version: number; compiler: string; target: string; model: string; plan: string; chat_format: string | null; weights_manifest: string; tokenizer: string[]; files: { path: string; bytes: number; sha256: string }[] }

export type Progress = (event: { stage: string; message?: string; loaded?: number; total?: number }) => void;

export interface PackageSource {
  /** identifies the package in the cache */
  readonly id: string;
  text(path: string): Promise<string>;
  /** `size`: the size of the file when the caller knows it (m2w.json lists it), which saves a round trip for the segmented download of a UrlSource */
  bytes(path: string, progress?: (loaded: number, total: number) => void, size?: number): Promise<ArrayBuffer>;
  /** bytes [start, end) of a file, without reading the rest (the per-layer embedding table of Gemma 4 is read one row per token) */
  slice?(path: string, start: number, end: number): Promise<ArrayBuffer>;
}

export interface UrlSourceOptions {
  /** big files are fetched as this many parallel byte ranges (a single connection is often slower than the link, e.g. from a CDN); 1 = one request per file. Default 4. */
  segments?: number;
  /** files smaller than this are fetched in one request. Default 48 MiB. */
  segmentMinBytes?: number;
}

export class UrlSource implements PackageSource {
  readonly base: string;
  readonly id: string;
  private segments: number; private segmentMinBytes: number;
  constructor(base: string, options: UrlSourceOptions = {}) {
    this.base = base.endsWith('/') ? base : base + '/';
    this.id = this.base;
    this.segments = Math.max(1, options.segments ?? 4); this.segmentMinBytes = options.segmentMinBytes ?? 48 * 2 ** 20;
  }
  private url(path: string) { return new URL(path, new URL(this.base, globalThis.location?.href ?? 'http://localhost/')).href; }
  async text(path: string) {
    const response = await fetch(this.url(path), { cache: 'no-store' });
    if (!response.ok) throw new Error(`failed to fetch ${this.url(path)}: ${response.status}`);
    return response.text();
  }
  async bytes(path: string, progress?: (loaded: number, total: number) => void, size?: number) {
    if (this.segments > 1) {
      const segmented = await this.segmentedBytes(path, progress, size);
      if (segmented) return segmented;
    }
    return this.singleBytes(path, progress);
  }

  /** A big file as parallel byte ranges written into one pre-sized buffer; null when the file is small or the server does not serve ranges (the caller then fetches it in one request). */
  async slice(path: string, start: number, end: number): Promise<ArrayBuffer> {
    const response = await fetch(this.url(path), { headers: { Range: `bytes=${start}-${end - 1}` }, cache: 'no-store' });
    if (response.status === 206) return response.arrayBuffer();
    if (!response.ok) throw new Error(`failed to fetch ${this.url(path)}: ${response.status}`);
    return (await response.arrayBuffer()).slice(start, end);   // a server without ranges answers with the whole file
  }

  private async segmentedBytes(path: string, progress: ((loaded: number, total: number) => void) | undefined, size?: number): Promise<ArrayBuffer | null> {
    const url = this.url(path);
    let total = size ?? 0;
    if (!total) {
      const head = await fetch(url, { method: 'HEAD', cache: 'no-store' }).catch(() => null);
      if (!head?.ok || head.headers.get('Accept-Ranges') !== 'bytes') return null;
      total = +(head.headers.get('Content-Length') ?? 0);
    }
    if (total < this.segmentMinBytes) return null;
    const parts = this.segments, step = Math.ceil(total / parts);
    const out = new Uint8Array(total), loadedBy = new Array<number>(parts).fill(0);
    try {
      await Promise.all(Array.from({ length: parts }, async (_, i) => {
        const from = i * step, to = Math.min(total, from + step) - 1;
        const response = await fetch(url, { headers: { Range: `bytes=${from}-${to}` }, cache: 'no-store' });
        if (response.status !== 206 || !response.body) throw new Error(`no byte ranges: ${response.status}`);
        const reader = response.body.getReader();
        let at = from;
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          if (at + value.byteLength > to + 1) throw new Error(`${path}: a range delivered more bytes than requested`);
          out.set(value, at);
          at += value.byteLength;
          loadedBy[i] = at - from;
          progress?.(loadedBy.reduce((a, b) => a + b, 0), total);
        }
        if (at !== to + 1) throw new Error(`${path}: incomplete range ${from}-${to}`);
      }));
    } catch { return null; }   // whatever went wrong with ranges (proxy, CORS, a server that answers 200): fetch the file in one request
    return out.buffer;
  }

  private async singleBytes(path: string, progress?: (loaded: number, total: number) => void) {
    const response = await fetch(this.url(path), { cache: 'no-store' });
    if (!response.ok || !response.body) throw new Error(`failed to fetch ${this.url(path)}: ${response.status}`);
    const total = +(response.headers.get('Content-Length') ?? 0);
    if (!total) return response.arrayBuffer();
    const out = new Uint8Array(total);   // pre-sized: the file is never held twice in memory
    const reader = response.body.getReader();
    let loaded = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (loaded + value.byteLength > total) throw new Error(`${path}: more bytes than Content-Length`);
      out.set(value, loaded);
      loaded += value.byteLength;
      progress?.(loaded, total);
    }
    if (loaded !== total) throw new Error(`incomplete download of ${path}: ${loaded} of ${total} bytes`);
    return out.buffer;
  }
}

/** A folder picked with showDirectoryPicker() (a package written by `m2w compile`). */
export class DirectorySource implements PackageSource {
  readonly id: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(private root: any) { this.id = `dir:${root.name}`; }
  private async file(path: string): Promise<File> {
    let directory = this.root;
    const parts = path.split('/');
    for (const part of parts.slice(0, -1)) directory = await directory.getDirectoryHandle(part);
    return (await directory.getFileHandle(parts[parts.length - 1])).getFile();
  }
  async text(path: string) { return (await this.file(path)).text(); }
  async bytes(path: string) { return (await this.file(path)).arrayBuffer(); }
  async slice(path: string, start: number, end: number) { return (await this.file(path)).slice(start, end).arrayBuffer(); }
}

/** Wraps a source with an OPFS cache keyed by the package id AND the sha256 of every file (from m2w.json): a recompiled file with the same size is a different cache entry. */
export class CachedSource implements PackageSource {
  readonly id: string;
  private files = new Map<string, { bytes: number; sha256: string }>();
  constructor(private inner: PackageSource, private cacheName = 'm2w-packages') { this.id = inner.id; }
  setFiles(files: { path: string; bytes: number; sha256: string }[]) { for (const f of files) this.files.set(f.path, { bytes: f.bytes, sha256: f.sha256 }); }
  text(path: string) { return this.inner.text(path); }
  slice(path: string, start: number, end: number) {
    if (!this.inner.slice) throw new Error('this package source cannot read byte ranges');
    return this.inner.slice(path, start, end);
  }
  private async directory(create: boolean) {
    const root = await navigator.storage.getDirectory();
    const cache = await root.getDirectoryHandle(this.cacheName, { create });
    return cache.getDirectoryHandle(encodeURIComponent(this.id), { create });
  }
  /** cache entry name: <encoded path>@<first 16 hex digits of the sha256> */
  static entryName(path: string, sha256: string) { return `${encodeURIComponent(path)}@${sha256.slice(0, 16)}`; }
  async bytes(path: string, progress?: (loaded: number, total: number) => void) {
    const info = this.files.get(path);
    if (!info) return this.inner.bytes(path, progress);   // unknown file: nothing to validate against, do not cache
    const name = CachedSource.entryName(path, info.sha256);
    try {
      const file = await (await (await this.directory(false)).getFileHandle(name)).getFile();
      if (file.size === info.bytes) return file.arrayBuffer();
    } catch { /* not cached */ }
    const data = await this.inner.bytes(path, progress, info.bytes);
    try {
      const directory = await this.directory(true);
      const writable = await (await directory.getFileHandle(name, { create: true })).createWritable();
      await writable.write(data);
      await writable.close();
      const prefix = `${encodeURIComponent(path)}@`;   // older versions of this file are garbage now
      for await (const [entry] of (directory as any).entries()) if (entry.startsWith(prefix) && entry !== name) await directory.removeEntry(entry).catch(() => {});
    } catch { /* quota or private mode: run without the cache */ }
    return data;
  }
  async clear() {
    try { await (await navigator.storage.getDirectory()).removeEntry(this.cacheName, { recursive: true }); } catch { /* nothing cached */ }
  }
}

export function sourceFrom(src: string | PackageSource, options: UrlSourceOptions = {}): PackageSource {
  if (typeof src !== 'string') return src;
  const hf = /^hf:([^@/]+\/[^@/]+)(?:@(.+))?$/.exec(src);
  if (hf) return new UrlSource(`https://huggingface.co/${hf[1]}/resolve/${hf[2] ?? 'main'}/`, options);
  return new UrlSource(src, options);
}

export interface ModelPackage {
  source: PackageSource; index: PackageIndex; model: ModelSpecJson; plan: Plan; chatFormat: ChatFormat | null; manifest: WeightsManifest;
}

export async function openPackage(src: string | PackageSource, cache = false, download: UrlSourceOptions = {}): Promise<ModelPackage> {
  let source = sourceFrom(src, download);
  const index: PackageIndex = JSON.parse(await source.text('m2w.json'));
  if (index.format !== 'm2w-package') throw new Error('not an m2w package (m2w.json has another format)');
  if (index.version !== 0) throw new Error(`package format version ${index.version} is newer than this runtime understands (0); update model2webnn`);
  if (cache && typeof navigator !== 'undefined' && typeof (navigator.storage as any)?.getDirectory === 'function') {
    const cached = new CachedSource(source);
    cached.setFiles(index.files);
    source = cached;
  }
  const [model, plan, manifest, chatFormat] = await Promise.all([
    source.text(index.model).then(JSON.parse), source.text(index.plan).then(JSON.parse), source.text(index.weights_manifest).then(JSON.parse),
    index.chat_format ? source.text(index.chat_format).then(JSON.parse) : Promise.resolve(null),
  ]);
  return { source, index, model, plan, chatFormat, manifest };
}

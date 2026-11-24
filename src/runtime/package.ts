// Model packages: where the files come from (URL / Hugging Face / a local folder), an OPFS cache so that the second visit needs no network, and the
// typed view of m2w.json / model.json / plan.json / chat_format.json / the weights manifest written by the Python compiler (python/m2w).
/* eslint-disable @typescript-eslint/no-explicit-any */
import type { RopeEntry } from './rope.js';

export interface ModelSpecJson {
  architecture: string; hidden: number; num_layers: number; num_heads: number; num_kv_heads: number; head_dim: number; ffn: number;
  vocab: number; vocab_padded: number; eps: number; qk_norm: boolean; tie_embeddings: boolean; max_position_embeddings: number;
  layer_types: string[]; sliding_window: number | null; rope: Record<string, RopeEntry>; eos_token_ids: number[]; bos_token_id: number | null; group_size: number;
}
export interface GraphTable { head: string; block: string; tail: string; flush?: string; block_groups?: Record<string, string> }
export interface Plan {
  max_lengths: number[]; prefill_chunks: number[]; staging_rows: number;
  graphs: { lm_head: string; decode: Record<string, GraphTable>; prefill: Record<string, Record<string, GraphTable>> };
  weights: { attention: string[]; ffn: string[]; final_norm: string[]; embedding: string[] };
  profile: { compute_type: string; kv_style: string; merge_projections: boolean };
}
export interface ManifestTensor { name: string; dtype: string; shape: number[]; offset: number; bytes: number }
export interface ManifestFile { file: string; role: 'layer' | 'final_norm' | 'embedding'; layer?: number; bytes: number; tensors: ManifestTensor[] }
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
  bytes(path: string, progress?: (loaded: number, total: number) => void): Promise<ArrayBuffer>;
}

export class UrlSource implements PackageSource {
  readonly base: string;
  readonly id: string;
  constructor(base: string) {
    this.base = base.endsWith('/') ? base : base + '/';
    this.id = this.base;
  }
  private url(path: string) { return new URL(path, new URL(this.base, globalThis.location?.href ?? 'http://localhost/')).href; }
  async text(path: string) {
    const response = await fetch(this.url(path), { cache: 'no-store' });
    if (!response.ok) throw new Error(`failed to fetch ${this.url(path)}: ${response.status}`);
    return response.text();
  }
  async bytes(path: string, progress?: (loaded: number, total: number) => void) {
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
}

/** Wraps a source with an OPFS cache keyed by the package id AND the sha256 of every file (from m2w.json): a recompiled file with the same size is a different cache entry. */
export class CachedSource implements PackageSource {
  readonly id: string;
  private files = new Map<string, { bytes: number; sha256: string }>();
  constructor(private inner: PackageSource, private cacheName = 'm2w-packages') { this.id = inner.id; }
  setFiles(files: { path: string; bytes: number; sha256: string }[]) { for (const f of files) this.files.set(f.path, { bytes: f.bytes, sha256: f.sha256 }); }
  text(path: string) { return this.inner.text(path); }
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
    const data = await this.inner.bytes(path, progress);
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

export function sourceFrom(src: string | PackageSource): PackageSource {
  if (typeof src !== 'string') return src;
  const hf = /^hf:([^@/]+\/[^@/]+)(?:@(.+))?$/.exec(src);
  if (hf) return new UrlSource(`https://huggingface.co/${hf[1]}/resolve/${hf[2] ?? 'main'}/`);
  return new UrlSource(src);
}

export interface ModelPackage {
  source: PackageSource; index: PackageIndex; model: ModelSpecJson; plan: Plan; chatFormat: ChatFormat | null; manifest: WeightsManifest;
}

export async function openPackage(src: string | PackageSource, cache = false): Promise<ModelPackage> {
  let source = sourceFrom(src);
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

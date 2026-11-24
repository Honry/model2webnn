import { afterEach, describe, expect, it } from 'vitest';
import { CachedSource, type PackageSource } from './package.js';

// A minimal in-memory OPFS: just what CachedSource touches.
class FakeFile { constructor(public data: ArrayBuffer) {} get size() { return this.data.byteLength; } arrayBuffer() { return Promise.resolve(this.data); } }
class FakeDirectory {
  directories = new Map<string, FakeDirectory>(); files = new Map<string, FakeFile>();
  async getDirectoryHandle(name: string, o?: { create?: boolean }) { if (!this.directories.has(name)) { if (!o?.create) throw new Error('NotFound'); this.directories.set(name, new FakeDirectory()); } return this.directories.get(name) as FakeDirectory; }
  async getFileHandle(name: string, o?: { create?: boolean }) {
    if (!this.files.has(name)) { if (!o?.create) throw new Error('NotFound'); this.files.set(name, new FakeFile(new ArrayBuffer(0))); }
    const files = this.files;
    return { getFile: async () => files.get(name) as FakeFile, createWritable: async () => ({ write: async (d: ArrayBuffer) => { files.set(name, new FakeFile(d)); }, close: async () => {} }) };
  }
  async removeEntry(name: string) { this.files.delete(name); this.directories.delete(name); }
  async *entries() { for (const k of [...this.files.keys(), ...this.directories.keys()]) yield [k, null]; }
}
const bytesOf = (...values: number[]) => new Uint8Array(values).buffer;

describe('CachedSource', () => {
  afterEach(() => { delete (globalThis as any).navigator; });
  const setup = () => {
    const root = new FakeDirectory();
    Object.defineProperty(globalThis, 'navigator', { value: { storage: { getDirectory: async () => root } }, configurable: true });
    return root;
  };
  const inner = (content: { value: ArrayBuffer }, calls: string[]): PackageSource => ({ id: 'pkg', text: async () => '', bytes: async (p) => { calls.push(p); return content.value; } });

  it('serves a cached file without asking the source again', async () => {
    setup();
    const calls: string[] = [], content = { value: bytesOf(1, 2, 3, 4) };
    const source = new CachedSource(inner(content, calls));
    source.setFiles([{ path: 'weights/a.bin', bytes: 4, sha256: 'aaaa'.repeat(16) }]);
    expect(new Uint8Array(await source.bytes('weights/a.bin'))).toEqual(new Uint8Array([1, 2, 3, 4]));
    expect(new Uint8Array(await source.bytes('weights/a.bin'))).toEqual(new Uint8Array([1, 2, 3, 4]));
    expect(calls.length).toBe(1);
  });

  it('a recompiled file with the SAME size but another sha256 is fetched again, and the stale entry is deleted', async () => {
    const root = setup();
    const calls: string[] = [], content = { value: bytesOf(1, 2, 3, 4) };
    const source = new CachedSource(inner(content, calls));
    source.setFiles([{ path: 'weights/a.bin', bytes: 4, sha256: 'aaaa'.repeat(16) }]);
    await source.bytes('weights/a.bin');
    content.value = bytesOf(9, 9, 9, 9);   // same size, new content (what a recompile of Aion's embedding scale did)
    source.setFiles([{ path: 'weights/a.bin', bytes: 4, sha256: 'bbbb'.repeat(16) }]);
    expect(new Uint8Array(await source.bytes('weights/a.bin'))).toEqual(new Uint8Array([9, 9, 9, 9]));
    expect(calls.length).toBe(2);
    const directory = await (await root.getDirectoryHandle('m2w-packages')).getDirectoryHandle('pkg');
    expect([...directory.files.keys()]).toEqual([CachedSource.entryName('weights/a.bin', 'bbbb'.repeat(16))]);
  });

  it('does not cache files the index does not list', async () => {
    const root = setup();
    const calls: string[] = [], content = { value: bytesOf(1) };
    const source = new CachedSource(inner(content, calls));
    await source.bytes('verification.json');
    await source.bytes('verification.json');
    expect(calls.length).toBe(2);
    expect(root.directories.size).toBe(0);
  });
});

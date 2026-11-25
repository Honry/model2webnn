import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { UrlSource } from './package.js';

// A tiny static server with (optional) byte ranges, to test the segmented download of big files.
const content = new Uint8Array(3 * 2 ** 20 + 12345).map((_, i) => (i * 31 + (i >> 8)) & 255);
const requests: { method: string; range?: string }[] = [];
let ranges = true, server: Server, base = '';

function handle(request: IncomingMessage, response: ServerResponse) {
  requests.push({ method: request.method as string, range: request.headers.range });
  const headers: Record<string, string | number> = { 'Content-Type': 'application/octet-stream' };
  if (ranges) headers['Accept-Ranges'] = 'bytes';
  const match = /^bytes=(\d+)-(\d+)$/.exec(request.headers.range ?? '');
  if (ranges && match) {
    const from = +match[1], to = Math.min(+match[2], content.length - 1);
    response.writeHead(206, { ...headers, 'Content-Length': to - from + 1, 'Content-Range': `bytes ${from}-${to}/${content.length}` });
    response.end(content.subarray(from, to + 1));
    return;
  }
  response.writeHead(200, { ...headers, 'Content-Length': content.length });
  response.end(request.method === 'HEAD' ? undefined : content);
}

beforeAll(async () => {
  server = createServer(handle);
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${(server.address() as AddressInfo).port}/`;
});
afterAll(() => new Promise<void>((resolve) => server.close(() => resolve())));

const same = (buffer: ArrayBuffer) => buffer.byteLength === content.length && new Uint8Array(buffer).every((v, i) => v === content[i]);

describe('UrlSource segmented download', () => {
  it('fetches a big file as parallel byte ranges into one buffer, with monotonic progress, using the size the caller knows', async () => {
    ranges = true; requests.length = 0;
    const seen: number[] = [];
    const data = await new UrlSource(base, { segments: 4, segmentMinBytes: 1 << 20 }).bytes('a.bin', (loaded, total) => { seen.push(loaded); expect(total).toBe(content.length); }, content.length);
    expect(same(data)).toBe(true);
    expect(requests.map((r) => r.method)).toEqual(['GET', 'GET', 'GET', 'GET']);   // no HEAD: the size was given
    expect(requests.map((r) => r.range).every((r) => /^bytes=\d+-\d+$/.test(r as string))).toBe(true);
    expect(seen[seen.length - 1]).toBe(content.length);
    expect(seen.every((v, i) => i === 0 || v >= seen[i - 1])).toBe(true);
  });

  it('asks the size with a HEAD request when it is not known', async () => {
    ranges = true; requests.length = 0;
    expect(same(await new UrlSource(base, { segments: 3, segmentMinBytes: 1 << 20 }).bytes('a.bin'))).toBe(true);
    expect(requests[0].method).toBe('HEAD');
    expect(requests.filter((r) => r.range).length).toBe(3);
  });

  it('uses one request for small files and for segments = 1', async () => {
    ranges = true; requests.length = 0;
    expect(same(await new UrlSource(base, { segments: 4, segmentMinBytes: 10 * 2 ** 20 }).bytes('a.bin', undefined, content.length))).toBe(true);
    expect(same(await new UrlSource(base, { segments: 1, segmentMinBytes: 1 }).bytes('a.bin', undefined, content.length))).toBe(true);
    expect(requests.length).toBe(2);
    expect(requests.every((r) => r.method === 'GET' && !r.range)).toBe(true);
  });

  it('falls back to a single request when the server does not serve ranges', async () => {
    ranges = false; requests.length = 0;
    expect(same(await new UrlSource(base, { segments: 4, segmentMinBytes: 1 << 20 }).bytes('a.bin'))).toBe(true);   // HEAD says no Accept-Ranges
    expect(requests.filter((r) => r.range).length).toBe(0);
    requests.length = 0;
    expect(same(await new UrlSource(base, { segments: 4, segmentMinBytes: 1 << 20 }).bytes('a.bin', undefined, content.length))).toBe(true);   // size given: the ranges are tried, answered 200 -> one plain request
    expect(requests.filter((r) => !r.range).length).toBe(1);
    ranges = true;
  });
});

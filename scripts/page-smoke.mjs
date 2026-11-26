// Smoke test of a bench page WITHOUT a device: headless Chrome / Edge loads the page from a local static server, `navigator.ml` is a validating mock (src/runtime/mock-ml.ts: every dispatch is checked
// by name, shape and dtype; reads return zeros) and the transformers.js CDN import is replaced by a tiny stub tokenizer, so the page logic (DOM, tabs, URL parameters, runtime calls) runs end to end
// on the tiny Qwen3.5 package (tests/golden/tiny-qwen35-package, linked as bench/packages/tiny-qwen35-pkg). It checks that the page does not crash; numbers and answers are meaningless.
//
//   node scripts/page-smoke.mjs "bench/raw/vision-suite.html?model=tiny-qwen35-pkg&maxNewTokens=3" --wait "Done" [--eval "js expression"]... [--browser path] [--timeout 120]
// prints the text of the page's #log / #out, the console errors, and the result of every --eval (run in the page after --wait matched; a string result is printed as is).
import { spawn } from 'node:child_process';
import { existsSync, mkdtempSync, readFileSync, statSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildSync } from 'esbuild';

const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const args = process.argv.slice(2), positional = args.filter((a, i) => !a.startsWith('--') && !(args[i - 1] ?? '').startsWith('--'));
const option = (name) => args.flatMap((a, i) => (a === `--${name}` ? [args[i + 1]] : []));
const target = positional[0], waitFor = option('wait')[0] ?? 'Done', evals = option('eval'), timeout = +(option('timeout')[0] ?? 120) * 1000;
const browser = option('browser')[0] ?? ['C:/Program Files/Google/Chrome/Application/chrome.exe', 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'].find(existsSync);
if (!target || !browser) { console.error('usage: node scripts/page-smoke.mjs <page url relative to the repo root> [--wait text] [--eval js] [--browser path]'); process.exit(2); }

const mockSource = buildSync({ entryPoints: [join(root, 'src/runtime/mock-ml.ts')], bundle: true, format: 'esm', write: false }).outputFiles[0].text;
const CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js';
const stubTransformers = `
const SPECIAL = { '<|vision_start|>': 288, '<|vision_end|>': 289, '<|image_pad|>': 290 };
export class PreTrainedTokenizer {
  constructor() {}
  encode(text) { const out = []; for (const part of String(text).split(/(<\\|vision_start\\|>|<\\|vision_end\\|>|<\\|image_pad\\|>)/)) { if (part in SPECIAL) out.push(SPECIAL[part]); else for (const c of part) out.push(3 + (c.codePointAt(0) % 250)); } return out; }
  decode(ids) { return Array.from(ids).map((t) => String.fromCharCode(97 + (t % 26))).join(''); }
}
export const AutoTokenizer = { from_pretrained: async () => new PreTrainedTokenizer() };
export const env = {};
`;
const NL = '\n';
const chatFormat = { format: 'm2w-chat', version: 0, first_prefix: `<|im_start|>user${NL}`, first_prefix_with_system: `<|im_start|>system${NL}<<system>><|im_end|>${NL}<|im_start|>user${NL}`,
  user_suffix_generation_prompt: `<|im_end|>${NL}<|im_start|>assistant${NL}`, assistant_suffix: `<|im_end|>${NL}`, next_user_prefix: `<|im_start|>user${NL}`, user_suffix_assistant_prefix: `<|im_end|>${NL}<|im_start|>assistant${NL}` };
const mime = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.png': 'image/png', '.txt': 'text/plain', '.css': 'text/css' };
const server = createServer((req, res) => {
  const url = new URL(req.url, 'http://x');
  if (url.pathname === '/__mock/mock-ml.js') { res.writeHead(200, { 'content-type': 'text/javascript' }); res.end(mockSource); return; }
  if (url.pathname === '/log') { res.writeHead(204); res.end(); return; }
  const tiny = /^\/bench\/packages\/tiny-qwen35-pkg\/(.*)$/.exec(url.pathname)?.[1];   // the committed tiny package has no tokenizer and no chat format: fake ones (the stub tokenizer ignores the json)
  if (tiny === 'tokenizer/tokenizer.json' || tiny === 'tokenizer/tokenizer_config.json') { res.writeHead(200, { 'content-type': 'application/json' }); res.end('{}'); return; }
  if (tiny === 'chat_format.json') { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(chatFormat)); return; }
  if (tiny === 'm2w.json') { res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify({ ...JSON.parse(readFileSync(join(root, 'tests/golden/tiny-qwen35-package/m2w.json'), 'utf8')), chat_format: 'chat_format.json' })); return; }
  const file = normalize(join(root, decodeURIComponent(url.pathname)));
  if (!file.startsWith(root) || !existsSync(file) || statSync(file).isDirectory()) { res.writeHead(404); res.end('not found'); return; }
  res.writeHead(200, { 'content-type': mime[extname(file)] ?? 'application/octet-stream', 'accept-ranges': 'none' });
  res.end(readFileSync(file));
});
await new Promise((ok) => server.listen(0, '127.0.0.1', ok));
const origin = `http://127.0.0.1:${server.address().port}`;

const port = 9300 + Math.floor(Math.random() * 500), profile = mkdtempSync(join(tmpdir(), 'smoke-'));
const chrome = spawn(browser, ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--no-first-run', '--disable-gpu', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
let wsUrl = null;
for (let i = 0; i < 80 && !wsUrl; i++) {
  try { wsUrl = (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find((t) => t.type === 'page')?.webSocketDebuggerUrl; } catch { /* not up yet */ }
  if (!wsUrl) await sleep(250);
}
if (!wsUrl) { chrome.kill(); server.close(); throw new Error('the browser did not start'); }
const ws = new WebSocket(wsUrl);
await new Promise((ok) => (ws.onopen = ok));
let nextId = 1;
const pending = new Map(), errors = [];
ws.onmessage = ({ data }) => {
  const m = JSON.parse(data);
  if (m.id) { pending.get(m.id)?.(m); pending.delete(m.id); return; }
  if (m.method === 'Fetch.requestPaused') {
    const { requestId, request } = m.params;
    if (request.url === CDN) send('Fetch.fulfillRequest', { requestId, responseCode: 200, responseHeaders: [{ name: 'content-type', value: 'text/javascript' }, { name: 'access-control-allow-origin', value: '*' }], body: Buffer.from(stubTransformers).toString('base64') });
    else send('Fetch.continueRequest', { requestId });
  } else if (m.method === 'Runtime.exceptionThrown') errors.push('exception: ' + (m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text));
  else if (m.method === 'Runtime.consoleAPICalled' && m.params.type === 'error') errors.push('console.error: ' + m.params.args.map((a) => a.value ?? a.description).join(' '));
  else if (m.method === 'Log.entryAdded' && m.params.entry.level === 'error' && !/favicon/.test(m.params.entry.url ?? '')) errors.push(`log: ${m.params.entry.text} ${m.params.entry.url ?? ''}`);
};
const send = (method, params = {}) => new Promise((ok) => { const id = nextId++; pending.set(id, ok); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (r.result?.exceptionDetails) return 'EVAL ERROR: ' + (r.result.exceptionDetails.exception?.description ?? r.result.exceptionDetails.text);
  const v = r.result?.result?.value;
  return typeof v === 'string' ? v : JSON.stringify(v);
};
await send('Page.enable'); await send('Runtime.enable'); await send('Log.enable');
await send('Fetch.enable', { patterns: [{ urlPattern: CDN }] });
await send('Page.addScriptToEvaluateOnNewDocument', { source: `
  Object.defineProperty(navigator, 'ml', { configurable: true, value: { createContext: async () => { const m = await import('/__mock/mock-ml.js'); m.installMockBuilder(); const c = new m.MockContext(); window.__mockContext = c; return c; } } });
` });
await send('Page.navigate', { url: `${origin}/${target}` });

const text = () => evaluate(`(document.getElementById('log') || document.getElementById('out') || document.body).textContent`);
const started = Date.now();
let page = '';
while (Date.now() - started < timeout) {
  page = await text();
  if (page.includes(waitFor) || /FAILED|"fatal"|Failed:/.test(page)) break;
  await sleep(500);
}
console.log(`--- page text after ${((Date.now() - started) / 1000).toFixed(1)} s ---\n${page.slice(-4000)}`);
for (const expression of evals) console.log(`--- eval: ${expression}\n${await evaluate(expression)}`);
console.log(`--- ${errors.length} browser error(s) ---\n${errors.join('\n')}`);
ws.close(); chrome.kill(); server.close();
process.exit(errors.length || /FAILED|Failed:/.test(page) ? 1 : 0);

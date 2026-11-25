// Structural equivalence check (no browser, no GPU): replay the hand-written graph builders of qwen3_subgraphs.js and the Python-generated graph JSON
// (through ir-loader) on the SAME recording mock builder, then compare what WebNN would actually compile: the sub-graph REACHABLE from the outputs.
// Every operand gets a canonical hash (op, options, hashes of its operands; graph inputs by name / type / shape; constants by type / shape / value),
// so creation order and dead nodes (e.g. the unused f16 copies of the new key / value rows in the hand-written code) do not matter.
//
//   node bench/tools/diff-handwritten.mjs [rawDir] [graphDir]
//   rawDir   : the folder with the hand-written graphs, bench/raw   (qwen3_subgraphs.js, layer.js)
//   graphDir : python/_out/qwen3-4b                    (python -m m2w.cli graphs --preset qwen3-4b --out ... --max-length 1152 --chunk 64 --blocks-per-graph 5)
// Needs the runtime built to lib/ (npm run build:lib).
import { createHash } from 'node:crypto';
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, '$1')), '../..');
const rawDir = process.argv[2] ?? path.join(root, 'bench/raw');
const graphDir = process.argv[3] ?? path.join(root, 'python/_out/qwen3-4b');
// the benchmark folder has no package.json ("type": "module"), so Node would load its .js files as CommonJS: use a copy in _out/handwritten
const copyDir = path.join(root, '_out/handwritten');
mkdirSync(copyDir, { recursive: true });
writeFileSync(path.join(copyDir, 'package.json'), '{"type":"module"}');
for (const file of ['qwen3_subgraphs.js', 'layer.js']) copyFileSync(path.join(rawDir, file), path.join(copyDir, file));
const sub = await import(pathToFileURL(path.join(copyDir, 'qwen3_subgraphs.js')).href);
const { loadGraph } = await import(pathToFileURL(path.join(root, 'lib/runtime/ir-loader.js')).href);

// ---------------------------------------------------------------------------------------------------------------- recording mock builder
const count = (shape) => shape.reduce((a, b) => a * b, 1);
const sha = (text) => createHash('sha1').update(text).digest('hex').slice(0, 12);
function broadcast(a, b) {
  const out = [];
  for (let i = 1; i <= Math.max(a.length, b.length); i++) { const x = a[a.length - i] ?? 1, y = b[b.length - i] ?? 1; if (x !== y && x !== 1 && y !== 1) throw new Error(`broadcast ${a} ${b}`); out.unshift(x === 1 ? y : x); }
  return out;
}
// a constant's value signature: external / dummy (no data), uniform fill, or the first values of a small literal
function constantSignature(data) {
  if (!data || data.length === 0) return 'external';
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  const first = bytes[0];
  if (bytes.every((b) => b === first)) return `uniform:${Buffer.from(bytes.subarray(0, Math.min(bytes.length, 8))).toString('hex')}:${bytes.length}`;
  return `data:${sha(Buffer.from(bytes).toString('hex'))}`;
}
function makeBuilder() {
  const inputs = {};
  const operand = (dataType, shape, hash, node) => ({ dataType, shape: [...shape], hash, node });
  const make = (op, ins, options, dataType, shape) => {
    const type = `${dataType}[${shape}]`;
    const hash = sha(JSON.stringify([op, options, ins.map((o) => (o ? o.hash : null)), type]));
    return operand(dataType, shape, hash, { op, options, ins: ins.filter(Boolean), hash, type, inputTypes: ins.map((o) => (o ? `${o.dataType}[${o.shape}]` : null)) });
  };
  const elementwise = (op) => (a, b) => make(op, [a, b], {}, a.dataType, broadcast(a.shape, b.shape));
  const reduce = (op) => (x, o = {}) => {
    const axes = (o.axes ?? x.shape.map((_, i) => i)).map((a) => (a < 0 ? a + x.shape.length : a)), keep = o.keepDimensions ?? false;
    return make(op, [x], { axes: o.axes, keepDimensions: keep }, x.dataType, keep ? x.shape.map((d, i) => (axes.includes(i) ? 1 : d)) : x.shape.filter((_, i) => !axes.includes(i)));
  };
  return {
    inputs,
    input: (name, d) => (inputs[name] = operand(d.dataType, d.shape, `in:${name}:${d.dataType}[${d.shape}]`, null)),
    constant: (d, data) => operand(d.dataType, d.shape, `const:${d.dataType}[${d.shape}]:${constantSignature(data)}`, null),
    add: elementwise('add'), sub: elementwise('sub'), mul: elementwise('mul'), div: elementwise('div'), pow: elementwise('pow'),
    sqrt: (x) => make('sqrt', [x], {}, x.dataType, x.shape), sigmoid: (x) => make('sigmoid', [x], {}, x.dataType, x.shape),
    cast: (x, to) => make('cast', [x], { to }, to, x.shape),
    reshape: (x, shape) => { if (count(shape) !== count(x.shape)) throw new Error(`reshape ${x.shape} -> ${shape}`); return make('reshape', [x], { newShape: shape }, x.dataType, shape); },
    transpose: (x, o) => make('transpose', [x], { permutation: o.permutation }, x.dataType, o.permutation.map((p) => x.shape[p])),
    slice: (x, starts, sizes) => make('slice', [x], { starts, sizes }, x.dataType, sizes),
    concat: (xs, axis) => make('concat', xs, { axis }, xs[0].dataType, xs[0].shape.map((d, i) => (i === axis ? xs.reduce((s, x) => s + x.shape[axis], 0) : d))),
    expand: (x, shape) => make('expand', [x], { newShape: shape }, x.dataType, shape),
    reduceMean: reduce('reduceMean'), reduceMax: reduce('reduceMax'), reduceSum: reduce('reduceSum'),
    softmax: (x, axis) => make('softmax', [x], { axis }, x.dataType, x.shape),
    matmul: (a, c) => make('matmul', [a, c], {}, a.dataType, [...broadcast(a.shape.slice(0, -2), c.shape.slice(0, -2)), a.shape.at(-2), c.shape.at(-1)]),
    gemm: (a, c, o = {}) => {
      const [m, k] = o.aTranspose ? [a.shape[1], a.shape[0]] : a.shape, [k2, n] = o.bTranspose ? [c.shape[1], c.shape[0]] : c.shape;
      if (k !== k2) throw new Error(`gemm ${a.shape} ${c.shape}`);
      return make('gemm', [a, c], { ...(o.bTranspose ? { bTranspose: true } : {}), ...(o.aTranspose ? { aTranspose: true } : {}) }, a.dataType, [m, n]);
    },
    dequantizeLinear: (x, scale, zp) => make('dequantizeLinear', [x, scale, zp ?? null], {}, scale.dataType, x.shape),
    tensorScatter: (x, u, i, o) => make('tensorScatter', [x, u, i], { axis: o.axis }, x.dataType, x.shape),
  };
}
// what WebNN compiles: the nodes reachable from the outputs, and the graph inputs they use
function reachable(outputs) {
  const nodes = new Map(), usedInputs = new Set(), seen = new Set();
  const visit = (o) => {
    if (seen.has(o)) return; seen.add(o);
    if (!o.node) { if (o.hash.startsWith('in:')) usedInputs.add(o.hash.slice(3)); return; }
    nodes.set(o, o.node);
    for (const i of o.node.ins) visit(i);
  };
  for (const o of Object.values(outputs)) visit(o);
  return { nodes: [...nodes.values()], usedInputs: [...usedInputs].sort() };
}
function describe(builder, outputs) {
  const { nodes, usedInputs } = reachable(outputs);
  const histogram = new Map();
  for (const n of nodes) histogram.set(n.hash, (histogram.get(n.hash) ?? 0) + 1);
  return {
    nodes, usedInputs, histogram,
    declaredInputs: Object.entries(builder.inputs).map(([n, o]) => o.hash.slice(3)).sort(),
    outputs: Object.entries(outputs).map(([n, o]) => `${n}:${o.dataType}[${o.shape}]=${o.hash}`).sort(),
  };
}

// ---------------------------------------------------------------------------------------------------------------- cases
sub.setMergedProjections(true);
const P = 1151, R = 64, C = 64;
const optionsFor = (kvStyle) => ({ matmulStyle: 'gemm', zeroPoint: 'const', dequantStyle: 'dequantizeLinear', computeType: 'float32', fuseKvUpdate: true, kvStyle, valueLayout: 'natural', stagingRows: R });
const decode = { sequenceLength: 1, pastLength: P, maskDataType: 'float16', options: optionsFor('staged') };
const prefill = { sequenceLength: C, pastLength: P, maskDataType: 'float16', options: optionsFor('staged') };
const cases = [
  ['decode_head', (b) => sub.buildHeadGraph(b, decode)],
  ['decode_block', (b) => sub.buildBlockGraph(b, decode)],
  ['decode_block_x5', (b) => sub.buildBlockGroupGraph(b, { ...decode, blockCount: 5 })],
  ['decode_tail', (b) => sub.buildTailGraph(b, decode)],
  ['lm_head', (b) => sub.buildLmHeadGraph(b, { weightMode: 'constant', weightProvider: () => new Uint8Array(0), options: decode.options })],
  ['kv_flush', (b) => sub.buildKvUpdateGraph(b, { sequenceLength: R, pastLength: P, valueLayout: 'natural' })],
  [`prefill_head_c${C}`, (b) => sub.buildHeadGraph(b, prefill)],
  [`prefill_block_c${C}`, (b) => sub.buildBlockGraph(b, prefill)],
  [`prefill_tail_c${C}`, (b) => sub.buildTailGraph(b, prefill)],
];

let failures = 0;
for (const [name, create] of cases) {
  const handBuilder = makeBuilder();
  const hand = describe(handBuilder, create(handBuilder).outputs);
  const doc = JSON.parse(readFileSync(path.join(graphDir, `${name}.json`), 'utf8'));
  const irBuilder = makeBuilder();
  const ir = describe(irBuilder, loadGraph(irBuilder, doc, () => new Uint8Array(0)).outputs);
  const problems = [];
  const sameList = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  if (!sameList(hand.declaredInputs, ir.declaredInputs)) problems.push(`declared inputs differ:\n    hand-only ${hand.declaredInputs.filter((x) => !ir.declaredInputs.includes(x))}\n    IR-only   ${ir.declaredInputs.filter((x) => !hand.declaredInputs.includes(x))}`);
  if (!sameList(hand.usedInputs, ir.usedInputs)) problems.push(`inputs used by the outputs differ:\n    hand-only ${hand.usedInputs.filter((x) => !ir.usedInputs.includes(x))}\n    IR-only   ${ir.usedInputs.filter((x) => !hand.usedInputs.includes(x))}`);
  if (!sameList(ir.declaredInputs, ir.usedInputs)) problems.push(`IR graph declares inputs that no output uses (WebNN would drop them and break the binding): ${ir.declaredInputs.filter((x) => !ir.usedInputs.includes(x))}`);
  const declaredOutputs = doc.outputs.map((o) => `${o.name}:${o.dataType}[${o.shape}]`).sort();
  if (!sameList(declaredOutputs, ir.outputs.map((s) => s.split('=')[0]))) problems.push(`the document declares other output shapes than the mock infers`);
  if (!sameList(hand.outputs, ir.outputs)) {
    problems.push(`output hashes differ (the computations are not the same graph):\n    hand ${hand.outputs.join(' ')}\n    IR   ${ir.outputs.join(' ')}`);
    const onlyHand = [...hand.histogram.keys()].filter((h) => !ir.histogram.has(h)), onlyIr = [...ir.histogram.keys()].filter((h) => !hand.histogram.has(h));
    const show = (nodes, only) => nodes.filter((n) => only.includes(n.hash)).slice(0, 4).map((n) => `${n.op}(${n.inputTypes.join(', ')}) ${JSON.stringify(n.options)} -> ${n.type}`).join('\n      ');
    problems.push(`reachable nodes only in hand (${onlyHand.length}):\n      ${show(hand.nodes, onlyHand)}\n    only in IR (${onlyIr.length}):\n      ${show(ir.nodes, onlyIr)}`);
  }
  if (hand.nodes.length !== ir.nodes.length) problems.push(`reachable node count: hand ${hand.nodes.length}, IR ${ir.nodes.length}`);
  if (problems.length) { failures++; console.log(`DIFF  ${name}\n  ${problems.join('\n  ')}`); }
  else console.log(`SAME  ${name}: ${hand.nodes.length} reachable nodes, ${hand.declaredInputs.length} inputs, ${hand.outputs.length} outputs (hand-written graph had ${'-'} dead nodes ignored)`.replace(' (hand-written graph had - dead nodes ignored)', ''));
}
process.exit(failures ? 1 : 0);

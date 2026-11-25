// Times every graph of a package ON ITS OWN (random inputs, N back-to-back dispatches of the same tensors, one readback): which graph of a model is slow?
// URL: m2wgraphs.html?package=../packages/aion-2.3b-i8/&device=npu&maxLength=1152&chunk=64&chain=20&variant=onehot-float32   (variant: default = the package's default variant)
import { openPackage, loadGraph } from "../../lib/runtime/index.js";
import { buildLmHeadGraph, setLmHeadFormat } from "./qwen3_subgraphs.js?v=20261124a";   // the hand-written lm_head (Qwen3-4B dims), timed side by side with the package graph

const query = new URLSearchParams(location.search);
const param = (name, fallback) => query.get(name) ?? fallback;
const packageUrl = new URL(param("package", "../packages/aion-2.3b-i8/"), location.href).href, deviceType = param("device", "npu"), chain = +param("chain", 20), iterations = +param("iters", 5);
const only = (query.get("only") || "").split(",").filter(Boolean);
const byId = (id) => document.getElementById(id);
const start = performance.now();
const log = (message) => { byId("out").textContent += `[${((performance.now() - start) / 1000).toFixed(1).padStart(6)} s] ${message}\n`; console.log(message); fetch("/log", { method: "POST", body: JSON.stringify({ experiment: "m2wgraphs", message }) }).catch(() => {}); };
const bytesOf = { float32: 4, float16: 2, int32: 4, int64: 8, int8: 1, uint8: 1, int4: 0.5, uint4: 0.5 };
const count = (shape) => shape.reduce((a, b) => a * b, 1);
const median = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];

let seed = 12345;
const pool = new Uint8Array(16 << 20);
for (let i = 0; i < pool.length; i++) { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; pool[i] = seed & 255; }
const randomBytes = (n) => { const out = new Uint8Array(n); for (let o = 0; o < n; o += pool.length) out.set(pool.subarray(0, Math.min(pool.length, n - o)), o); return out; };

function dataFor(spec) {
    const n = count(spec.shape);
    if (spec.dataType === "int4") return randomBytes(Math.ceil(n / 2));
    if (spec.dataType === "uint8" || spec.dataType === "int8") return randomBytes(n);
    if (spec.dataType === "int32") return new Int32Array(n);
    const half = new Float16Array(n);   // activations, scales, gammas, caches, masks (0 = visible), selectors
    if (spec.name.endsWith("_scale")) half.fill(0.002); else if (spec.name.endsWith("_gamma")) half.fill(1); else if (spec.role === "weight") half.fill(0.02); else if (spec.role === "activation" || spec.role === "kv") half.fill(0.05);
    return spec.dataType === "float32" ? Float32Array.from(half) : half;
}

// plausible lm_head data: random u8 weights, zero point 128, scale 0.002 (random bytes as f16 scales would be NaN / inf)
function lmHeadConstant(c) {
    const n = count(c.shape);
    if (c.external === "lm_head_zero_point") return new Uint8Array(n).fill(128);
    if (c.external === "lm_head_scale") return new Uint8Array(new Float16Array(n).fill(0.002).buffer);
    return randomBytes(n);
}

// time one built graph with the given inputs / outputs (descriptors { name, dataType, shape })
async function timeGraph(ctx, name, graph, inputSpecs, outputSpecs, extra) {
    const inputTensors = {}, outputTensors = {};
    for (const spec of inputSpecs) { inputTensors[spec.name] = await ctx.createTensor({ dataType: spec.dataType, shape: spec.shape, writable: true }); ctx.writeTensor(inputTensors[spec.name], dataFor(spec)); }
    for (const o of outputSpecs) outputTensors[o.name] = await ctx.createTensor({ dataType: o.dataType, shape: o.shape, readable: true });
    const first = outputSpecs[0], sink = first.dataType === "float32" ? new Float32Array(count(first.shape)) : new Float16Array(count(first.shape));
    const run = async (n) => { const s0 = performance.now(); for (let i = 0; i < n; i++) ctx.dispatch(graph, inputTensors, outputTensors); await ctx.readTensor(outputTensors[first.name], sink); return performance.now() - s0; };
    await run(2);
    const single = [], chained = [];
    for (let i = 0; i < iterations; i++) { single.push(await run(1)); chained.push((await run(chain)) / chain); }
    log(`${name.padEnd(16)} ${extra} | single ${median(single).toFixed(2).padStart(7)} ms | chained ${median(chained).toFixed(3).padStart(7)} ms/dispatch`);
    for (const t of Object.values(inputTensors).concat(Object.values(outputTensors))) t.destroy?.();
}

// lmVariants=1: the lm_head graph of the package with (a) synthetic data (b) the package's real embedding table (c) the real table with the scale of zero-scale rows set to 1e-3
async function lmHeadVariants(ctx, pkg) {
    const plan = pkg.plan, doc = JSON.parse(await pkg.source.text(plan.graphs[param("variant", plan.default_variant)].lm_head));
    const fileOf = (name) => pkg.manifest.files.find((f) => f.role === "embedding" && f.tensors[0].name === name);
    const real = {};
    for (const name of plan.weights.embedding) {   // int8-sym tables have no zero point; the tensor sits inside its .safetensors shard
        const file = fileOf(name), tensor = file.tensors[0];
        real[name] = new Uint8Array(await pkg.source.bytes(`weights/${file.file}`), tensor.offset, tensor.bytes).slice();
    }
    const scale = new Float16Array(real.lm_head_scale.buffer.slice(0));
    let zeroScaleRows = 0, smallest = Infinity, largest = 0;   // (no Math.min(...array): 200k arguments overflow the stack)
    for (const v of scale) { if (v === 0) zeroScaleRows++; else if (v < smallest) smallest = v; if (v > largest) largest = v; }
    const fixed = new Float16Array(scale); for (let i = 0; i < fixed.length; i++) if (fixed[i] === 0) fixed[i] = 0.001;
    log(`real table: ${scale.length} rows, ${zeroScaleRows} rows with scale 0, nonzero scale min ${smallest}, max ${largest}`);
    const variants = [
        ["synthetic data", lmHeadConstant],
        ["real data", (c) => real[c.external]],
        ["real, zero scales -> 1e-3", (c) => (c.external === "lm_head_scale" ? new Uint8Array(fixed.buffer) : real[c.external])],
    ];
    for (const [label, provider] of variants) {
        const builder = new MLGraphBuilder(ctx), t0 = performance.now();
        const { outputs } = loadGraph(builder, doc, provider);
        const graph = await builder.build(outputs);
        await timeGraph(ctx, "lm_head", graph, doc.inputs, doc.outputs, `build ${((performance.now() - t0) / 1000).toFixed(1).padStart(5)} s | ${label.padEnd(26)}`);
        graph.destroy?.();
    }
}

// lmInt8=1: the hand-written lm_head in its two formats (Qwen3-4B dims), each with normal scales and with 35 rows whose scale is 0 (weight 0):
//   uint8 asymmetric (per-row zero point: what the OpenVINO export and the packages use)  and  int8 symmetric (no zero point, the experiment format of qwen3_subgraphs.js)
async function lmHeadInt8(ctx) {
    const V = 151936, E = 2560, options = { matmulStyle: "gemm", zeroPoint: "const", dequantStyle: "dequantizeLinear", computeType: "float32" };
    const make = (format, zeroRows) => ([name]) => {
        if (name === "lm_head_weight") { const w = randomBytes(V * E); if (zeroRows) w.fill(format === "uint8Asymmetric" ? 128 : 0, 0, zeroRows * E); return w; }   // dead rows: q = zero point (u8) / 0 (i8): they dequantize to 0
        if (name === "lm_head_zero_point") return new Uint8Array(V).fill(128);
        const scale = new Float16Array(V).fill(0.002); for (let i = 0; i < zeroRows; i++) scale[i] = 0;
        return new Uint8Array(scale.buffer);
    };
    for (const format of ["uint8Asymmetric", "int8Symmetric"]) for (const zeroRows of [0, 35]) {
        setLmHeadFormat(format);
        const builder = new MLGraphBuilder(ctx), t0 = performance.now();
        const description = buildLmHeadGraph(builder, { weightMode: "constant", weightProvider: make(format, zeroRows), options });
        const graph = await builder.build(description.outputs);
        await timeGraph(ctx, "lm_head_HAND", graph, Object.entries(description.inputs).map(([n, o]) => ({ name: n, dataType: o.dataType, shape: o.shape, role: "activation" })),
            Object.entries(description.outputs).map(([n, o]) => ({ name: n, dataType: o.dataType, shape: o.shape })), `build ${((performance.now() - t0) / 1000).toFixed(1).padStart(5)} s | ${format.padEnd(16)} | ${String(zeroRows).padStart(2)} rows with scale 0`);
        graph.destroy?.();
    }
    setLmHeadFormat("uint8Asymmetric");
}

// lmBuild=1: build time of the two lm_head formats (Qwen3-4B dims, no zero scales), alternating, 4 rounds each: the median is what counts
async function lmHeadBuildAB(ctx) {
    const V = 151936, E = 2560, options = { matmulStyle: "gemm", zeroPoint: "const", dequantStyle: "dequantizeLinear", computeType: "float32" };
    const provider = ([name]) => (name === "lm_head_weight" ? randomBytes(V * E) : name === "lm_head_zero_point" ? new Uint8Array(V).fill(128) : new Uint8Array(new Float16Array(V).fill(0.002).buffer));
    const builds = { uint8Asymmetric: [], int8Symmetric: [] }, runs = { uint8Asymmetric: [], int8Symmetric: [] };
    for (let round = 1; round <= 4; round++) for (const format of Object.keys(builds)) {
        setLmHeadFormat(format);
        const builder = new MLGraphBuilder(ctx), t0 = performance.now();
        const description = buildLmHeadGraph(builder, { weightMode: "constant", weightProvider: provider, options });
        const graph = await builder.build(description.outputs), ms = performance.now() - t0;
        const inputs = {}, outputs = {};
        for (const [n, o] of Object.entries(description.inputs)) { inputs[n] = await ctx.createTensor({ dataType: o.dataType, shape: o.shape, writable: true }); ctx.writeTensor(inputs[n], new Float16Array(count(o.shape)).fill(0.05)); }
        for (const [n, o] of Object.entries(description.outputs)) outputs[n] = await ctx.createTensor({ dataType: o.dataType, shape: o.shape, readable: true });
        const sink = new Float16Array(V), name = Object.keys(outputs)[0];
        for (let i = 0; i < 2; i++) { ctx.dispatch(graph, inputs, outputs); await ctx.readTensor(outputs[name], sink); }
        const t1 = performance.now(); for (let i = 0; i < 10; i++) ctx.dispatch(graph, inputs, outputs); await ctx.readTensor(outputs[name], sink);
        builds[format].push(ms); runs[format].push((performance.now() - t1) / 10);
        log(`round ${round} ${format.padEnd(16)} build ${(ms / 1000).toFixed(2)} s, run ${runs[format].at(-1).toFixed(2)} ms/dispatch`);
        for (const t of Object.values(inputs).concat(Object.values(outputs))) t.destroy?.();
        graph.destroy?.();
    }
    for (const format of Object.keys(builds)) log(`RESULT ${format.padEnd(16)} build median ${(median(builds[format]) / 1000).toFixed(2)} s (min ${(Math.min(...builds[format]) / 1000).toFixed(2)}, max ${(Math.max(...builds[format]) / 1000).toFixed(2)}), run median ${median(runs[format]).toFixed(2)} ms`);
    setLmHeadFormat("uint8Asymmetric");
}

async function main() {
    const pkg = await openPackage(packageUrl, false);
    const key = param("maxLength", String(Math.min(...pkg.plan.max_lengths))), chunk = param("chunk", String(Math.min(...Object.keys(pkg.plan.graphs[pkg.plan.default_variant].prefill[key]).map(Number))));
    const variant = param("variant", pkg.plan.default_variant), table = pkg.plan.graphs[variant];
    const decode = table.decode[key], prefill = table.prefill[key][chunk];
    const list = [["lm_head", table.lm_head], ["decode_head", decode.head], ["decode_block", decode.block], ["decode_tail", decode.tail], ...(decode.flush ? [["kv_flush", decode.flush]] : []),
        ...Object.entries(decode.block_groups ?? {}).map(([k, p]) => [`decode_block_x${k}`, p]),
        ["prefill_head", prefill.head], ["prefill_block", prefill.block], ["prefill_tail", prefill.tail]].filter(([name]) => !only.length || only.includes(name));
    const ctx = await navigator.ml.createContext({ deviceType });
    log(`${pkg.model.architecture} (${pkg.model.num_layers} layers), max length ${key}, chunk ${chunk}, variant ${variant}, device ${deviceType}, chain ${chain}`);
    if (query.get("lmBuild") === "1") { await lmHeadBuildAB(ctx); log("DONE"); byId("status").textContent = "done"; return; }
    if (query.get("lmInt8") === "1") { await lmHeadInt8(ctx); log("DONE"); byId("status").textContent = "done"; return; }
    if (query.get("lmVariants") === "1") { await lmHeadVariants(ctx, pkg); log("DONE"); byId("status").textContent = "done"; return; }
    if (pkg.model.vocab_padded === 151936 && pkg.model.hidden === 2560) {
        const options = { matmulStyle: "gemm", zeroPoint: "const", dequantStyle: "dequantizeLinear", computeType: "float32" };
        const builder = new MLGraphBuilder(ctx), t0 = performance.now();
        const description = buildLmHeadGraph(builder, { weightMode: "constant", weightProvider: ([n]) => lmHeadConstant({ external: n, shape: n === "lm_head_weight" ? [151936, 2560] : [151936, 1] }), options });
        const graph = await builder.build(description.outputs);
        await timeGraph(ctx, "lm_head_HAND", graph, Object.entries(description.inputs).map(([n, o]) => ({ name: n, dataType: o.dataType, shape: o.shape, role: "activation" })),
            Object.entries(description.outputs).map(([n, o]) => ({ name: n, dataType: o.dataType, shape: o.shape })), `build ${((performance.now() - t0) / 1000).toFixed(1).padStart(5)} s (hand-written JS graph, same data)`);
        graph.destroy?.();
    }
    for (const [name, path] of list) {
        const doc = JSON.parse(await pkg.source.text(path));
        const builder = new MLGraphBuilder(ctx);
        const external = lmHeadConstant;
        let t = performance.now();
        const { inputs, outputs } = loadGraph(builder, doc, external);
        const graph = await builder.build(outputs), buildMs = performance.now() - t;
        const inputTensors = {}, outputTensors = {};
        for (const spec of doc.inputs) { inputTensors[spec.name] = await ctx.createTensor({ dataType: spec.dataType, shape: spec.shape, writable: true }); ctx.writeTensor(inputTensors[spec.name], dataFor(spec)); }
        for (const o of doc.outputs) outputTensors[o.name] = await ctx.createTensor({ dataType: o.dataType, shape: o.shape, readable: true });
        const first = doc.outputs[0], sink = first.dataType === "float32" ? new Float32Array(count(first.shape)) : new Float16Array(count(first.shape));
        const run = async (n) => { const s = performance.now(); for (let i = 0; i < n; i++) ctx.dispatch(graph, inputTensors, outputTensors); await ctx.readTensor(outputTensors[first.name], sink); return performance.now() - s; };
        await run(2);
        const single = [], chained = [];
        for (let i = 0; i < iterations; i++) { single.push(await run(1)); chained.push((await run(chain)) / chain); }
        const weights = doc.inputs.filter((i) => i.role === "weight").length, weightMb = doc.inputs.filter((i) => i.role === "weight").reduce((s, i) => s + count(i.shape) * bytesOf[i.dataType], 0) / 2 ** 20;
        log(`${name.padEnd(16)} build ${(buildMs / 1000).toFixed(1).padStart(5)} s | single ${median(single).toFixed(2).padStart(7)} ms | chained ${median(chained).toFixed(3).padStart(7)} ms/dispatch | ${doc.nodes.length} nodes, ${doc.inputs.length} inputs (${weights} weights, ${weightMb.toFixed(0)} MB)`);
        for (const t of Object.values(inputTensors).concat(Object.values(outputTensors))) t.destroy?.();
        graph.destroy?.();
    }
    log("DONE");
    byId("status").textContent = "done";
}
main().catch((e) => { log(`ERROR ${e?.stack || e}`); byId("status").textContent = "failed"; });

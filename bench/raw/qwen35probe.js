// Runs the linear attention half of layer 0 of the Qwen3.5 package (the HEAD graph with every intermediate result as an extra output) on the device with the real weights and the real prompt, and
// compares each intermediate result with the numpy interpreter's (python tools/qwen35_probe.py -> bench/data/qwen35-probe/): which operation first goes wrong?
// URL: qwen35probe.html?package=../packages/qwen35-4b-pkg/&probe=../data/qwen35-probe/&device=npu&which=prefill,decode&threshold=0.05
import { loadGraph } from "../../lib/runtime/index.js";

const query = new URLSearchParams(location.search);
const param = (name, fallback) => query.get(name) ?? fallback;
const packageUrl = new URL(param("package", "../packages/qwen35-4b-pkg/"), location.href).href, probeUrl = new URL(param("probe", "../data/qwen35-probe/"), location.href).href;
const threshold = +param("threshold", 0.05), deviceType = param("device", "npu");
const byId = (id) => document.getElementById(id);
const start = performance.now();
const log = (message) => { byId("out").textContent += `[${((performance.now() - start) / 1000).toFixed(1).padStart(6)} s] ${message}\n`; console.log(message); fetch("/log", { method: "POST", body: JSON.stringify({ experiment: "qwen35probe", message }) }).catch(() => {}); };
const count = (shape) => shape.reduce((a, b) => a * b, 1);
const ROW_AXIS = { p_normalized: 1, p_qkvz: 1, p_ba: 1, p_beta: 1, p_g: 1, p_conv: 1, p_normed: 1, p_projected: 1, residual: 1, attention_output: 1, p_q: 2, p_k: 2, p_v: 2, p_core: 2 };   // outputs with one row per token

/** error over the REAL rows (the left-padded chunk has `pad` garbage rows in front) */
function compare(name, got, want, shape, pad) {
    const axis = ROW_AXIS[name], inner = axis === undefined ? 1 : count(shape.slice(axis + 1)), rows = axis === undefined ? 1 : shape[axis];
    let maxAbs = 0, errSq = 0, refSq = 0, bad = 0;
    for (let i = 0; i < want.length; i++) {
        if (axis !== undefined && Math.floor(i / inner) % rows < pad) continue;
        const g = got[i];
        if (!Number.isFinite(g)) { bad++; continue; }
        const d = g - want[i]; maxAbs = Math.max(maxAbs, Math.abs(d)); errSq += d * d; refSq += want[i] * want[i];
    }
    return { maxAbs, rel: Math.sqrt(errSq / (refSq + 1e-30)), bad, scale: Math.sqrt(refSq / Math.max(1, want.length)) };
}

async function runOne(ctx, name, probe, shard) {
    const doc = await (await fetch(`${probeUrl}${name}.json`)).json(), io = await (await fetch(`${probeUrl}${name}.io.json`)).json(), blob = new Uint8Array(await (await fetch(`${probeUrl}${name}.bin`)).arrayBuffer());
    const builder = new MLGraphBuilder(ctx);
    const { outputs } = loadGraph(builder, doc);
    const graph = await builder.build(outputs);
    const inputs = {}, outs = {};
    for (const spec of io.inputs) { inputs[spec.name] = await ctx.createTensor({ dataType: spec.dataType, shape: spec.shape, writable: true }); ctx.writeTensor(inputs[spec.name], blob.slice(spec.offset, spec.offset + spec.bytes)); }
    for (const weight of io.weights) {
        const entry = probe.tensors.find((t) => t.name === weight);
        inputs[weight] = await ctx.createTensor({ dataType: entry.dtype, shape: entry.shape, writable: true });
        ctx.writeTensor(inputs[weight], new Uint8Array(shard, entry.offset, entry.bytes));
    }
    for (const o of doc.outputs) outs[o.name] = await ctx.createTensor({ dataType: o.dataType, shape: o.shape, readable: true });
    log(`${name}: dispatching (${doc.nodes.length} nodes, ${Object.keys(outs).length} outputs)`);
    ctx.dispatch(graph, inputs, outs);
    const pad = name === "probe_prefill" ? probe.pad : 0;
    let first = null;
    for (const want of io.expected) {
        const o = doc.outputs.find((x) => x.name === want.name), size = count(o.shape), raw = new Uint8Array(size * (o.dataType === "float16" ? 2 : 4));
        await ctx.readTensor(outs[want.name], raw);
        const got = o.dataType === "float16" ? new Float16Array(raw.buffer) : new Float32Array(raw.buffer);
        const c = compare(want.name, got, new Float32Array(blob.buffer.slice(want.offset, want.offset + want.bytes)), o.shape, pad);
        const flag = c.rel > threshold || c.bad ? "  <== DEVIATES" : "";
        if (flag && !first && want.name.startsWith("p_")) first = want.name;
        log(`${want.name.padEnd(18)} [${o.shape}]  rel ${c.rel.toExponential(2)}  max abs ${c.maxAbs.toExponential(2)}  rms of the reference ${c.scale.toExponential(2)}${c.bad ? `  NOT FINITE x${c.bad}` : ""}${flag}`);
    }
    log(first ? `${name}: the first intermediate result (in the order of the computation) that deviates: ${first}` : `${name}: every intermediate result is within ${threshold}`);
}

async function main() {
    const probe = await (await fetch(`${probeUrl}probe.json`)).json();
    const shard = await (await fetch(`${packageUrl}${probe.shard}`)).arrayBuffer();
    const ctx = await navigator.ml.createContext({ deviceType });
    log(`device ${deviceType}; layer 0 weights ${(shard.byteLength / 2 ** 20).toFixed(0)} MB; prompt ${probe.tokens} tokens in a chunk of ${probe.rows} rows (${probe.pad} pad rows); threshold ${threshold}`);
    for (const which of param("which", "prefill,decode").split(",")) await runOne(ctx, `probe_${which}`, probe, shard);
    log("DONE");
    byId("status").textContent = "done";
}
main().catch((e) => { log(`ERROR ${e?.stack || e}`); byId("status").textContent = "failed"; });

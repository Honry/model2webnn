// Gated DeltaNet recurrence (linear attention of Qwen3.5) as loop-free graphs, one layer, 32 heads, dk = dv = 128: accuracy against the float64 token loop AND time on the device.
// Graphs + inputs + expected outputs come from  python tools/gdn_experiment.py --export ../bench/data/gdn
// URL: gdn.html?device=npu&data=mixed-corr&chain=20&only=float16        (data: mixed = typical decays, mixed-corr = adversarial nearly parallel keys; only: substring filter of the graph names)
import { loadGraph } from "../../lib/runtime/index.js";

const query = new URLSearchParams(location.search);
const param = (name, fallback) => query.get(name) ?? fallback;
const deviceType = param("device", "npu"), dataset = param("data", "mixed"), chain = +param("chain", 20), iterations = +param("iters", 5), only = (query.get("only") || "").split(",").filter(Boolean);
const dir = new URL(param("dir", "../data/gdn/"), location.href).href;
const byId = (id) => document.getElementById(id);
const start = performance.now();
const log = (message) => { byId("out").textContent += `[${((performance.now() - start) / 1000).toFixed(1).padStart(6)} s] ${message}\n`; console.log(message); fetch("/log", { method: "POST", body: JSON.stringify({ experiment: "gdn", message }) }).catch(() => {}); };
const median = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
const count = (shape) => shape.reduce((a, b) => a * b, 1);
const decode = (dataType, bytes) => (dataType === "float16" ? Float32Array.from(new Float16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2)) : new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)));

function compare(got, want) {
    let maxAbs = 0, errSq = 0, refSq = 0, bad = 0;
    for (let i = 0; i < want.length; i++) {
        const g = got[i];
        if (!Number.isFinite(g)) { bad++; continue; }
        const d = g - want[i]; maxAbs = Math.max(maxAbs, Math.abs(d)); errSq += d * d; refSq += want[i] * want[i];
    }
    return `abs ${maxAbs.toExponential(2)} rel ${Math.sqrt(errSq / (refSq + 1e-30)).toExponential(2)}${bad ? `  NOT FINITE x${bad}` : ""}`;
}

async function runCase(ctx, name) {
    const doc = await (await fetch(`${dir}${name}.json`)).json();
    const io = await (await fetch(`${dir}${name}.${dataset}.io.json`)).json(), blob = new Uint8Array(await (await fetch(`${dir}${name}.${dataset}.bin`)).arrayBuffer());
    const builder = new MLGraphBuilder(ctx), t0 = performance.now();
    const { outputs } = loadGraph(builder, doc);
    const graph = await builder.build(outputs), buildMs = performance.now() - t0;
    const inputTensors = {}, outputTensors = {};
    for (const spec of io.inputs) { inputTensors[spec.name] = await ctx.createTensor({ dataType: spec.dataType, shape: spec.shape, writable: true }); ctx.writeTensor(inputTensors[spec.name], blob.slice(spec.offset, spec.offset + spec.bytes)); }
    for (const o of doc.outputs) outputTensors[o.name] = await ctx.createTensor({ dataType: o.dataType, shape: o.shape, readable: true });
    ctx.dispatch(graph, inputTensors, outputTensors);
    const accuracy = [];
    for (const want of io.expected) {
        const o = doc.outputs.find((x) => x.name === want.name), raw = new Uint8Array(count(o.shape) * (o.dataType === "float16" ? 2 : 4));
        await ctx.readTensor(outputTensors[want.name], raw);
        accuracy.push(`${want.name} ${compare(decode(o.dataType, raw), new Float32Array(blob.buffer.slice(want.offset, want.offset + want.bytes)))}`);
    }
    const first = doc.outputs[0], sink = new Uint8Array(count(first.shape) * (first.dataType === "float16" ? 2 : 4));
    const run = async (n) => { const s = performance.now(); for (let i = 0; i < n; i++) ctx.dispatch(graph, inputTensors, outputTensors); await ctx.readTensor(outputTensors[first.name], sink); return performance.now() - s; };
    await run(2);
    const single = [], chained = [];
    for (let i = 0; i < iterations; i++) { single.push(await run(1)); chained.push((await run(chain)) / chain); }
    log(`${name.padEnd(32)} ${String(doc.nodes.length).padStart(4)} nodes, build ${(buildMs / 1000).toFixed(1).padStart(5)} s | single ${median(single).toFixed(2).padStart(7)} ms | chained ${median(chained).toFixed(3).padStart(7)} ms | vs float64 token loop: ${accuracy.join(" | ")}`);
    for (const t of Object.values(inputTensors).concat(Object.values(outputTensors))) t.destroy?.();
    graph.destroy?.();
}

async function main() {
    const index = await (await fetch(`${dir}index.json`)).json();
    const ctx = await navigator.ml.createContext({ deviceType });
    log(`device ${deviceType}, dataset ${dataset}, chain ${chain}; core = 128 tokens x 32 heads (dk = dv = 128), c<N>_b<M> = sub-chunk N tokens, Neumann base block M`);
    for (const name of index.graphs.filter((n) => !only.length || only.some((o) => n.includes(o)))) {
        try { await runCase(ctx, name); } catch (e) { log(`${name.padEnd(32)} FAILED ${e?.message || e}`); }
    }
    log("DONE");
    byId("status").textContent = "done";
}
main().catch((e) => { log(`ERROR ${e?.stack || e}`); byId("status").textContent = "failed"; });

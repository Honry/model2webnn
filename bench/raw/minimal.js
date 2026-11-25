// Minimal comparison: the SAME K-layer graph with its weights as graph INPUTS (MLTensors written once, before timing)
// vs as graph CONSTANTS (absorbed by builder.build). Nothing else: no embedding, no lm_head, no per-step host writes.
// Each iteration = ctx.dispatch(graph, inputs, outputs) + await ctx.readTensor(output)   (the usual inference step)
// URL params: device=gpu|npu  mode=input|const  constScale=1 scaleAfter=1 (NPUW DQ form)  constNorm=1 (diagnostic: keep block scales / RMSNorm gains as constants in input mode)  K=1 (layers in the graph)  T=512  iters=50  warm=5  tag=<label>
import { CFG, buildLayers, weightSpecs, isGraphInput, halfFill } from "./layer.js";

const P = new URLSearchParams(location.search);
const DEVICE = P.get("device") || "gpu", MODE = P.get("mode") || "input", K = +(P.get("K") || 1), T = +(P.get("T") || 512), S = +(P.get("S") || 1);
const SCALE_AFTER = P.get("scaleAfter") === "1";
const CONST_SCALE = P.get("constScale") === "1", CONST_NORM = P.get("constNorm") === "1";
const ITERS = +(P.get("iters") || 50), WARM = +(P.get("warm") || 5), TAG = P.get("tag") || "";
const statusEl = document.getElementById("status"), outEl = document.getElementById("out");
const emit = (o) => {
    const s = JSON.stringify({ exp: "minimal", tag: TAG, device: DEVICE, mode: MODE, constScale: CONST_SCALE, constNorm: CONST_NORM, scaleAfter: SCALE_AFTER, K, T, ...o });
    console.log(s); outEl.textContent += s + "\n"; fetch("/log", { method: "POST", body: s }).catch(() => {});
};
const stats = (a) => { const s = [...a].sort((x, y) => x - y), r = (x) => Math.round(x * 1000) / 1000; return { n: a.length, p50: r(s[Math.floor(s.length / 2)]), min: r(s[0]), max: r(s[s.length - 1]) }; };
const randomBytes = (n) => { const a = new Uint8Array(n); for (let o = 0; o < n; o += 65536) crypto.getRandomValues(a.subarray(o, Math.min(n, o + 65536))); return a; };
const weight = ([name, dataType, , bytes]) => (dataType === "int4" ? randomBytes(bytes) : halfFill(bytes / 2, name.endsWith("_s") ? 0.0008 : 1.0));

async function main() {
    const ctx = await navigator.ml.createContext({ deviceType: DEVICE });
    const mk = (dt, shape, o = {}) => ctx.createTensor({ dataType: dt, shape, writable: !!o.w, readable: !!o.r });
    const specs = weightSpecs("int4", SCALE_AFTER);

    // build
    const builder = new MLGraphBuilder(ctx);
    const { outputs } = buildLayers(builder, { S, T, layers: K, weightMode: MODE, wq: "int4", kv: "shift", constScale: CONST_SCALE, constNorm: CONST_NORM, scaleAfter: SCALE_AFTER, weightProvider: (_l, _n, sp) => weight(sp) });
    const t0 = performance.now();
    const graph = await builder.build(outputs);
    const compileMs = Math.round(performance.now() - t0);

    // bind: create tensors and write everything once (input mode: includes the weights)
    const t1 = performance.now();
    const inputs = {
        hidden: await mk("float16", [1, S, CFG.E], { w: true }),
        cos: await mk("float16", [1, S, 1, CFG.D], { w: true }),
        sin: await mk("float16", [1, S, 1, CFG.D], { w: true }),
        mask: await mk("float16", [1, 1, S, T], { w: true }),
    };
    ctx.writeTensor(inputs.hidden, halfFill(S * CFG.E, 0.1));
    ctx.writeTensor(inputs.cos, halfFill(S * CFG.D, 1));
    ctx.writeTensor(inputs.sin, halfFill(S * CFG.D, 0));
    ctx.writeTensor(inputs.mask, halfFill(S * T, 0));
    const outs = { hidden_out: await mk("float16", [1, S, CFG.E], { r: true }) };
    const kv = [1, CFG.KVH, T, CFG.D];
    for (let l = 0; l < K; l++) {
        for (const k of ["k", "v"]) {
            inputs[`past_${k}.${l}`] = await mk("float16", kv, { w: true });
            ctx.writeTensor(inputs[`past_${k}.${l}`], halfFill(kv.reduce((a, b) => a * b), 0.01));
            outs[`present_${k}.${l}`] = await mk("float16", kv);
        }
        if (MODE === "input") for (const sp of specs.filter((s) => isGraphInput(s[0], MODE, { constScale: CONST_SCALE, constNorm: CONST_NORM }))) {
            const t = await mk(sp[1], sp[2], { w: true });
            ctx.writeTensor(t, weight(sp));
            inputs[`l${l}_${sp[0]}`] = t;
        }
    }
    const buf = new Float16Array(S * CFG.E);
    await ctx.readTensor(outs.hidden_out, buf); // make sure the uploads are done
    const bindMs = Math.round(performance.now() - t1);
    emit({ event: "built", compileMs, bindMs, graphInputs: Object.keys(inputs).length });

    // inference: dispatch + readTensor, same inputs every time
    const call = [], total = [];
    for (let i = 0; i < WARM + ITERS; i++) {
        const a = performance.now();
        console.time("---webnn dispatch ---");
        ctx.dispatch(graph, inputs, outs);
        console.timeEnd("---webnn dispatch ---");
        const b = performance.now();
        console.time("---webnn read ---");
        await ctx.readTensor(outs.hidden_out, buf);
        const c = performance.now();
        console.timeEnd("---webnn read ---");
        if (i >= WARM) { call.push(b - a); total.push(c - a); }
    }
    emit({ event: "result", dispatchCall_ms: stats(call), dispatchPlusRead_ms: stats(total), perLayer_ms: Math.round((stats(total).p50 / K) * 100) / 100, finite: Number.isFinite(buf[0]) });
}
(async () => {
    try { await main(); statusEl.textContent = "done"; } catch (e) { emit({ event: "fatal", err: String(e && e.stack || e).slice(0, 800) }); }
    emit({ event: "done" });
})();

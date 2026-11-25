// RMSNorm-only graphs to check whether the backend fuses the decomposition (and with what precision).
// v=div_const      y = x / sqrt(mean(x^2)+eps) * gamma(const)          <- matches the NPU compiler's FuseRMSNorm
//   div_input      same, gamma is a graph input
//   mulrecip_const y = x * (1 / sqrt(mean(x^2)+eps)) * gamma(const)    <- the form of OpenVINO's RMSFusion
//   div_f32_const  div_const with an explicit cast to float32 around the normalisation (what NPUW's precision option avoids)
// amp=<n>: inputs ~ N(0,1)*amp. amp=1000 makes x^2 overflow fp16 (65504), exposing a non-fused / low precision path.
// Reports max abs error against a float64 reference and dispatch+read latency. With run_bench.ps1 -Dump the OV IR that
// OVEP receives is dumped: look for an RMS op in it.
import { halfFill } from "./layer.js";

const P = new URLSearchParams(location.search);
const DEVICE = P.get("device") || "npu", V = P.get("v") || "div_const", AMP = +(P.get("amp") || 1), E = +(P.get("E") || 2560), S = +(P.get("S") || 1);
const TAG = P.get("tag") || "";
const out = document.getElementById("out"), st = document.getElementById("status");
const emit = (o) => { const s = JSON.stringify({ exp: "rms", tag: TAG, device: DEVICE, v: V, amp: AMP, E, S, ...o }); console.log(s); out.textContent += s + "\n"; fetch("/log", { method: "POST", body: s }).catch(() => {}); };

async function main() {
    const ctx = await navigator.ml.createContext({ deviceType: DEVICE });
    const b = new MLGraphBuilder(ctx);
    const c1 = (v) => b.constant({ dataType: "float16", shape: [1] }, halfFill(1, v));
    const gammaData = Float16Array.from({ length: E }, (_, i) => 0.5 + (i % 7) / 7);
    const x = b.input("x", { dataType: "float16", shape: [1, S, E] });
    const gamma = V.endsWith("_input") ? b.input("gamma", { dataType: "float16", shape: [E] }) : b.constant({ dataType: "float16", shape: [E] }, gammaData);
    const eps = 1e-6;
    let h = V === "div_f32_const" ? b.cast(x, "float32") : x;
    const f = (v) => (V === "div_f32_const" ? b.constant({ dataType: "float32", shape: [1] }, new Float32Array([v])) : c1(v));
    const ms = b.reduceMean(b.pow(h, f(2)), { axes: [2], keepDimensions: true });
    const root = b.sqrt(b.add(ms, f(eps)));
    let y = V === "mulrecip_const" ? b.mul(h, b.div(f(1), root)) : b.div(h, root);
    if (V === "div_f32_const") y = b.cast(y, "float16");
    y = b.mul(y, gamma);
    const t0 = performance.now();
    const graph = await b.build({ y });
    const compileMs = Math.round(performance.now() - t0);

    // inputs: N(0,1) * amp
    const xs = new Float16Array(S * E);
    for (let i = 0; i < xs.length; i++) { let u = 0; for (let k = 0; k < 12; k++) u += Math.random(); xs[i] = (u - 6) * AMP; } // ~N(0,1)
    const xt = await ctx.createTensor({ dataType: "float16", shape: [1, S, E], writable: true });
    const yt = await ctx.createTensor({ dataType: "float16", shape: [1, S, E], readable: true });
    ctx.writeTensor(xt, xs);
    const inputs = { x: xt };
    if (V.endsWith("_input")) {
        const gt = await ctx.createTensor({ dataType: "float16", shape: [E], writable: true });
        ctx.writeTensor(gt, gammaData);
        inputs.gamma = gt;
    }
    const buf = new Float16Array(S * E);
    const times = [];
    for (let i = 0; i < 25; i++) {
        const a = performance.now();
        ctx.dispatch(graph, inputs, { y: yt });
        await ctx.readTensor(yt, buf);
        if (i >= 5) times.push(performance.now() - a);
    }
    // float64 reference on the fp16-rounded input
    let maxErr = 0, nonFinite = 0, maxRef = 0;
    for (let s = 0; s < S; s++) {
        let sum = 0;
        for (let i = 0; i < E; i++) sum += xs[s * E + i] ** 2;
        const inv = 1 / Math.sqrt(sum / E + eps);
        for (let i = 0; i < E; i++) {
            const ref = xs[s * E + i] * inv * gammaData[i], got = buf[s * E + i];
            if (!Number.isFinite(got)) { nonFinite++; continue; }
            maxErr = Math.max(maxErr, Math.abs(got - ref));
            maxRef = Math.max(maxRef, Math.abs(ref));
        }
    }
    times.sort((a, b2) => a - b2);
    emit({ event: "result", compileMs, p50_ms: Math.round(times[times.length >> 1] * 1000) / 1000, maxAbsErr: maxErr, maxAbsRef: maxRef, nonFinite });
}
(async () => {
    try { await main(); st.textContent = "done"; } catch (e) { emit({ event: "fatal", err: String((e && e.stack) || e).slice(0, 600) }); }
    emit({ event: "done" });
})();

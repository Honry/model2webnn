// Raw WebNN (no ort-web) benchmark that mimics the demo pipeline with per-layer(-group) graphs.
//
// One decode/prefill step =
//   host:   embedding rows of the S input tokens -> write the input hidden MLTensor, write the shared per-step inputs
//           (cos/sin, mask, KV one-hot or scatter indices)
//   device: L/K layer-group dispatches (hidden stays in MLTensors between groups, KV is double-buffered and swapped,
//           nothing is read back) + one head dispatch (final norm + lm_head)
//   host:   read back only the logits, argmax, feed the sampled token to the next step
//
// URL params: device=npu|gpu|cpu  mode=input|const  wq=int4|uint4  kv=onehot|shift|scatter
//             headMode=input|const (default = mode)  L=12 (layers)  K=1 (layers per graph)  S=1 (tokens per step)  T=512 (KV capacity)  pos=100 (first position)
//             iters=100 warm=10  tag=<label>   (needs pos + (warm+iters)*S <= T)
// Weights are random (performance only). mode=input: ONE compiled graph per kind (layer group / head) bound to L/K weight
// sets; mode=const: L/K separately compiled graphs with constants. The head is built in the same mode.
// Reported: compileMs (builder.build), bindMs (tensor creation + weight upload = part of the build stage), and per step
// host_ms / enqueue_ms / read_ms (logits wait) / sample_ms; device_ms = enqueue + read; total_ms = the full step.
import { CFG, buildLayers, buildHead, weightSpecs, headSpecs, ropeRows, maskRows, scatterIdx, onehotRows, halfFill } from "./layer.js";

const P = new URLSearchParams(location.search);
const DEVICE = P.get("device") || "npu", MODE = P.get("mode") || "input";
const L = +(P.get("L") || 12), K = +(P.get("K") || 1), S = +(P.get("S") || 1), T = +(P.get("T") || 512), POS0 = +(P.get("pos") || 100);
const WQ = P.get("wq") || "int4", KV = P.get("kv") || "onehot", HEAD_MODE = P.get("headMode") || MODE;
const ITERS = +(P.get("iters") || 100), WARM = +(P.get("warm") || 10), TAG = P.get("tag") || "";
const statusEl = document.getElementById("status"), outEl = document.getElementById("out");

function emit(obj) {
    const s = JSON.stringify({ exp: "raw", tag: TAG, mode: MODE, headMode: HEAD_MODE, wq: WQ, kv: KV, L, K, S, T, device: DEVICE, ...obj });
    console.log(s);
    outEl.textContent += s + "\n";
    fetch("/log", { method: "POST", body: s }).catch(() => {});
}
const setStatus = (s) => (statusEl.textContent = s);
function stats(a) {
    if (!a.length) return {};
    const s = [...a].sort((x, y) => x - y), q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))], r = (x) => Math.round(x * 1000) / 1000;
    return { n: a.length, mean: r(a.reduce((x, y) => x + y, 0) / a.length), p50: r(q(0.5)), p95: r(q(0.95)), min: r(s[0]), max: r(s[s.length - 1]) };
}
function randomBytes(n) {
    const a = new Uint8Array(n);
    for (let o = 0; o < n; o += 65536) crypto.getRandomValues(a.subarray(o, Math.min(n, o + 65536)));
    return a;
}
// random weights: int4/int8 -> random bytes, scales -> small constant, norm gains -> 1
function randomWeight([name, dataType, , bytes]) {
    if (dataType === "int4" || dataType === "uint4" || dataType === "int8") return randomBytes(bytes);
    return halfFill(bytes / 2, name.endsWith("_s") ? 0.0008 : name.endsWith("_w") ? 0.01 : 1.0);
}
const nel = (shape) => shape.reduce((a, b) => a * b, 1);

async function main() {
    if (ITERS > 0 && POS0 + (WARM + ITERS) * S > T) throw new Error(`pos + (warm+iters)*S = ${POS0 + (WARM + ITERS) * S} exceeds T=${T}`);
    const ctx = await navigator.ml.createContext({ deviceType: DEVICE });
    emit({ event: "ctx", ua: navigator.userAgent });
    const specs = weightSpecs(WQ), hspecs = headSpecs();
    const inst = L / K;
    const mkT = (dataType, shape, { writable = false, readable = false } = {}) => ctx.createTensor({ dataType, shape, writable, readable });

    // ---- compile -------------------------------------------------------------------------------------------------
    const compileMs = [];
    const layerGraphs = [];
    const nLayerGraphs = MODE === "input" ? 1 : inst;
    for (let g = 0; g < nLayerGraphs; g++) {
        setStatus(`building layer graph ${g + 1}/${nLayerGraphs}`);
        const builder = new MLGraphBuilder(ctx);
        const { outputs } = buildLayers(builder, { S, T, layers: K, weightMode: MODE, wq: WQ, kv: KV, weightProvider: (_l, _n, spec) => randomWeight(spec) });
        const tb = performance.now();
        layerGraphs.push(await builder.build(outputs));
        compileMs.push(Math.round(performance.now() - tb));
        emit({ event: "compiled", what: "layers", graph: g, compileMs: compileMs[g] });
    }
    setStatus("building head graph");
    const hb = new MLGraphBuilder(ctx);
    const head = buildHead(hb, { S, weightMode: HEAD_MODE, weightProvider: (_n, spec) => randomWeight(spec) });
    const th = performance.now();
    const headGraph = await hb.build(head.outputs);
    const headCompileMs = Math.round(performance.now() - th);
    emit({ event: "compiled", what: "head", compileMs: headCompileMs });

    // ---- bind: create tensors + upload weights/initial KV (build stage, NOT part of the per-step timing) ----------------
    setStatus("creating and uploading tensors");
    const tBind0 = performance.now();
    const shared = {
        cos: await mkT("float16", [1, S, 1, CFG.D], { writable: true }),
        sin: await mkT("float16", [1, S, 1, CFG.D], { writable: true }),
        mask: await mkT("float16", [1, 1, S, T], { writable: true }),
    };
    if (KV === "onehot") {
        shared.kv_keep = await mkT("float16", [1, 1, T, 1], { writable: true });
        shared.kv_onehot = await mkT("float16", [1, 1, T, S], { writable: true });
    }
    if (KV === "scatter") shared.scatter_idx = await mkT("int32", [1, S, CFG.KVH, 3], { writable: true });
    const hbuf = [];
    for (let i = 0; i <= inst; i++) hbuf.push(await mkT("float16", [1, S, CFG.E], { writable: i === 0 }));
    const logitsT = await mkT("float16", [1, 1, CFG.VOCAB], { readable: true });
    const kvShape = [1, CFG.KVH, T, CFG.D];
    // calls[parity][g]: KV is double-buffered per layer; parity p reads buffer p and writes buffer 1-p (swap by alternating)
    const calls = [[], []];
    for (let g = 0; g < inst; g++) {
        const sets = [{ hidden: hbuf[g], ...shared }, { hidden: hbuf[g], ...shared }];
        const outs = [{ hidden_out: hbuf[g + 1] }, { hidden_out: hbuf[g + 1] }];
        for (let l = 0; l < K; l++) {
            for (const kind of ["k", "v"]) {
                const a = await mkT("float16", kvShape, { writable: true }), b = await mkT("float16", kvShape, { writable: true });
                ctx.writeTensor(a, halfFill(nel(kvShape), 0.01));
                ctx.writeTensor(b, halfFill(nel(kvShape), 0.01));
                sets[0][`past_${kind}.${l}`] = a; outs[0][`present_${kind}.${l}`] = b;
                sets[1][`past_${kind}.${l}`] = b; outs[1][`present_${kind}.${l}`] = a;
            }
            if (MODE === "input") {
                for (const sp of specs) {
                    const t = await mkT(sp[1], sp[2], { writable: true });
                    ctx.writeTensor(t, randomWeight(sp));
                    sets[0][`l${l}_${sp[0]}`] = t; sets[1][`l${l}_${sp[0]}`] = t;
                }
            }
        }
        const graph = layerGraphs[MODE === "input" ? 0 : g];
        for (const p of [0, 1]) calls[p].push({ graph, inputs: sets[p], outputs: outs[p] });
        if (g % 6 === 5) setStatus(`tensors for group ${g + 1}/${inst}`);
    }
    const headInputs = { hidden: hbuf[inst] };
    if (HEAD_MODE === "input") for (const sp of hspecs) {
        const t = await mkT(sp[1], sp[2], { writable: true });
        ctx.writeTensor(t, randomWeight(sp));
        headInputs[sp[0]] = t;
    }
    // sync point so the uploads are really done before the clock for the steps starts
    await ctx.readTensor(logitsT, new Float16Array(CFG.VOCAB));
    const bindMs = Math.round(performance.now() - tBind0);
    emit({ event: "bound", bindMs, layerGroups: inst, coldStartMs: compileMs.reduce((a, b) => a + b, 0) + headCompileMs + bindMs });

    // ---- decode loop (embedding on host, logits-only readback, argmax feeds the next token) ----------------------------
    const EMB_ROWS = 2048;
    const emb = new Float16Array(EMB_ROWS * CFG.E);
    for (let i = 0; i < emb.length; i++) emb[i] = (Math.random() - 0.5) * 0.2;
    const logitsBuf = new Float16Array(CFG.VOCAB);
    const hostIn = new Float16Array(S * CFG.E);
    let tokens = Array.from({ length: S }, (_, i) => (7 + i * 13) % EMB_ROWS);
    let pos = POS0;
    const hostMs = [], enq = [], read = [], sample = [], total = [], generated = [];
    for (let it = 0; it < WARM + ITERS && ITERS > 0; it++) {
        const t0 = performance.now();
        for (let s = 0; s < S; s++) hostIn.set(emb.subarray((tokens[s] % EMB_ROWS) * CFG.E, (tokens[s] % EMB_ROWS + 1) * CFG.E), s * CFG.E);
        ctx.writeTensor(hbuf[0], hostIn);
        const r = ropeRows(pos, S);
        ctx.writeTensor(shared.cos, r.cos);
        ctx.writeTensor(shared.sin, r.sin);
        ctx.writeTensor(shared.mask, maskRows(pos, S, T));
        if (KV === "scatter") ctx.writeTensor(shared.scatter_idx, scatterIdx(pos, S));
        if (KV === "onehot") { const o = onehotRows(pos, S, T); ctx.writeTensor(shared.kv_keep, o.keep); ctx.writeTensor(shared.kv_onehot, o.onehot); }
        const t1 = performance.now();
        for (const c of calls[it & 1]) ctx.dispatch(c.graph, c.inputs, c.outputs);
        ctx.dispatch(headGraph, headInputs, { logits: logitsT });
        const t2 = performance.now();
        await ctx.readTensor(logitsT, logitsBuf);
        const t3 = performance.now();
        let best = 0, bv = -Infinity;
        for (let i = 0; i < CFG.VOCAB; i++) { const v = logitsBuf[i]; if (v > bv) { bv = v; best = i; } }
        const t4 = performance.now();
        if (it >= WARM) { hostMs.push(t1 - t0); enq.push(t2 - t1); read.push(t3 - t2); sample.push(t4 - t3); total.push(t4 - t0); generated.push(best); }
        tokens = S === 1 ? [best] : tokens; // prefill-like chunks reuse the token ids
        pos += S;
    }
    const finite = Number.isFinite(logitsBuf[0]) && Number.isFinite(logitsBuf[CFG.VOCAB - 1]);
    emit({
        event: "result", dispatchesPerStep: inst + 1, compileMs, headCompileMs, compileSum: compileMs.reduce((a, b) => a + b, 0) + headCompileMs, bindMs,
        host_ms: stats(hostMs), enqueue_ms: stats(enq), read_ms: stats(read), sample_ms: stats(sample),
        device_ms: stats(enq.map((e, i) => e + read[i])), total_ms: stats(total), finite, tokens: generated.slice(0, 8),
    });
}

(async () => {
    try { await main(); setStatus("done"); }
    catch (e) { emit({ event: "fatal", err: String(e && e.stack ? e.stack : e).slice(0, 1500) }); setStatus("FAILED: " + e); }
    emit({ event: "done" });
})();

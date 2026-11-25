// int4 dequantization as a small *build-time* subgraph (NPUW-style), so the converted f16 weight stays on the device.
//
// What NPUW does (openvino/src/plugins/intel_npu/src/plugin/npuw):
//   * Weights in the OV IR are the standard compressed form  Const(i4 [N, K/B, B]) -> Convert(f16) -> Multiply(scale [N, K/B, 1])
//     -> Reshape [N, K] -> MatMul(x, w, transpose_b)   (partitioning/patterns/compute.cpp: DQMatMulGQi4 = Convert, Multiply,
//     Reshape, MatMul; symmetric int4 => NO zero point; the u4 variant DQMatMulGQu4 adds Subtract(zp) before the Multiply).
//   * For the DCOFF / "unpack" path the same math is done once on the host (util.cpp unpack -> XARCH::unpack_i4f16_scale:
//     out = f16(i4) * scale, written to a host f16 tensor that the weights bank hands to the compiled layer as a Parameter).
//   Here we do the same thing ONCE at build time, but as a tiny WebNN graph, so the f16 result lives in an MLTensor produced
//   by the device and is later bound as the f16 weight input of the layer graph (no host unpack, no 4x host memory).
//
// Variants (URL param v):
//   cast : cast(int4 -> f16) -> mul(scale) -> reshape [N,K]            (exactly NPUW's Convert -> Multiply -> Reshape)
//   dql  : dequantizeLinear(int4, scale, zero-point 0) -> reshape [N,K] (what the demo/ort-web export emits; OV sees
//                                                                        Convert -> Subtract(0) -> Multiply)
// Layout (URL param layout): "nk" keeps NPUW's [N,K] (OV MatMul has transpose_b); "kn" appends transpose so the consumer can
// use WebNN matmul(x, w) directly ([K,N]).
// The scale is a graph CONSTANT by default (scale=const): it is only used by this throw-away prep graph, it does not need to be
// an input. scale=input binds it as an MLTensor instead.
import { CFG, halfFill } from "./layer.js";

const P = new URLSearchParams(location.search);
const DEVICE = P.get("device") || "npu", VAR = P.get("v") || "dql", LAYOUT = P.get("layout") || "kn", SCALE = P.get("scale") || "const";
const PROJ = P.get("proj") || "q", S = +(P.get("S") || 1), ITERS = +(P.get("iters") || 50), WARM = +(P.get("warm") || 5), TAG = P.get("tag") || "";
const DWQ = P.get("wq") || "int4";   // int4 | uint4 (uint4: weights are shifted, zero point 8, like the demo's uint4 export)
const statusEl = document.getElementById("status"), outEl = document.getElementById("out");
function emit(obj) {
    const s = JSON.stringify({ exp: "dq", tag: TAG, v: VAR, layout: LAYOUT, scale: SCALE, proj: PROJ, wq: DWQ, device: DEVICE, ...obj });
    console.log(s); outEl.textContent += s + "\n";
    fetch("/log", { method: "POST", body: s }).catch(() => {});
}
const setStatus = (s) => (statusEl.textContent = s);
const p50 = (a) => { const s = [...a].sort((x, y) => x - y); return Math.round(s[Math.floor(s.length / 2)] * 1000) / 1000; };

// ---- the dequant subgraph ---------------------------------------------------------------------------------------------
// w: int4 [N, kb, B] (input), scale: f16 [N, kb, 1]  ->  f16 [K, N] (layout kn) or [N, K] (layout nk)
export function buildDequant(builder, { K, N, block = CFG.BLOCK, variant = "dql", layout = "kn", scaleMode = "const", scaleData, wq = "int4" }) {
    const kb = K / block;
    const w = builder.input("w", { dataType: wq, shape: [N, kb, block] });
    const s = scaleMode === "input"
        ? builder.input("scale", { dataType: "float16", shape: [N, kb, 1] })
        : builder.constant({ dataType: "float16", shape: [N, kb, 1] }, scaleData);
    let dq;
    if (variant === "cast") {
        const wf = builder.cast(w, "float16");                       // OV Convert(i4 -> f16)
        dq = wq === "uint4"
            ? builder.mul(builder.sub(wf, builder.constant({ dataType: "float16", shape: [1] }, halfFill(1, 8))), s) // u4: Subtract(zp) -> Multiply
            : builder.mul(wf, s);                                    // OV Multiply(scale)  (symmetric i4: no zero point)
    } else {
        const zp = builder.constant({ dataType: wq, shape: [N, kb, 1] }, new Uint8Array(Math.ceil((N * kb) / 2)).fill(wq === "uint4" ? 0x88 : 0));
        dq = builder.dequantizeLinear(w, s, zp);
    }
    let out = builder.reshape(dq, [N, K]);                           // OV Reshape [N, K/B, B] -> [N, K]
    if (layout === "kn") out = builder.transpose(out, { permutation: [1, 0] });
    return { inputs: { w }, outputs: { wf16: out } };
}

// host reference: nibble i of the packed buffer (low nibble first) -> signed int4 -> * scale -> [K, N] (always [K,N], caller maps)
function nibble(buf, i, unsigned) { const b = buf[i >> 1], v = i & 1 ? b >> 4 : b & 15; return unsigned ? v - 8 : v >= 8 ? v - 16 : v; }

async function main() {
    const { E, NH, KVH, D, FF } = CFG;
    const [K, N] = ({ q: [E, NH * D], k: [E, KVH * D], o: [NH * D, E], gate: [E, FF], down: [FF, E] })[PROJ] || [E, NH * D];
    const kb = K / CFG.BLOCK, B = CFG.BLOCK;
    const ctx = await navigator.ml.createContext({ deviceType: DEVICE });
    emit({ event: "ctx", K, N, ua: navigator.userAgent });

    // probe=<name>: only try to build a minimal variant of the graph (which op combination does the NPU compiler reject?)
    const probe = P.get("probe");
    if (probe) {
        const pb = new MLGraphBuilder(ctx), sc = pb.constant({ dataType: "float16", shape: [N, kb, 1] }, new Float16Array(N * kb).fill(0.001));
        let o;
        if (probe === "f16mul") o = pb.reshape(pb.mul(pb.input("w", { dataType: "float16", shape: [N, kb, B] }), sc), [N, K]);       // no int4
        else if (probe === "i4dql") o = pb.dequantizeLinear(pb.input("w", { dataType: DWQ, shape: [N, kb, B] }), sc, pb.constant({ dataType: DWQ, shape: [N, kb, 1] }, new Uint8Array(Math.ceil((N * kb) / 2))));  // no reshape
        else if (probe === "i8cast") o = pb.reshape(pb.mul(pb.cast(pb.input("w", { dataType: "int8", shape: [N, kb, B] }), "float16"), sc), [N, K]);   // unpacked int8 weight, no matmul
        else if (probe === "i4idmm") { // int4 -> dequant -> matmul with an identity input: out = I @ W = the f16 weight, produced by a matmul consumer
            const wq = pb.reshape(pb.dequantizeLinear(pb.input("w", { dataType: DWQ, shape: [N, kb, B] }), sc, pb.constant({ dataType: DWQ, shape: [N, kb, 1] }, new Uint8Array(Math.ceil((N * kb) / 2)))), [N, K]);
            o = pb.matmul(pb.input("eye", { dataType: "float16", shape: [K, K] }), pb.transpose(wq, { permutation: [1, 0] }));
        }
        else if (probe === "i4mm") { // known-good pattern: int4 input -> dequant -> matmul
            const wq = pb.reshape(pb.dequantizeLinear(pb.input("w", { dataType: DWQ, shape: [N, kb, B] }), sc, pb.constant({ dataType: DWQ, shape: [N, kb, 1] }, new Uint8Array(Math.ceil((N * kb) / 2)))), [N, K]);
            o = pb.matmul(pb.input("x", { dataType: "float16", shape: [1, 1, K] }), pb.transpose(wq, { permutation: [1, 0] }));
        }
        try { await pb.build({ o }); emit({ event: "probe", probe, ok: true }); setStatus(`probe ${probe}: BUILD OK`); } catch (e) { emit({ event: "probe", probe, ok: false, err: String(e).slice(0, 200) }); setStatus(`probe ${probe}: BUILD FAILED (${e})`); throw e; }
        return;
    }
    // random weights + scales
    const packed = new Uint8Array((N * K) / 2); for (let o = 0; o < packed.length; o += 65536) crypto.getRandomValues(packed.subarray(o, Math.min(packed.length, o + 65536)));
    const scaleData = new Float16Array(N * kb); for (let i = 0; i < scaleData.length; i++) scaleData[i] = 0.0005 + 0.001 * Math.random();

    // ---- prep graph: compile, run once on the device, keep the output as an MLTensor ------------------------------------
    setStatus("building dequant graph");
    const b = new MLGraphBuilder(ctx);
    const { outputs } = buildDequant(b, { K, N, variant: VAR, layout: LAYOUT, scaleMode: SCALE, scaleData, wq: DWQ });
    const tb = performance.now();
    const prep = await b.build(outputs);
    const compileMs = Math.round(performance.now() - tb);
    const shape = LAYOUT === "kn" ? [K, N] : [N, K];
    const wIn = await ctx.createTensor({ dataType: DWQ, shape: [N, kb, B], writable: true });
    const wDev = await ctx.createTensor({ dataType: "float16", shape, readable: true });
    ctx.writeTensor(wIn, packed);
    const inputs = { w: wIn };
    if (SCALE === "input") { inputs.scale = await ctx.createTensor({ dataType: "float16", shape: [N, kb, 1], writable: true }); ctx.writeTensor(inputs.scale, scaleData); }
    const got = new Float16Array(N * K);
    const times = [];
    for (let i = 0; i < 1 + Math.min(ITERS, 5); i++) {
        const t0 = performance.now();
        ctx.dispatch(prep, inputs, { wf16: wDev });
        await ctx.readTensor(wDev, got); // full readback as the sync point (one-time build cost; includes the f16 readback)
        if (i > 0) times.push(performance.now() - t0);
    }
    emit({ event: "prep", compileMs, dispatchPlusReadback_ms: times.length ? p50(times) : null });

    // ---- correctness vs host reference ------------------------------------------------------------------------------------
    let maxErr = 0, nBad = 0;
    for (let n = 0; n < N; n += 7) for (let k = 0; k < K; k += 5) {
        const ref = nibble(packed, n * K + k, DWQ === "uint4") * scaleData[n * kb + Math.floor(k / B)];
        const g = LAYOUT === "kn" ? got[k * N + n] : got[n * K + k];
        const e = Math.abs(g - ref); if (!(e <= 1e-3 * Math.max(1, Math.abs(ref)) + 1e-4)) nBad++; if (e > maxErr) maxErr = e;
    }
    emit({ event: "check", maxAbsErr: maxErr, nBad, sample: [got[0], got[1], got[N]] });

    // ---- consumer: y = x @ w (f16 weight as INPUT) — device-produced weight tensor vs host-written weight tensor ---------
    if (LAYOUT === "kn") {
        setStatus("consumer graph");
        const cb = new MLGraphBuilder(ctx);
        const x = cb.input("x", { dataType: "float16", shape: [1, S, K] }), w = cb.input("w", { dataType: "float16", shape: [K, N] });
        const tc = performance.now();
        const cons = await cb.build({ y: cb.matmul(x, w) });
        const cCompile = Math.round(performance.now() - tc);
        const xT = await ctx.createTensor({ dataType: "float16", shape: [1, S, K], writable: true }); ctx.writeTensor(xT, halfFill(S * K, 0.01));
        const yT = await ctx.createTensor({ dataType: "float16", shape: [1, S, N], readable: true });
        const wHost = await ctx.createTensor({ dataType: "float16", shape: [K, N], writable: true }); ctx.writeTensor(wHost, got);
        const yBuf = new Float16Array(S * N);
        const run = async (wt) => {
            const ts = [];
            for (let i = 0; i < WARM + ITERS; i++) {
                const t0 = performance.now();
                ctx.dispatch(cons, { x: xT, w: wt }, { y: yT });
                await ctx.readTensor(yT, yBuf);
                if (i >= WARM) ts.push(performance.now() - t0);
            }
            return p50(ts);
        };
        const host = await run(wHost), dev = await run(wDev), host2 = await run(wHost);
        emit({ event: "consumer", consumerCompileMs: cCompile, hostWritten_ms: host, deviceProduced_ms: dev, hostWritten_again_ms: host2, S });
    }
}
(async () => {
    try { await main(); setStatus("done"); }
    catch (e) { emit({ event: "fatal", err: String(e && e.stack ? e.stack : e).slice(0, 1500) }); setStatus("FAILED: " + e); }
    emit({ event: "done" });
})();

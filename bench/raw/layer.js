// Hand-written WebNN graph for Qwen3-4B decoder layers (pure WebNN JS API, no ort-web).
//
// Design (NPUW-like: the host owns positions):
//   inputs : hidden [1,S,E]  cos/sin [1,S,1,D]  mask [1,1,S,T] (additive)  scatter_idx [1,S,KVH,3] (int32)
//            past_k.i / past_v.i [1,KVH,T,D]
//   outputs: hidden_out [1,S,E]  present_k.i / present_v.i [1,KVH,T,D]   (fixed-capacity KV, scatterND at the position)
//   weights: int4 block-128 for q/k/v/o/gate/up/down (+ f16 scales), f16 norm weights; as graph *inputs* (weightMode
//            "input", bound once to MLTensors) or as graph *constants* (weightMode "const").
// The attention follows the static GQA decomposition of the WebNN EP: scatterND into the past buffer, expand KV heads,
// q*k^T*scale + mask, softmax, *v.

export const CFG = { E: 2560, NH: 32, KVH: 8, D: 128, FF: 9728, BLOCK: 128, EPS: 1e-6, VOCAB: 151936 };

export const halfFill = (n, v) => new Float16Array(n).fill(v);

// [name, K (in), N (out)] of the seven projections of one layer
export function projections() {
    const { E, NH, KVH, D, FF } = CFG;
    return [["q", E, NH * D], ["k", E, KVH * D], ["v", E, KVH * D], ["o", NH * D, E], ["gate", E, FF], ["up", E, FF], ["down", FF, E]];
}
// Weight tensor specs of one layer: [name, dataType, shape, byteLength]
export function weightSpecs(wq = "int4", scaleAfter = false) {
    const { BLOCK, E, D } = CFG;
    const specs = [];
    for (const [n, K, N] of projections()) {
        const kb = K / BLOCK;
        if (wq === "fp16") {
            specs.push([`${n}_w`, "float16", [K, N], K * N * 2]); // plain f16 weight, no dequantize (control experiment)
        } else if (scaleAfter) {
            // NPUW "DQ" form: weights pre-permuted on the host to [G, BLOCK, N], scales as [G, 1, N]
            specs.push([`${n}_w`, wq, [kb, BLOCK, N], (N * K) / 2]);
            specs.push([`${n}_s`, "float16", [kb, 1, N], N * kb * 2]);
        } else {
            specs.push([`${n}_w`, wq, [N, kb, BLOCK], (N * K) / 2]);
            specs.push([`${n}_s`, "float16", [N, kb, 1], N * kb * 2]);
        }
    }
    for (const [n, dim] of [["ln1", E], ["ln2", E], ["qn", D], ["kn", D]]) specs.push([`${n}_g`, "float16", [dim], dim * 2]);
    return specs;
}

// Build one graph with `layers` consecutive decoder layers.
//   weightProvider(layerIdx, name, spec) -> typed array (only used for weightMode "const")
// Which weights are graph inputs? weightMode "input" makes everything an input, except the pieces that can be kept as
// constants for diagnosis: constScale (block scales) and constNorm (RMSNorm gains). NOTE: none of the float tensors is the
// same in every layer of the real model (weights, scales, norm gains all differ), so keeping them constant forces one
// compiled graph per layer. Only the zero point (symmetric quantization) and the structural constants are shared.
export function isGraphInput(name, weightMode, { constScale = false, constNorm = false } = {}) {
    if (weightMode !== "input") return false;
    if (constScale && name.endsWith("_s")) return false;
    if (constNorm && name.endsWith("_g")) return false;
    return true;
}
export function buildLayers(builder, { S, T, layers, weightMode, weightProvider, wq = "int4", kv = "scatter", constScale = false, constNorm = false, scaleAfter = false }) {
    const { E, NH, KVH, D, FF, BLOCK, EPS } = CFG;
    const inputs = {}, outputs = {};
    const inp = (name, dataType, shape) => (inputs[name] = builder.input(name, { dataType, shape }));
    const cst = (dataType, shape, data) => builder.constant({ dataType, shape }, data);
    const scalar = (v) => cst("float16", [1], halfFill(1, v));
    const epsC = scalar(EPS), two = scalar(2), scaleC = scalar(1 / Math.sqrt(D));

    let h = inp("hidden", "float16", [1, S, E]);
    const cos = inp("cos", "float16", [1, S, 1, D]);
    const sin = inp("sin", "float16", [1, S, 1, D]);
    const mask = inp("mask", "float16", [1, 1, S, T]);
    const sidx = kv === "scatter" ? inp("scatter_idx", "int32", [1, S, KVH, 3]) : null;
    // "onehot": position-indexed update without scatterND: present = past * keep + onehot @ new
    const keep = kv === "onehot" ? inp("kv_keep", "float16", [1, 1, T, 1]) : null;
    const onehot = kv === "onehot" ? inp("kv_onehot", "float16", [1, 1, T, S]) : null;

    const weight = (layer, spec) => {
        const [name, dataType, shape] = spec;
        const key = `l${layer}_${name}`;
        return isGraphInput(name, weightMode, { constScale, constNorm })
            ? inp(key, dataType, shape)
            : cst(dataType, shape, weightProvider(layer, name, spec));
    };
    const zpCache = new Map();
    const zeroPoint = (N, kb) => {
        const k = `${N}x${kb}`;
        if (!zpCache.has(k)) zpCache.set(k, cst(wq, [N, kb, 1], new Uint8Array(Math.ceil((N * kb) / 2)).fill(wq === "uint4" ? 0x88 : 0)));
        return zpCache.get(k);
    };
    // scaleAfter (NPUW "DQ"/GQi): the block scale is NOT applied to the dequantized weight (1e8 elements), but to the
    // per-group matmul outputs (small):  y = sum_g ( x_g @ int4(W_g) ) * scale_g.  Weights are [G,BLOCK,N], scales [G,1,N].
    let unitScale = null, unitZero = null;
    const linear = (x, w, s, K, N) => {
        if (wq === "fp16") return builder.matmul(x, w);
        const kb = K / BLOCK;
        if (scaleAfter) {
            unitScale = unitScale || cst("float16", [1, 1, 1], halfFill(1, 1));
            unitZero = unitZero || cst(wq, [1, 1, 1], new Uint8Array(1));
            const wf = builder.dequantizeLinear(w, unitScale, unitZero); // int4 -> f16 only, [G,BLOCK,N]
            const rows = S;
            const xg = builder.transpose(builder.reshape(x, [rows, kb, BLOCK]), { permutation: [1, 0, 2] }); // [G,rows,BLOCK]
            const part = builder.mul(builder.matmul(xg, wf), s); // [G,rows,N] * [G,1,N]
            return builder.reshape(builder.reduceSum(part, { axes: [0], keepDimensions: false }), [1, rows, N]);
        }
        const dq = builder.dequantizeLinear(w, s, zeroPoint(N, kb)); // [N,kb,BLOCK] f16
        const wt = builder.transpose(builder.reshape(dq, [N, K]), { permutation: [1, 0] }); // [K,N]
        return builder.matmul(x, wt);
    };
    // ================================================================================================================
    // !!! RMSNorm: a CROSS-BACKEND PROBLEM, keep this in mind for every WebNN backend !!!
    // WebNN has no RMSNorm operator, so it is expressed as x / sqrt(mean(x^2) + eps) * gamma. Whether that runs fast AND
    // accurately depends entirely on the backend recognizing the decomposition and fusing it:
    //  * Intel NPU compiler: pass FuseRMSNorm (npu_compiler/.../fuse_rms_norm.cpp) matches
    //      Power|Multiply -> ReduceMean/ReduceSum -> Add(eps) -> Sqrt -> Divide(x, .) -> Multiply(CONSTANT gamma)
    //    and turns it into one IE.RMS op. So: keep the exact Divide(x, sqrt) form below, and keep gamma a CONSTANT
    //    (NPUW deliberately keeps these constants for the same reason). With gamma as a graph input only the Divide part
    //    fuses.
    //  * Precision: x^2 overflows fp16 when |x| > 256 (LLM hidden states do have such outliers). NPUW passes
    //    compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm to the compiler for this. WebNN cannot set
    //    that option, and an explicit fp32 cast is NOT equivalent (the compiler narrows fp32 to fp16 unless told
    //    otherwise, and a backend that forbids CPU fallback would fail to compile unsupported types).
    //  => other backends need either their own fusion of this pattern, or a WebNN RMSNorm op / a recognised precision hint.
    // ================================================================================================================
    const rms = (x, g, axisLast) => {
        const ms = builder.reduceMean(builder.pow(x, two), { axes: [axisLast], keepDimensions: true });
        return builder.mul(builder.div(x, builder.sqrt(builder.add(ms, epsC))), g);
    };
    const rope = (x, nh) => {
        // x: [1,S,nh,D]
        const x1 = builder.slice(x, [0, 0, 0, 0], [1, S, nh, D / 2]);
        const x2 = builder.slice(x, [0, 0, 0, D / 2], [1, S, nh, D / 2]);
        const rot = builder.concat([builder.neg(x2), x1], 3);
        return builder.add(builder.mul(x, cos), builder.mul(rot, sin));
    };
    const specs = weightSpecs(wq, scaleAfter);
    const rep = NH / KVH;
    for (let l = 0; l < layers; l++) {
        const W = {};
        for (const sp of specs) W[sp[0]] = weight(l, sp);
        const pk = inp(`past_k.${l}`, "float16", [1, KVH, T, D]);
        const pv = inp(`past_v.${l}`, "float16", [1, KVH, T, D]);

        const x = rms(h, W.ln1_g, 2);
        let q = builder.reshape(linear(x, W.q_w, W.q_s, E, NH * D), [1, S, NH, D]);
        let k = builder.reshape(linear(x, W.k_w, W.k_s, E, KVH * D), [1, S, KVH, D]);
        const v = builder.reshape(linear(x, W.v_w, W.v_s, E, KVH * D), [1, S, KVH, D]);
        q = rope(rms(q, W.qn_g, 3), NH);
        k = rope(rms(k, W.kn_g, 3), KVH);

        let presentK, presentV;
        if (kv === "scatter") {
            presentK = builder.scatterND(pk, sidx, k);
            presentV = builder.scatterND(pv, sidx, v);
        } else if (kv === "onehot") {
            const bnsd = (t) => builder.transpose(t, { permutation: [0, 2, 1, 3] });
            const upd = (past, nw) => builder.add(builder.mul(past, keep), builder.matmul(onehot, bnsd(nw)));
            presentK = upd(pk, k);
            presentV = upd(pv, v);
        } else {
            // "shift": drop the oldest S entries and append the new ones (fixed capacity, no scatter)
            const tail = (t) => builder.slice(t, [0, 0, S, 0], [1, KVH, T - S, D]);
            const bnsd = (t) => builder.transpose(t, { permutation: [0, 2, 1, 3] });
            presentK = builder.concat([tail(pk), bnsd(k)], 2);
            presentV = builder.concat([tail(pv), bnsd(v)], 2);
        }
        outputs[`present_k.${l}`] = presentK;
        outputs[`present_v.${l}`] = presentV;

        const expand = (t) =>
            builder.reshape(builder.expand(builder.reshape(t, [1, KVH, 1, T, D]), [1, KVH, rep, T, D]), [1, NH, T, D]);
        const kt = builder.transpose(expand(presentK), { permutation: [0, 1, 3, 2] }); // [1,NH,D,T]
        const qh = builder.transpose(q, { permutation: [0, 2, 1, 3] }); // [1,NH,S,D]
        let sc = builder.add(builder.mul(builder.matmul(qh, kt), scaleC), mask);
        const p = builder.softmax(sc, 3);
        const o = builder.matmul(p, expand(presentV)); // [1,NH,S,D]
        const o2 = builder.reshape(builder.transpose(o, { permutation: [0, 2, 1, 3] }), [1, S, NH * D]);
        h = builder.add(h, linear(o2, W.o_w, W.o_s, NH * D, E));

        const y = rms(h, W.ln2_g, 2);
        const gate = linear(y, W.gate_w, W.gate_s, E, FF);
        const up = linear(y, W.up_w, W.up_s, E, FF);
        const act = builder.mul(builder.mul(gate, builder.sigmoid(gate)), up);
        h = builder.add(h, linear(act, W.down_w, W.down_s, FF, E));
    }
    outputs["hidden_out"] = h;
    return { inputs, outputs };
}

// Host-side per-step inputs ------------------------------------------------------------------------------------------
// RoPE tables (HF style: cos/sin of pos*inv_freq, the same value in both halves of D)
export function ropeRows(startPos, S, theta = 1e6) {
    const { D } = CFG;
    const cos = new Float32Array(S * D), sin = new Float32Array(S * D);
    for (let s = 0; s < S; s++)
        for (let i = 0; i < D / 2; i++) {
            const a = (startPos + s) * Math.pow(theta, (-2 * i) / D);
            cos[s * D + i] = cos[s * D + i + D / 2] = Math.cos(a);
            sin[s * D + i] = sin[s * D + i + D / 2] = Math.sin(a);
        }
    return { cos: Float16Array.from(cos), sin: Float16Array.from(sin) };
}
// additive causal mask [1,1,S,T] for S new tokens written at startPos..startPos+S-1
export function maskRows(startPos, S, T) {
    const m = new Float16Array(S * T);
    for (let s = 0; s < S; s++) for (let j = 0; j < T; j++) m[s * T + j] = j <= startPos + s ? 0 : -65504;
    return m;
}
// scatterND indices [1,S,KVH,3] = (batch 0, head h, position startPos+s)
export function scatterIdx(startPos, S) {
    const { KVH } = CFG;
    const a = new Int32Array(S * KVH * 3);
    let o = 0;
    for (let s = 0; s < S; s++) for (let h = 0; h < KVH; h++) { a[o++] = 0; a[o++] = h; a[o++] = startPos + s; }
    return a;
}

// onehot KV update inputs for S new tokens written at startPos..startPos+S-1
export function onehotRows(startPos, S, T) {
    const keep = halfFill(T, 1), onehot = new Float16Array(T * S);
    for (let s = 0; s < S; s++) { keep[startPos + s] = 0; onehot[(startPos + s) * S + s] = 1; }
    return { keep, onehot };
}

// Output head: final RMSNorm (+ optional last-row slice) + lm_head (per-channel int8 DequantizeLinear + MatMul, as in the
// ov-perchannel Qwen3 export). weightMode "input": norm gain / head weight / head scale are graph inputs.
export function headSpecs() {
    const { E, VOCAB } = CFG;
    return [["lnf_g", "float16", [E], E * 2], ["head_w", "int8", [E, VOCAB], E * VOCAB], ["head_s", "float16", [1, VOCAB], VOCAB * 2]];
}
export function buildHead(builder, { S, weightMode, weightProvider }) {
    const { E, VOCAB, EPS } = CFG;
    const inputs = {};
    const inp = (name, dataType, shape) => (inputs[name] = builder.input(name, { dataType, shape }));
    const cst = (dataType, shape, data) => builder.constant({ dataType, shape }, data);
    const w = (spec) => {
        const [name, dataType, shape] = spec;
        return weightMode === "input" ? inp(name, dataType, shape) : cst(dataType, shape, weightProvider(name, spec));
    };
    const [gS, wS, sS] = headSpecs();
    const h = inp("hidden", "float16", [1, S, E]);
    const g = w(gS), hw = w(wS), hs = w(sS);
    const zp = cst("int8", [1, VOCAB], new Int8Array(VOCAB));
    const ms = builder.reduceMean(builder.pow(h, cst("float16", [1], halfFill(1, 2))), { axes: [2], keepDimensions: true });
    let x = builder.mul(builder.div(h, builder.sqrt(builder.add(ms, cst("float16", [1], halfFill(1, EPS))))), g);
    if (S > 1) x = builder.slice(x, [0, S - 1, 0], [1, 1, E]); // logits of the last position only
    const logits = builder.matmul(x, builder.dequantizeLinear(hw, hs, zp)); // [1,1,VOCAB]
    return { inputs, outputs: { logits } };
}

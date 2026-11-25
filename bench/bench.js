// NPU graph-splitting benchmark (ort-web + WebNN, static shapes).
//
// URL params
//   exp=m0|chain|compile        which experiment (see below)
//   device=npu|gpu|cpu          WebNN deviceType (default npu)
//   ort=cdn|<url>               ORT-web script (default: jsdelivr dev build used by the demo)
//   iters=100 warm=10           timed / warm-up iterations
//   mode=await|nowait           chain: await every run (default) | fire all runs back-to-back (timing only)
//   opt=all|basic|disabled      ORT graphOptimizationLevel (default all)
//
// exp=m0     smoke test: tiny graph, 1-layer block (dq_const, matmulnbits, dq_input) -> does it compile/run on the NPU?
// exp=chain  L=36 S=1,3,12 q=1 t=512 w=dq_const   split L layers into S sequential graphs, time the whole chain
//            names=a,b,c                          or an explicit list of model names (chained in order)
// exp=compile names=a,b,c mode=seq|par           cold createSession time per model (bytes are prefetched, not timed)
//
// Every result line is also POSTed to /log (server.py) -> results/<run>.jsonl.  A final {"event":"done"} line is emitted.

const P = new URLSearchParams(location.search);
const EXP = P.get("exp") || "m0";
const DEVICE = P.get("device") || "npu";
const DEV_ORT = "1.32.0-dev.20261102-d460f01a9";
const ITERS = +(P.get("iters") || 100);
const WARM = +(P.get("warm") || 10);
const MODE = P.get("mode") || "await";
const statusEl = document.getElementById("status");
const outEl = document.getElementById("out");

function emit(obj) {
    const s = JSON.stringify({ exp: EXP, ...obj });
    console.log(s);
    outEl.textContent += s + "\n";
    fetch("/log", { method: "POST", body: s }).catch(() => {});
}
const setStatus = s => (statusEl.textContent = s);
const sleep = ms => new Promise(r => setTimeout(r, ms));

function stats(a) {
    if (!a.length) return {};
    const s = [...a].sort((x, y) => x - y);
    const q = p => s[Math.min(s.length - 1, Math.floor(p * s.length))];
    const r = x => Math.round(x * 1000) / 1000;
    return { n: a.length, mean: r(a.reduce((x, y) => x + y, 0) / a.length), p50: r(q(0.5)), p95: r(q(0.95)), min: r(s[0]), max: r(s[s.length - 1]) };
}

async function loadOrt() {
    let url = P.get("ort") || "cdn";
    if (url === "cdn") url = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${DEV_ORT}/dist/ort.webgpu.min.js`;
    await new Promise((res, rej) => {
        const s = document.createElement("script");
        s.src = url;
        s.crossOrigin = "anonymous";
        s.onload = res;
        s.onerror = () => rej(new Error("failed to load " + url));
        document.head.appendChild(s);
    });
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.logLevel = P.get("verbose") ? "verbose" : "warning";
    emit({ event: "ort", url, version: ort.env.versions?.web ?? null });
}

// ---- fp16 helpers -----------------------------------------------------------------------------------------------
function toHalfBits(v) {
    const f = new Float32Array([v]);
    const x = new Uint32Array(f.buffer)[0];
    const sign = (x >>> 16) & 0x8000;
    let e = ((x >>> 23) & 0xff) - 127 + 15;
    const m = x & 0x7fffff;
    if (v === 0) return sign;
    if (e <= 0) return sign;
    if (e >= 31) return sign | 0x7c00;
    return sign | (e << 10) | (m >>> 13);
}
function halfFill(n, v) {
    const a = new Uint16Array(n);
    a.fill(toHalfBits(v));
    return a;
}
function halfToFloat(h) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    if (e === 0) return s * Math.pow(2, -14) * (m / 1024);
    if (e === 31) return m ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + m / 1024);
}

// ---- WebNN / ORT helpers ----------------------------------------------------------------------------------------
const ORT_TYPE = { FLOAT16: "float16", INT4: "int4", UINT4: "uint4", FLOAT: "float32", INT64: "int64" };
const numel = dims => dims.reduce((a, b) => a * b, 1);

async function mkTensor(ctx, onnxType, dims, { readable = false, fill } = {}) {
    const dataType = ORT_TYPE[onnxType];
    const ml = await ctx.createTensor({ dataType, shape: dims, writable: true, readable });
    const t = ort.Tensor.fromMLTensor(ml, { dataType, dims });
    if (fill !== undefined) {
        if (onnxType === "FLOAT16") ctx.writeTensor(ml, halfFill(numel(dims), fill));
        else if (onnxType === "INT4" || onnxType === "UINT4") {
            const b = new Uint8Array(Math.ceil(numel(dims) / 2));
            for (let i = 0; i < b.length; i++) b[i] = (Math.random() * 256) | 0;
            ctx.writeTensor(ml, b);
        } else throw new Error("fill not implemented for " + onnxType);
    }
    return { ml, t, dims, onnxType };
}

async function fetchBuf(url, optional = false) {
    let r;
    try {
        r = await fetch(url);
    } catch (e) {
        if (optional) return null;
        throw e;
    }
    if (!r.ok) {
        if (optional) return null;
        throw new Error(`${url}: HTTP ${r.status}`);
    }
    return new Uint8Array(await r.arrayBuffer());
}

async function loadFiles(name) {
    const meta = await (await fetch(`models/${name}.json`)).json();
    const [model, data] = await Promise.all([fetchBuf(`models/${meta.file}`), fetchBuf(`models/${meta.data}`, true)]);
    return { name, meta, model, data };
}

async function createSession(ctx, files) {
    const opts = {
        executionProviders: [{ name: "webnn", deviceType: DEVICE, context: ctx }],
        graphOptimizationLevel: P.get("opt") === "disabled" ? "disabled" : P.get("opt") || "all",
    };
    if (files.data) opts.externalData = [{ path: files.meta.data, data: files.data }];
    const t0 = performance.now();
    const session = await ort.InferenceSession.create(files.model, opts);
    return { session, ms: performance.now() - t0 };
}

// Build one chained segment: feeds/fetches for a block model. `chainIn`/`chainOut` are the hidden-state tensors.
async function buildSegment(ctx, files, shared, chainIn, chainOut) {
    const { session, ms } = await createSession(ctx, files);
    const feeds = {}, fetches = {}, own = [];
    for (const [name, type, dims] of files.meta.inputs) {
        if (name === "hidden") feeds[name] = chainIn.t;
        else {
            const key = /^(cos|sin|mask)$/.test(name) ? name + dims.join("x") : null;
            let tn = key && shared.get(key);
            if (!tn) {
                const fill = name === "cos" ? 0.5 : name === "sin" ? 0.25 : name === "mask" ? 0 : name.startsWith("past_") ? 0.01 : 0.0012;
                tn = await mkTensor(ctx, type, dims, { fill });
                if (key) shared.set(key, tn);
                else own.push(tn);
            }
            feeds[name] = tn.t;
        }
    }
    for (const [name, type, dims] of files.meta.outputs) {
        if (name === "hidden_out") fetches[name] = chainOut.t;
        else {
            const tn = await mkTensor(ctx, type, dims);
            fetches[name] = tn.t;
            own.push(tn);
        }
    }
    return { name: files.name, session, compileMs: ms, feeds, fetches, own };
}

async function disposeSegments(segs, shared, bufs) {
    for (const s of segs) {
        try { await s.session.release(); } catch (e) {}
        for (const t of s.own) try { t.ml.destroy(); } catch (e) {}
    }
    for (const t of shared.values()) try { t.ml.destroy(); } catch (e) {}
    for (const t of bufs) try { t.ml.destroy(); } catch (e) {}
}

// Run names[] as a chain; returns timing summary.
async function benchChain(ctx, names, label, extra = {}) {
    const files = [];
    const tFetch0 = performance.now();
    for (const n of names) files.push(await loadFiles(n));
    const fetchMs = performance.now() - tFetch0;
    const hiddenDims = files[0].meta.inputs.find(i => i[0] === "hidden")[2];
    const bufs = [];
    for (let i = 0; i <= names.length; i++) bufs.push(await mkTensor(ctx, "FLOAT16", hiddenDims, { readable: i === names.length, fill: i === 0 ? 0.1 : undefined }));
    const shared = new Map();
    const segs = [];
    const t0 = performance.now();
    for (let i = 0; i < names.length; i++) {
        setStatus(`${label}: compiling ${names[i]} (${i + 1}/${names.length})`);
        segs.push(await buildSegment(ctx, files[i], shared, bufs[i], bufs[i + 1]));
        emit({ event: "compiled", label, model: names[i], ms: Math.round(segs[i].compileMs) });
        files[i].model = files[i].data = null;
    }
    const compileWall = performance.now() - t0;
    const readBuf = new Uint16Array(numel(hiddenDims));
    const tRuns = [], tTotal = [], tRead = [], perRun = segs.map(() => []);
    for (let it = 0; it < WARM + ITERS; it++) {
        const a = performance.now();
        if (MODE === "nowait") {
            await Promise.all(segs.map(s => s.session.run(s.feeds, s.fetches)));
        } else {
            for (let j = 0; j < segs.length; j++) {
                const b = performance.now();
                await segs[j].session.run(segs[j].feeds, segs[j].fetches);
                if (it >= WARM) perRun[j].push(performance.now() - b);
            }
        }
        const b = performance.now();
        await ctx.readTensor(bufs[names.length].ml, readBuf);
        const c = performance.now();
        if (it >= WARM) { tRuns.push(b - a); tRead.push(c - b); tTotal.push(c - a); }
        if (it === 0) emit({ event: "first-run", label, ms: Math.round(c - a), out0: halfToFloat(readBuf[0]) });
    }
    const res = {
        event: "chain", label, mode: MODE, segments: names.length, names, ...extra,
        fetchMs: Math.round(fetchMs), compileMsSum: Math.round(segs.reduce((x, s) => x + s.compileMs, 0)), compileWallMs: Math.round(compileWall),
        runs_ms: stats(tRuns), read_ms: stats(tRead), total_ms: stats(tTotal),
        perRun_p50_ms: perRun.map(a => stats(a).p50),
        finite: Number.isFinite(halfToFloat(readBuf[0])),
    };
    emit(res);
    await disposeSegments(segs, shared, bufs);
    return res;
}

const blkName = (w, a, b, q, t) => `blk_${w}_L${a}-${b}_q${q}_t${t}`;
function splitNames(L, S, q, t, w) {
    const names = [];
    const per = L / S;
    for (let i = 0; i < S; i++) names.push(blkName(w, i * per, (i + 1) * per, q, t));
    return names;
}

// ---- experiments ------------------------------------------------------------------------------------------------
async function step(name, fn) {
    setStatus("m0: " + name);
    const t0 = performance.now();
    try {
        const r = await fn();
        emit({ event: "step", step: name, ok: true, ms: Math.round(performance.now() - t0), ...r });
    } catch (e) {
        emit({ event: "step", step: name, ok: false, ms: Math.round(performance.now() - t0), err: String(e && e.message ? e.message : e).slice(0, 600) });
    }
}

async function expM0(ctx) {
    await step("tiny", async () => {
        const f = await loadFiles("tiny");
        const inT = await mkTensor(ctx, "FLOAT16", [1, 1, 2560], { fill: 0.1 });
        const outT = await mkTensor(ctx, "FLOAT16", [1, 1, 2560], { readable: true });
        const { session, ms } = await createSession(ctx, f);
        await session.run({ hidden: inT.t }, { hidden_out: outT.t });
        const buf = new Uint16Array(2560);
        await ctx.readTensor(outT.ml, buf);
        await session.release();
        return { compileMs: Math.round(ms), out0: halfToFloat(buf[0]), expect: 1.1 };
    });
    const W = ["dq_const", "matmulnbits", "dq_input"];
    for (const w of W) {
        const name = blkName(w, 0, 1, 1, 512);
        await step(`block1:${w}`, async () => {
            const r = await benchChain(ctx, [name], `m0:${w}`, { w });
            return { compileMs: r.compileMsSum, total_p50: r.total_ms.p50, finite: r.finite };
        });
    }
}

async function expChain(ctx) {
    const q = +(P.get("q") || 1), t = +(P.get("t") || 512), w = P.get("w") || "dq_const", L = +(P.get("L") || 36);
    if (P.get("groups")) {
        // groups=a,b;c;d,e  -> run each ';'-separated group as its own chain (fresh sessions each time)
        for (const g of P.get("groups").split(";")) {
            const names = g.split(",");
            await benchChain(ctx, names, "group:" + names.join("+"), { q, t, w });
            await sleep(1000);
        }
        return;
    }
    if (P.get("names")) {
        const names = P.get("names").split(",");
        await benchChain(ctx, names, "chain:" + names.length, { q, t, w });
        return;
    }
    for (const S of (P.get("S") || "1").split(",").map(Number)) {
        await benchChain(ctx, splitNames(L, S, q, t, w), `L${L}xS${S}`, { L, S, q, t, w });
        await sleep(1000);
    }
}

async function expCompile(ctx) {
    const names = (P.get("names") || "").split(",").filter(Boolean);
    const par = (P.get("mode") || "seq") === "par";
    const files = [];
    for (const n of names) files.push(await loadFiles(n));
    const t0 = performance.now();
    const sessions = [];
    const one = async f => {
        setStatus("compile " + f.name);
        const r = await createSession(ctx, f);
        sessions.push(r.session);
        f.model = f.data = null; // don't let the harness' own copies inflate the memory numbers
        emit({ event: "compiled", model: f.name, ms: Math.round(r.ms), par });
        return r.ms;
    };
    const ms = par ? await Promise.all(files.map(one)) : (await (async () => { const o = []; for (const f of files) o.push(await one(f)); return o; })());
    emit({ event: "compile-total", par, names, perModelMs: ms.map(Math.round), wallMs: Math.round(performance.now() - t0) });
    const hold = +(P.get("hold") || 0);
    if (hold) { emit({ event: "hold-start", s: hold }); await sleep(hold * 1000); emit({ event: "hold-end" }); }
    for (const s of sessions) await s.release();
}

// ---- exp=qwen3: the real Qwen3-4B (ORT-GenAI GQA export) made static with freeDimensionOverrides --------------------
// params: T=512 (KV capacity)  C=64 (prefill chunk)  n=192 (prompt tokens, needs qwen3_prompt_<n>.json; n % C == 0)
//         gen=16 (greedy tokens after the prompt)   paths=ref,chunk (ref = token-by-token reference; chunk = chunked prefill)
const QL = 36, QKVH = 8, QD = 128, QVOCAB = 151936;
const kvName = i => `${i >> 1}.${i & 1 ? "value" : "key"}`;

async function qwenSession(ctx, model, data, S, T) {
    const opts = {
        executionProviders: [{ name: "webnn", deviceType: DEVICE, context: ctx }],
        externalData: [{ path: "model.onnx.data", data }],
        freeDimensionOverrides: { batch_size: 1, sequence_length: S, past_sequence_length: T, total_sequence_length: T, kv_cache_dim: QD },
        graphOptimizationLevel: P.get("opt") || "all",
    };
    if (P.get("verbose")) { opts.logSeverityLevel = 0; opts.logVerbosityLevel = 1; }
    const t0 = performance.now();
    const session = await ort.InferenceSession.create(model, opts);
    return { session, ms: performance.now() - t0 };
}

async function newKv(ctx, T) {
    const past = [], present = [];
    for (let i = 0; i < QL * 2; i++) {
        past.push(await mkTensor(ctx, "FLOAT16", [1, QKVH, T, QD], { fill: 0 }));
        present.push(await mkTensor(ctx, "FLOAT16", [1, QKVH, T, QD], { fill: 0 }));
    }
    return { past, present };
}
const freeKv = kv => { for (const t of [...kv.past, ...kv.present]) try { t.ml.destroy(); } catch (e) {} };

// Same input handling as the demo's WebNN decode path: input_ids / attention_mask are pre-allocated writable
// MLTensors (rewritten every step), position_ids is a small CPU tensor.
const qwenIo = {};
async function qwenIoFor(ctx, S, T) {
    if (!qwenIo[S]) qwenIo[S] = { ids: await mkTensor(ctx, "INT64", [1, S]), mask: await mkTensor(ctx, "INT64", [1, T]) };
    return qwenIo[S];
}

async function qwenStep(ctx, session, kv, logits, ids, start, T) {
    const S = ids.length;
    const io = await qwenIoFor(ctx, S, T);
    const mask = new BigInt64Array(T);
    mask.fill(1n, 0, start + S);
    ctx.writeTensor(io.mask.ml, mask);
    ctx.writeTensor(io.ids.ml, BigInt64Array.from(ids, v => BigInt(v)));
    const feeds = {
        input_ids: io.ids.t,
        attention_mask: io.mask.t,
        position_ids: new ort.Tensor("int64", BigInt64Array.from({ length: S }, (_, i) => BigInt(start + i)), [1, S]),
    };
    const fetches = { logits: logits.t };
    for (let i = 0; i < QL * 2; i++) {
        const n = kvName(i);
        feeds[`past_key_values.${n}`] = kv.past[i].t;
        fetches[`present.${n}`] = kv.present[i].t;
    }
    await session.run(feeds, fetches);
    [kv.past, kv.present] = [kv.present, kv.past]; // double-buffered KV, as in the demo
}

async function readLogits(ctx, logits, buf) {
    await ctx.readTensor(logits.ml, buf);
    const f = typeof Float16Array !== "undefined" ? new Float16Array(buf.buffer) : null;
    const out = new Float32Array(QVOCAB);
    let bi = 0, best = -Infinity;
    for (let i = 0; i < QVOCAB; i++) {
        const v = f ? f[i] : halfToFloat(buf[i]);
        out[i] = v;
        if (v > best) { best = v; bi = i; }
    }
    return { logits: out, argmax: bi };
}

async function expQwen3(ctx) {
    const T = +(P.get("T") || 512), C = +(P.get("C") || 64), N = +(P.get("n") || 192), GEN = +(P.get("gen") || 16);
    const paths = (P.get("paths") || "ref,chunk").split(",");
    const ids = await (await fetch(`qwen3_prompt_${N}.json`)).json();
    if (ids.length !== N || N % C) throw new Error(`prompt has ${ids.length} tokens; need n=${N} divisible by C=${C}`);
    const tf = performance.now();
    const [model, data] = await Promise.all([fetchBuf("/qwen3/model.onnx"), fetchBuf("/qwen3/model.onnx.data")]);
    emit({ event: "qwen3-fetched", ms: Math.round(performance.now() - tf), modelMB: Math.round(model.length / 1e6), dataMB: Math.round(data.length / 1e6) });

    setStatus("compiling decode (S=1)");
    const dec = await qwenSession(ctx, model, data, 1, T);
    emit({ event: "qwen3-compiled", which: "decode", S: 1, T, ms: Math.round(dec.ms) });
    let pre = null;
    if (paths.includes("chunk")) {
        setStatus(`compiling prefill (S=${C})`);
        try {
            pre = await qwenSession(ctx, model, data, C, T);
            emit({ event: "qwen3-compiled", which: "prefill", S: C, T, ms: Math.round(pre.ms) });
        } catch (e) {
            emit({ event: "qwen3-compiled", which: "prefill", S: C, T, ok: false, err: String(e.message || e).slice(0, 800) });
        }
    }
    const hold = +(P.get("hold") || 0);
    if (hold) { emit({ event: "hold-start", s: hold }); await sleep(hold * 1000); emit({ event: "hold-end" }); }

    const logits = await mkTensor(ctx, "FLOAT16", [1, 1, QVOCAB], { readable: true });
    const buf = new Uint16Array(QVOCAB);
    const result = {};

    const generate = async (label, kv, first, startPos) => {
        const toks = [first.argmax];
        const t = [];
        let pos = startPos;
        for (let g = 1; g < GEN && pos < T; g++) {
            const a = performance.now();
            await qwenStep(ctx, dec.session, kv, logits, [toks[g - 1]], pos, T);
            const b = performance.now();
            const r = await readLogits(ctx, logits, buf);
            const c = performance.now();
            toks.push(r.argmax);
            pos++;
            t.push({ run: b - a, read: c - b, total: c - a });
        }
        emit({ event: "qwen3-decode", label, tokens: toks, stepMs: stats(t.map(x => x.total)), runMs: stats(t.map(x => x.run)), readMs: stats(t.map(x => x.read)) });
        return toks;
    };

    if (paths.includes("ref")) {
        setStatus("reference: token-by-token prefill");
        const kv = await newKv(ctx, T);
        const times = [];
        let last;
        for (let i = 0; i < N; i++) {
            const a = performance.now();
            await qwenStep(ctx, dec.session, kv, logits, [ids[i]], i, T);
            if (i === N - 1) last = await readLogits(ctx, logits, buf);
            else await ctx.readTensor(logits.ml, buf); // keep the sync pattern of a real decode step
            times.push(performance.now() - a);
        }
        emit({ event: "qwen3-prefill", label: "ref-token-by-token", n: N, totalMs: Math.round(times.reduce((x, y) => x + y, 0)), stepMs: stats(times), argmax: last.argmax });
        result.ref = last;
        result.refTokens = await generate("ref", kv, last, N);
        freeKv(kv);
    }

    if (pre) {
        setStatus("chunked prefill");
        const kv = await newKv(ctx, T);
        const times = [];
        let last;
        const t0 = performance.now();
        for (let c = 0; c < N / C; c++) {
            const a = performance.now();
            await qwenStep(ctx, pre.session, kv, logits, ids.slice(c * C, (c + 1) * C), c * C, T);
            if (c === N / C - 1) last = await readLogits(ctx, logits, buf);
            else await ctx.readTensor(logits.ml, buf);
            times.push(performance.now() - a);
        }
        emit({ event: "qwen3-prefill", label: `chunk${C}`, n: N, chunks: N / C, ttftMs: Math.round(performance.now() - t0), chunkMs: stats(times), firstChunkMs: Math.round(times[0]), argmax: last.argmax });
        result.chunk = last;
        result.chunkTokens = await generate("chunk", kv, last, N);
        freeKv(kv);
    }

    if (result.ref && result.chunk) {
        const a = result.ref.logits, b = result.chunk.logits;
        let maxd = 0, sumd = 0;
        for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > maxd) maxd = d; sumd += d; }
        const top = x => Array.from(x.keys()).sort((i, j) => x[j] - x[i]).slice(0, 5);
        const ta = top(a), tb = top(b);
        let match = 0;
        const m = Math.min(result.refTokens.length, result.chunkTokens.length);
        for (let i = 0; i < m; i++) if (result.refTokens[i] === result.chunkTokens[i]) match++;
        emit({
            event: "qwen3-compare", maxAbsDiff: maxd, meanAbsDiff: sumd / a.length, argmaxEqual: result.ref.argmax === result.chunk.argmax,
            top5Ref: ta, top5Chunk: tb, top5Overlap: ta.filter(x => tb.includes(x)).length, genTokensMatch: `${match}/${m}`,
        });
    }
    try { await dec.session.release(); } catch (e) {}
    if (pre) try { await pre.session.release(); } catch (e) {}
}

// ---- exp=qwen3rem: prompts that are not a multiple of the chunk size ----------------------------------------------
// The exported lm_head is pruned to the last position of the run, so a chunk must be full (no padding) and the
// remainder has to go through smaller runs. Compare, against the token-by-token reference:
//   A: chunks of C1, remainder with the decode graph (S=1)
//   B: chunks of C1, then chunks of C2, remainder with the decode graph
// params: ns=70,127,200,300 (needs qwen3_prompt_<n>.json)  C1=64  C2=8  T=512  gen=8  hold=0
async function expQwen3Rem(ctx) {
    const T = +(P.get("T") || 512), C1 = +(P.get("C1") || 64), C2 = +(P.get("C2") || 8), GEN = +(P.get("gen") || 8);
    const ns = (P.get("ns") || "70,127,200,300").split(",").map(Number);
    const [model, data] = await Promise.all([fetchBuf("/qwen3/model.onnx"), fetchBuf("/qwen3/model.onnx.data")]);
    const sess = {};
    for (const S of [1, C1, C2]) {
        setStatus(`compiling S=${S}`);
        sess[S] = await qwenSession(ctx, model, data, S, T);
        emit({ event: "qwen3-compiled", S, T, ms: Math.round(sess[S].ms) });
    }
    const hold = +(P.get("hold") || 0);
    if (hold) { emit({ event: "hold-start", s: hold }); await sleep(hold * 1000); emit({ event: "hold-end" }); }
    const logits = await mkTensor(ctx, "FLOAT16", [1, 1, QVOCAB], { readable: true });
    const buf = new Uint16Array(QVOCAB);

    // Feed ids[from..n) with the given plan; returns the logits after the last token and the timing.
    const prefill = async (kv, ids, plan) => {
        const n = ids.length;
        let pos = 0, steps = 0;
        const t0 = performance.now();
        for (const C of plan) {
            while (n - pos >= C && (C > 1 || n - pos > 0)) {
                await qwenStep(ctx, sess[C].session, kv, logits, ids.slice(pos, pos + C), pos, T);
                pos += C; steps++;
            }
        }
        const r = await readLogits(ctx, logits, buf);
        return { ...r, ms: performance.now() - t0, steps };
    };
    const generate = async (kv, first, start) => {
        const toks = [first.argmax];
        let pos = start;
        for (let g = 1; g < GEN && pos < T; g++) {
            await qwenStep(ctx, sess[1].session, kv, logits, [toks[g - 1]], pos, T);
            toks.push((await readLogits(ctx, logits, buf)).argmax);
            pos++;
        }
        return toks;
    };

    for (const n of ns) {
        const ids = await (await fetch(`qwen3_prompt_${n}.json`)).json();
        const plans = { ref: [1], A: [C1, 1], B: [C1, C2, 1] };
        const res = {};
        for (const name of ["ref", "A", "B"]) {
            setStatus(`n=${n} ${name}`);
            const kv = await newKv(ctx, T);
            const r = await prefill(kv, ids, plans[name]);
            r.tokens = await generate(kv, r, n);
            freeKv(kv);
            res[name] = r;
            emit({ event: "qwen3rem-prefill", n, plan: name, steps: r.steps, ttftMs: Math.round(r.ms), argmax: r.argmax });
        }
        for (const name of ["A", "B"]) {
            const a = res.ref.logits, b = res[name].logits;
            let maxd = 0, sumd = 0;
            for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > maxd) maxd = d; sumd += d; }
            let match = 0;
            for (let i = 0; i < GEN; i++) if (res.ref.tokens[i] === res[name].tokens[i]) match++;
            emit({ event: "qwen3rem-compare", n, plan: name, maxAbsDiff: maxd, meanAbsDiff: sumd / a.length, argmaxEqual: res.ref.argmax === res[name].argmax, genTokensMatch: `${match}/${GEN}`, ttftMs: Math.round(res[name].ms), refMs: Math.round(res.ref.ms) });
        }
    }
    for (const S of Object.keys(sess)) try { await sess[S].session.release(); } catch (e) {}
}

window.addEventListener("unhandledrejection", e => {
    const r = e.reason;
    emit({ event: "unhandled", err: String((r && r.stack) || r).slice(0, 1500) });
});

(async () => {
    try {
        setStatus("loading ORT…");
        await loadOrt();
        if (!navigator.ml) throw new Error("navigator.ml undefined (missing --enable-features=WebMachineLearningNeuralNetwork?)");
        const ctx = await navigator.ml.createContext({ deviceType: DEVICE });
        emit({ event: "ctx", deviceType: DEVICE, ua: navigator.userAgent });
        if (EXP === "m0") await expM0(ctx);
        else if (EXP === "chain") await expChain(ctx);
        else if (EXP === "compile") await expCompile(ctx);
        else if (EXP === "qwen3") await expQwen3(ctx);
        else if (EXP === "qwen3rem") await expQwen3Rem(ctx);
        else throw new Error("unknown exp " + EXP);
        setStatus("done");
    } catch (e) {
        emit({ event: "fatal", err: String(e && e.stack ? e.stack : e).slice(0, 1000) });
        setStatus("FAILED: " + e);
    }
    emit({ event: "done" });
})();

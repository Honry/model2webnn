// Numerical equivalence check of the decode attention half on the chosen device (use the NPU: graph forms must be verified where they run):
//   reference : fused one-hot update, attention over concat(cache, new token)                          (kvStyle=onehot, the form that is known to work)
//   candidate : kvStyle=staged (default) | broadcast | select, optionally attendInCache=1               (any form passed via the URL)
// Both are built from the same random weights / activations and the same logical KV content, then attention_output is compared.
// Cases (staged): A = empty staging buffer (filled=0), B = staging buffer holding `filled` rows after the main cache rows [0, base).
// URL: device=npu  P=511 (main cache rows)  R=64 (staging rows)  base=20  filled=3  kvStyle=staged  attendInCache=0  tag=
import { buildHeadGraph, buildBlockGraph, attentionWeightSpecs, feedForwardWeightSpecs, byteLengthOf } from "./qwen3_subgraphs.js?v=20261124a";
import { ropeRows } from "./layer.js?v=20261119x";

const P = new URLSearchParams(location.search);
const DEVICE = P.get("device") || "npu", PAST = +(P.get("P") || 511), ROWS = +(P.get("R") || 64), BASE = +(P.get("base") || 20), FILLED = +(P.get("filled") || 3);
const STYLE = P.get("kvStyle") || "staged", ATTEND_IN_CACHE = P.get("attendInCache") === "1", TAG = P.get("tag") || "";
const HEADS = 8, DIM = 128;
// graph=block: test the BLOCK graph (feed-forward half of layer i + attention half of layer i+1, what chat.html dispatches 35 times) instead of the attention-only head graph
const GRAPH = P.get("graph") || "head", buildGraphFor = (builder, settings) => (GRAPH === "block" ? buildBlockGraph : buildHeadGraph)(builder, settings);
// ref=incache: the REFERENCE is also "attention reads the updated cache" (one-hot update, total length P), so candidate and reference differ only in the update op (scatterND vs one-hot)
// cands=<comma list>: candidate forms tried one after the other in ONE page run (default: incache = the form chat.html uses). All use kvStyle=scatterND unless noted.
const PRESETS = {
    incache: {},                                                   // attention reads the scatterND result (chat.html's form)
    guard_allValues: { attentionGuard: "allValues" }, guard_values: { attentionGuard: "values" }, guard_scores: { attentionGuard: "scores" },
    guard_context: { attentionGuard: "context" }, guard_probabilities: { attentionGuard: "probabilities" }, guard_allKeys: { attentionGuard: "allKeys" },
    concat: { scatterAttention: "concat" },                        // attention reads concat(past, new); scatterND only writes the cache
    concat_guard_values: { scatterAttention: "concat", attentionGuard: "values" },
    grouped: { gqaStyle: "grouped" }, float16_attention: { attentionType: "float16" },
    natural_noguard: { kvStyle: "onehot", valueLayout: "natural", attentionGuard: null },            // prefill chunk: natural V layout, NO attention guard (chat.html's prefill graphs use the guard)
    natural_guard: { kvStyle: "onehot", valueLayout: "natural" },
    onehot_incache: { kvStyle: "onehot", attendInCache: true, valueLayout: "natural" },   // not scatter: one-hot update, attention reads it
};
const CANDIDATES = (P.get("cands") || "incache").split(",");
const isScatterOptions = (options) => options.kvStyle === "scatterND";
const REFERENCE_OPTIONS = P.get("ref") === "incache" ? { attendInCache: true } : {};
const outputNameOfAttention = GRAPH === "block" ? "next_attention_output" : "attention_output";
// S>1: a PREFILL CHUNK of S rows at cache position `start` (chat.html's prefill graph form: attention guard "values", left-padded last chunk: pads=<comma list of pad counts>)
const CHUNK = +(P.get("S") || 1), START = +(P.get("start") || 64), PADS = (P.get("pads") || "0,44").split(",").map(Number);
const statusEl = document.getElementById("status"), outEl = document.getElementById("out");
function emit(obj) {
    const s = JSON.stringify({ exp: "kvcheck", tag: TAG, device: DEVICE, style: STYLE, ...obj });
    console.log(s); outEl.textContent += s + "\n";
    fetch("/log", { method: "POST", body: s }).catch(() => {});
}
let seed = 12345; const rand = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; };
const randomHalf = (count, amplitude) => { const a = new Float16Array(count); for (let i = 0; i < count; i++) a[i] = (rand() - 0.5) * 2 * amplitude; return a; };

async function main() {
    const context = await navigator.ml.createContext({ deviceType: DEVICE });
    const weightData = {};
    for (const spec of [...attentionWeightSpecs(), ...(GRAPH === "block" ? feedForwardWeightSpecs() : [])]) {
        const [name, dataType, shape] = spec, count = shape.reduce((a, b) => a * b, 1);
        weightData[name] = dataType === "int4" ? Uint8Array.from({ length: Math.ceil(count / 2) }, () => (rand() * 256) | 0)
            : name.endsWith("_scale") ? Float16Array.from({ length: count }, () => 0.01 + 0.01 * rand()) : new Float16Array(count).fill(1);
    }
    if (CHUNK > 1) return chunkCheck(context, weightData);
    const hidden = randomHalf(2560, 1), blockAttentionOutput = randomHalf(4096, 1);   // (the same data for the reference and the candidate)
    // logical KV content: positions [0, BASE) in the main cache; for case B also positions [BASE, BASE+FILLED) in the staging buffer
    const keyRows = randomHalf(HEADS * (BASE + FILLED) * DIM, 1), valueRows = randomHalf(HEADS * (BASE + FILLED) * DIM, 1);   // [head, position, dim]
    const at = (h, pos, d, total) => (h * total + pos) * DIM + d;
    const total = BASE + FILLED;
    const makeKeyValue = (capacity, firstPosition, count, natural = false) => {   // key [1,8,capacity,128], value transposed [1,8,128,capacity] (natural: [1,8,capacity,128])
        const key = new Float16Array(HEADS * capacity * DIM), value = new Float16Array(HEADS * DIM * capacity);
        for (let h = 0; h < HEADS; h++) for (let i = 0; i < count; i++) for (let d = 0; d < DIM; d++) {
            key[(h * capacity + i) * DIM + d] = keyRows[at(h, firstPosition + i, d, total)];
            value[natural ? (h * capacity + i) * DIM + d : (h * DIM + d) * capacity + i] = valueRows[at(h, firstPosition + i, d, total)];
        }
        return { key, value };
    };
    const tensorFor = async (name, dataType, shape, data, readable) => {
        const tensor = await context.createTensor({ dataType, shape, writable: !readable, readable: !!readable });
        if (data) context.writeTensor(tensor, data);
        return tensor;
    };

    async function run(label, options, { caseFilled }) {
        const builder = new MLGraphBuilder(context);
        const description = buildGraphFor(builder, { sequenceLength: 1, pastLength: PAST, maskDataType: "float16", options: { fuseKvUpdate: true, ...options } });
        const t0 = performance.now(), graph = await builder.build(description.outputs), buildMs = Math.round(performance.now() - t0);
        const staged = options.kvStyle === "staged", scatter = options.kvStyle === "scatterND", inCache = options.attendInCache === true || (scatter && options.scatterAttention !== "concat");
        // cache content: staged -> main = positions [0,BASE), staging = positions [BASE, BASE+filled); others -> one cache with all positions [0, total)
        const logicalTotal = BASE + caseFilled;   // number of cached positions before the new token
        const rope = ropeRows(logicalTotal, 1), cos = Float16Array.from(rope.cos), sin = Float16Array.from(rope.sin);
        const natural = options.valueLayout === "natural" || scatter;
        const main = staged ? makeKeyValue(PAST, 0, BASE, natural) : makeKeyValue(PAST, 0, logicalTotal, natural);
        const staging = staged ? makeKeyValue(ROWS, BASE, caseFilled, natural) : null;
        const maskLength = staged ? PAST + ROWS : inCache ? PAST : PAST + 1;
        const mask = new Float16Array(maskLength).fill(-65504);
        if (staged) { mask.fill(0, 0, BASE); mask.fill(0, PAST, PAST + caseFilled + 1); }
        else if (inCache) mask.fill(0, 0, logicalTotal + 1);
        else { mask.fill(0, 0, logicalTotal); mask[PAST] = 0; }
        const keep = new Float16Array(staged ? ROWS : PAST).fill(1), selector = new Float16Array(staged ? ROWS : PAST);
        const newRow = staged ? caseFilled : logicalTotal;
        keep[newRow] = 0; selector[newRow] = 1;
        const blockInputs = GRAPH === "block" ? { residual: hidden, attention_output: blockAttentionOutput } : { hidden_states: hidden };
        const data = { ...blockInputs, rope_cos: cos, rope_sin: sin, attention_mask: mask, past_key: main.key, past_value: main.value,
                       ...(staged ? { staging_key: staging.key, staging_value: staging.value, staging_rows_to_keep: keep, staging_row_selector: selector }
                                  : scatter ? { kv_indices: Int32Array.from({ length: HEADS * 3 }, (_, i) => (i % 3 === 0 ? 0 : i % 3 === 1 ? Math.floor(i / 3) : newRow)) }   // (0, head, row) of the new token
                                  : { rows_to_keep: keep, row_selector: selector }), ...weightData };
        const inputs = {};
        for (const [name, operand] of Object.entries(description.inputs)) {
            if (!(name in data)) throw new Error(`no data for input ${name}`);
            inputs[name] = await tensorFor(name, operand.dataType, operand.shape, data[name]);
        }
        const outputs = {};
        for (const [name, operand] of Object.entries(description.outputs)) outputs[name] = await tensorFor(name, operand.dataType, operand.shape, null, true);
        context.dispatch(graph, inputs, outputs);
        const attention = new Float16Array(4096); await context.readTensor(outputs[outputNameOfAttention], attention);
        const updatedKey = new Float16Array(HEADS * (staged ? ROWS : PAST) * DIM); await context.readTensor(outputs.updated_key, updatedKey);
        emit({ event: "built", label, buildMs, inputs: Object.keys(description.inputs).length });
        return { attention, updatedKey, scatter };
    }
    const compare = (a, b) => {
        let maxAbs = 0, maxRef = 0, nan = 0, sum = 0;
        for (let i = 0; i < a.length; i++) { if (!Number.isFinite(a[i])) nan++; maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i])); maxRef = Math.max(maxRef, Math.abs(b[i])); sum += a[i]; }
        return { maxAbsDiff: maxAbs, maxAbsReference: maxRef, nonFinite: nan, sumCandidate: sum };
    };
    const cases = STYLE === "staged" ? [0, FILLED] : [FILLED];
    for (const caseFilled of cases) {
        const ref = await run(`reference one-hot (${BASE + caseFilled} cached positions)`, { ...REFERENCE_OPTIONS }, { caseFilled });
        for (const name of CANDIDATES) {
            try {
                const options = { kvStyle: STYLE, attendInCache: ATTEND_IN_CACHE ? true : undefined, stagingRows: ROWS, ...(P.get("valueLayout") || STYLE === "scatterND" ? { valueLayout: P.get("valueLayout") || "natural" } : {}), ...(P.get("guard") ? { attentionGuard: P.get("guard") } : {}), ...PRESETS[name] };
                const candidate = await run(`candidate ${name} filled=${caseFilled}`, options, { caseFilled });
                const keyCompare = candidate.scatter ? { updatedKey: compare(candidate.updatedKey, ref.updatedKey) } : {};   // scatterND: the whole updated K cache must equal the one-hot result
                const c = compare(candidate.attention, ref.attention);
                emit({ event: "compare", candidate: name, verdict: c.nonFinite === 0 && c.maxAbsDiff < 0.05 ? "OK" : "BAD", filled: caseFilled, ...c, ...keyCompare, candidateFirst: [...candidate.attention.slice(0, 4)], referenceFirst: [...ref.attention.slice(0, 4)] });
            } catch (error) { emit({ event: "candidate_failed", candidate: name, err: String(error).slice(0, 200) }); }
        }
    }
}
async function chunkCheck(context, weightData) {
    const tensorFor = async (dataType, shape, data, readable) => {
        const tensor = await context.createTensor({ dataType, shape, writable: !readable, readable: !!readable });
        if (data) context.writeTensor(tensor, data);
        return tensor;
    };
    for (const pad of PADS) {
        const valid = CHUNK - pad, total = START + valid;
        const keyAll = randomHalf(HEADS * total * DIM, 1), valueAll = randomHalf(HEADS * total * DIM, 1);   // logical content [head, position, dim] of positions [0,total)
        const hiddenStates = randomHalf(CHUNK * 2560, 1), chunkAttentionOutput = randomHalf(CHUNK * 4096, 1);
        const rope = ropeRows(START, valid), cos = new Float16Array(CHUNK * DIM), sin = new Float16Array(CHUNK * DIM);
        for (let row = 0; row < CHUNK; row++) { const index = Math.max(0, row - pad); cos.set(rope.cos.subarray(index * DIM, (index + 1) * DIM), row * DIM); sin.set(rope.sin.subarray(index * DIM, (index + 1) * DIM), row * DIM); }
        const runVariant = async (label, options, scatter) => {
            const builder = new MLGraphBuilder(context);
            const description = buildGraphFor(builder, { sequenceLength: CHUNK, pastLength: PAST, maskDataType: "float16", options: { fuseKvUpdate: true, attentionGuard: "values", ...options } });
            const t0 = performance.now(), graph = await builder.build(description.outputs), buildMs = Math.round(performance.now() - t0);
            const natural = scatter || options.valueLayout === "natural";
            const key = new Float16Array(HEADS * PAST * DIM), value = new Float16Array(HEADS * PAST * DIM);   // the cache before this chunk: positions [0, START)
            for (let h = 0; h < HEADS; h++) for (let pos = 0; pos < START; pos++) for (let d = 0; d < DIM; d++) {
                key[(h * PAST + pos) * DIM + d] = keyAll[(h * total + pos) * DIM + d];
                value[natural ? (h * PAST + pos) * DIM + d : (h * DIM + d) * PAST + pos] = valueAll[(h * total + pos) * DIM + d];
            }
            // the chunk's own new K / V rows are NOT given: the graph computes them from hidden_states (same random weights in both variants)
            const inCache = (scatter && options.scatterAttention !== "concat") || options.attendInCache === true;
            const width = inCache ? PAST : PAST + CHUNK, mask = new Float16Array(CHUNK * width).fill(-65504);
            for (let row = 0; row < CHUNK; row++) {   // identical to chat.js writePrefillChunkInputs
                const base = row * width;
                if (inCache) { if (row < pad) mask[base + (scatter ? Math.min(PAST - 1, START + valid + row) : 0)] = 0; else mask.fill(0, base, base + START + (row - pad) + 1); continue; }
                if (row < pad) { mask[base + PAST + row] = 0; continue; }
                mask.fill(0, base, base + START); mask.fill(0, base + PAST + pad, base + PAST + row + 1);
            }
            const extra = {};
            if (scatter) {
                const indices = new Int32Array(HEADS * CHUNK * 3);
                for (let h = 0; h < HEADS; h++) for (let row = 0; row < CHUNK; row++) { const at = (h * CHUNK + row) * 3; indices[at] = 0; indices[at + 1] = h; indices[at + 2] = row < pad ? Math.min(PAST - 1, START + valid + row) : START + (row - pad); }
                extra.kv_indices = indices;
            } else {
                const keep = new Float16Array(PAST).fill(1), selector = new Float16Array(PAST * CHUNK);
                for (let i = 0; i < valid; i++) { keep[START + i] = 0; selector[(START + i) * CHUNK + pad + i] = 1; }
                extra.rows_to_keep = keep; extra.row_selector = selector;
            }
            const data = { ...(GRAPH === "block" ? { residual: hiddenStates, attention_output: chunkAttentionOutput } : { hidden_states: hiddenStates }), rope_cos: cos, rope_sin: sin, attention_mask: mask, past_key: key, past_value: value, ...extra, ...weightData };
            const inputs = {}, outputs = {};
            for (const [name, operand] of Object.entries(description.inputs)) { if (!(name in data)) throw new Error(`no data for input ${name}`); inputs[name] = await tensorFor(operand.dataType, operand.shape, data[name]); }
            for (const [name, operand] of Object.entries(description.outputs)) outputs[name] = await tensorFor(operand.dataType, operand.shape, null, true);
            context.dispatch(graph, inputs, outputs);
            const attention = new Float16Array(CHUNK * 4096); await context.readTensor(outputs[outputNameOfAttention], attention);
            const updatedKey = new Float16Array(HEADS * PAST * DIM); await context.readTensor(outputs.updated_key, updatedKey);
            emit({ event: "built", label, buildMs, chunk: CHUNK, pad });
            return { attention, updatedKey };
        };
        const reference = await runVariant("reference one-hot chunk", { ...REFERENCE_OPTIONS }, false);
        for (const name of CANDIDATES) {
        try {
        const candidateOptions = { kvStyle: "scatterND", valueLayout: "natural", ...PRESETS[name] };
        const candidate = await runVariant(`candidate ${name} chunk`, candidateOptions, candidateOptions.kvStyle === "scatterND");
        // compare the REAL rows only (pad rows hold garbage by design)
        const compareRange = (a, b, from, to, stride) => { let maxAbs = 0, nan = 0; for (let i = from * stride; i < to * stride; i++) { if (!Number.isFinite(a[i])) nan++; maxAbs = Math.max(maxAbs, Math.abs(a[i] - b[i])); } return { maxAbsDiff: maxAbs, nonFinite: nan }; };
        let keyDiff = 0, keyNan = 0;
        for (let h = 0; h < HEADS; h++) { const r = compareRange(candidate.updatedKey.subarray(h * PAST * DIM, (h + 1) * PAST * DIM), reference.updatedKey.subarray(h * PAST * DIM, (h + 1) * PAST * DIM), 0, total, DIM); keyDiff = Math.max(keyDiff, r.maxAbsDiff); keyNan += r.nonFinite; }
        const attentionCompare = compareRange(candidate.attention, reference.attention, pad, CHUNK, 4096);
        emit({ event: "compare_chunk", candidate: name, verdict: attentionCompare.nonFinite === 0 && attentionCompare.maxAbsDiff < 0.2 ? "OK" : "BAD", chunk: CHUNK, start: START, pad, valid, attention: attentionCompare, updatedKey: { maxAbsDiff: keyDiff, nonFinite: keyNan } });
        } catch (error) { emit({ event: "candidate_failed", candidate: name, err: String(error).slice(0, 200) }); }
        }
    }
}
(async () => {
    try { await main(); statusEl.textContent = "done"; }
    catch (error) { emit({ event: "fatal", err: String(error && error.stack ? error.stack : error).slice(0, 1500) }); statusEl.textContent = "FAILED: " + error; }
    emit({ event: "done" });
})();

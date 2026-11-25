// Run ONE of the Qwen3 subgraphs (qwen3_subgraphs.js) on its own with random inputs: build it, dispatch it a few times, report the
// time of one dispatch and check that the outputs are finite. Used to time each graph separately and to dump / inspect one graph.
//
// URL parameters:
//   only=head|block|tail|lmhead|kvupdate        which subgraph
//   device=npu|gpu
//   sequenceLength|S = 1 (decode)  or 1024 (prefill)        pastLength|P = 1151 (decode) or 0 (prefill)
//   fuseKvUpdate=0 (default 1: head/block graphs include the KV-cache update, like chat.html)
//   iterations|iters = 10   warmup|warm = 3     tag=<label>     chain=N (N back-to-back dispatches, one readback: pipelined cost per dispatch)
//   distinctOnly=weights|state|all (with distinct): which tensors differ between the K sets (weights / scales / gammas, or KV + activations + outputs)
//   distinct=K (with chain): rotate through K separate sets of input/output tensors instead of binding the same tensors every dispatch (chat.html binds different weights / KV per layer)
//   matmulStyle|mm = gemm|matmul   zeroPoint|zp = const|none   dequantStyle|deq = dequantizeLinear|castMultiply|scaleAfterMatmul   computeType|dt = float32|float16
//   lmHeadWeights|lmw = constant|input      constants=weight,scale,gamma (experiment: these weight kinds become graph constants)
// The page prints one readable line per step and one JSON line per event to the result file.
import { CFG, halfFill, ropeRows } from "./layer.js?v=20261119x";
import {
    byteLengthOf, lmHeadWeightSpecs,
    buildHeadGraph, buildBlockGraph, buildTailGraph, buildLmHeadGraph, buildKvUpdateGraph, setWeightLayout, setLmHeadFormat, setMergedProjections, buildBlockGroupGraph,
} from "./qwen3_subgraphs.js?v=20261124a";

const query = new URLSearchParams(location.search);
const param = (name, legacyName, fallback) => query.get(name) ?? (legacyName ? query.get(legacyName) : null) ?? fallback;
const deviceType = param("device", null, "npu"), subgraphName = param("only", null, "block");
const sequenceLength = +param("sequenceLength", "S", 1), pastLength = +param("pastLength", "P", 1151);
const distinctSets = Math.max(1, +param("distinct", null, 1));   // distinct=K: the chain rotates through K different sets of input / output tensors (real weights differ per layer), default 1 = the same tensors every dispatch
const distinctOnly = param("distinctOnly", null, "all");   // with distinct=K: "weights" = only the weight / scale / gamma inputs differ between the K sets, "state" = only everything else (KV, activations, mask, outputs) differs, "all" = both
const chainLength = +param("chain", null, 1);   // chain=N: N back-to-back dispatches of the graph, ONE readback at the end (pipelined cost per dispatch)
const iterations = +param("iterations", "iters", 10), warmupIterations = +param("warmup", "warm", 3), resultTag = param("tag", null, "");
const graphOptions = {
    matmulStyle: param("matmulStyle", "mm", "gemm"), zeroPoint: param("zeroPoint", "zp", "const"),
    dequantStyle: param("dequantStyle", "deq", "dequantizeLinear"), computeType: param("computeType", "dt", "float32"),
};
setWeightLayout(query.get("weightLayout") || "groupMajor");
if (query.get("mergeProjections") === "1") setMergedProjections(true);   // experiment: q/k/v as one projection and gate/up as one projection (timing only: the random weights are not merged real weights)   // experiment: groupMajor (NPUW) | outputMajor | float16 (build-time dequantized weights)
setLmHeadFormat(query.get("lmHeadFormat") || "uint8Asymmetric");   // experiment: uint8Asymmetric | int8Symmetric
if (query.get("debugOutputs") === "1") graphOptions.debugOutputs = true;   // head graph also returns scores / probabilities / expanded keys / scaled query
if (query.get("attentionGuard")) graphOptions.attentionGuard = query.get("attentionGuard");   // scores | probabilities | values | context   // experiment: tiny extra output that prevents attention fusion on the NPU
if (query.get("fuseKvUpdate") !== "0") graphOptions.fuseKvUpdate = true;   // ON by default like in chat.html (KV-cache update fused into head/block); fuseKvUpdate=0 = the plain block
if (query.get("attendInCache") === "1") graphOptions.attendInCache = true;   // attention reads the updated KV cache (length P) instead of concat(cache, new rows)
for (const name of ["attentionType", "rmsStyle", "valueLayout", "gqaStyle", "blockPart", "kvStyle", "gammaType", "stagingUpdate", "scatterAttention", "scatterIndexType"])
    if (query.get(name)) graphOptions[name] = query.get(name);   // per-block diagnostics, see qwen3_subgraphs.js
if (query.get("mergeRope") === "1") graphOptions.mergeRope = true;               // one "rope" input [1,2,S,128] instead of rope_cos / rope_sin
if (query.get("deriveRowsToKeep") === "1") graphOptions.deriveRowsToKeep = true;   // rows_to_keep = 1 - row_selector inside the graph (S = 1)
if (query.get("keepPresentRows") === "1") graphOptions.keepPresentRows = true;   // A/B: keep the (unused) present_key / present_value outputs of the fused graphs
if (query.get("stagingRows")) graphOptions.stagingRows = +query.get("stagingRows");
// like chat.html: decode (S = 1) blocks default to kvStyle=staged (+ the natural value layout it needs: with the transposed layout the NPU compiler returns NaN, see 05 notes 27); kvStyle=onehot gives the whole-cache update
if (!query.get("kvStyle") && graphOptions.fuseKvUpdate && sequenceLength === 1 && ["head", "block", "blockgroup"].includes(subgraphName)) { graphOptions.kvStyle = "staged"; if (!query.get("valueLayout")) graphOptions.valueLayout = "natural"; }   // kvStyle=staged: rows of the small staging buffer (default 64)
const lmHeadWeightMode = param("lmHeadWeights", "lmw", "constant");
// experiment: make some weight kinds graph constants instead of inputs, e.g. constants=scale  or  constants=weight,scale,gamma
const constantKinds = (query.get("constants") || "").split(",").filter(Boolean);
if (constantKinds.length) { graphOptions.constants = constantKinds; graphOptions.weightProvider = ([name, dataType, shape]) => dataForInput(name, dataType, shape); }
const maskDataType = "float16";            // (the NPUW dump has an f32 prefill mask; we keep f16 everywhere)
const { VOCAB: vocabularySize } = CFG;
const byId = (id) => document.getElementById(id);
const elementCount = (shape) => shape.reduce((a, b) => a * b, 1);
const startTime = performance.now();
function log(eventName, message, fields = {}) {
    byId("out").textContent += `[${((performance.now() - startTime) / 1000).toFixed(1).padStart(6)} s] ${message}\n`;
    const record = JSON.stringify({ experiment: "single_subgraph", tag: resultTag, subgraph: subgraphName, device: deviceType, sequence_length: sequenceLength,
        past_length: pastLength, ...graphOptions, event: eventName, message, ...fields });
    console.log(record);
    fetch("/log", { method: "POST", body: record }).catch(() => {});
}
const setStatus = (text) => (byId("status").textContent = text);
const summarize = (values) => {
    if (!values.length) return {};
    const sorted = [...values].sort((a, b) => a - b), round = (x) => Math.round(x * 1000) / 1000;
    return { count: values.length, median_ms: round(sorted[Math.floor(sorted.length / 2)]), min_ms: round(sorted[0]), max_ms: round(sorted[sorted.length - 1]) };
};

// random but plausible data for an input of the given name / type (timing and finiteness only, not model quality)
const randomBytePool = new Uint8Array(16 << 20);
{ let state = 123456789; for (let i = 0; i < randomBytePool.length; i++) { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; randomBytePool[i] = state & 255; } }   // fixed seed: runs are comparable
function dataForInput(name, dataType, shape) {
    const count = elementCount(shape), bytes = byteLengthOf([name, dataType, shape]);
    if (dataType === "int4" || ((dataType === "uint8" || dataType === "int8") && name === "lm_head_weight")) {
        const data = new Uint8Array(bytes);
        for (let offset = 0; offset < bytes; offset += randomBytePool.length) data.set(randomBytePool.subarray(0, Math.min(randomBytePool.length, bytes - offset)), offset);
        return data;
    }
    if (dataType === "uint8") return new Uint8Array(count).fill(128);                                     // zero points
    if (dataType === "float16") return halfFill(count, name.endsWith("_scale") ? 0.002 : name.endsWith("_gamma") ? 1 : 0.05);   // scales / gammas / activations
    if (dataType === "float32") return new Float32Array(count).fill(name.endsWith("_gamma") ? 1 : 0.05);  // gammas / f32 activations
    throw new Error(`no test data for ${dataType} input ${name}`);
}
const typedArrayFor = (dataType, count) => (dataType === "float16" ? new Float16Array(count) : dataType === "float32" ? new Float32Array(count) : new Uint8Array(count));

async function main() {
    const mlContext = await navigator.ml.createContext({ deviceType });
    log("context_created", `WebNN context created (device=${deviceType})`, { user_agent: navigator.userAgent });
    const common = { sequenceLength, pastLength, maskDataType, options: graphOptions };
    const createGraph = {
        head: (b) => buildHeadGraph(b, common),
        block: (b) => buildBlockGraph(b, common),
        blockgroup: (b) => buildBlockGroupGraph(b, { ...common, blockCount: +param("blocks", null, 5) }),   // blocks=K: K consecutive blocks in one graph
        tail: (b) => buildTailGraph(b, common),
        lmhead: (b) => buildLmHeadGraph(b, { weightMode: lmHeadWeightMode, weightProvider: ([name, dataType, shape]) => dataForInput(name, dataType, shape), options: graphOptions }),
        kvupdate: (b) => buildKvUpdateGraph(b, { sequenceLength, pastLength }),
    }[subgraphName];
    if (!createGraph) throw new Error(`unknown subgraph "${subgraphName}"`);

    setStatus(`building ${subgraphName}`);
    const builder = new MLGraphBuilder(mlContext), description = createGraph(builder), buildStart = performance.now();
    const graph = await builder.build(description.outputs), buildMs = Math.round(performance.now() - buildStart);
    log("graph_built", `Built graph "${subgraphName}" in ${buildMs} ms (${Object.keys(description.inputs).length} inputs)`, { build_ms: buildMs });

    // bind: one MLTensor per graph input (random data) and per output
    const inputTensors = {}, outputTensors = {}, inputData = {};
    for (const [name, operand] of Object.entries(description.inputs)) {
        inputTensors[name] = await mlContext.createTensor({ dataType: operand.dataType, shape: operand.shape, writable: true });
        const data = name === "kv_write_index" ? (operand.dataType === "int64" ? BigInt64Array.of(0n) : Int32Array.of(0))   // tensorScatter: the new rows go to cache rows 0..S-1 (timing test only)
            : name === "kv_indices" ? kvIndices()
            : name === "attention_mask" ? maskFor(operand.dataType)
            : name === "rope_cos" || name === "rope_sin" ? ropeRows(pastLength, sequenceLength)[name === "rope_cos" ? "cos" : "sin"]
            : dataForInput(name, operand.dataType, operand.shape);
        inputData[name] = data;
        mlContext.writeTensor(inputTensors[name], data);
    }
    for (const [name, operand] of Object.entries(description.outputs)) outputTensors[name] = await mlContext.createTensor({ dataType: operand.dataType, shape: operand.shape, readable: true });
    const inputSets = [inputTensors], outputSets = [outputTensors];
    for (let k = 1; k < distinctSets; k++) {   // extra tensor sets for distinct=K (same contents, other buffers)
        setStatus(`creating tensor set ${k + 1}/${distinctSets}`);
        const ins = {}, outs = {};
        const isWeight = (name) => /(weight|scale|gamma)/.test(name);
        for (const [name, operand] of Object.entries(description.inputs)) {
            if ((distinctOnly === "weights" && !isWeight(name)) || (distinctOnly === "state" && isWeight(name))) { ins[name] = inputTensors[name]; continue; }   // shared with set 0
            ins[name] = await mlContext.createTensor({ dataType: operand.dataType, shape: operand.shape, writable: true }); mlContext.writeTensor(ins[name], inputData[name]);
        }
        for (const [name, operand] of Object.entries(description.outputs)) outs[name] = distinctOnly === "weights" ? outputTensors[name] : await mlContext.createTensor({ dataType: operand.dataType, shape: operand.shape, readable: true });
        inputSets.push(ins); outputSets.push(outs);
    }
    function kvIndices() {   // (batch 0, kv head, cache row) for every new row; the rows are written at the start of the cache (timing test only)
        const heads = 8, indices = new Int32Array(heads * sequenceLength * 3);
        for (let h = 0; h < heads; h++) for (let s = 0; s < sequenceLength; s++) { const at = (h * sequenceLength + s) * 3; indices[at] = 0; indices[at + 1] = h; indices[at + 2] = s; }
        return indices;
    }
    function maskFor(dataType) { // additive causal mask over [past cache + new rows]: everything visible except the future new rows
        const total = description.inputs.attention_mask.shape[3], inCache = total === pastLength, mask = typedArrayFor(dataType, sequenceLength * total);
        for (let row = 0; row < sequenceLength; row++) for (let column = 0; column < total; column++) mask[row * total + column] = column <= (inCache ? pastLength - sequenceLength + row : pastLength + row) ? 0 : -65504;
        return mask;
    }

    // time dispatch + readback of the first output (the readback is the synchronisation point)
    const firstName = Object.keys(description.outputs)[0], firstOperand = description.outputs[firstName];
    const firstBuffer = typedArrayFor(firstOperand.dataType, elementCount(firstOperand.shape)), timings = [];
    for (let i = 0; i < warmupIterations + iterations; i++) {
        setStatus(`dispatching ${subgraphName} (${i + 1}/${warmupIterations + iterations})`);
        const start = performance.now();
        mlContext.dispatch(graph, inputTensors, outputTensors);
        await mlContext.readTensor(outputTensors[firstName], firstBuffer);
        if (i >= warmupIterations) timings.push(performance.now() - start);
    }
    const outputReport = {};
    for (const [name, operand] of Object.entries(description.outputs)) {
        const buffer = typedArrayFor(operand.dataType, elementCount(operand.shape));
        await mlContext.readTensor(outputTensors[name], buffer);
        outputReport[name] = { shape: operand.shape, dataType: operand.dataType, all_finite: buffer.slice(0, 4096).every(Number.isFinite), first_values: [...buffer.slice(0, 3)], sum_of_first_4096: buffer.slice(0, 4096).reduce((a, b) => a + b, 0), max_abs_of_first_4096: buffer.slice(0, 4096).reduce((a, b) => Math.max(a, Math.abs(b)), 0) };
    }
    let chain = null;
    if (chainLength > 1) {   // pipelined: enqueue N dispatches without waiting, then read back once; the per-dispatch time includes everything that is not overlapped
        const chainTimes = [], enqueueTimes = [];
        for (let i = 0; i < warmupIterations + iterations; i++) {
            const start = performance.now();
            for (let c = 0; c < chainLength; c++) mlContext.dispatch(graph, inputSets[c % distinctSets], outputSets[c % distinctSets]);
            const enqueued = performance.now();
            await mlContext.readTensor(outputSets[(chainLength - 1) % distinctSets][firstName], firstBuffer);
            if (i >= warmupIterations) { chainTimes.push(performance.now() - start); enqueueTimes.push(enqueued - start); }
        }
        const total = summarize(chainTimes), enqueue = summarize(enqueueTimes);
        chain = { chain_length: chainLength, distinct_sets: distinctSets, distinct_only: distinctOnly, total_median_ms: total.median_ms, per_dispatch_ms: Math.round(total.median_ms / chainLength * 1000) / 1000, enqueue_median_ms: enqueue.median_ms };
        log("chain_result", `Chain of ${chainLength} dispatches of "${subgraphName}"${distinctSets > 1 ? ` over ${distinctSets} distinct tensor sets (${distinctOnly})` : ""}: ${total.median_ms} ms total = ${chain.per_dispatch_ms} ms per dispatch (JS enqueue part: ${enqueue.median_ms} ms)`, chain);
    }
    const timing = summarize(timings);
    log("result", `One dispatch of "${subgraphName}" (incl. readback): median ${timing.median_ms} ms, min ${timing.min_ms} ms, max ${timing.max_ms} ms; build ${buildMs} ms`,
        { build_ms: buildMs, dispatch_with_readback: timing, chain, outputs: outputReport });
}
(async () => {
    try { await main(); setStatus("done"); }
    catch (error) { log("fatal", "Failed: " + String(error && error.stack ? error.stack : error).slice(0, 1500)); setStatus("FAILED: " + error); }
    log("done", "Done");
})();

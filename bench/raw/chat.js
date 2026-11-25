// Qwen3-4B text generation with the NPUW-style WebNN subgraphs (qwen3_subgraphs.js) and REAL weights.
//
// Weights come from the OpenVINO GenAI model, exported by bench/tools/extract_ov_weights.py into
// <root>/weights/qwen3_4b/ (manifest.json + one file per layer + the embedding table).
//
// Pipeline (see the diagram on the page):
//   1. tokenizer (transformers.js)            prompt text -> token ids
//   2. host embedding lookup                  token id -> row of the u8 table -> dequantize -> f16 hidden_states
//   3. PREFILL, CHUNKED  the prompt is processed `prefillChunkSize` tokens at a time. Every chunk runs
//        HEAD -> BLOCK x 35 (-> TAIL -> LM HEAD on the last chunk only), attends to the KV cache (everything written by the
//        earlier chunks) plus its own rows, and writes its own key/value rows into the cache (KV UPDATE after every attention half).
//        The LAST chunk is left-padded so that its last row is the last prompt token; its logits give the first output token.
//   4. DECODE   (one token per step; the KV cache holds `maxLength - 1` rows + the new token)  the same chain with S = 1
//   5. read back the logits, greedy argmax, loop to step 2
// All weights are MLTensors that the prefill and decode graphs share (weights-as-inputs). Hidden state, attention output and
// the KV cache stay on the device; only the logits come back to the host.
//
// What depends on what:
//   shared for every setting     : weights, embedding table, lm_head graph
//   rebuilt when max length or chunk size changes ("session"): prefill HEAD/BLOCK/TAIL (chunk size, past = max length - 1), decode
//                                  HEAD/BLOCK/TAIL (S = 1), both KV-update graphs, the KV cache tensors, the activation tensors
//
// URL parameters (long name | legacy alias; the session options kvStyle, stagingRows, blocksPerGraph, prefillBlocksPerGraph, prefillChunkSizes, computeType can also be picked on the page):
//   device=npu|gpu     maxLength = 1152 (= pastLength|P + 1)     prefillChunkSize|chunk = 256     maxNewTokens|maxNew = 64     layers|L = 36
//   prompt=...         autoRun|auto = 1 (load immediately, generate once, finish; otherwise the page waits for the "Load model" button)
//   fuseKvUpdate=0 (separate KV-update graphs instead of the default in-graph update)   kvStyle=staged (default: decode updates a small staging buffer, flushed into the main cache every stagingRows=64 steps) | onehot (whole-cache update) | scatterND (in-graph KV update as scatterND, value cache in natural layout)   kvInPlace=1 (KV update writes into the tensor it reads, no second cache)   tag=<label for the result file>     parallelDownloads=6 (weight files downloaded at the same time into the OPFS cache)
//   matmulStyle|mm = gemm|matmul    zeroPoint|zp = const|none    dequantStyle|deq = dequantizeLinear|castMultiply|scaleAfterMatmul
//   computeType|dt = float32 (default) | float16 (no f32 inside the graphs; the RMSNorm x^2 can overflow where it is not fused, e.g. on a GPU)
//   weightFormat|wf = int4 (default) | float16 (build-time dequantization, see above)    dequantizeAt = device|host
import { AutoTokenizer, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js";
import { CFG, ropeRows } from "./layer.js?v=20261119x";
import { listPackages, peekPackage, loadPackageModel, setupPackageControls, clearPackageCache } from "./chat-package.js?v=3";
import {
    attentionWeightSpecs, feedForwardWeightSpecs,
    buildHeadGraph, buildBlockGraph, buildTailGraph, buildLmHeadGraph, buildKvUpdateGraph, buildDequantizeWeightGraph, setWeightLayout, setMergedProjections, buildBlockGroupGraph,
} from "./qwen3_subgraphs.js?v=20261124a";

// ---------------------------------------------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------------------------------------------
const query = new URLSearchParams(location.search);
const param = (name, legacyName, fallback) => query.get(name) ?? (legacyName ? query.get(legacyName) : null) ?? fallback;
const deviceType = param("device", null, "npu");
const modelId = param("model", null, "js");   // "js": the built-in hand-written Qwen3-4B graphs (every option below); anything else: a package from ../packages/models.json (m2w compile)
// default prefill chunk 256 (was 64): every prefill dispatch dequantizes the int4 weights again whatever the chunk size, so a bigger chunk is much cheaper per token (05 notes 27.20-27.22; a short
// prompt costs somewhat more than with 64). NPUW does the same with one static prefill graph (its default is 1024, too slow for us).
const defaultChunkSize = +param("prefillChunkSize", "chunk", 256);
const defaultMaxLength = query.get("maxLength") ? +query.get("maxLength") : +param("pastLength", "P", 1151) + 1;   // KV cache capacity
const defaultMaxNewTokens = +param("maxNewTokens", "maxNew", 64);
const layerCount = +param("layers", "L", 36);
const autoRun = param("autoRun", "auto", "0") === "1";
const resultTag = param("tag", null, "");
// profile=1: after a generation, time the decode step in segments (head / blocks / tail / lm_head, each followed by a sync) and as one chain; shown in the "Decode segment profile" card.
// It makes the residual / final-hidden tensors readable (the syncs read them), so it is a page-load option, not a session option.
const profileEnabled = param("profile", null, "0") === "1";
// scatterIndexType=int32|int64: the data type of the write index of kvStyle=tensorScatter (the op follows ONNX TensorScatter, whose write_indices are int64)
const scatterIndexType = param("scatterIndexType", null, "int32") === "int64" ? "int64" : "int32";
const writeIndexData = (row) => (scatterIndexType === "int64" ? BigInt64Array.of(BigInt(row)) : Int32Array.of(row));
// fuseKvUpdate (default 1): the head/block graphs update the KV cache themselves (extra inputs/outputs), there is no separate KV-update dispatch
const fuseKvUpdate = param("fuseKvUpdate", null, "1") !== "0";
// kvStyle=scatterND (experiment, needs fuseKvUpdate): the in-graph KV update is a scatterND of the new rows (kv_indices input) instead of the one-hot
// multiply-add; the value cache then has the natural layout [1,kvHeads,T,128] like the key cache. Combine with kvInPlace=1 for "same tensor in and out".
// kvStyle: staged (DEFAULT, decode only; see below) | onehot (the whole-cache update: past*keep + selector@new) | broadcast (decode: selector*new, no matmul) | select (decode: where(selector>0, new, past))
// attendInCache=1: attention reads the updated cache (length P) instead of concat(cache, new rows) (length P+S); always the case for scatterND
// kvStyle, stagingRows, blocksPerGraph, prefillBlocksPerGraph, prefillChunkSizesParam and everything derived from them are SESSION options: the page can change them (selectors under the "Load model"
// button) and they are applied by applySessionOptions() when the session is (re)built; the URL values below are only the initial ones. All others (weightFormat, computeType, mergeProjections, ...) are fixed at page load.
let kvStyle, scatterKv, tensorScatterKv, attendInCache, stagedKv, stagingRows, naturalValueLayout, prefillGuard, blocksPerGraph, prefillBlocksPerGraph, prefillChunkSizesParam, graphOptions, computeType;
// kvStyle=staged (decode only): the O(cache) one-hot update is replaced by an update of a small STAGING buffer with the newest `stagingRows` rows; attention reads
// concat(main cache, staging). The main cache is written by a flush graph (per layer) once every `stagingRows` decode steps, and by the prefill chunks as before.
// attention guard of the PREFILL chunk graphs: a tiny extra output that keeps the NPU compiler from fusing the attention. With the TRANSPOSED value cache that fused path returned wrong
// results for S > 1 with a past cache (05 notes ch. 17); with the natural value layout (the default) it is not needed (kvcheck.html: identical outputs on the devices checked). prefillGuard=1 forces it.
// blocksPerGraph=K (default 1): K consecutive BLOCKs in one graph (35 blocks -> ceil(35/K) dispatches); needs weights/KV names with a "b<j>_" prefix
// mergeProjections (default 1): q/k/v are ONE projection and gate/up ONE projection (NPUW's parallel MatMul merge): fewer weight inputs and MatMuls per block. The weight files stay as they
// are; the int4 weights / f16 scales are concatenated along the output channels when a layer is loaded.
const mergeProjections = param("mergeProjections", "mp", "1") === "1" && param("weightFormat", "wf", "int4") === "int4";   // ON by default (fewer dispatches per token); mergeProjections=0 turns it off; not with weightFormat=float16
setMergedProjections(mergeProjections);
// attention guard of the PREFILL chunk graphs (extra tiny output that keeps the NPU compiler from fusing the attention: with the TRANSPOSED value cache that fused path gave wrong
// results for S > 1 with a past cache, 05 notes ch. 17). With the natural value layout (the default) it is not needed (checked with kvcheck.html on the devices tried): prefillGuard=1 forces it.
// prefillChunkSizes|chunks (default: only the chunk size picked on the page): several prefill chunk sizes, e.g. 64,256. One set of prefill graphs is built per size and every prompt uses the
// chunk sequence with the lowest estimated cost. A prefill dispatch costs about (prefillFixedTokens + chunk size) "token units": the int4 weights are dequantized again by EVERY dispatch,
// whatever the chunk size (05 notes 27.20: the time per block is a fixed part plus a part linear in S), which makes big chunks cheaper per token but wasteful for short prompts.
const prefillFixedTokens = +param("prefillFixedTokens", null, 200);
// prefillBlocksPerGraph|pbpg (default: same as blocksPerGraph): blocks per graph of the PREFILL chunk graphs only, e.g. blocksPerGraph=5&prefillBlocksPerGraph=1 groups only the decode graphs
const groupSizeOf = (phase) => (phase === "prefill" ? prefillBlocksPerGraph : blocksPerGraph);
const SESSION_KV_STYLES = ["onehot", "broadcast", "select", "scatterND", "tensorScatter", "staged"];
const parseChunkSizes = (text) => String(text ?? "").split(",").map(Number).filter((size) => size > 0);
// requested: { kvStyle, stagingRows, blocksPerGraph, prefillBlocksPerGraph (0 = same as blocksPerGraph), chunkSizes, computeType }
function applySessionOptions(requested) {
    kvStyle = SESSION_KV_STYLES.includes(requested.kvStyle) ? requested.kvStyle : "staged";
    scatterKv = fuseKvUpdate && kvStyle === "scatterND";
    tensorScatterKv = fuseKvUpdate && kvStyle === "tensorScatter";   // MLGraphBuilder.tensorScatter(cache, rows, write_index, {axis: 2}): ONE start row per dispatch, the rows go in contiguously (see writePrefillChunkInputsTensorScatter)
    attendInCache = fuseKvUpdate && (scatterKv || tensorScatterKv || param("attendInCache", null, "0") === "1");
    stagedKv = fuseKvUpdate && kvStyle === "staged";
    stagingRows = requested.stagingRows;
    naturalValueLayout = stagedKv || scatterKv || tensorScatterKv || param("valueLayout", null, "transposed") === "natural";   // staged needs it: with the transposed value cache the NPU compiler returns NaN for concat(main, staging) along the last axis
    prefillGuard = !naturalValueLayout || param("prefillGuard", null, "0") === "1" ? "values" : null;
    blocksPerGraph = Math.max(1, requested.blocksPerGraph);
    prefillBlocksPerGraph = Math.max(1, requested.prefillBlocksPerGraph || blocksPerGraph);
    prefillChunkSizesParam = requested.chunkSizes;
    computeType = requested.computeType === "float16" ? "float16" : "float32";
    keyValueNoInput = keyValueNoInputRequested && tensorScatterKv;   // the other KV styles ignore kvNoInput
    graphOptions = {
        matmulStyle: param("matmulStyle", "mm", "gemm"), zeroPoint: param("zeroPoint", "zp", "const"),
        dequantStyle: param("dequantStyle", "deq", "dequantizeLinear"), computeType, fuseKvUpdate, ...(kvStyle !== "onehot" ? { kvStyle } : {}), ...(attendInCache ? { attendInCache: true } : {}), ...(tensorScatterKv ? { scatterIndexType } : {}), ...(keyValueNoInput ? { kvNoInput: true, ...(scatterOptionsTemplate ? { scatterOptions: scatterOptionsTemplate } : {}) } : {}), ...(stagedKv ? { stagingRows } : {}), ...(naturalValueLayout ? { valueLayout: "natural" } : {}),
    };
}
const sessionOptionKey = () => [kvStyle, stagingRows, blocksPerGraph, prefillBlocksPerGraph, computeType].join("|");
const urlSessionOptions = { kvStyle: param("kvStyle", null, "staged"), stagingRows: +param("stagingRows", null, 64), blocksPerGraph: +param("blocksPerGraph", "bpg", 1),
    prefillBlocksPerGraph: +param("prefillBlocksPerGraph", "pbpg", 0), chunkSizes: parseChunkSizes(param("prefillChunkSizes", "chunks", "")), computeType: param("computeType", "dt", "float32") };
applySessionOptions(urlSessionOptions);
// weightFormat=float16: dequantize int4 -> f16 ONCE while loading (a small graph per projection shape, output kept on the device, no readback)
// and bind the f16 weights to the block graphs (no dequantization inside the graphs). dequantizeAt=device|host chooses where that happens.
const weightFormat = param("weightFormat", "wf", "int4"), dequantizeAt = param("dequantizeAt", null, "device");
// kvInPlace=1: the KV-update graph reads and writes the SAME MLTensor (past_key == updated_key, past_value == updated_value) instead of ping-ponging
// between two buffers; saves the spare cache (about 0.3 GB at max length 1152). Whether the runtime accepts an input tensor as an output is what this tests.
const keyValueInPlace = param("kvInPlace", null, "0") === "1";
// kvNoInput=1 (with kvStyle=tensorScatter; needs the PoC browser build): MLGraphBuilder.tensorScatter(update, writeIndices, {axis, ...}) WITHOUT the cache as input. The head / block graphs
// have no past_key / past_value inputs; updated_key / updated_value ARE the cache tensors (bound as outputs only, one set of KV tensors per layer, no spare). scatterOptions={"axis":2,"shape":"$shape"}
// overrides the options passed to the op ("$shape" = the cache shape [1,kvHeads,P,128]; default: axis 2 plus the shape as both `shape` and `outputShape`).
const keyValueNoInputRequested = param("kvNoInput", null, "0") === "1";
const scatterOptionsTemplate = (() => { try { return query.get("scatterOptions") ? JSON.parse(query.get("scatterOptions")) : null; } catch (error) { throw new Error("scatterOptions is not valid JSON: " + error); } })();
let keyValueNoInput = false;
const keyValueZeroInit = param("kvZeroInit", null, "1") !== "0";   // kvNoInput: the cache tensors are zero-filled once when a session is created (rows that are not written yet are masked, but NaN x 0 would still poison attention); kvZeroInit=0 skips it

if (weightFormat === "float16") setWeightLayout("float16");
const { E: hiddenSize, NH: numAttentionHeads, KVH: numKeyValueHeads, D: headDim, VOCAB: vocabularySize } = CFG;
const attentionWidth = numAttentionHeads * headDim;
const downProjectionElementCount = hiddenSize * CFG.FF;   // f16 down_projection weight [2560, 9728]
// paths are resolved relative to this script (<root>/raw/chat.js -> <root>/weights/...), so any server root works
const rootPath = new URL("../", import.meta.url).pathname;
const weightsPath = rootPath + "weights/qwen3_4b/";
const END_OF_TEXT_TOKENS = new Set([151645, 151643]);                   // <|im_end|>, <|endoftext|>

// ---------------------------------------------------------------------------------------------------------------------
// Page helpers: readable log + structured result line (the result file is read by the benchmark scripts)
// ---------------------------------------------------------------------------------------------------------------------
const byId = (id) => document.getElementById(id);
const pageStartTime = performance.now();
const seconds = (milliseconds) => (milliseconds / 1000).toFixed(1) + " s";
const rounded = (value) => Math.round(value * 10) / 10;
function log(eventName, message, fields = {}) {
    const elapsed = ((performance.now() - pageStartTime) / 1000).toFixed(1).padStart(7);
    byId("log").textContent += `[${elapsed} s] ${message}\n`;
    byId("log").scrollTop = byId("log").scrollHeight;
    if (eventName === "fatal") byId("log").closest("details")?.setAttribute("open", "");
    const record = JSON.stringify({ experiment: "chat", tag: resultTag, device: deviceType, layers: layerCount, weight_format: weightFormat, kv_in_place: keyValueInPlace, kv_no_input: keyValueNoInput, fuse_kv_update: fuseKvUpdate, kv_style: kvStyle, value_layout: naturalValueLayout ? 'natural' : 'transposed', blocks_per_graph: blocksPerGraph, ...graphOptions, event: eventName, message, ...fields });
    console.log(record);
    fetch("/log", { method: "POST", body: record }).catch(() => {});
}
const setStatus = (text) => (byId("status").textContent = text);
// summary cell: a short value, the long explanation as a hover tip; headline tile: big value + one line below it
const setCell = (id, value, tip = "") => { const cell = byId(id); cell.textContent = value; if (tip) cell.dataset.tip = tip; else delete cell.dataset.tip; };
const setMetric = (id, value, unit = "", sub = "") => { byId(id).innerHTML = unit ? `${value}<small>${unit}</small>` : value; byId(id + "-sub").textContent = sub; };
// rows: [label, ms, tip (hover), kind (colour)]; the last row is the total, every other row gets a bar for its share of it
function fillTimingTable(rows) {
    const table = byId("timingTable");
    while (table.rows.length > 1) table.deleteRow(1);
    const total = rows[rows.length - 1][1] || 1;
    for (const [label, value, tip, kind] of rows) {
        const row = table.insertRow(); row.className = `t-${kind}`;
        const name = row.insertCell(); name.textContent = label; name.dataset.tip = tip; name.classList.add("has-tip");
        const number = row.insertCell(); number.textContent = value; number.className = "num";
        const share = row.insertCell(); share.className = "share";
        const percent = Math.min(100, Math.round((value / total) * 100));
        share.innerHTML = `<span class="bar"><i style="width:${percent}%"></i></span><span class="pct">${percent}%</span>`;
    }
}
function highlight(...ids) { // mark the pipeline steps that are running right now
    document.querySelectorAll(".active").forEach((element) => element.classList.remove("active"));
    for (const id of ids) byId(id)?.classList.add("active");
}

// ---------------------------------------------------------------------------------------------------------------------
// Weight files: downloaded in parallel into the Origin Private File System (OPFS) and read back from there.
// The second visit needs no network at all. Modeled after getModelOPFS() in webnn-developer-preview/assets/js/common_utils.js.
// ---------------------------------------------------------------------------------------------------------------------
const OPFS_CACHE_DIRECTORY = "webnn-qwen3-4b-weights", PARALLEL_DOWNLOADS = +param("parallelDownloads", null, 6);
async function openCacheFile(fileName, create) {
    const directory = await (await navigator.storage.getDirectory()).getDirectoryHandle(OPFS_CACHE_DIRECTORY, { create });
    return directory.getFileHandle(fileName, { create });
}
async function clearWeightCache() {
    try { await (await navigator.storage.getDirectory()).removeEntry(OPFS_CACHE_DIRECTORY, { recursive: true }); } catch { /* nothing cached */ }
}
// make sure `fileName` is in the OPFS cache (stream network -> disk, never holding the whole file in memory); resolves to { fromCache, bytes }
async function ensureCachedFile(fileName) {
    const url = weightsPath + fileName;
    try {   // cache hit: same size as on the server (a HEAD request that cannot tell the size counts as a match)
        const file = await (await openCacheFile(fileName, false)).getFile();
        let serverSize = 0;
        try { serverSize = +(await fetch(url, { method: "HEAD", cache: "no-store" })).headers.get("Content-Length") || 0; } catch { /* offline: trust the cache */ }
        if (file.size > 0 && (serverSize === 0 || serverSize === file.size)) return { fromCache: true, bytes: file.size };
    } catch { /* not cached yet */ }
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(`failed to fetch ${url}: ${response.status}`);
    const total = +response.headers.get("Content-Length") || 0;
    const writable = await (await openCacheFile(fileName, true)).createWritable();
    let written = 0;
    try {
        const reader = response.body.getReader();
        for (;;) { const chunk = await reader.read(); if (chunk.done) break; await writable.write(chunk.value); written += chunk.value.byteLength; }
        await writable.close();
    } catch (error) { await writable.abort().catch(() => {}); throw error; }
    if (total > 0 && written !== total) throw new Error(`incomplete download of ${fileName}: ${written} of ${total} bytes`);
    return { fromCache: false, bytes: written };
}
const readCachedFile = async (fileName) => (await (await openCacheFile(fileName, false)).getFile()).arrayBuffer();
// run the downloads with at most PARALLEL_DOWNLOADS in flight; returns { fileName -> promise } so consumers can wait for exactly the file they need
function startDownloads(fileNames) {
    const promises = new Map(), queue = [];
    let running = 0;
    const pump = () => {
        while (running < PARALLEL_DOWNLOADS && queue.length) {
            const { fileName, resolve, reject } = queue.shift();
            running++;
            ensureCachedFile(fileName).then(resolve, reject).finally(() => { running--; pump(); });
        }
    };
    for (const fileName of fileNames) promises.set(fileName, new Promise((resolve, reject) => { queue.push({ fileName, resolve, reject }); }));
    pump();
    return promises;
}
const keyValueCacheMegabytes = (maxLength) => (layerCount * 2 /* key+value */ * 2 /* double buffer */ * numKeyValueHeads * (attendInCache ? maxLength : maxLength - 1) * headDim * 2 /* f16 bytes */) / 2 ** 20;

// Loads everything that does not depend on max length / chunk size and returns the functions the page needs.
async function loadSharedModel() {
    if (fuseKvUpdate) for (const id of ["prefill-kv_update", "decode-kv_update"]) byId(id).innerHTML = "KV UPDATE fused into HEAD / BLOCK (fuseKvUpdate=1): no separate dispatch";
    document.querySelectorAll(".layerCount").forEach((e) => (e.textContent = layerCount));
    document.querySelectorAll(".layerCountMinusOne").forEach((e) => (e.textContent = layerCount - 1));

    // -------------------------------------------------------------------------------------------------------------------
    // Startup: tokenizer, WebNN context, embedding table, lm_head graph, weights
    // -------------------------------------------------------------------------------------------------------------------
    setStatus("loading tokenizer");
    env.localModelPath = rootPath + "weights/tok/"; env.allowRemoteModels = false; env.allowLocalModels = true;
    const tokenizer = await AutoTokenizer.from_pretrained("qwen3");
    log("tokenizer_loaded", "Tokenizer loaded");

    const mlContext = await navigator.ml.createContext({ deviceType });
    log("context_created", `WebNN context created (device=${deviceType})`, { user_agent: navigator.userAgent });
    // While a session is being built, every tensor and graph is also recorded here (see createSession) so that all of it can be released, also after a failed build.
    let tensorSink = null, graphSink = null;
    const createTensor = async (dataType, shape, { writable = false, readable = false } = {}) => {
        const tensor = await mlContext.createTensor({ dataType, shape, writable, readable });
        tensorSink?.push(tensor);
        return tensor;
    };
    const createWritableTensor = (dataType, shape) => createTensor(dataType, shape, { writable: true });

    let weightLoadMs = 0, sharedBuildMs = 0;
    // start downloading EVERY weight file right away (embedding first, then layer 0, 1, ...); the graphs are compiled while they arrive
    const manifest = await (await fetch(weightsPath + "manifest.json", { cache: "no-store" })).json();
    const embeddingFiles = ["embed.bin", "embed_zp.bin", "embed_s.bin", "final_norm.bin"];
    const downloadStart = performance.now(), downloads = startDownloads([...embeddingFiles, ...manifest.layers.slice(0, layerCount).map((layer) => layer.file)]);
    let cachedFileCount = 0, downloadedFileCount = 0, downloadedBytes = 0;
    for (const promise of downloads.values()) promise.then((result) => { if (result.fromCache) cachedFileCount++; else { downloadedFileCount++; downloadedBytes += result.bytes; } }).catch(() => {});
    Promise.all(downloads.values()).then(() => log("downloads_finished",
        `All ${downloads.size} weight files are in the OPFS cache after ${seconds(performance.now() - downloadStart)} (${cachedFileCount} from cache, ${downloadedFileCount} downloaded, ${Math.round(downloadedBytes / 2 ** 20)} MB; ${PARALLEL_DOWNLOADS} parallel)`,
        { download_wall_ms: Math.round(performance.now() - downloadStart), files_from_cache: cachedFileCount, files_downloaded: downloadedFileCount, downloaded_mb: Math.round(downloadedBytes / 2 ** 20) })).catch(() => {});
    setStatus("loading embedding table (389 MB)");
    let stepStart = performance.now();
    await Promise.all(embeddingFiles.map((fileName) => downloads.get(fileName)));
    const embeddingWeight = new Uint8Array(await readCachedFile("embed.bin")), embeddingZeroPoint = new Uint8Array(await readCachedFile("embed_zp.bin"));
    const embeddingScale = new Float16Array(await readCachedFile("embed_s.bin")), finalNormBytes = new Uint8Array(await readCachedFile("final_norm.bin"));
    weightLoadMs += performance.now() - stepStart;
    log("embedding_table_loaded", `Embedding table loaded (${seconds(performance.now() - stepStart)}); also used as the lm_head weight`);
    // host lookup: dequantize one u8 row (per-row zero point and scale) into f16 at `outputOffset`
    const lookupEmbeddingRow = (tokenId, output, outputOffset) => {
        const zeroPoint = embeddingZeroPoint[tokenId], scale = embeddingScale[tokenId], rowStart = tokenId * hiddenSize;
        for (let i = 0; i < hiddenSize; i++) output[outputOffset + i] = (embeddingWeight[rowStart + i] - zeroPoint) * scale;
    };

    // build one graph, log it, add it to the per-graph table; returns { graph, buildMs }
    const buildGraph = async (label, createGraph, timeLabelId) => {
        setStatus(`building graph: ${label}`);
        const builder = new MLGraphBuilder(mlContext), description = createGraph(builder), start = performance.now();
        const graph = await builder.build(description.outputs), elapsed = performance.now() - start;
        graphSink?.push(graph);
        log("graph_built", `Built graph ${label.padEnd(30)} ${seconds(elapsed)}`, { graph: label, build_ms: Math.round(elapsed) });
        const row = byId("buildTable").insertRow(); row.insertCell().textContent = label; row.insertCell().textContent = Math.round(elapsed);
        const timeLabel = timeLabelId && byId(timeLabelId);
        if (timeLabel) timeLabel.textContent = ` built in ${seconds(elapsed)}`;
        return { graph, buildMs: elapsed };
    };
    const lmHeadProvider = ([name]) => (name === "lm_head_weight" ? embeddingWeight : name === "lm_head_zero_point" ? embeddingZeroPoint : new Uint8Array(embeddingScale.buffer));
    // lm_head is shared by all sessions, but its inner compute type follows the page option: rebuilt (outside any session) when that changes
    let lmHeadGraph = null, lmHeadComputeType = null, lmHeadBuildMs = 0;
    const ensureLmHead = async () => {
        if (lmHeadGraph && lmHeadComputeType === computeType) return;
        if (lmHeadGraph) {
            lmHeadGraph.destroy?.();
            const table = byId("buildTable");
            for (let i = table.rows.length - 1; i >= 1; i--) if (table.rows[i].cells[0].textContent === "lm_head") table.deleteRow(i);
        }
        const built = await buildGraph("lm_head", (b) => buildLmHeadGraph(b, { weightMode: "constant", weightProvider: lmHeadProvider, options: graphOptions }), "time-lm_head");
        lmHeadGraph = built.graph; lmHeadComputeType = computeType; sharedBuildMs += built.buildMs - lmHeadBuildMs; lmHeadBuildMs = built.buildMs;
    };
    await ensureLmHead();

    // weights: one set of MLTensors per layer, shared by the prefill and decode graphs
    const layerWeights = [];
    const specByName = Object.fromEntries([...attentionWeightSpecs(), ...feedForwardWeightSpecs()].map((spec) => [spec[0], spec]));
    const projectionNames = ["query", "key", "value", "output_projection", "gate_projection", "up_projection", "down_projection"];
    // ---- build-time dequantization (weightFormat=float16 only) ----
    const mergeBuffers = {};
    let dequantizeMs = 0, dequantizeOnHost = dequantizeAt === "host", mergeCopyMs = 0, mergeUploadMs = 0, mergeCreateMs = 0;   // mergeProjections: JS concatenation / tensor creation + upload
    const dequantizeGraphs = new Map(), temporaries = [];
    let syncBuffer = null;
    async function dequantizeGraphFor(inputChannels, outputChannels) {
        const key = `${inputChannels}x${outputChannels}`;
        if (!dequantizeGraphs.has(key)) {
            const builder = new MLGraphBuilder(mlContext), start = performance.now();
            dequantizeGraphs.set(key, await builder.build(buildDequantizeWeightGraph(builder, { inputChannels, outputChannels }).outputs));
            log("graph_built", `Built graph ${("dequantize_" + key).padEnd(30)} ${seconds(performance.now() - start)}`, { graph: "dequantize_" + key, build_ms: Math.round(performance.now() - start) });
        }
        return dequantizeGraphs.get(key);
    }
    // packed int4 [groups, outputChannels, 128] (low nibble first) -> f16 [outputChannels, groups * 128]
    function dequantizeOnHostToF16(packed, scales, groups, outputChannels) {
        const inputChannels = groups * 128, output = new Float16Array(outputChannels * inputChannels);
        for (let group = 0; group < groups; group++) for (let channel = 0; channel < outputChannels; channel++) {
            const scale = scales[group * outputChannels + channel], base = (group * outputChannels + channel) * 128, outBase = channel * inputChannels + group * 128;
            for (let j = 0; j < 128; j++) {
                const element = base + j, nibble = (packed[element >> 1] >> ((element & 1) * 4)) & 15;
                output[outBase + j] = (nibble >= 8 ? nibble - 16 : nibble) * scale;
            }
        }
        return output;
    }
    stepStart = performance.now();
    for (let layer = 0; layer < layerCount; layer++) {
        setStatus(`loading weights: layer ${layer + 1}/${layerCount}`);
        await downloads.get(manifest.layers[layer].file);                              // wait for exactly this layer (later layers keep downloading)
        const bytes = await readCachedFile(manifest.layers[layer].file), tensors = {}, quantized = {}, partsToMerge = {}, uploads = [];   // uploads: { name, dataType, shape, data } of this layer, created concurrently below
        for (const entry of manifest.layers[layer].tensors) {
            if (mergeProjections && /^(query|key|value|gate_projection|up_projection)_(weight|scale)$/.test(entry.name)) { partsToMerge[entry.name] = entry; continue; }   // merged below
            if (weightFormat === "float16" && (entry.name.endsWith("_weight") || entry.name.endsWith("_scale"))) { quantized[entry.name] = entry; continue; }
            const spec = specByName[entry.name];
            const gammaFromF32 = spec && spec[1] === "float16" && entry.dtype === "float32";   // gammas: the weight file holds f32, the graphs take f16
            if (!spec || JSON.stringify(spec[2]) !== JSON.stringify(entry.shape) || (spec[1] !== entry.dtype && !gammaFromF32)) throw new Error(`weight mismatch for ${entry.name}: manifest ${entry.dtype} ${entry.shape}`);
            uploads.push({ name: entry.name, dataType: spec[1], shape: entry.shape,
                           data: gammaFromF32 ? Float16Array.from(new Float32Array(bytes.slice(entry.offset, entry.offset + entry.bytes))) : new Uint8Array(bytes, entry.offset, entry.bytes) });
        }
        if (mergeProjections) {   // int4 weights [groups, N, 128] and f16 scales [groups, N, 1]: concatenate the parts along N, group by group
            for (const [mergedName, parts] of [["qkv_projection", ["query", "key", "value"]], ["gate_up_projection", ["gate_projection", "up_projection"]]]) {
                const groups = partsToMerge[`${parts[0]}_weight`].shape[0], rowCounts = parts.map((part) => partsToMerge[`${part}_weight`].shape[1]), totalRows = rowCounts.reduce((a, b) => a + b, 0);
                for (const [kind, bytesPerRow, dataType, trailing] of [["weight", 64, "int4", 128], ["scale", 2, "float16", 1]]) {   // 128 int4 values = 64 bytes per (group, row); one f16 scale = 2 bytes
                    const copyStart = performance.now();
                    const mergedKey = `${mergedName}_${kind}`, mergedBytes = groups * totalRows * bytesPerRow;
                    const merged = (mergeBuffers[mergedKey] ??= new Uint8Array(mergedBytes));   // one buffer per merged tensor kind, reused for every layer (writeTensor copies the data when it is called)
                    for (let group = 0; group < groups; group++) {
                        let row = 0;
                        parts.forEach((part, index) => {
                            const entry = partsToMerge[`${part}_${kind}`];
                            merged.set(new Uint8Array(bytes, entry.offset + group * rowCounts[index] * bytesPerRow, rowCounts[index] * bytesPerRow), (group * totalRows + row) * bytesPerRow);
                            row += rowCounts[index];
                        });
                    }
                    mergeCopyMs += performance.now() - copyStart;
                    uploads.push({ name: mergedKey, dataType, shape: [groups, totalRows, trailing], data: merged });
                }
            }
        }
        {   // create all tensors of the layer at once (one round trip instead of one per tensor), then write them
            const uploadStart = performance.now();
            const created = await Promise.all(uploads.map((upload) => createWritableTensor(upload.dataType, upload.shape)));
            const writeStart = performance.now(); mergeCreateMs += writeStart - uploadStart;
            uploads.forEach((upload, index) => { tensors[upload.name] = created[index]; mlContext.writeTensor(created[index], upload.data); });
            mergeUploadMs += performance.now() - uploadStart;
        }
        if (weightFormat === "float16") {
            const dequantizeStart = performance.now();
            for (const name of projectionNames) {
                const weightEntry = quantized[`${name}_weight`], scaleEntry = quantized[`${name}_scale`];
                const [groups, outputChannels] = weightEntry.shape, inputChannels = groups * 128;
                const packed = new Uint8Array(bytes, weightEntry.offset, weightEntry.bytes), scaleBytes = new Uint8Array(bytes, scaleEntry.offset, scaleEntry.bytes);
                if (!dequantizeOnHost) {
                    try { await dequantizeGraphFor(inputChannels, outputChannels); }
                    catch (error) { log("dequantize_fallback", `Dequantization graph failed to build (${String(error).slice(0, 120)}); dequantizing on the host instead`); dequantizeOnHost = true; }
                }
                if (dequantizeOnHost) {
                    tensors[`${name}_weight`] = await createWritableTensor("float16", [outputChannels, inputChannels]);
                    mlContext.writeTensor(tensors[`${name}_weight`], dequantizeOnHostToF16(packed, new Float16Array(scaleBytes.slice().buffer), groups, outputChannels));
                } else {
                    const weightTensor = await createWritableTensor("int4", weightEntry.shape), scaleTensor = await createWritableTensor("float16", scaleEntry.shape);
                    mlContext.writeTensor(weightTensor, packed); mlContext.writeTensor(scaleTensor, scaleBytes);
                    const output = await createTensor("float16", [outputChannels, inputChannels], { readable: name === "down_projection" });   // the last projection is read back as the sync point
                    mlContext.dispatch(await dequantizeGraphFor(inputChannels, outputChannels), { weight: weightTensor, scale: scaleTensor }, { weight_f16: output });
                    tensors[`${name}_weight`] = output; temporaries.push(weightTensor, scaleTensor);
                }
            }
            if (!dequantizeOnHost) {                 // wait until this layer's dequantization has really run, then release the int4 temporaries
                syncBuffer = syncBuffer ?? new Float16Array(downProjectionElementCount);
                await mlContext.readTensor(tensors.down_projection_weight, syncBuffer);
                temporaries.splice(0).forEach((tensor) => tensor.destroy?.());
            }
            dequantizeMs += performance.now() - dequantizeStart;
        }
        layerWeights.push(tensors);
    }
    const finalNormGamma = await createWritableTensor("float16", [1, 1, hiddenSize]);
    mlContext.writeTensor(finalNormGamma, Float16Array.from(new Float32Array(finalNormBytes.slice().buffer)));   // the weight file holds f32 gammas
    weightLoadMs += performance.now() - stepStart;
    log("weights_loaded", `Loaded weights of ${layerCount} layers (${seconds(performance.now() - stepStart)})` +
        (weightFormat === "float16" ? `; build-time int4 -> f16 dequantization ${seconds(dequantizeMs)} (${dequantizeOnHost ? "on the host" : "on the device"})` : ""),
        { dequantize_ms: Math.round(dequantizeMs), dequantize_on_host: dequantizeOnHost, tensor_copy_ms: Math.round(mergeCopyMs), tensor_create_and_upload_ms: Math.round(mergeUploadMs), tensor_create_ms: Math.round(mergeCreateMs) });
    if (weightFormat === "float16") setCell("sum-dequantize", seconds(dequantizeMs), `${dequantizeOnHost ? "on the host" : "on the device, output kept as MLTensors"}; included in "Load weights"`);
    else setCell("sum-dequantize", "not used", "int4 weights stay int4 and are dequantized inside the graphs");
    const pickWeights = (layer, specs) => Object.fromEntries(specs.map(([name]) => [name, layerWeights[layer][name]]));
    const attentionWeights = (layer) => pickWeights(layer, attentionWeightSpecs());
    const feedForwardWeights = (layer) => pickWeights(layer, feedForwardWeightSpecs());

    const logitsTensor = await createTensor("float16", [1, 1, vocabularySize], { readable: true }), logits = new Float16Array(vocabularySize);

    // Activation tensors of one phase: everything the chain passes between graphs (all stay on the device)
    const createActivations = async (sequenceLength, totalMaskLength, maskDataType) => {
        const state = { sequenceLength };
        state.hiddenStates = await createWritableTensor("float16", [1, sequenceLength, hiddenSize]);
        state.ropeCos = await createWritableTensor("float16", [1, 1, sequenceLength, headDim]);
        state.ropeSin = await createWritableTensor("float16", [1, 1, sequenceLength, headDim]);
        state.attentionMask = await createWritableTensor(maskDataType, [1, 1, sequenceLength, totalMaskLength]);
        state.residuals = []; state.attentionOutputs = []; state.presentRows = [];
        for (let layer = 0; layer < layerCount; layer++) {
            state.residuals.push(await createTensor("float16", [1, sequenceLength, hiddenSize], { readable: profileEnabled }));
            state.attentionOutputs.push(await createTensor("float16", [1, sequenceLength, attentionWidth]));
            state.presentRows.push(fuseKvUpdate ? {} : { present_key: await createTensor("float16", [1, numKeyValueHeads, sequenceLength, headDim]),
                                     present_value: await createTensor("float16", naturalValueLayout ? [1, numKeyValueHeads, sequenceLength, headDim] : [1, numKeyValueHeads, headDim, sequenceLength]) });
        }
        state.finalHiddenStates = await createTensor("float16", [1, 1, hiddenSize], { readable: profileEnabled });
        if (tensorScatterKv) state.kvWriteIndex = await createWritableTensor(scatterIndexType, [1]);   // first cache row this dispatch writes
        if (scatterKv) state.kvIndices = await createWritableTensor("int32", [1, numKeyValueHeads, sequenceLength, 3]);   // (0, kv head, cache row) of every new row
        if (sequenceLength > 1 && prefillGuard) {   // sinks of the attention guard outputs (one per block of a group)
            const guardShape = sequenceLength > 1 ? [1, numAttentionHeads, totalMaskLength, 1] : [1, numKeyValueHeads, headDim, 1];   // prefill guard: max over the expanded values; staged decode guard: max over the un-expanded value cache
            state.attentionGuard = await createTensor("float16", guardShape);
            state.attentionGuards = [];
            for (let i = 0; i < prefillBlocksPerGraph; i++) state.attentionGuards.push(await createTensor("float16", guardShape));
        }
        state.tensors = () => [state.hiddenStates, state.ropeCos, state.ropeSin, state.attentionMask, state.finalHiddenStates,
            ...(state.attentionGuard ? [state.attentionGuard, ...state.attentionGuards] : []), ...state.residuals, ...state.attentionOutputs, ...state.presentRows.flatMap((rows) => [rows.present_key, rows.present_value]).filter(Boolean)];
        return state;
    };

    // -------------------------------------------------------------------------------------------------------------------
    // A "session" = everything that depends on the max length (KV cache capacity) and the prefill chunk size.
    // Only one session is kept alive at a time.
    // -------------------------------------------------------------------------------------------------------------------
    let currentSession = null;
    async function buildSession(maxLength, requestedChunkSizes) {
        const chunkSizes = [...new Set(requestedChunkSizes)].sort((a, b) => a - b), chunkSize = chunkSizes[0];   // chunkSize: the smallest one (the active set is switched per chunk)
        // cache rows: the concat forms (one-hot, staged) attend over cache + 1 new row, so the cache has maxLength - 1 rows; the in-cache forms (attendInCache, scatterND) attend over the
        // cache itself, so it has maxLength rows: the attention total length is then maxLength (a multiple of 64 for the usual 512 / 1024 / 1152 / 2048 / 4096; NPU decode attention
        // returned NaN for totals that are not, see 05 notes 27.4 / 27.14)
        const pastLength = attendInCache ? maxLength : maxLength - 1, session = { maxLength, chunkSize, chunkSizes, chunkKey: chunkSizes.join(","), optionKey: sessionOptionKey(), pastLength };
        if (maxLength % 64 !== 0 && deviceType === "npu" && !stagedKv) log("warning", `max length ${maxLength} is not a multiple of 64: on the NPU the decode attention (S=1) returned NaN for such total lengths in our tests (see 05 notes, chapter 27); kvStyle=staged (default, natural value layout) is not affected`);
        if (pastLength < chunkSizes[chunkSizes.length - 1]) throw new Error(`max length ${maxLength} is too small for prefill chunk size ${chunkSizes[chunkSizes.length - 1]}`);
        byId("pastLengthLabel").textContent = `${pastLength}`; byId("chunkSizeLabel").textContent = chunkSizes.join(" / ");
        let sessionBuildMs = 0;
        const label = (name, size = null) => `${name} (max ${maxLength}${size ? `, chunk ${size}` : ""})`;
        // prefill chunk graphs: S = chunk size, past = the whole KV cache (rows not written yet are masked); f32 mask as in the NPUW dump
        // attentionGuard "values": the NPU compiler returned WRONG attention results for S > 1 with a past cache unless the expanded value tensor has a
        // second (tiny) consumer; that extra output keeps it from fusing the value path into the attention matmul (see 05 notes, chapter 17)
        const prefillOptions = prefillGuard ? { ...graphOptions, attentionGuard: prefillGuard } : graphOptions;
        const decodeSettings = { sequenceLength: 1, pastLength, maskDataType: "float16", options: graphOptions };
        session.prefillSets = new Map(); session.decodeGraphs = {};   // prefillSets: chunk size -> { size, graphs, activations, selectors }
        const phaseEntries = [...chunkSizes.map((size) => {
            const set = { size, graphs: {} }; session.prefillSets.set(size, set);
            return ["prefill", { sequenceLength: size, pastLength, maskDataType: "float16", options: prefillOptions }, set.graphs, size];
        }), ["decode", decodeSettings, session.decodeGraphs, null]];
        for (const [phase, settings, graphs, size] of phaseEntries) {
            for (const [name, createGraph] of [["head", buildHeadGraph], ["block", buildBlockGraph], ["tail", buildTailGraph]]) {
                const groupSize = groupSizeOf(phase);
                if (name === "block" && groupSize > 1) {   // groups of K blocks (+ one smaller group for the remainder)
                    const totalBlocks = layerCount - 1, remainder = totalBlocks % groupSize;
                    graphs.blockGroups = {};
                    for (const count of [groupSize, ...(remainder ? [remainder] : [])]) {
                        if (count > totalBlocks) continue;
                        const built = await buildGraph(label(`${phase}_block_x${count}`, size), (b) => buildBlockGroupGraph(b, { ...settings, blockCount: count }), `time-${phase}-block`);
                        graphs.blockGroups[count] = built.graph; sessionBuildMs += built.buildMs;
                    }
                    continue;
                }
                const built = await buildGraph(label(`${phase}_${name}`, size), (b) => createGraph(b, settings), `time-${phase}-${name}`);
                graphs[name] = built.graph; sessionBuildMs += built.buildMs;
            }
            if (!fuseKvUpdate) {   // fused: the head/block graphs do the update themselves
                const kvUpdate = await buildGraph(label(`${phase}_kv_update`, size), (b) => buildKvUpdateGraph(b, { sequenceLength: settings.sequenceLength, pastLength }), `time-${phase}-kv_update`);
                graphs.kvUpdate = kvUpdate.graph; sessionBuildMs += kvUpdate.buildMs;
            }
        }
        if (stagedKv) {   // flush graph: main cache <- the staging rows (same math as the prefill chunk update with S = stagingRows)
            const flush = await buildGraph(label("kv_flush"), (b) => buildKvUpdateGraph(b, { sequenceLength: stagingRows, pastLength, valueLayout: naturalValueLayout ? "natural" : "transposed" }), "time-decode-kv_update");
            session.flushGraph = flush.graph; sessionBuildMs += flush.buildMs;
        }
        session.buildMs = sessionBuildMs;

        // device state: KV cache (double buffered per layer, the update graph cannot write in place) + activations of both phases
        session.keyValueCache = []; session.keyValueCacheSpare = [];
        for (let layer = 0; layer < layerCount; layer++) {
            const create = async () => ({
                key: await createWritableTensor("float16", [1, numKeyValueHeads, pastLength, headDim]),
                value: await createWritableTensor("float16", naturalValueLayout ? [1, numKeyValueHeads, pastLength, headDim] : [1, numKeyValueHeads, headDim, pastLength]),   // value cache: transposed (default) or natural
            });
            const first = await create(), second = keyValueInPlace || keyValueNoInput ? first : await create();   // in place / no cache input: the "spare" cache IS the cache
            if (!(keyValueNoInput && !keyValueZeroInit)) for (const tensor of new Set([first.key, first.value, second.key, second.value])) mlContext.writeTensor(tensor, new Float16Array(numKeyValueHeads * pastLength * headDim));   // kvNoInput + kvZeroInit=0: the cache is never written by the host, prefill fills it
            session.keyValueCache.push(first); session.keyValueCacheSpare.push(second);
        }
        const attentionLength = (newRows) => (attendInCache ? pastLength : pastLength + newRows);   // attendInCache: attention reads the updated cache itself (length P); else concat(cache, new rows)
        const createSelectors = async (sequenceLength) => ({
            rowsToKeep: await createWritableTensor("float16", [1, 1, pastLength, 1]), rowSelector: await createWritableTensor("float16", [1, 1, pastLength, sequenceLength]) });
        for (const set of session.prefillSets.values()) { set.activations = await createActivations(set.size, attentionLength(set.size), "float16"); set.selectors = await createSelectors(set.size); }
        session.decodeActivations = await createActivations(1, stagedKv ? pastLength + stagingRows : attentionLength(1), "float16");
        session.decodeSelectors = await createSelectors(1);
        session.activatePrefill = (size) => {   // the prefill code reads session.prefillGraphs / prefillActivations / prefillSelectors / chunkSize: point them at the set of this chunk size
            const set = session.prefillSets.get(size);
            session.prefillGraphs = set.graphs; session.prefillActivations = set.activations; session.prefillSelectors = set.selectors; session.chunkSize = size;
        };
        session.activatePrefill(chunkSize);
        if (stagedKv) {   // staging buffers (double buffered per layer like the main cache) and their selectors; the flush graph has its own selectors
            const createStaging = async () => ({ key: await createWritableTensor("float16", [1, numKeyValueHeads, stagingRows, headDim]),
                                                 value: await createWritableTensor("float16", naturalValueLayout ? [1, numKeyValueHeads, stagingRows, headDim] : [1, numKeyValueHeads, headDim, stagingRows]) });
            session.stagingCache = []; session.stagingSpare = [];
            for (let layer = 0; layer < layerCount; layer++) {
                const first = await createStaging(), second = await createStaging();
                for (const tensor of [first.key, first.value, second.key, second.value]) mlContext.writeTensor(tensor, new Float16Array(numKeyValueHeads * headDim * stagingRows));
                session.stagingCache.push(first); session.stagingSpare.push(second);
            }
            session.stagingSelectors = { rowsToKeep: await createWritableTensor("float16", [1, 1, stagingRows, 1]), rowSelector: await createWritableTensor("float16", [1, 1, stagingRows, 1]) };
            session.flushSelectors = await createSelectors(stagingRows);
            session.staging = { base: 0, filled: 0 };   // main cache holds positions < base; the staging buffer holds base .. base+filled-1
            session.flushCount = 0;
        }
        log("session_ready", `Session ready: max length ${maxLength}, prefill chunk ${chunkSizes.join("+")}; build ${seconds(sessionBuildMs)}, KV cache ${Math.round(keyValueCacheMegabytes(maxLength))} MB`,
            { max_length: maxLength, prefill_chunk_size: chunkSize, prefill_chunk_sizes: chunkSizes, session_build_ms: Math.round(sessionBuildMs), kv_cache_mb: Math.round(keyValueCacheMegabytes(maxLength)) });
        return session;
    }
    // Release everything one session created: its MLTensors (KV cache, staging buffers, activations, selectors, ...) and, where the browser has MLGraph.destroy(), its compiled graphs
    // (otherwise they go with the garbage collector). Weights, the embedding table and the lm_head graph belong to the page, not to a session.
    const releaseSession = (tensors, graphs, reason = "") => {
        tensors.forEach((tensor) => tensor.destroy?.());
        let destroyedGraphs = 0;
        for (const graph of graphs) if (typeof graph.destroy === "function") { graph.destroy(); destroyedGraphs++; }
        const hasGraphDestroy = typeof MLGraph !== "undefined" && "destroy" in MLGraph.prototype;
        log("session_released", `Session released${reason ? " " + reason : ""}: ${tensors.length} tensors destroyed, ${hasGraphDestroy ? `${destroyedGraphs} graphs destroyed` : `${graphs.length} graphs left to the garbage collector (this browser has no MLGraph.destroy())`}`);
    };
    async function createSession(maxLength, requestedChunkSizes) {
        await ensureLmHead();   // not recorded as part of the session
        const tensors = [], graphs = [];
        const table = byId("buildTable");
        for (let i = table.rows.length - 1; i >= 1; i--) if (table.rows[i].cells[0].textContent !== "lm_head") table.deleteRow(i);   // the table lists the graphs of the current session (and the shared lm_head)
        tensorSink = tensors; graphSink = graphs;
        try {
            const session = await buildSession(maxLength, requestedChunkSizes);
            session.destroy = () => releaseSession(tensors, graphs);
            return session;
        } catch (error) {
            releaseSession(tensors, graphs, "after a failed build");   // a failed build must not leave tensors / graphs behind
            throw error;
        } finally { tensorSink = null; graphSink = null; }
    }
    // requested: the session options picked on the page (see applySessionOptions); a different option set rebuilds the session (graphs + KV cache), the weights stay
    async function getSession(maxLength, chunkSize, requested) {
        applySessionOptions(requested);
        const chunkSizes = prefillChunkSizesParam.length ? prefillChunkSizesParam : [chunkSize];
        if (currentSession && currentSession.maxLength === maxLength && currentSession.optionKey === sessionOptionKey() && currentSession.chunkKey === [...new Set(chunkSizes)].sort((a, b) => a - b).join(",")) return currentSession;
        currentSession?.destroy(); currentSession = null;
        currentSession = await createSession(maxLength, chunkSizes);
        return currentSession;
    }
    // Which chunk sizes to use for a prompt of `tokenCount` tokens: every chunk but the last is full, the last one is left-padded. Minimises sum(prefillFixedTokens + size); ties: fewer chunks,
    // bigger chunks first (so the padded one is the small one).
    function planPrefill(session, tokenCount) {
        const sizes = [...session.chunkSizes].sort((a, b) => b - a), memo = new Map();
        const best = (remaining) => {
            if (memo.has(remaining)) return memo.get(remaining);
            let result = null;
            for (const size of sizes) {
                const cost = prefillFixedTokens + size;
                const candidate = size >= remaining ? { cost, plan: [size] } : (() => { const rest = best(remaining - size); return { cost: cost + rest.cost, plan: [size, ...rest.plan] }; })();
                if (!result || candidate.cost < result.cost || (candidate.cost === result.cost && candidate.plan.length < result.plan.length)) result = candidate;
            }
            memo.set(remaining, result); return result;
        };
        return best(tokenCount).plan;
    }

    // -------------------------------------------------------------------------------------------------------------------
    // One forward pass: enqueue every subgraph of the chain (all host inputs of the phase are already written).
    // `lastChunk` false (non-final prefill chunk): only the KV cache matters, so TAIL and LM HEAD are skipped.
    // -------------------------------------------------------------------------------------------------------------------
    function enqueueForwardPass(session, phase, lastChunk = true, parts = null) {   // parts: { head, blocks, tail, lmHead } = run only these segments (profile=1); null = all
        const run = (name) => !parts || parts[name];
        const isPrefill = phase === "prefill";
        const graphs = isPrefill ? session.prefillGraphs : session.decodeGraphs;
        const state = isPrefill ? session.prefillActivations : session.decodeActivations, selectors = isPrefill ? session.prefillSelectors : session.decodeSelectors;
        const dispatch = (graph, inputs, outputs) => mlContext.dispatch(graph, inputs, outputs);
        // fused: the attention half writes the updated cache to the spare buffers (extra inputs/outputs of head/block); afterwards only swap
        const stagedDecode = stagedKv && !isPrefill;   // decode with a staging buffer: the fused update writes the STAGING buffer, the main cache is only read
        const updatedCaches = stagedDecode ? session.stagingSpare : session.keyValueCacheSpare, currentCaches = stagedDecode ? session.stagingCache : session.keyValueCache;
        const stagingOfLayer = (layer) => (stagedDecode ? { staging_key: session.stagingCache[layer].key, staging_value: session.stagingCache[layer].value } : {});
        const fusedInputs = !fuseKvUpdate ? {} : stagedDecode ? { staging_rows_to_keep: session.stagingSelectors.rowsToKeep, staging_row_selector: session.stagingSelectors.rowSelector } : tensorScatterKv ? { kv_write_index: state.kvWriteIndex } : scatterKv ? { kv_indices: state.kvIndices } : { ...(kvStyle === "select" && !isPrefill ? {} : { rows_to_keep: selectors.rowsToKeep }), row_selector: selectors.rowSelector };   // decode "select" graphs have no rows_to_keep input
        const fusedOutputs = (layer) => (fuseKvUpdate ? { updated_key: updatedCaches[layer].key, updated_value: updatedCaches[layer].value } : {});
        const updateKeyValueCache = (layer) => {
            if (fuseKvUpdate) { [currentCaches[layer], updatedCaches[layer]] = [updatedCaches[layer], currentCaches[layer]]; return; }
            dispatch(graphs.kvUpdate, {
                past_key: session.keyValueCache[layer].key, past_value: session.keyValueCache[layer].value,
                new_key: state.presentRows[layer].present_key, new_value: state.presentRows[layer].present_value,
                rows_to_keep: selectors.rowsToKeep, row_selector: selectors.rowSelector,
            }, { updated_key: session.keyValueCacheSpare[layer].key, updated_value: session.keyValueCacheSpare[layer].value });
            [session.keyValueCache[layer], session.keyValueCacheSpare[layer]] = [session.keyValueCacheSpare[layer], session.keyValueCache[layer]];
        };
        const pastOfLayer = (layer) => (keyValueNoInput ? {} : { past_key: session.keyValueCache[layer].key, past_value: session.keyValueCache[layer].value });   // kvNoInput: the graphs have no cache inputs
        const shared = { rope_cos: state.ropeCos, rope_sin: state.ropeSin, attention_mask: state.attentionMask };
        const guardOutput = state.attentionGuard ? { attention_guard: state.attentionGuard } : {};

        // HEAD: embedding -> attention half of layer 0
        if (run("head")) dispatch(graphs.head, { hidden_states: state.hiddenStates, ...shared, ...pastOfLayer(0), ...stagingOfLayer(0), ...fusedInputs, ...attentionWeights(0) },
            { residual: state.residuals[0], attention_output: state.attentionOutputs[0], ...state.presentRows[0], ...guardOutput, ...fusedOutputs(0) });
        if (run("head")) updateKeyValueCache(0);
        // BLOCK i: feed-forward half of layer i + attention half of layer i+1
        const groupSize = groupSizeOf(phase);
        if (!run("blocks")) { /* skipped */ } else if (groupSize > 1) {
            for (let start = 0; start < layerCount - 1; start += groupSize) {
                const count = Math.min(groupSize, layerCount - 1 - start), inputs = { residual: state.residuals[start], attention_output: state.attentionOutputs[start], ...shared, ...fusedInputs };
                const outputs = { next_residual: state.residuals[start + count], next_attention_output: state.attentionOutputs[start + count] };
                for (let j = 0; j < count; j++) {   // block j of the group: feed-forward half of layer start+j, attention half of layer start+j+1
                    const layer = start + j, prefix = `b${j}_`, attentionLayer = layer + 1;
                    for (const [name, tensor] of Object.entries({ ...feedForwardWeights(layer), ...attentionWeights(attentionLayer), ...pastOfLayer(attentionLayer), ...stagingOfLayer(attentionLayer) })) inputs[prefix + name] = tensor;
                    if (!fuseKvUpdate) { outputs[prefix + "present_key"] = state.presentRows[attentionLayer].present_key; outputs[prefix + "present_value"] = state.presentRows[attentionLayer].present_value; }
                    if (state.attentionGuards) outputs[prefix + "attention_guard"] = state.attentionGuards[j];
                    if (fuseKvUpdate) { outputs[prefix + "updated_key"] = updatedCaches[attentionLayer].key; outputs[prefix + "updated_value"] = updatedCaches[attentionLayer].value; }
                }
                dispatch(graphs.blockGroups[count], inputs, outputs);
                for (let j = 0; j < count; j++) updateKeyValueCache(start + j + 1);
            }
        } else
        for (let layer = 0; layer < layerCount - 1; layer++) {
            dispatch(graphs.block,
                { residual: state.residuals[layer], attention_output: state.attentionOutputs[layer], ...shared, ...pastOfLayer(layer + 1), ...stagingOfLayer(layer + 1), ...fusedInputs, ...feedForwardWeights(layer), ...attentionWeights(layer + 1) },
                { next_residual: state.residuals[layer + 1], next_attention_output: state.attentionOutputs[layer + 1], ...state.presentRows[layer + 1], ...guardOutput, ...fusedOutputs(layer + 1) });
            updateKeyValueCache(layer + 1);
        }
        if (!lastChunk) return;
        // TAIL: feed-forward half of the last layer + final norm; LM HEAD: logits
        const last = layerCount - 1;
        if (run("tail")) dispatch(graphs.tail, { residual: state.residuals[last], attention_output: state.attentionOutputs[last], ...feedForwardWeights(last), final_norm_gamma: finalNormGamma },
            { final_hidden_states: state.finalHiddenStates });
        if (run("lmHead")) dispatch(lmHeadGraph, { final_hidden_states: state.finalHiddenStates }, { logits: logitsTensor });
    }
    const argmaxOfLogits = () => { let best = 0, bestValue = -Infinity; for (let i = 0; i < vocabularySize; i++) if (logits[i] > bestValue) { bestValue = logits[i]; best = i; } return best; };

    // -------------------------------------------------------------------------------------------------------------------
    // Prefill and decode: write the host inputs, run the chain, (read the logits)
    // -------------------------------------------------------------------------------------------------------------------
    async function runPhase(session, phase, writeHostInputs, { lastChunk = true } = {}) {
        const timing = {};
        let t = performance.now();
        highlight("stage-embedding", "stage-inputs");
        writeHostInputs();
        timing.hostPrepareMs = performance.now() - t; t = performance.now();
        highlight(`phase-${phase}`, `${phase}-head`, `${phase}-block`, `${phase}-tail`, `${phase}-lm_head`, `${phase}-kv_update`);
        enqueueForwardPass(session, phase, lastChunk);
        timing.enqueueMs = performance.now() - t; t = performance.now();
        if (!lastChunk) return { token: null, timing };        // non-final chunk: keep queueing, the next chunk's inputs are ordered after this work
        highlight(`phase-${phase}`, "stage-readback");
        await mlContext.readTensor(logitsTensor, logits);        // waits for the whole device chain
        timing.deviceWaitMs = performance.now() - t; t = performance.now();
        highlight("stage-sampling");
        const token = argmaxOfLogits();
        timing.samplingMs = performance.now() - t;
        return { token, timing };
    }

    // Chunk layout: chunks 0..k-2 are full (chunkSize tokens each, in order); the last chunk holds the remaining tokens
    // RIGHT-aligned (left padded) so that its last row is the last prompt token.
    //   chunk rows:  [ pad ... pad | token ... token ]           padCount = chunkSize - validCount
    //   attention mask row r (columns = [cache rows 0..pastLength-1 | chunk rows 0..chunkSize-1]):
    //       pad row   : sees only itself
    //       real row  : sees the cache rows written by earlier chunks (columns < startPosition) and the real rows of this chunk up to itself
    //   KV update   : cache row startPosition+i receives chunk row padCount+i  (i < validCount)
    // kvStyle=tensorScatter writes the S rows of a chunk CONTIGUOUSLY from one start row, so the left padding cannot sit in the last chunk (it would overwrite the rows of the
    // chunk before). Instead the whole prompt is padded at its FRONT: with n chunks, pad = sum(chunk sizes) - prompt length rows of padding come first, then the prompt:
    //   cache row of prompt token t = pad + t          (the pad rows sit in cache rows 0..pad-1 and are masked for every real row; RoPE uses the true position t)
    // Every chunk is full, the last chunk ends with the last prompt token (the TAIL graph still takes the last row), and decode writes row pad + position.
    function writePrefillChunkInputsTensorScatter(session, tokenIds, startPosition) {   // startPosition = first cache row of this chunk
        const { chunkSize, pastLength } = session, state = session.prefillActivations, pad = session.promptPad;
        const hiddenStates = new Float16Array(chunkSize * hiddenSize), cos = new Float16Array(chunkSize * headDim), sin = new Float16Array(chunkSize * headDim);
        const mask = new Float16Array(chunkSize * pastLength).fill(-65504);
        const firstReal = Math.max(0, pad - startPosition), firstToken = Math.max(0, startPosition - pad);   // first real row of this chunk, and its prompt token index
        const rope = ropeRows(firstToken, chunkSize - firstReal);
        for (let row = 0; row < chunkSize; row++) {
            const cacheRow = startPosition + row, token = cacheRow - pad, base = row * pastLength, index = Math.max(0, row - firstReal);
            cos.set(rope.cos.subarray(index * headDim, (index + 1) * headDim), row * headDim);
            sin.set(rope.sin.subarray(index * headDim, (index + 1) * headDim), row * headDim);
            if (token >= 0) { lookupEmbeddingRow(tokenIds[token], hiddenStates, row * hiddenSize); mask.fill(0, base + pad, base + cacheRow + 1); }   // real row: the real rows up to itself
            else mask[base + cacheRow] = 0;                                                                                                       // pad row: only itself (keeps the softmax finite)
        }
        mlContext.writeTensor(state.hiddenStates, hiddenStates); mlContext.writeTensor(state.ropeCos, cos); mlContext.writeTensor(state.ropeSin, sin);
        mlContext.writeTensor(state.attentionMask, mask); mlContext.writeTensor(state.kvWriteIndex, writeIndexData(startPosition));
    }
    function writePrefillChunkInputs(session, tokenIds, startPosition) {
        if (tensorScatterKv) return writePrefillChunkInputsTensorScatter(session, tokenIds, startPosition);
        const { chunkSize, pastLength } = session, state = session.prefillActivations;   // chunkSize: the size of the ACTIVE prefill set
        const validCount = Math.min(chunkSize, tokenIds.length - startPosition), padCount = chunkSize - validCount;
        const totalLength = attendInCache ? pastLength : pastLength + chunkSize;
        const hiddenStates = new Float16Array(chunkSize * hiddenSize);
        for (let i = 0; i < validCount; i++) lookupEmbeddingRow(tokenIds[startPosition + i], hiddenStates, (padCount + i) * hiddenSize);
        const rope = ropeRows(startPosition, validCount), cos = new Float16Array(chunkSize * headDim), sin = new Float16Array(chunkSize * headDim);
        for (let row = 0; row < chunkSize; row++) {                 // padded rows reuse the first real position
            const index = Math.max(0, row - padCount);
            cos.set(rope.cos.subarray(index * headDim, (index + 1) * headDim), row * headDim);
            sin.set(rope.sin.subarray(index * headDim, (index + 1) * headDim), row * headDim);
        }
        const mask = new Float16Array(chunkSize * totalLength).fill(-65504);
        for (let row = 0; row < chunkSize; row++) {
            const base = row * totalLength;
            if (attendInCache) {   // columns are cache rows only: the chunk's rows were written into the cache before attention
                if (row < padCount) mask[base + (scatterKv ? Math.min(pastLength - 1, startPosition + validCount + row) : 0)] = 0;   // pad row: sees only the cache row it is parked in (one-hot: pad rows are not stored, it sees row 0 just to keep the softmax finite)
                else mask.fill(0, base, base + startPosition + (row - padCount) + 1);                               // real row: earlier chunks + this chunk's real rows up to itself
                continue;
            }
            if (row < padCount) { mask[base + pastLength + row] = 0; continue; }
            mask.fill(0, base, base + startPosition);                                        // cache rows from earlier chunks
            mask.fill(0, base + pastLength + padCount, base + pastLength + row + 1);         // real rows of this chunk up to the row itself
        }
        const rowsToKeep = new Float16Array(pastLength).fill(1), rowSelector = new Float16Array(pastLength * chunkSize);
        for (let i = 0; i < validCount; i++) { rowsToKeep[startPosition + i] = 0; rowSelector[(startPosition + i) * chunkSize + padCount + i] = 1; }
        mlContext.writeTensor(state.hiddenStates, hiddenStates); mlContext.writeTensor(state.ropeCos, cos); mlContext.writeTensor(state.ropeSin, sin);
        mlContext.writeTensor(state.attentionMask, mask);
        if (scatterKv) {   // real rows go to cache rows startPosition.., pad rows are parked behind them (those rows are masked / overwritten later)
            const indices = new Int32Array(numKeyValueHeads * chunkSize * 3);
            for (let h = 0; h < numKeyValueHeads; h++) for (let row = 0; row < chunkSize; row++) {
                const cacheRow = row < padCount ? Math.min(pastLength - 1, startPosition + validCount + row) : startPosition + (row - padCount);
                const at = (h * chunkSize + row) * 3; indices[at] = 0; indices[at + 1] = h; indices[at + 2] = cacheRow;
            }
            mlContext.writeTensor(state.kvIndices, indices);
        } else {
            mlContext.writeTensor(session.prefillSelectors.rowsToKeep, rowsToKeep); mlContext.writeTensor(session.prefillSelectors.rowSelector, rowSelector);
        }
    }
    function writeDecodeInputs(session, tokenId, position) {
        const state = session.decodeActivations, { pastLength } = session, totalLength = stagedKv ? pastLength + stagingRows : attendInCache ? pastLength : pastLength + 1;
        const hiddenStates = new Float16Array(hiddenSize); lookupEmbeddingRow(tokenId, hiddenStates, 0);
        const rope = ropeRows(position, 1);
        const mask = new Float16Array(totalLength).fill(-65504);
        if (stagedKv) {   // columns = [main cache rows 0..P-1 | staging rows 0..R-1]: main rows below `base` and the staging rows up to (and including) the new token
            const { base, filled } = session.staging;
            mask.fill(0, 0, base); mask.fill(0, pastLength, pastLength + filled + 1);
            const stagingKeep = new Float16Array(stagingRows).fill(1), stagingSelector = new Float16Array(stagingRows);
            stagingKeep[filled] = 0; stagingSelector[filled] = 1;
            mlContext.writeTensor(session.stagingSelectors.rowsToKeep, stagingKeep); mlContext.writeTensor(session.stagingSelectors.rowSelector, stagingSelector);
        } else if (attendInCache) mask.fill(0, tensorScatterKv ? session.promptPad : 0, (tensorScatterKv ? session.promptPad : 0) + position + 1);           // cache rows 0..position (the new token was written into row `position`)
        else { mask.fill(0, 0, position); mask[pastLength] = 0; }          // valid past rows + the new token (last column)
        const rowsToKeep = new Float16Array(pastLength).fill(1), rowSelector = new Float16Array(pastLength);
        rowsToKeep[position] = 0; rowSelector[position] = 1;                                                   // the new token's row goes to cache row `position`
        mlContext.writeTensor(state.hiddenStates, hiddenStates); mlContext.writeTensor(state.ropeCos, rope.cos); mlContext.writeTensor(state.ropeSin, rope.sin);
        mlContext.writeTensor(state.attentionMask, mask);
        if (stagedKv) {
            // nothing else: the update of the staging buffer uses the staging selectors written above
        } else if (tensorScatterKv) {
            mlContext.writeTensor(state.kvWriteIndex, writeIndexData(session.promptPad + position));   // cache row of the new token
        } else if (scatterKv) {
            const indices = new Int32Array(numKeyValueHeads * 3);
            for (let h = 0; h < numKeyValueHeads; h++) { indices[h * 3] = 0; indices[h * 3 + 1] = h; indices[h * 3 + 2] = position; }
            mlContext.writeTensor(state.kvIndices, indices);
        } else {
            mlContext.writeTensor(session.decodeSelectors.rowsToKeep, rowsToKeep); mlContext.writeTensor(session.decodeSelectors.rowSelector, rowSelector);
        }
    }

    // staged KV: when the staging buffer is full, move its rows into the main cache (one flush dispatch per layer, enqueued like everything else)
    function flushStaging(session) {
        const { base } = session.staging, rows = session.pastLength;
        const rowsToKeep = new Float16Array(rows).fill(1), rowSelector = new Float16Array(rows * stagingRows);
        for (let i = 0; i < stagingRows && base + i < rows; i++) { rowsToKeep[base + i] = 0; rowSelector[(base + i) * stagingRows + i] = 1; }   // staging row i -> main cache row base+i
        mlContext.writeTensor(session.flushSelectors.rowsToKeep, rowsToKeep); mlContext.writeTensor(session.flushSelectors.rowSelector, rowSelector);
        for (let layer = 0; layer < layerCount; layer++) {
            mlContext.dispatch(session.flushGraph, {
                past_key: session.keyValueCache[layer].key, past_value: session.keyValueCache[layer].value,
                new_key: session.stagingCache[layer].key, new_value: session.stagingCache[layer].value,
                rows_to_keep: session.flushSelectors.rowsToKeep, row_selector: session.flushSelectors.rowSelector,
            }, { updated_key: session.keyValueCacheSpare[layer].key, updated_value: session.keyValueCacheSpare[layer].value });
            [session.keyValueCache[layer], session.keyValueCacheSpare[layer]] = [session.keyValueCacheSpare[layer], session.keyValueCache[layer]];
        }
        session.staging.base += stagingRows; session.staging.filled = 0; session.flushCount++;
    }

    // -------------------------------------------------------------------------------------------------------------------
    // Generation
    // -------------------------------------------------------------------------------------------------------------------
    const blockDispatchCount = () => (blocksPerGraph > 1 ? Math.ceil((layerCount - 1) / blocksPerGraph) : layerCount - 1);
    const dispatchCountPerStep = () => 1 + blockDispatchCount() + 1 + 1 + (fuseKvUpdate ? 0 : layerCount);   // head + blocks + tail + lm_head (+ KV updates)
    // profile=1: where does a decode step go? Re-runs the step the next token would take (same inputs, results are discarded; the real generation is over) in two ways:
    //  (a) segment by segment, each followed by a readTensor sync: head | blocks | tail | lm_head  -> JS enqueue time and device wait of every segment
    //  (b) as one chain with a single sync at the end, exactly like a normal step
    // plus the round trip of a readTensor on an idle device. sum(a) - (b) is what the three extra syncs cost.
    async function profileDecode(session, token, position) {
        setStatus("profiling decode segments");
        if (stagedKv && session.staging.filled >= stagingRows) flushStaging(session);
        const decodeState = session.decodeActivations, last = layerCount - 1, hiddenBuffer = new Float16Array(hiddenSize);
        const segments = [
            ["HEAD", { head: true }, decodeState.residuals[0], hiddenBuffer, "layer 0 attention half: 1 dispatch"],
            [`BLOCKs (${blockDispatchCount()} dispatches)`, { blocks: true }, decodeState.residuals[last], hiddenBuffer, "FFN of layer i + attention of layer i+1, 35 times"],
            ["TAIL", { tail: true }, decodeState.finalHiddenStates, hiddenBuffer, "last FFN + final RMSNorm"],
            ["LM HEAD + read logits", { lmHead: true }, logitsTensor, logits, "389 MB weights; the sync reads 151936 logits"],
        ];
        const totals = segments.map(() => ({ enqueue: 0, wait: 0 })), full = { host: 0, enqueue: 0, wait: 0 }, idle = { wait: 0 };
        const warmup = 3, runs = 20;
        for (let i = 0; i < warmup + runs; i++) {
            const measure = i >= warmup;
            let start = performance.now();
            writeDecodeInputs(session, token, position);
            if (measure) full.host += performance.now() - start;
            for (let k = 0; k < segments.length; k++) {      // (a) every segment followed by a sync
                const [, parts, target, buffer] = segments[k];
                start = performance.now(); enqueueForwardPass(session, "decode", true, parts); const enqueued = performance.now();
                await mlContext.readTensor(target, buffer); const done = performance.now();
                if (measure) { totals[k].enqueue += enqueued - start; totals[k].wait += done - enqueued; }
            }
            start = performance.now(); await mlContext.readTensor(logitsTensor, logits);     // the device is idle: this is the pure round trip
            if (measure) idle.wait += performance.now() - start;
            start = performance.now(); enqueueForwardPass(session, "decode", true); const enqueued = performance.now();   // (b) one chain, one sync
            await mlContext.readTensor(logitsTensor, logits); const done = performance.now();
            if (measure) { full.enqueue += enqueued - start; full.wait += done - enqueued; }
        }
        const mean = (value) => rounded(value / runs);
        const rows = segments.map(([name, , , , tip], k) => [name, mean(totals[k].enqueue), mean(totals[k].wait), mean(totals[k].enqueue + totals[k].wait), tip, "segment"]);
        const sumTotal = totals.reduce((sum, t) => sum + t.enqueue + t.wait, 0), fullTotal = full.enqueue + full.wait;
        rows.push(["Sum of the 4 segments (4 syncs)", mean(totals.reduce((sum, t) => sum + t.enqueue, 0)), mean(totals.reduce((sum, t) => sum + t.wait, 0)), mean(sumTotal), "each segment waits for the device before the next one is queued", "sum"]);
        rows.push(["Whole chain, 1 sync (= a normal step)", mean(full.enqueue), mean(full.wait), mean(fullTotal), "all dispatches queued back to back, one readTensor at the end; host input writing is not included", "full"]);
        rows.push(["Cost of the 3 extra syncs", "", "", mean(sumTotal - fullTotal), "sum of segments minus the whole chain", "diff"]);
        rows.push(["Idle readTensor round trip", "", mean(idle.wait), mean(idle.wait), "readTensor of the logits when nothing is queued: the floor of every sync", "idle"]);
        rows.push(["Host: write step inputs", "", "", mean(full.host), "embedding lookup + writing rope / mask / selectors", "host"]);
        const table = byId("profileTable");
        while (table.rows.length > 1) table.deleteRow(1);
        for (const [name, enqueue, wait, total, tip, kind] of rows) {
            const row = table.insertRow(); row.className = `p-${kind}`;
            const nameCell = row.insertCell(); nameCell.textContent = name; nameCell.dataset.tip = tip; nameCell.classList.add("has-tip");
            for (const value of [enqueue, wait, total]) { const cell = row.insertCell(); cell.textContent = value; cell.className = "num"; }
        }
        byId("profileCard").style.display = "";
        log("decode_profile", `Decode profile (${runs} runs): ` + rows.map(([name, , , total]) => `${name} ${total} ms`).join("; "),
            { runs, segments: rows.map(([name, enqueue, wait, total]) => ({ name, enqueue_ms: enqueue, wait_ms: wait, total_ms: total })) });
    }
    async function generate(session, userText, maxNewTokens) {
        const prompt = `<|im_start|>user\n${userText}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;   // Qwen3, thinking disabled
        highlight("stage-tokenizer");
        const tokenIds = Array.from(tokenizer.encode(prompt));
        if (tokenIds.length >= session.pastLength) throw new Error(`prompt has ${tokenIds.length} tokens but max length is ${session.maxLength}`);
        const plan = planPrefill(session, tokenIds.length), chunkCount = plan.length, chunkDescription = plan.join("+");
        const planRows = plan.reduce((sum, size) => sum + size, 0);
        session.promptPad = tensorScatterKv ? planRows - tokenIds.length : 0;   // tensorScatter: padding rows in FRONT of the prompt (see writePrefillChunkInputsTensorScatter)
        if (tensorScatterKv && planRows > session.pastLength) throw new Error(`the prefill chunks (${planRows} rows) do not fit into the ${session.pastLength}-row cache`);
        byId("answer").textContent = ""; byId("summaryLine").textContent = `prompt: ${tokenIds.length} tokens …`;
        log("generation_started", `Generation started: max length ${session.maxLength}, ${tokenIds.length} prompt tokens in ${chunkCount} chunk(s) (${chunkDescription}), up to ${maxNewTokens} new tokens`,
            { max_length: session.maxLength, prompt_tokens: tokenIds.length, prefill_chunks: chunkCount, prefill_plan: plan });
        const generationStart = performance.now();

        // prefill: chunk by chunk; only the last chunk reads back logits
        let prefill = null;
        const chunkEnqueueMs = [];
        let chunkStart = 0;
        for (let chunk = 0; chunk < chunkCount; chunk++) {
            setStatus(`prefill chunk ${chunk + 1}/${chunkCount}`);
            const lastChunk = chunk === chunkCount - 1, startPosition = chunkStart;
            session.activatePrefill(plan[chunk]); chunkStart += plan[chunk];
            const result = await runPhase(session, "prefill", () => writePrefillChunkInputs(session, tokenIds, startPosition), { lastChunk });
            chunkEnqueueMs.push(result.timing.hostPrepareMs + result.timing.enqueueMs);
            if (lastChunk) prefill = result;
        }
        const prefillMs = performance.now() - generationStart;
        log("prefill_finished", `Prefill finished in ${seconds(prefillMs)} (time to first token): ${chunkCount} chunk(s), last chunk host=${rounded(prefill.timing.hostPrepareMs)}ms enqueue=${rounded(prefill.timing.enqueueMs)}ms device wait=${rounded(prefill.timing.deviceWaitMs)}ms`,
            { max_length: session.maxLength, prefill_chunks: chunkCount, prefill_ms: Math.round(prefillMs), ...prefill.timing });

        const generated = [], stepTimings = [];
        let token = prefill.token, position = tokenIds.length;
        if (stagedKv) { session.staging.base = tokenIds.length; session.staging.filled = 0; session.flushCount = 0; }   // the prompt is in the main cache, the staging buffer is empty
        for (let step = 0; step < maxNewTokens && !END_OF_TEXT_TOKENS.has(token) && session.promptPad + position < session.pastLength; step++) {
            generated.push(token);
            byId("answer").textContent = tokenizer.decode(generated, { skip_special_tokens: true });
            setStatus(`decode step ${step + 1}/${maxNewTokens}`);
            const stepStart = performance.now();
            if (stagedKv && session.staging.filled === stagingRows) flushStaging(session);   // counted in this step's time
            const result = await runPhase(session, "decode", () => writeDecodeInputs(session, token, position));
            token = result.token; position++;
            if (stagedKv) session.staging.filled++;
            result.timing.totalMs = performance.now() - stepStart;
            stepTimings.push(result.timing);
            if (step < 3 || (step + 1) % 16 === 0) log("decode_step", `Decode step ${step + 1}: ${rounded(result.timing.totalMs)} ms (host ${rounded(result.timing.hostPrepareMs)}, enqueue ${rounded(result.timing.enqueueMs)}, device wait ${rounded(result.timing.deviceWaitMs)}, sampling ${rounded(result.timing.samplingMs)})`, { max_length: session.maxLength, step: step + 1, ...result.timing });
        }
        if (!END_OF_TEXT_TOKENS.has(token)) generated.push(token);
        const answer = tokenizer.decode(generated, { skip_special_tokens: true });
        byId("answer").textContent = answer;
        highlight();

        const inferenceMs = performance.now() - generationStart;
        const average = (key) => (stepTimings.length ? stepTimings.reduce((sum, timing) => sum + timing[key], 0) / stepTimings.length : 0);
        const timePerOutputTokenMs = average("totalMs"), tokensPerSecond = timePerOutputTokenMs ? 1000 / timePerOutputTokenMs : 0;
        const blockDispatches = blockDispatchCount();
        setCell("sum-prefill", seconds(prefillMs), `${tokenIds.length} prompt tokens in ${chunkCount} chunk(s): ${chunkDescription}`);
        setCell("sum-decode", `${rounded(timePerOutputTokenMs)} ms/token`, `${rounded(tokensPerSecond)} tokens/s over ${stepTimings.length} decode steps`);
        setCell("sum-inference", seconds(inferenceMs));
        setMetric("metric-ttft", (prefillMs / 1000).toFixed(2), "s", `${tokenIds.length} prompt tokens · ${chunkCount} chunk${chunkCount > 1 ? "s" : ""} (${chunkDescription})`);
        setMetric("metric-tpot", rounded(timePerOutputTokenMs), "ms/token", `${stepTimings.length} decode steps · ${dispatchCountPerStep()} dispatches per step`);
        setMetric("metric-tps", rounded(tokensPerSecond), "tokens/s", `KV ${kvStyle}${stagedKv ? ` ${stagingRows}` : ""} · blocks/graph ${blocksPerGraph}${prefillBlocksPerGraph !== blocksPerGraph ? ` (prefill ${prefillBlocksPerGraph})` : ""}`);
        byId("summaryLine").textContent = `max length ${session.maxLength} · prompt ${tokenIds.length} tokens · ${generated.length} generated · total inference ${seconds(inferenceMs)}`;
        fillTimingTable([
            ["Host: inputs", rounded(average("hostPrepareMs")), "JavaScript work before the dispatches: embedding lookup + writing the step inputs", "host"],
            ["Enqueue", rounded(average("enqueueMs")), `${dispatchCountPerStep()} dispatches (head, ${blockDispatches} block graph(s)${blocksPerGraph > 1 ? ` of up to ${blocksPerGraph} blocks` : ""}, tail, lm_head${fuseKvUpdate ? ", KV update fused" : `, ${layerCount} KV updates`}); only queued here`, "enqueue"],
            ["Device wait", rounded(average("deviceWaitMs")), "read logits: time until the device finished the whole chain", "device"],
            ["Host: argmax", rounded(average("samplingMs")), "greedy sampling over 151936 logits", "sample"],
            ["Total = TPOT", rounded(timePerOutputTokenMs), "time per generated token", "total"],
        ]);
        log("generation_finished", `Generated ${generated.length} tokens: TTFT ${seconds(prefillMs)}, TPOT ${rounded(timePerOutputTokenMs)} ms/token (${rounded(tokensPerSecond)} tokens/s)`,
            { max_length: session.maxLength, prefill_chunk_size: session.chunkSize, prefill_plan: plan, prompt_tokens: tokenIds.length, prefill_chunks: chunkCount, ttft_ms: Math.round(prefillMs),
              tpot_ms: rounded(timePerOutputTokenMs), tokens_per_second: rounded(tokensPerSecond), decode_steps: stepTimings.length, inference_ms: Math.round(inferenceMs),
              generated_token_ids: generated, text: answer, ...(stagedKv ? { kv_flushes: session.flushCount, staging_rows: stagingRows } : {}) });
        if (profileEnabled && session.promptPad + position < session.pastLength) await profileDecode(session, token, position);
        setStatus("ready");
    }

    byId("sum-load").textContent = seconds(weightLoadMs);
    log("shared_ready", `Shared part ready in ${seconds(performance.now() - pageStartTime)}: load weights ${seconds(weightLoadMs)} + lm_head graph ${seconds(sharedBuildMs)} + other`,
        { load_weights_ms: Math.round(weightLoadMs), shared_build_ms: Math.round(sharedBuildMs) });
    const countPromptTokens = (userText) => tokenizer.encode(`<|im_start|>user
${userText}<|im_end|>
<|im_start|>assistant
<think>

</think>

`).length;
    return { getSession, currentSession: () => currentSession, generate, get sharedBuildMs() { return sharedBuildMs; }, countPromptTokens };
}

// ---------------------------------------------------------------------------------------------------------------------
// Page flow: nothing is loaded until the user picks a max length / chunk size and presses "Load model".
// ---------------------------------------------------------------------------------------------------------------------
async function run() {
    const maxLengthSelect = byId("maxLengthSelect"), chunkSizeSelect = byId("chunkSizeSelect");
    for (const length of [512, 1024, 2048, 4096, defaultMaxLength].sort((a, b) => a - b)) {
        if (![...maxLengthSelect.options].some((option) => +option.value === length)) maxLengthSelect.add(new Option("", length));
    }
    for (const option of maxLengthSelect.options) option.textContent = `${option.value} tokens  (KV cache ${Math.round(keyValueCacheMegabytes(+option.value))} MB)`;
    for (const size of [32, 64, 128, 256, 512, defaultChunkSize].sort((a, b) => a - b)) {
        if (![...chunkSizeSelect.options].some((option) => +option.value === size)) chunkSizeSelect.add(new Option(`${size} tokens`, size));
    }
    maxLengthSelect.value = defaultMaxLength; chunkSizeSelect.value = defaultChunkSize;
    // session options (applied when the session is (re)built, i.e. at "Load model" / the next "Generate"): initial values come from the URL, changes are written back to the URL
    const deviceSelect = byId("deviceSelect");
    for (const device of ["npu", "gpu", "cpu"]) deviceSelect.add(new Option(device.toUpperCase(), device));
    deviceSelect.value = deviceType;
    deviceSelect.addEventListener("change", () => { syncUrl(); location.reload(); });   // a different device needs a new WebNN context and new weight tensors: simplest and safest is a fresh page
    // model: the built-in hand-written Qwen3-4B graphs (all options) or a package written by `m2w compile` (see chat-package.js); a different model reloads the page
    const modelSelect = byId("modelSelect");
    const packages = (await listPackages()).filter((entry) => !entry.dev || query.get("dev") === "1" || entry.id === modelId);
    modelSelect.add(new Option("Qwen3-4B · built-in JS graphs (all options)", "js"));
    for (const entry of packages) modelSelect.add(new Option(entry.label, entry.id));
    if (![...modelSelect.options].some((option) => option.value === modelId)) modelSelect.add(new Option(`${modelId} (not found in packages/models.json)`, modelId));
    modelSelect.value = modelId;
    modelSelect.addEventListener("change", () => {
        const url = new URL(location.href);
        url.searchParams.set("model", modelSelect.value);
        for (const key of ["maxLength", "prefillChunkSize", "blocksPerGraph", "prefillBlocksPerGraph", "prefillChunkSizes", "kvStyle", "stagingRows", "computeType", "pastLength", "P", "chunk", "bpg", "pbpg", "chunks", "dt"]) url.searchParams.delete(key);
        location.href = url;
    });
    const packageEntry = modelId === "js" ? null : packages.find((entry) => entry.id === modelId);
    const packageMode = !!packageEntry;
    const computeTypeSelect = byId("computeTypeSelect"), kvStyleSelect = byId("kvStyleSelect"), stagingRowsSelect = byId("stagingRowsSelect"), blocksSelect = byId("blocksPerGraphSelect"), prefillBlocksSelect = byId("prefillBlocksPerGraphSelect"), chunkSizesInput = byId("chunkSizesInput");
    const addChoice = (select, value, text = String(value)) => { if (![...select.options].some((option) => option.value === String(value))) select.add(new Option(text, value)); };
    const kvStyleLabels = { staged: "staged (default)", onehot: "one-hot (whole cache every step)", scatterND: "scatterND (in-graph)", tensorScatter: "tensorScatter (in-graph)" };
    for (const style of ["staged", "onehot", "scatterND", "tensorScatter"]) addChoice(kvStyleSelect, style, kvStyleLabels[style]);
    for (const rows of [32, 64, 128]) addChoice(stagingRowsSelect, rows);
    for (const count of [1, 2, 3, 5, 7, 35]) { addChoice(blocksSelect, count); addChoice(prefillBlocksSelect, count); }
    addChoice(kvStyleSelect, kvStyle); addChoice(stagingRowsSelect, urlSessionOptions.stagingRows); addChoice(blocksSelect, urlSessionOptions.blocksPerGraph);
    if (urlSessionOptions.prefillBlocksPerGraph) addChoice(prefillBlocksSelect, urlSessionOptions.prefillBlocksPerGraph);
    for (const [value, text] of [["float32", "float32 (default)"], ["float16", "float16"]]) addChoice(computeTypeSelect, value, text);
    computeTypeSelect.value = computeType;
    kvStyleSelect.value = kvStyle; stagingRowsSelect.value = urlSessionOptions.stagingRows; blocksSelect.value = urlSessionOptions.blocksPerGraph;
    prefillBlocksSelect.value = urlSessionOptions.prefillBlocksPerGraph || 0; chunkSizesInput.value = urlSessionOptions.chunkSizes.join(",");
    let packageControls = null;
    if (packageMode) {   // the options of a package are what it was compiled with: fill the selects from its plan and lock the rest
        const peek = await peekPackage(packageEntry.url);
        packageControls = setupPackageControls(peek, { maxLengthSelect, chunkSizeSelect, blocksSelect, prefillBlocksSelect, kvStyleSelect, stagingRowsSelect, computeTypeSelect, chunkSizesInput },
            { kvStyle: query.get("kvStyle"), computeType: query.get("computeType") ?? query.get("dt"), stagingRows: query.get("stagingRows"), blocksPerGraph: query.get("blocksPerGraph") ?? query.get("bpg"), prefillBlocksPerGraph: query.get("prefillBlocksPerGraph") ?? query.get("pbpg") });
        const want = (select, value) => { if (value && [...select.options].some((option) => option.value === String(value))) select.value = value; };
        want(maxLengthSelect, query.get("maxLength")); maxLengthSelect.dispatchEvent(new Event("change"));
        want(chunkSizeSelect, query.get("prefillChunkSize") ?? query.get("chunk"));
        document.querySelector("header h1").childNodes[0].textContent = `${packageEntry.label} on WebNN `;
        document.querySelector("header h1 small").textContent = `model package · compiled by m2w (${peek.index.compiler}) · target ${peek.index.target}`;
    }
    const updateOptionControls = () => {
        if (packageMode) { packageControls.lock(); return; }
        stagingRowsSelect.disabled = kvStyleSelect.value !== "staged" || !fuseKvUpdate; kvStyleSelect.disabled = !fuseKvUpdate;
    };
    byId("fixedOptions").textContent = packageMode ? `Package: ${packageEntry.url}  (the options come from the package plan; the built-in JS graphs offer the full option set: choose "Qwen3-4B · built-in JS graphs")` : `Fixed at page load (URL): profile=${profileEnabled ? 1 : 0}  weightFormat=${weightFormat}  mergeProjections=${mergeProjections ? 1 : 0}  fuseKvUpdate=${fuseKvUpdate ? 1 : 0}  matmulStyle=${graphOptions.matmulStyle}  zeroPoint=${graphOptions.zeroPoint}  dequantStyle=${graphOptions.dequantStyle}  layers=${layerCount}`;
    byId("maxNewTokensInput").value = defaultMaxNewTokens;
    setStatus("choose max length and prefill chunk size, then press \"Load model\"");

    // prompt presets of different length (the paragraph is repeated to reach roughly the stated number of prompt tokens)
    const paragraph = "The history of computing stretches from mechanical calculators to modern neural processors. Early machines filled whole rooms and executed a few " +
        "operations per second, while today a thin laptop carries a dedicated accelerator that performs trillions of low-precision operations every second. " +
        "Engineers keep moving work from general purpose processors to specialised units because moving data costs far more energy than computing on it. ";
    const repeated = (count) => "Summarize the following text in two sentences.\n\n" + Array(count).fill(paragraph).join("\n");
    const presets = [
        ["Default: short poem about the sea", "Write a short poem about the sea."],
        ["Short (~25 tokens): capital of France", "What is the capital of France?"],
        ["Medium (~250 tokens)", repeated(3)], ["Long (~500 tokens)", repeated(6)], ["Long (~900 tokens)", repeated(11)],
    ];
    const presetSelect = byId("promptPreset");
    presets.forEach(([name], index) => presetSelect.add(new Option(name, index)));
    presetSelect.add(new Option("Custom (edited text / ?prompt=… in the URL)", "custom"));
    presetSelect.onchange = () => { if (presetSelect.value !== "custom") byId("prompt").value = presets[+presetSelect.value][1]; updateTokenCount(); };
    if (query.get("prompt")) { byId("prompt").value = query.get("prompt"); presetSelect.value = "custom"; }
    else { presetSelect.value = 0; byId("prompt").value = presets[0][1]; }
    const updateTokenCount = () => { byId("promptTokenCount").textContent = model ? `${model.countPromptTokens(byId("prompt").value)} prompt tokens (with chat template)` : "(token count shown after the model is loaded)"; };
    byId("prompt").oninput = () => { const preset = presets[+presetSelect.value]; if (!preset || preset[1] !== byId("prompt").value) presetSelect.value = "custom"; updateTokenCount(); };

    let model = null;
    updateTokenCount();
    const sessionOptionsOnPage = () => ({ kvStyle: kvStyleSelect.value, stagingRows: +stagingRowsSelect.value, blocksPerGraph: +blocksSelect.value, prefillBlocksPerGraph: +prefillBlocksSelect.value, chunkSizes: parseChunkSizes(chunkSizesInput.value), computeType: computeTypeSelect.value });
    const selected = () => ({ maxLength: +maxLengthSelect.value, chunkSize: +chunkSizeSelect.value, maxNewTokens: +byId("maxNewTokensInput").value, options: sessionOptionsOnPage() });
    const optionControls = [maxLengthSelect, chunkSizeSelect, computeTypeSelect, kvStyleSelect, stagingRowsSelect, blocksSelect, prefillBlocksSelect, chunkSizesInput];
    const setBusy = (busy) => { for (const id of ["loadButton", "runButton"]) byId(id).disabled = busy; optionControls.forEach((control) => (control.disabled = busy)); if (!busy) updateOptionControls(); };
    // keep the URL in step with the page so that a link to it reproduces the setting (long names only)
    const syncUrl = () => {
        const url = new URL(location.href), { maxLength, chunkSize, options } = selected();
        for (const key of ["bpg", "pbpg", "chunk", "chunks", "P", "pastLength", "dt"]) url.searchParams.delete(key);
        const set = (key, value, keep) => (keep ? url.searchParams.set(key, value) : url.searchParams.delete(key));
        set("model", modelId, modelId !== "js"); set("device", deviceSelect.value, true); set("maxLength", maxLength, true); set("prefillChunkSize", chunkSize, true); set("kvStyle", options.kvStyle, true); set("stagingRows", options.stagingRows, options.kvStyle === "staged");
        set("blocksPerGraph", options.blocksPerGraph, options.blocksPerGraph > 1); set("prefillBlocksPerGraph", options.prefillBlocksPerGraph, options.prefillBlocksPerGraph > 0);
        set("prefillChunkSizes", options.chunkSizes.join(","), options.chunkSizes.length > 0); set("computeType", options.computeType, options.computeType !== "float32");
        history.replaceState(null, "", url);
    };
    optionControls.forEach((control) => control.addEventListener("change", () => { updateOptionControls(); syncUrl(); }));
    updateOptionControls();
    const loadModel = async () => {
        setBusy(true);
        try {
            const { maxLength, chunkSize, options } = selected();
            if (!packageMode) applySessionOptions(options);   // before the first load: lm_head is built with the compute type picked on the page
            const loadStart = performance.now(), firstLoad = !model, previousSession = model?.currentSession?.();   // startup time = this load only, not the time since the page was opened
            if (!model) model = packageMode ? await loadPackageModel({ byId, log, setStatus, setCell, setMetric, fillTimingTable, seconds, rounded }, { url: packageEntry.url, deviceType })
                : await loadSharedModel();             // first time: tokenizer, weights, lm_head graph
            const session = await model.getSession(maxLength, chunkSize, options);   // builds the max-length / chunk-size / session-option specific graphs + KV cache
            const startupMs = performance.now() - loadStart, reused = !firstLoad && session === previousSession;
            setCell("sum-build", seconds(model.sharedBuildMs + session.buildMs), `shared ${seconds(model.sharedBuildMs)} + max-length / chunk / option specific ${seconds(session.buildMs)}`);
            setCell("sum-startup", seconds(startupMs), firstLoad ? "tokenizer + load weights + build graphs" : reused ? "same settings: the session was already loaded" : "reload: weights were already loaded, only the session graphs were rebuilt");
            setMetric("metric-startup", (startupMs / 1000).toFixed(1), "s", firstLoad ? `incl. weights; graphs built in ${seconds(model.sharedBuildMs + session.buildMs)}` : reused ? "already loaded" : `reload, weights kept; graphs ${seconds(session.buildMs)}`);
            byId("loadedLabel").textContent = packageMode ? `loaded: ${session.label}` : `loaded: max length ${maxLength}, chunk ${session.chunkSizes.join("+")}, kv ${kvStyle}${stagedKv ? ` (${stagingRows} rows)` : ""}, ${computeType}, blocks/graph ${blocksPerGraph}${prefillBlocksPerGraph !== blocksPerGraph ? ` (prefill ${prefillBlocksPerGraph})` : ""}`;
            updateTokenCount();
            setStatus("ready");
        } catch (error) { log("fatal", "Load failed: " + String(error && error.stack ? error.stack : error).slice(0, 1500)); setStatus("FAILED: " + error); }
        finally { setBusy(false); byId("runButton").disabled = !model; }
    };
    const generateOnce = async () => {
        const { maxLength, chunkSize, maxNewTokens, options } = selected();
        await model.generate(await model.getSession(maxLength, chunkSize, options), byId("prompt").value, maxNewTokens);
    };
    byId("loadButton").onclick = loadModel;
    byId("clearCacheButton").onclick = async () => { await (packageMode ? clearPackageCache(packageEntry.url) : clearWeightCache()); log("cache_cleared", "Weight cache (OPFS) cleared"); };
    byId("runButton").onclick = async () => {
        setBusy(true);
        try { await generateOnce(); } catch (error) { log("fatal", "Generation failed: " + error); setStatus("FAILED: " + error); }
        finally { setBusy(false); }
    };
    if (autoRun) {
        await loadModel();
        if (model) { try { await generateOnce(); } catch (error) { log("fatal", "Generation failed: " + error); } }
        log("done", "Done");
    }
}
run().catch((error) => { log("fatal", "Failed: " + String(error && error.stack ? error.stack : error).slice(0, 1500)); setStatus("FAILED: " + error); });

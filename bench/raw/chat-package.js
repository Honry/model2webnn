// The "package" engine of chat.html: runs a model package written by `m2w compile` (python/ in model2webnn) with the model2webnn runtime (../../lib/runtime, built by `npm run build:lib`),
// behind the same interface the page already uses for the hand-written Qwen3 graphs:  { getSession, currentSession, generate, sharedBuildMs, countPromptTokens }.
// What a package offers is fixed when it is compiled (max lengths, prefill chunks, blocks per graph, compute type, staging rows): the page shows exactly that.
import { LLM, CachedSource, UrlSource, openPackage, buildPrompt, tokenizerFromTransformersJs } from "../../lib/runtime/index.js";

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js";
const packagesUrl = new URL("../packages/", import.meta.url);

/** [{ id, label, url, dev }] from packages/models.json (written by bench/tools/sync-packages.sh) */
export async function listPackages() {
    try {
        const response = await fetch(new URL("models.json", packagesUrl), { cache: "no-store" });
        if (!response.ok) return [];
        return (await response.json()).map((entry) => ({ ...entry, url: new URL(entry.url, packagesUrl).href }));
    } catch { return []; }
}
export const peekPackage = (url) => openPackage(url, true);
export const clearPackageCache = (url) => new CachedSource(new UrlSource(url)).clear();

const kvCacheMegabytes = (model, maxLength) => (model.num_layers * 2 * 2 * model.num_kv_heads * (maxLength - 1) * model.head_dim * 2) / 2 ** 20;

/** Fill and lock the option selects of the page from what the package contains. Returns { lock } to call whenever the page re-enables its controls. */
export function setupPackageControls(pkg, c) {
    const { model, plan } = pkg;
    const lengths = plan.max_lengths.slice().sort((a, b) => a - b);
    const chunksOf = (length) => Object.keys(plan.graphs.prefill[String(length)] ?? {}).map(Number).sort((a, b) => a - b);
    const replace = (select, entries) => { select.innerHTML = ""; for (const [value, text] of entries) select.add(new Option(text, value)); };
    replace(c.maxLengthSelect, lengths.map((l) => [l, `${l} tokens  (KV cache ${Math.round(kvCacheMegabytes(model, l))} MB)`]));
    const fillChunks = () => { const keep = c.chunkSizeSelect.value; replace(c.chunkSizeSelect, chunksOf(+c.maxLengthSelect.value).map((s) => [s, `${s} tokens`])); if ([...c.chunkSizeSelect.options].some((o) => o.value === keep)) c.chunkSizeSelect.value = keep; };
    fillChunks();
    c.maxLengthSelect.addEventListener("change", () => { fillChunks(); fillBlocks(); });
    const singleType = new Set(model.layer_types).size === 1;
    const fillBlocks = () => {
        const groups = Object.keys(plan.graphs.decode[c.maxLengthSelect.value]?.block_groups ?? {}).map(Number);
        const biggest = groups.length ? Math.max(...groups) : 0;
        replace(c.blocksSelect, [[1, "1"], ...(singleType && biggest > 1 ? [[biggest, String(biggest)]] : [])]);
    };
    fillBlocks();
    replace(c.kvStyleSelect, [["staged", "staged (the package's KV protocol)"]]);
    replace(c.stagingRowsSelect, [[plan.staging_rows, String(plan.staging_rows)]]);
    replace(c.computeTypeSelect, [[plan.profile.compute_type, plan.profile.compute_type + " (fixed when the package was compiled)"]]);
    replace(c.prefillBlocksSelect, [["0", "1 (single blocks)"]]);
    c.chunkSizesInput.value = ""; c.chunkSizesInput.placeholder = "not available for packages";
    const reasons = {
        kvStyleSelect: "A package carries the staged KV protocol only (one-hot / scatterND / tensorScatter are available with the built-in JS graphs).",
        stagingRowsSelect: "Fixed when the package was compiled (m2w compile --staging-rows).",
        computeTypeSelect: "Fixed when the package was compiled (m2w compile --compute-type).",
        prefillBlocksPerGraphSelect: "Prefill graphs of a package are single blocks.",
        chunkSizesInput: "Multi-size prefill planning is not implemented for packages yet.",
    };
    const fixed = [c.kvStyleSelect, c.stagingRowsSelect, c.computeTypeSelect, c.prefillBlocksSelect, c.chunkSizesInput];
    const lock = () => {
        fixed.forEach((control) => { control.disabled = true; control.title = reasons[control.id] ?? ""; });
        c.blocksSelect.disabled = c.blocksSelect.options.length < 2 || c.maxLengthSelect.disabled;
        if (!singleType) c.blocksSelect.title = "Blocks of one graph share a single mask and RoPE table: models with sliding + full attention layers run one block per graph.";
    };
    lock();
    return { lock };
}

/** the model of the page: tokenizer + (lazily, per max length / chunk / blocks-per-graph) a loaded LLM */
export async function loadPackageModel(ui, { url, deviceType }) {
    const { byId, log, setStatus, setCell, setMetric, fillTimingTable, seconds, rounded } = ui;
    setStatus("opening the package");
    const pkg = await openPackage(url, true);
    setStatus("loading the tokenizer");
    const lib = await import(TRANSFORMERS_URL);
    const tokenizer = await tokenizerFromTransformersJs(lib, pkg);
    log("tokenizer_loaded", "Tokenizer loaded (transformers.js, from the package)");
    const format = pkg.chatFormat, stop = new Set(pkg.model.eos_token_ids);
    const promptIds = (text) => tokenizer.encode(format ? buildPrompt(format, [{ user: text }]) : text);
    let llm = null, current = null;

    async function getSession(maxLength, chunkSize, options) {
        const blocksPerGraph = options.blocksPerGraph || 1, key = [maxLength, chunkSize, blocksPerGraph].join("|");
        if (current && current.key === key) return current;
        if (llm) { llm.destroy(); llm = null; current = null; }
        llm = await LLM.load(url, {
            device: deviceType, maxLength, chunkSize, blocksPerGraph, onProgress: (p) => {
                setStatus(p.stage === "download" ? `downloading weights ${Math.round(p.loaded / 2 ** 20)} / ${Math.round(p.total / 2 ** 20)} MB` : `${p.stage}${p.message ? ": " + p.message : ""}`);
            },
        });
        byId("pastLengthLabel").textContent = String(maxLength - 1); byId("chunkSizeLabel").textContent = String(chunkSize);
        current = { key, maxLength, chunkSize, chunkSizes: [chunkSize], blocksPerGraph, buildMs: llm.stats.buildMs, weightsMs: llm.stats.weightsMs, loadMs: llm.stats.loadMs,
            label: `${pkg.model.architecture}: max length ${maxLength}, chunk ${chunkSize}, kv staged (${pkg.plan.staging_rows} rows), ${pkg.plan.profile.compute_type}, blocks/graph ${blocksPerGraph}` };
        setCell("sum-load", seconds(llm.stats.weightsMs), "download (or OPFS cache) + upload of the weights to the device");
        setCell("sum-dequantize", "not used", "int4 weights stay int4 and are dequantized inside the graphs");
        log("session_ready", `Session ready: ${current.label}; load ${seconds(llm.stats.loadMs)} (weights ${seconds(llm.stats.weightsMs)}, graph build ${seconds(llm.stats.buildMs)})`,
            { max_length: maxLength, prefill_chunk_size: chunkSize, blocks_per_graph: blocksPerGraph, build_ms: Math.round(llm.stats.buildMs), weights_ms: Math.round(llm.stats.weightsMs), load_ms: Math.round(llm.stats.loadMs), package: url });
        return current;
    }

    async function generate(session, userText, maxNewTokens) {
        const ids = promptIds(userText);
        if (ids.length >= session.maxLength - 1) throw new Error(`prompt has ${ids.length} tokens but max length is ${session.maxLength}`);
        byId("answer").textContent = ""; byId("summaryLine").textContent = `prompt: ${ids.length} tokens …`;
        const chunks = Math.ceil(ids.length / session.chunkSize);
        log("generation_started", `Generation started: max length ${session.maxLength}, ${ids.length} prompt tokens in ${chunks} chunk(s), up to ${maxNewTokens} new tokens`, { max_length: session.maxLength, prompt_tokens: ids.length, prefill_chunks: chunks });
        setStatus("prefill");
        const start = performance.now(), generated = [];
        for await (const token of llm.generate(ids, { maxNewTokens })) {
            if (stop.has(token)) break;
            generated.push(token);
            byId("answer").textContent = tokenizer.decode(generated);
            setStatus(`decode step ${generated.length}/${maxNewTokens}`);
        }
        const inferenceMs = performance.now() - start, s = llm.stats, steps = s.steps;
        const avg = (key) => (steps.length ? steps.reduce((sum, step) => sum + step[key], 0) / steps.length : 0);
        const tpot = avg("totalMs"), tps = tpot ? 1000 / tpot : 0, prefillMs = s.prefillMs;
        const dispatches = 1 + Math.ceil((pkg.model.num_layers - 1) / session.blocksPerGraph) + 2;
        setCell("sum-build", seconds(session.buildMs), "graph build of this session (all graphs of the package variant)");
        setCell("sum-prefill", seconds(prefillMs), `${ids.length} prompt tokens in ${chunks} chunk(s)`);
        setCell("sum-decode", `${rounded(tpot)} ms/token`, `${rounded(tps)} tokens/s over ${steps.length} decode steps`);
        setCell("sum-inference", seconds(inferenceMs));
        setMetric("metric-ttft", (prefillMs / 1000).toFixed(2), "s", `${ids.length} prompt tokens · ${chunks} chunk${chunks > 1 ? "s" : ""}`);
        setMetric("metric-tpot", rounded(tpot), "ms/token", `${steps.length} decode steps · ${dispatches} dispatches per step`);
        setMetric("metric-tps", rounded(tps), "tokens/s", `KV staged ${pkg.plan.staging_rows} · blocks/graph ${session.blocksPerGraph} · ${s.flushes} flushes`);
        byId("summaryLine").textContent = `max length ${session.maxLength} · prompt ${ids.length} tokens · ${generated.length} generated · total inference ${seconds(inferenceMs)}`;
        fillTimingTable([
            ["Host: inputs", rounded(avg("hostMs")), "JavaScript work before the dispatches: writing the step inputs (embedding row, RoPE tables, masks, selectors)", "host"],
            ["Enqueue", rounded(avg("enqueueMs")), `${dispatches} dispatches queued per step (a flush of the staging buffer counts here when due)`, "enqueue"],
            ["Device wait", rounded(avg("waitMs")), "read logits: time until the device finished the whole chain", "device"],
            ["Host: argmax", rounded(avg("samplingMs")), `greedy sampling over ${pkg.model.vocab} logits`, "sample"],
            ["Total = TPOT", rounded(tpot), "time per generated token", "total"],
        ]);
        const answer = tokenizer.decode(generated);
        const first = s.stepMs.slice(0, 4).map(rounded).join(", ");
        log("generation_finished", `Generated ${generated.length} tokens: TTFT ${seconds(prefillMs)}, TPOT ${rounded(tpot)} ms/token (${rounded(tps)} tokens/s); first decode steps ms: ${first}`,
            { max_length: session.maxLength, prefill_chunk_size: session.chunkSize, prompt_tokens: ids.length, prefill_chunks: chunks, ttft_ms: Math.round(prefillMs), tpot_ms: rounded(tpot),
              tokens_per_second: rounded(tps), decode_steps: steps.length, inference_ms: Math.round(inferenceMs), generated_token_ids: generated, text: answer, kv_flushes: s.flushes, package: url });
        setStatus("ready");
    }

    return {
        getSession, currentSession: () => current, generate, sharedBuildMs: 0, pkg,
        countPromptTokens: (text) => promptIds(text).length,
    };
}

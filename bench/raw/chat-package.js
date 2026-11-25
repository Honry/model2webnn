// The "package" engine of chat.html: runs a model package written by `m2w compile` (python/ in model2webnn) with the model2webnn runtime (../../lib/runtime, built by `npm run build:lib`),
// behind the same interface the page already uses for the hand-written Qwen3 graphs:  { getSession, currentSession, generate, sharedBuildMs, countPromptTokens }.
// What a package offers is fixed when it is compiled (max lengths, prefill chunks, graph variants = KV protocol x compute type x staging rows, blocks per graph): the page offers exactly that,
// and a change of any session option rebuilds the graphs of the loaded model (LLM.reconfigure) while the weights stay on the device.
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

/** Fill and lock the option selects of the page from what the package contains. `initial`: the session options of the URL. Returns { lock } to call whenever the page re-enables its controls. */
export function setupPackageControls(pkg, c, initial = {}) {
    const { model, plan } = pkg;
    const variants = plan.variants, defaultVariant = variants.find((v) => v.id === plan.default_variant);
    const lengths = plan.max_lengths.slice().sort((a, b) => a - b);
    const chunksOf = (length) => Object.keys(plan.graphs[defaultVariant.id].prefill[String(length)] ?? {}).map(Number).sort((a, b) => a - b);
    const replace = (select, entries, keep) => { select.innerHTML = ""; for (const [value, text] of entries) select.add(new Option(text, value)); if (keep !== undefined && [...select.options].some((o) => o.value === String(keep))) select.value = keep; };
    replace(c.maxLengthSelect, lengths.map((l) => [l, `${l} tokens  (KV cache ${Math.round(kvCacheMegabytes(model, l))} MB)`]));
    const fillChunks = () => replace(c.chunkSizeSelect, chunksOf(+c.maxLengthSelect.value).map((s) => [s, `${s} tokens`]), c.chunkSizeSelect.value);
    fillChunks();
    const singleType = new Set(model.layer_types).size === 1 || plan.block_group_controls === "per_block";   // hybrid models: groups take a mask / RoPE table per block
    const groupsOf = (table) => Object.keys(table?.block_groups ?? {}).map(Number);
    const fillBlocks = () => {
        const length = c.maxLengthSelect.value, decode = groupsOf(plan.graphs[defaultVariant.id].decode[length]), biggest = decode.length ? Math.max(...decode) : 0;
        replace(c.blocksSelect, [[1, "1"], ...(singleType && biggest > 1 ? [[biggest, String(biggest)]] : [])], c.blocksSelect.value);
        const prefill = Object.values(plan.graphs[defaultVariant.id].prefill[length] ?? {}).map(groupsOf).reduce((common, list, i) => (i === 0 ? list : common.filter((n) => list.includes(n))), []);
        replace(c.prefillBlocksSelect, [["0", "same as decode"], ["1", "1 (single blocks)"], ...(singleType ? prefill.map((n) => [n, String(n)]) : [])], c.prefillBlocksSelect.value);
    };
    fillBlocks();
    // KV protocol x compute type x staging rows: only the combinations the package contains
    const kvLabels = { staged: "staged (staging buffer + flush)", onehot: "one-hot (whole cache every step)", tensorScatter: "tensorScatter (in-graph)" };
    const kvStyles = [...new Set(variants.map((v) => v.kv_style))];
    replace(c.kvStyleSelect, kvStyles.map((k) => [k, kvLabels[k] ?? k]), initial.kvStyle ?? defaultVariant.kv_style);
    const fillVariantControls = () => {
        const kv = c.kvStyleSelect.value, ofKv = variants.filter((v) => v.kv_style === kv);
        replace(c.computeTypeSelect, [...new Set(ofKv.map((v) => v.compute_type))].map((t) => [t, t]), c.computeTypeSelect.value || initial.computeType || defaultVariant.compute_type);
        replace(c.stagingRowsSelect, kv === "staged" ? [...new Set(ofKv.map((v) => v.staging_rows))].map((r) => [r, String(r)]) : [["0", "-"]], c.stagingRowsSelect.value || initial.stagingRows || defaultVariant.staging_rows);
    };
    fillVariantControls();
    c.kvStyleSelect.addEventListener("change", fillVariantControls);   // registered before the page's own listeners: they see the refilled selects
    c.maxLengthSelect.addEventListener("change", () => { fillChunks(); fillBlocks(); });
    if (initial.blocksPerGraph) replace(c.blocksSelect, [...c.blocksSelect.options].map((o) => [o.value, o.text]), initial.blocksPerGraph);
    if (initial.prefillBlocksPerGraph) replace(c.prefillBlocksSelect, [...c.prefillBlocksSelect.options].map((o) => [o.value, o.text]), initial.prefillBlocksPerGraph);
    c.chunkSizesInput.placeholder = `optional, e.g. ${plan.prefill_chunks.join(",")}`;
    const lock = () => {
        const only = (select) => select.options.length < 2;
        c.kvStyleSelect.disabled = only(c.kvStyleSelect); c.computeTypeSelect.disabled = only(c.computeTypeSelect);
        c.stagingRowsSelect.disabled = c.kvStyleSelect.value !== "staged" || only(c.stagingRowsSelect);
        c.blocksSelect.disabled = only(c.blocksSelect); c.prefillBlocksSelect.disabled = only(c.prefillBlocksSelect);
        if (!singleType) c.blocksSelect.title = c.prefillBlocksSelect.title = "Blocks of one graph share a single mask and RoPE table: models with sliding + full attention layers run one block per graph.";
        c.stagingRowsSelect.title = c.kvStyleSelect.value === "staged" ? "" : "Only the staged KV protocol has a staging buffer.";
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

    // a session = max length + prefill chunk sizes + graph variant (KV protocol, staging rows, compute type) + blocks per graph; the first one loads the model (weights go to the device once),
    // every other one is LLM.reconfigure(): graphs are rebuilt, the weights stay.
    async function getSession(maxLength, chunkSize, options) {
        const blocksPerGraph = options.blocksPerGraph || 1, chunkSizes = options.chunkSizes?.length ? options.chunkSizes : [chunkSize];
        const prefillGroups = Object.keys(pkg.plan.graphs[pkg.plan.default_variant].prefill[String(maxLength)]?.[String(chunkSizes[0])]?.block_groups ?? {}).map(Number);
        const prefillBlocksPerGraph = options.prefillBlocksPerGraph || (blocksPerGraph === 1 || prefillGroups.includes(blocksPerGraph) ? blocksPerGraph : 1);
        const kvStyle = options.kvStyle, computeType = options.computeType, stagingRows = kvStyle === "staged" ? options.stagingRows : undefined;
        const key = [maxLength, chunkSizes.join("+"), blocksPerGraph, prefillBlocksPerGraph, kvStyle, stagingRows, computeType].join("|");
        if (current && current.key === key) return current;
        const session = { maxLength, chunkSizes, blocksPerGraph, prefillBlocksPerGraph, kvStyle, computeType, ...(stagingRows ? { stagingRows } : {}), onProgress: (p) => {
            setStatus(p.stage === "download" ? `downloading weights ${Math.round(p.loaded / 2 ** 20)} / ${Math.round(p.total / 2 ** 20)} MB` : `${p.stage}${p.message ? ": " + p.message : ""}`);
        } };
        const reused = !!llm;
        if (llm) await llm.reconfigure(session);
        else {
            const query = new URLSearchParams(location.search);   // ?parallelDownloads=4&downloadSegments=4: see download.html for good values
            llm = await LLM.load(url, { device: deviceType, parallelDownloads: +query.get("parallelDownloads") || undefined, downloadSegments: +query.get("downloadSegments") || undefined, ...session });
        }
        const variant = llm.variant, sizes = llm.chunkSizes.join("+");
        byId("pastLengthLabel").textContent = String(maxLength - 1); byId("chunkSizeLabel").textContent = sizes;
        const kvText = variant.kv_style === "staged" ? `kv staged (${variant.staging_rows} rows)` : `kv ${variant.kv_style}`;
        const loadMs = llm.stats.loadMs, weightsMs = llm.stats.weightsMs, buildMs = llm.stats.buildMs;
        current = { key, maxLength, chunkSize: llm.chunkSize, chunkSizes: llm.chunkSizes, blocksPerGraph, prefillBlocksPerGraph, buildMs, weightsMs, loadMs, variant,
            label: `${pkg.model.architecture}: max length ${maxLength}, chunk ${sizes}, ${kvText}, ${variant.compute_type}, blocks/graph ${blocksPerGraph}${prefillBlocksPerGraph !== 1 ? `/${prefillBlocksPerGraph}` : ""}` };
        if (!reused) setCell("sum-load", seconds(weightsMs), "download (or OPFS cache) + upload of the weights to the device");
        setCell("sum-dequantize", "not used", "int4 weights stay int4 and are dequantized inside the graphs");
        log("session_ready", `Session ready: ${current.label}; ${reused ? "reconfigure" : "load"} ${seconds(reused ? buildMs : loadMs)} (weights ${seconds(weightsMs)}, graph build ${seconds(buildMs)})`,
            { max_length: maxLength, prefill_chunk_sizes: llm.chunkSizes, blocks_per_graph: blocksPerGraph, prefill_blocks_per_graph: prefillBlocksPerGraph, variant: variant.id, build_ms: Math.round(buildMs),
              weights_ms: Math.round(weightsMs), load_ms: Math.round(loadMs), reconfigured: reused, package: url });
        return current;
    }

    async function generate(session, userText, maxNewTokens) {
        const ids = promptIds(userText);
        if (ids.length >= session.maxLength - 1) throw new Error(`prompt has ${ids.length} tokens but max length is ${session.maxLength}`);
        byId("answer").textContent = ""; byId("summaryLine").textContent = `prompt: ${ids.length} tokens …`;
        const plan = llm.planFor(ids.length), chunks = plan.length;
        log("generation_started", `Generation started: max length ${session.maxLength}, ${ids.length} prompt tokens in ${chunks} chunk(s) (${plan.join("+")}), up to ${maxNewTokens} new tokens`, { max_length: session.maxLength, prompt_tokens: ids.length, prefill_chunks: chunks, prefill_plan: plan, variant: session.variant.id });
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
        const dispatches = 1 + Math.ceil((pkg.model.num_layers - 1) / session.blocksPerGraph) + 2, kv = session.variant.kv_style === "staged" ? `KV staged ${session.variant.staging_rows}` : `KV ${session.variant.kv_style}`;
        setCell("sum-build", seconds(session.buildMs), "graph build of this session (all graphs of the package variant)");
        setCell("sum-prefill", seconds(prefillMs), `${ids.length} prompt tokens in ${chunks} chunk(s): ${plan.join("+")}`);
        setCell("sum-decode", `${rounded(tpot)} ms/token`, `${rounded(tps)} tokens/s over ${steps.length} decode steps`);
        setCell("sum-inference", seconds(inferenceMs));
        setMetric("metric-ttft", (prefillMs / 1000).toFixed(2), "s", `${ids.length} prompt tokens · ${chunks} chunk${chunks > 1 ? "s" : ""}`);
        setMetric("metric-tpot", rounded(tpot), "ms/token", `${steps.length} decode steps · ${dispatches} dispatches per step`);
        setMetric("metric-tps", rounded(tps), "tokens/s", `${kv} · blocks/graph ${session.blocksPerGraph} · ${s.flushes} flushes`);
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
            { max_length: session.maxLength, prefill_chunk_size: session.chunkSize, prefill_plan: plan, variant: session.variant.id, prompt_tokens: ids.length, prefill_chunks: chunks, ttft_ms: Math.round(prefillMs), tpot_ms: rounded(tpot),
              tokens_per_second: rounded(tps), decode_steps: steps.length, inference_ms: Math.round(inferenceMs), generated_token_ids: generated, text: answer, kv_flushes: s.flushes, package: url });
        setStatus("ready");
    }

    return {
        getSession, currentSession: () => current, generate, sharedBuildMs: 0, pkg,
        countPromptTokens: (text) => promptIds(text).length,
    };
}

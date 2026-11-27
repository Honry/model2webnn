// The "package" engine of chat.html: runs a model package written by `m2w compile` (python/ in model2webnn) with the model2webnn runtime (../../lib/runtime, built by `npm run build:lib`),
// behind the same interface the page already uses for the hand-written Qwen3 graphs:  { getSession, currentSession, generate, sharedBuildMs, countPromptTokens, countTurns, multiTurn }
// (generate takes the whole conversation: options.turns, the pictures of all its turns in options.images, a signal to stop).
// What a package offers is fixed when it is compiled (max lengths, prefill chunks, graph variants = KV protocol x compute type x staging rows, blocks per graph): the page offers exactly that,
// and a change of any session option rebuilds the graphs of the loaded model (LLM.reconfigure) while the weights stay on the device.
import { loadModel, CachedSource, UrlSource, openPackage, buildPrompt, tokenizerFromTransformersJs, kvBytes, expectedMemory, formatBytes } from "../../lib/runtime/index.js";

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

const kvCacheMegabytes = (model, plan, maxLength) => kvBytes(model, plan, maxLength) / 2 ** 20;   // the runtime's formula (src/runtime/memory.ts): K + V, f16, double buffered, the layers that own a cache

/** Mark every max length option with the memory it is expected to need (data-memory: "~5.1 GB") and a tooltip with the breakdown. An estimate, not a measurement: the browser hides the real RAM. `vision`: the image tower is loaded. */
export function annotateMaxLengths(select, pkg, vision) {
    for (const option of select.options) {
        const need = expectedMemory({ model: pkg.model, plan: pkg.plan, manifest: pkg.manifest, maxLength: +option.value, vision: !!vision && !!pkg.plan.vision });
        option.dataset.memory = `~${formatBytes(need.totalBytes)}`;
        option.title = `max length ${need.maxLength}: about ${formatBytes(need.totalBytes)} of memory in total\n`
            + `device weights ${formatBytes(need.weightBytes)} + KV cache ${formatBytes(need.kvBytes)} + fixed overhead ${formatBytes(need.overheadBytes)} (browser, graph compiler, activations)\n`
            + "An estimate calibrated on one measurement, not a measurement of your machine: the browser hides the real RAM, so pick what your machine can hold.";
    }
}

/** Fill and lock the option selects of the page from what the package contains. `initial`: the session options of the URL. Returns { lock } to call whenever the page re-enables its controls. */
export function setupPackageControls(pkg, c, initial = {}) {
    const { model, plan } = pkg;
    const variants = plan.variants, defaultVariant = variants.find((v) => v.id === plan.default_variant);
    const lengths = plan.max_lengths.slice().sort((a, b) => a - b);
    const chunksOf = (length) => Object.keys(plan.graphs[defaultVariant.id].prefill[String(length)] ?? {}).map(Number).sort((a, b) => a - b);
    const replace = (select, entries, keep) => { select.innerHTML = ""; for (const [value, text] of entries) select.add(new Option(text, value)); if (keep !== undefined && [...select.options].some((o) => o.value === String(keep))) select.value = keep; };
    replace(c.maxLengthSelect, lengths.map((l) => [l, `${l} tokens  (KV cache ${Math.round(kvCacheMegabytes(model, plan, l))} MB)`]));
    const fillChunks = () => replace(c.chunkSizeSelect, chunksOf(+c.maxLengthSelect.value).map((s) => [s, `${s} tokens`]), c.chunkSizeSelect.value);
    fillChunks();
    const singleType = new Set(model.layer_types).size === 1 || plan.block_group_controls === "per_block";   // hybrid models: groups take a mask / RoPE table per block
    const gemma = ["gemma4", "qwen35"].includes(plan.architecture_kind);   // Gemma 4, Qwen3.5: decode block groups of any K the package lists (cut per sequence of attention kinds), no prefill groups
    const groupsOf = (table) => Object.keys(table?.block_groups ?? {}).map(Number);
    const fillBlocks = () => {
        const length = c.maxLengthSelect.value, decode = groupsOf(plan.graphs[defaultVariant.id].decode[length]), biggest = decode.length ? Math.max(...decode) : 0;
        replace(c.blocksSelect, [[1, "1"], ...(gemma ? decode.sort((a, b) => a - b).map((n) => [n, String(n)]) : singleType && biggest > 1 ? [[biggest, String(biggest)]] : [])], c.blocksSelect.value);
        const prefill = Object.values(plan.graphs[defaultVariant.id].prefill[length] ?? {}).map(groupsOf).reduce((common, list, i) => (i === 0 ? list : common.filter((n) => list.includes(n))), []);
        replace(c.prefillBlocksSelect, gemma ? [["1", "1 (decode groups only)"]] : [["0", "same as decode"], ["1", "1 (single blocks)"], ...(singleType ? prefill.map((n) => [n, String(n)]) : [])], c.prefillBlocksSelect.value);
    };
    fillBlocks();
    // KV protocol x compute type x staging rows: only the combinations the package contains
    const kvLabels = { staged: "staged (staging buffer + flush)", onehot: "one-hot (whole cache every step)", tensorScatter: "tensorScatter (in-graph)", scatterND: "scatterND (in-graph)" };
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
    // not chosen in the URL: the recommended group sizes of the package (plan.recommended: the best known for the model family) and say so in the option text
    const recommended = plan.recommended ?? {};
    const mark = (select, value) => { for (const o of select.options) if (+o.value === +value && !o.text.includes("recommended")) o.text += " (recommended)"; };
    if (!initial.blocksPerGraph && recommended.blocks_per_graph > 1 && [...c.blocksSelect.options].some((o) => +o.value === recommended.blocks_per_graph)) c.blocksSelect.value = String(recommended.blocks_per_graph);
    if (!initial.prefillBlocksPerGraph && recommended.prefill_blocks_per_graph && [...c.prefillBlocksSelect.options].some((o) => +o.value === recommended.prefill_blocks_per_graph)) c.prefillBlocksSelect.value = String(recommended.prefill_blocks_per_graph);
    if (recommended.blocks_per_graph) mark(c.blocksSelect, recommended.blocks_per_graph);
    if (recommended.prefill_blocks_per_graph) mark(c.prefillBlocksSelect, recommended.prefill_blocks_per_graph);
    if (initial.blocksPerGraph) replace(c.blocksSelect, [...c.blocksSelect.options].map((o) => [o.value, o.text]), initial.blocksPerGraph);
    if (initial.prefillBlocksPerGraph) replace(c.prefillBlocksSelect, [...c.prefillBlocksSelect.options].map((o) => [o.value, o.text]), initial.prefillBlocksPerGraph);
    // no chunk size in the URL: the recommended set of prefill chunks (the runtime plans every prompt with them)
    const availableChunks = chunksOf(+c.maxLengthSelect.value), recommendedChunks = (recommended.prefill_chunks ?? []).filter((n) => availableChunks.includes(n));
    if (!initial.chunkGiven && !initial.chunkSizes?.length && recommendedChunks.length) c.chunkSizesInput.value = recommendedChunks.join(",");
    c.chunkSizesInput.placeholder = `optional, e.g. ${plan.prefill_chunks.join(",")}`;
    const lock = () => {
        const only = (select) => select.options.length < 2;
        c.kvStyleSelect.disabled = only(c.kvStyleSelect); c.computeTypeSelect.disabled = only(c.computeTypeSelect);
        c.stagingRowsSelect.disabled = c.kvStyleSelect.value !== "staged" || only(c.stagingRowsSelect);
        c.blocksSelect.disabled = only(c.blocksSelect); c.prefillBlocksSelect.disabled = only(c.prefillBlocksSelect);
        if (gemma && only(c.blocksSelect)) c.blocksSelect.title = "This package has no decode block groups: compile with --blocks-per-graph 1 3 (or m2w rebuild-graphs).";
        else if (!singleType && !gemma) c.blocksSelect.title = c.prefillBlocksSelect.title = "Blocks of one graph share a single mask and RoPE table: models with sliding + full attention layers run one block per graph.";
        c.stagingRowsSelect.title = c.kvStyleSelect.value === "staged" ? "" : "Only the staged KV protocol has a staging buffer.";
    };
    lock();
    return { lock };
}

/** the model of the page: tokenizer + (lazily, per max length / chunk / blocks-per-graph) a loaded LLM */
export async function loadPackageModel(ui, { url, deviceType, vision = null }) {   // vision: null (no image tower), true (the tower; a picture gets any preset of the package) or [[h, w], ...] (only these presets); the graphs of a preset are built on its first use
    const { byId, log, setStatus, setCell, setMetric, fillTimingTable, seconds, rounded } = ui, report = (p) => ui.progress?.(p);   // report: the runtime's progress events (counts, bytes) for the page's progress bars
    setStatus("opening the package");
    const pkg = await openPackage(url, true);
    setStatus("loading the tokenizer");
    const lib = await import(TRANSFORMERS_URL);
    const tokenizer = await tokenizerFromTransformersJs(lib, pkg);
    log("tokenizer_loaded", "Tokenizer loaded (transformers.js, from the package)");
    const format = pkg.chatFormat, stop = new Set(pkg.model.eos_token_ids);
    let llm = null, current = null;
    const model = pkg.model, cacheLayers = (model.kv_producer?.length ? model.kv_producer.filter((p) => p < 0).length : model.num_layers) - model.layer_types.filter((t) => t === "linear_attention").length;
    document.querySelectorAll(".layerCount").forEach((e) => (e.textContent = model.num_layers));
    document.querySelectorAll(".layerCountMinusOne").forEach((e) => (e.textContent = model.num_layers - 1));
    document.querySelectorAll(".cacheLayerCount").forEach((e) => (e.textContent = cacheLayers));
    document.querySelectorAll(".hiddenSize").forEach((e) => (e.textContent = model.hidden));
    document.querySelectorAll(".vocabSize").forEach((e) => (e.textContent = model.vocab_padded));

    /** the "Build (compile) graphs" table and the build time of every box of the pipeline, from the runtime's per-graph timings of the current session */
    function showBuildTimes(builds) {
        const table = byId("buildTable");
        for (let i = table.rows.length - 1; i >= 1; i--) table.deleteRow(i);
        document.querySelectorAll("#phase-prefill .time, #phase-decode .time").forEach((e) => (e.textContent = ""));
        const boxes = new Map();   // pipeline element id -> { ms, count }
        for (const { label, ms } of builds) {
            const row = table.insertRow(); row.insertCell().textContent = label; row.insertCell().textContent = Math.round(ms);
            const phase = label.startsWith("prefill") ? "prefill" : "decode";
            const id = label === "lm_head" ? "time-lm_head" : /flush/.test(label) ? "time-decode-kv_flush" : /\bhead\b/.test(label) ? `time-${phase}-head` : /\b(block|group)\b/.test(label) ? `time-${phase}-block` : /\btail\b/.test(label) ? `time-${phase}-tail` : null;
            if (!id) continue;
            const entry = boxes.get(id) ?? { ms: 0, count: 0 };
            entry.ms += ms; entry.count++; boxes.set(id, entry);
        }
        for (const [id, { ms, count }] of boxes) { const element = byId(id); if (element) element.textContent = ` built in ${ui.seconds(ms)}${count > 1 ? ` (${count} graphs)` : ""}`; }
    }

    // a session = max length + prefill chunk sizes + graph variant (KV protocol, staging rows, compute type) + blocks per graph; the first one loads the model (weights go to the device once),
    // every other one is LLM.reconfigure(): graphs are rebuilt, the weights stay.
    async function getSession(maxLength, chunkSize, options) {
        const blocksPerGraph = options.blocksPerGraph || 1, chunkSizes = options.chunkSizes?.length ? options.chunkSizes : [chunkSize];
        const prefillGroups = Object.keys(pkg.plan.graphs[pkg.plan.default_variant].prefill[String(maxLength)]?.[String(chunkSizes[0])]?.block_groups ?? {}).map(Number);
        const prefillBlocksPerGraph = options.prefillBlocksPerGraph || (blocksPerGraph === 1 || prefillGroups.includes(blocksPerGraph) ? blocksPerGraph : 1);
        const kvStyle = options.kvStyle, computeType = options.computeType, stagingRows = kvStyle === "staged" ? options.stagingRows : undefined;
        const kvInPlace = new URLSearchParams(location.search).get("kvInPlace") === "1";   // scatterND / tensorScatter write the cache into the same tensor that is the input (page option, fixed at load)
        const key = [maxLength, chunkSizes.join("+"), blocksPerGraph, prefillBlocksPerGraph, kvStyle, stagingRows, computeType, kvInPlace].join("|");
        if (current && current.key === key) return current;
        const session = { maxLength, chunkSizes, blocksPerGraph, prefillBlocksPerGraph, kvStyle, computeType, kvInPlace, ...(stagingRows ? { stagingRows } : {}), onProgress: (p) => {
            report(p);
            setStatus(p.stage === "download" ? `downloading weights ${Math.round(p.loaded / 2 ** 20)} / ${Math.round(p.total / 2 ** 20)} MB` : `${p.stage}${p.message ? ": " + p.message : ""}`);
        } };
        const reused = !!llm;
        report({ stage: "begin", fresh: !llm });   // a load (download + upload) or a reconfigure (graphs only)
        if (llm) await llm.reconfigure(session);
        else {
            const query = new URLSearchParams(location.search);   // ?parallelDownloads=4&downloadSegments=4: see download.html for good values
            llm = await loadModel(url, { device: deviceType, parallelDownloads: +query.get("parallelDownloads") || undefined, downloadSegments: +query.get("downloadSegments") || undefined,
                ...(vision && pkg.plan.vision ? { vision: vision === true ? true : { grids: vision } } : {}), ...session });
            if (llm.vision) {
                llm.vision.progress = (p) => { report({ ...p, vision: true }); setStatus(p.message ?? p.stage); };   // "build vision 24x16 head": the graphs of a preset are built on its first use
                log("vision_ready", `Image tower ready: weights ${pkg.plan.vision.weights} loaded in ${seconds(llm.vision.weightsMs)}; presets a picture may get: ${llm.vision.grids.map((g) => g.join("x")).join(", ")} (graphs are built on first use)`, { grids: llm.vision.grids, weights: pkg.plan.vision.weights, weights_ms: Math.round(llm.vision.weightsMs) });
            }
        }
        const variant = llm.variant, sizes = llm.chunkSizes.join("+");
        byId("pastLengthLabel").textContent = String(maxLength - 1); byId("chunkSizeLabel").textContent = sizes;
        showBuildTimes(llm.stats.graphBuilds);
        const kvText = variant.kv_style === "staged" ? `kv staged (${variant.staging_rows} rows)` : `kv ${variant.kv_style}`;
        const loadMs = llm.stats.loadMs, weightsMs = llm.stats.weightsMs, buildMs = llm.stats.buildMs;
        current = { key, maxLength, chunkSize: llm.chunkSize, chunkSizes: llm.chunkSizes, blocksPerGraph, prefillBlocksPerGraph, buildMs, weightsMs, loadMs, variant,
            label: `${pkg.model.architecture}: max length ${maxLength}, chunk ${sizes}, ${kvText}, ${variant.compute_type}, blocks/graph ${blocksPerGraph}${prefillBlocksPerGraph !== 1 ? `/${prefillBlocksPerGraph}` : ""}` };
        if (!reused) setCell("sum-load", seconds(weightsMs), "download (or OPFS cache) + upload of the weights to the device");
        log("session_ready", `Session ready: ${current.label}; ${reused ? "reconfigure" : "load"} ${seconds(reused ? buildMs : loadMs)} (weights ${seconds(weightsMs)}, graph build ${seconds(buildMs)})`,
            { max_length: maxLength, prefill_chunk_sizes: llm.chunkSizes, blocks_per_graph: blocksPerGraph, prefill_blocks_per_graph: prefillBlocksPerGraph, variant: variant.id, build_ms: Math.round(buildMs),
              weights_ms: Math.round(weightsMs), load_ms: Math.round(loadMs), reconfigured: reused, package: url });
        return current;
    }

    // sampling: the package recommends parameters (plan.recommended.sampling: the model card's); ?temperature=0 = greedy (reproducible), ?temperature=0.7&topK=20&topP=0.8&presencePenalty=1.5 override
    const samplingFromUrl = () => {
        const query = new URLSearchParams(location.search), out = {};
        for (const [name, key] of [["temperature", "temperature"], ["topK", "topK"], ["topP", "topP"], ["presencePenalty", "presencePenalty"]]) if (query.get(name) !== null) out[key] = +query.get(name);
        return out;
    };
    const imagePrompt = pkg.model.vision?.image_prompt ?? "";
    /** the prompt of a conversation: the turns so far, the last one without an answer (turns: [{ user, assistant?, images: count }]) */
    const promptOf = (turns, system) => (format ? buildPrompt(format, turns, system, new Date(), imagePrompt) : turns[turns.length - 1].user);

    /** options: { turns: [{ user, assistant?, images }] (the whole conversation, the last turn is answered; default: just userText), images: [{ source (ImageBitmap / canvas / img), name, detail }] (in the order of the turns' pictures),
     *  system, sampling, onText(text) (the answer so far), signal (AbortSignal: stop) } -> a summary of the generation (stopped: the signal ended it) */
    async function generate(session, userText, maxNewTokens, options = {}) {
        const sources = options.images ?? [], turns = options.turns ?? [{ user: userText, images: sources.length }], showText = options.onText ?? (() => {});
        const images = [], imageInfos = [];
        for (const { source, name = "", detail = "auto" } of sources) {   // a picture: the grid preset is chosen from its size and the detail level, resized on a canvas, patchified; the pad token is expanded by the runtime
            if (!llm.vision) throw new Error("the image tower is not loaded (this package has none, or the page was opened with vision=0)");
            const started = performance.now(), prepared = llm.prepareImage(source, undefined, detail);
            const preprocessMs = performance.now() - started, buildMs = await llm.vision.ensure(prepared.grid);   // the first use of a preset builds its three graphs (about two seconds): not part of the vision encode
            const patch = pkg.model.vision.patch_size;
            images.push(prepared);
            imageInfos.push({ name, detail, grid: prepared.grid, pixels: `${prepared.grid[1] * patch}x${prepared.grid[0] * patch}`, width: prepared.width, height: prepared.height, tokens: prepared.tokens, preprocessMs, buildMs });
            log("image_prepared", `Image ${prepared.width}x${prepared.height} (detail ${detail}) -> grid ${prepared.grid.join("x")} (${imageInfos.at(-1).pixels} px, ${prepared.tokens} image tokens), preprocessed in ${Math.round(preprocessMs)} ms${buildMs ? `, graphs of the grid built in ${seconds(buildMs)}` : ""}`,
                { width: prepared.width, height: prepared.height, detail, grid: prepared.grid, image_tokens: prepared.tokens, preprocess_ms: Math.round(preprocessMs), build_ms: Math.round(buildMs) });
        }
        const ids = tokenizer.encode(promptOf(turns, options.system));
        const total = ids.length + images.reduce((n, image) => n + image.tokens - 1, 0);   // the pad token of every image becomes `tokens` tokens
        if (total >= session.maxLength - 1) throw new Error(`prompt has ${total} tokens but max length is ${session.maxLength}`);
        showText("");
        const plan = llm.planFor(total), chunks = plan.length;
        log("generation_started", `Generation started: max length ${session.maxLength}, ${total} prompt tokens in ${chunks} chunk(s) (${plan.join("+")}), up to ${maxNewTokens} new tokens`, { max_length: session.maxLength, prompt_tokens: total, prefill_chunks: chunks, prefill_plan: plan, variant: session.variant.id, image_tokens: total - ids.length + images.length, turns: turns.length, images: images.length });
        setStatus("prefill");
        const start = performance.now(), generated = [];
        for await (const token of llm.generate(ids, { maxNewTokens, images, signal: options.signal, onProgress: report, ...samplingFromUrl(), ...(options.sampling ?? {}) })) {
            if (stop.has(token)) break;
            generated.push(token);
            showText(tokenizer.decode(generated));
            setStatus(`decode step ${generated.length}${options.autoMax ? "" : `/${maxNewTokens}`}`);
        }
        const stopped = !!options.signal?.aborted, answer = tokenizer.decode(generated);
        const summary = { text: answer, promptTokens: total, imageTokens: llm.stats.imageTokens ?? 0, generated: generated.length, stopped, capacity: session.maxLength - 1, images: imageInfos };
        if (stopped && !generated.length) {   // stopped during the prefill: no tokens, nothing to report but the prompt
            log("generation_stopped", `Generation stopped during the prefill after ${seconds(performance.now() - start)}`, { max_length: session.maxLength, prompt_tokens: total });
            setStatus("ready");
            return { ...summary, imageTokens: 0, prefillMs: 0, visionMs: 0, buildMs: 0, preprocessMs: 0, tpotMs: 0, tps: 0, inferenceMs: performance.now() - start, grid: null, pixels: "", detail: "" };
        }
        const inferenceMs = performance.now() - start, s = llm.stats, steps = s.steps;
        const avg = (key) => (steps.length ? steps.reduce((sum, step) => sum + step[key], 0) / steps.length : 0);
        const tpot = avg("totalMs"), tps = tpot ? 1000 / tpot : 0, prefillMs = s.prefillMs;
        const dispatches = 1 + Math.ceil((pkg.model.num_layers - 1) / session.blocksPerGraph) + 2, kv = session.variant.kv_style === "staged" ? `KV staged ${session.variant.staging_rows}` : `KV ${session.variant.kv_style}`;
        setCell("sum-build", seconds(session.buildMs), "graph build of this session (all graphs of the package variant)");
        setCell("sum-prefill", seconds(prefillMs), `${total} prompt tokens in ${chunks} chunk(s): ${plan.join("+")}${images.length ? `; image tower ${seconds(llm.stats.visionMs ?? 0)}${llm.stats.visionBuildMs ? ` + graph build ${seconds(llm.stats.visionBuildMs)}` : ""} (inside)` : ""}`);
        setCell("sum-decode", `${rounded(tpot)} ms/token`, `${rounded(tps)} tokens/s over ${steps.length} decode steps`);
        setCell("sum-inference", seconds(inferenceMs));
        setMetric("metric-ttft", (prefillMs / 1000).toFixed(2), "s", `${total} prompt tokens · ${chunks} chunk${chunks > 1 ? "s" : ""}`);
        setMetric("metric-tpot", rounded(tpot), "ms/token", `${steps.length} decode steps · ${dispatches} dispatches per step`);
        setMetric("metric-tps", rounded(tps), "tokens/s", `${kv} · blocks/graph ${session.blocksPerGraph} · ${s.flushes} flushes`);
        byId("summaryLine").textContent = `max length ${session.maxLength} · prompt ${total} tokens · ${generated.length} generated · total inference ${seconds(inferenceMs)}${stopped ? " · stopped" : ""}`;
        fillTimingTable([
            ["Host: inputs", rounded(avg("hostMs")), "JavaScript work before the dispatches: writing the step inputs (embedding row, RoPE tables, masks, selectors)", "host"],
            ["Enqueue", rounded(avg("enqueueMs")), `${dispatches} dispatches queued per step (a flush of the staging buffer counts here when due)`, "enqueue"],
            ["Device wait", rounded(avg("waitMs")), "read logits: time until the device finished the whole chain", "device"],
            ["Host: sampling", rounded(avg("samplingMs")), `${pkg.plan.recommended?.sampling ? "temperature / top-k / top-p sampling" : "argmax"} over ${pkg.model.vocab} logits`, "sample"],
            ["Total = TPOT", rounded(tpot), "time per generated token", "total"],
        ]);
        const first = s.stepMs.slice(0, 4).map(rounded).join(", ");
        log("generation_finished", `Generated ${generated.length} tokens: TTFT ${seconds(prefillMs)}, TPOT ${rounded(tpot)} ms/token (${rounded(tps)} tokens/s); first decode steps ms: ${first}${stopped ? " (stopped)" : ""}`,
            { ...(images.length ? { image_tokens: llm.stats.imageTokens, vision_ms: Math.round(llm.stats.visionMs ?? 0), vision_build_ms: Math.round(llm.stats.visionBuildMs ?? 0), grid: imageInfos[0].grid, detail: imageInfos[0].detail, preprocess_ms: Math.round(imageInfos[0].preprocessMs) } : {}), max_length: session.maxLength, prefill_chunk_size: session.chunkSize, prefill_plan: plan, variant: session.variant.id, prompt_tokens: total, prefill_chunks: chunks, ttft_ms: Math.round(prefillMs), tpot_ms: rounded(tpot),
              tokens_per_second: rounded(tps), decode_steps: steps.length, inference_ms: Math.round(inferenceMs), generated_token_ids: generated, text: answer, kv_flushes: s.flushes, package: url, ...(stopped ? { stopped: true } : {}) });
        setStatus("ready");
        return { ...summary, prefillMs, visionMs: llm.stats.visionMs ?? 0, buildMs: imageInfos.reduce((n, i) => n + i.buildMs, 0), preprocessMs: imageInfos.reduce((n, i) => n + i.preprocessMs, 0),
            tpotMs: tpot, tps, inferenceMs, grid: imageInfos[0]?.grid ?? null, pixels: imageInfos[0]?.pixels ?? "", detail: imageInfos[0]?.detail ?? "" };
    }

    return {
        getSession, currentSession: () => current, generate, sharedBuildMs: 0, pkg,
        countPromptTokens: (text) => tokenizer.encode(promptOf([{ user: text }])).length,
        /** tokens of a conversation as the model reads it (a picture counts as its one pad token: the page adds the image tokens); the page keeps the history only when the package has a chat format */
        countTurns: (turns, system) => tokenizer.encode(promptOf(turns, system)).length,
        multiTurn: !!format,
    };
}

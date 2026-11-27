// The "package" engine of chat.html: runs a model package written by `m2w compile` (python/ in model2webnn) with the model2webnn runtime (../../lib/runtime, built by `npm run build:lib`),
// behind the same interface the page already uses for the hand-written Qwen3 graphs:  { getSession, currentSession, generate, sharedBuildMs, countPromptTokens }.
// What a package offers is fixed when it is compiled (max lengths, prefill chunks, graph variants = KV protocol x compute type x staging rows, blocks per graph): the page offers exactly that,
// and a change of any session option rebuilds the graphs of the loaded model (LLM.reconfigure) while the weights stay on the device.
import { loadModel, CachedSource, UrlSource, openPackage, buildPrompt, tokenizerFromTransformersJs } from "../../lib/runtime/index.js";
import { answerView } from "./answer-view.js?v=1";

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

/** K + V (double buffered) of the layers that have a cache of their own; the head dim / KV heads can differ per layer type (Gemma 4) */
const kvCacheMegabytes = (model, maxLength) => {
    let bytes = 0;
    for (let i = 0; i < model.num_layers; i++) {
        if (model.kv_producer && model.kv_producer[i] >= 0) continue;
        if (model.layer_types[i] === "linear_attention") continue;   // Gated DeltaNet layers carry a small state instead of a KV cache
        const d = model.layer_dims?.[model.layer_types[i]];
        bytes += 2 * 2 * (d?.num_kv_heads ?? model.num_kv_heads) * (maxLength - 1) * (d?.head_dim ?? model.head_dim) * 2;
    }
    return bytes / 2 ** 20;
};

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
            setStatus(p.stage === "download" ? `downloading weights ${Math.round(p.loaded / 2 ** 20)} / ${Math.round(p.total / 2 ** 20)} MB` : `${p.stage}${p.message ? ": " + p.message : ""}`);
        } };
        const reused = !!llm;
        if (llm) await llm.reconfigure(session);
        else {
            const query = new URLSearchParams(location.search);   // ?parallelDownloads=4&downloadSegments=4: see download.html for good values
            llm = await loadModel(url, { device: deviceType, parallelDownloads: +query.get("parallelDownloads") || undefined, downloadSegments: +query.get("downloadSegments") || undefined,
                ...(vision && pkg.plan.vision ? { vision: vision === true ? true : { grids: vision } } : {}), ...session });
            if (llm.vision) {
                llm.vision.progress = (p) => setStatus(p.message ?? p.stage);   // "build vision 24x16 head": the graphs of a preset are built on its first use
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

    /** options: { image: { source (ImageBitmap / canvas / img), name, detail }, onText(text) (default: the answer box of the Chat tab) } -> a summary of the generation */
    // sampling: the package recommends parameters (plan.recommended.sampling: the model card's); ?temperature=0 = greedy (reproducible), ?temperature=0.7&topK=20&topP=0.8&presencePenalty=1.5 override
    const samplingFromUrl = () => {
        const query = new URLSearchParams(location.search), out = {};
        for (const [name, key] of [["temperature", "temperature"], ["topK", "topK"], ["topP", "topP"], ["presencePenalty", "presencePenalty"]]) if (query.get(name) !== null) out[key] = +query.get(name);
        return out;
    };
    async function generate(session, userText, maxNewTokens, options = {}) {
        let ids, images = [], imageInfo = null;
        const chatAnswer = answerView(byId("answer")), showText = options.onText ?? ((text) => chatAnswer.set(text));   // the Chat tab shows Markdown; the log below gets the raw text
        if (options.image) {   // a picture with the prompt: the grid preset is chosen from its size and the detail level, resized on a canvas, patchified; the pad token is expanded by the runtime
            if (!llm.vision) throw new Error("the image tower is not loaded (this package has none, or the page was opened with vision=0)");
            const { source, name = "", detail = "auto" } = options.image, started = performance.now();
            images = [llm.prepareImage(source, undefined, detail)];
            const preprocessMs = performance.now() - started, buildMs = await llm.vision.ensure(images[0].grid);   // the first use of a preset builds its three graphs (about two seconds): not part of the vision encode
            const patch = pkg.model.vision.patch_size;
            imageInfo = { name, detail, grid: images[0].grid, pixels: `${images[0].grid[1] * patch}x${images[0].grid[0] * patch}`, width: images[0].width, height: images[0].height, preprocessMs, buildMs };
            log("image_prepared", `Image ${images[0].width}x${images[0].height} (detail ${detail}) -> grid ${images[0].grid.join("x")} (${imageInfo.pixels} px, ${images[0].tokens} image tokens), preprocessed in ${Math.round(preprocessMs)} ms${buildMs ? `, graphs of the grid built in ${seconds(buildMs)}` : ""}`,
                { width: images[0].width, height: images[0].height, detail, grid: images[0].grid, image_tokens: images[0].tokens, preprocess_ms: Math.round(preprocessMs), build_ms: Math.round(buildMs) });
            ids = tokenizer.encode(buildPrompt(format, [{ user: userText, images: 1 }], undefined, new Date(), pkg.model.vision.image_prompt));
        } else ids = promptIds(userText);
        const total = ids.length + images.reduce((n, image) => n + image.tokens - 1, 0);   // the pad token of every image becomes `tokens` tokens
        if (total >= session.maxLength - 1) throw new Error(`prompt has ${total} tokens but max length is ${session.maxLength}`);
        showText(""); byId("summaryLine").textContent = `prompt: ${total} tokens …`;
        const plan = llm.planFor(total), chunks = plan.length;
        log("generation_started", `Generation started: max length ${session.maxLength}, ${total} prompt tokens in ${chunks} chunk(s) (${plan.join("+")}), up to ${maxNewTokens} new tokens`, { max_length: session.maxLength, prompt_tokens: total, prefill_chunks: chunks, prefill_plan: plan, variant: session.variant.id, image_tokens: total - ids.length + images.length });
        setStatus("prefill");
        const start = performance.now(), generated = [];
        for await (const token of llm.generate(ids, { maxNewTokens, images, ...samplingFromUrl(), ...(options.sampling ?? {}) })) {
            if (stop.has(token)) break;
            generated.push(token);
            showText(tokenizer.decode(generated));
            setStatus(`decode step ${generated.length}/${maxNewTokens}`);
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
        byId("summaryLine").textContent = `max length ${session.maxLength} · prompt ${total} tokens · ${generated.length} generated · total inference ${seconds(inferenceMs)}`;
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
            { ...(images.length ? { image_tokens: llm.stats.imageTokens, vision_ms: Math.round(llm.stats.visionMs ?? 0), vision_build_ms: Math.round(llm.stats.visionBuildMs ?? 0), grid: imageInfo.grid, detail: imageInfo.detail, preprocess_ms: Math.round(imageInfo.preprocessMs) } : {}), max_length: session.maxLength, prefill_chunk_size: session.chunkSize, prefill_plan: plan, variant: session.variant.id, prompt_tokens: total, prefill_chunks: chunks, ttft_ms: Math.round(prefillMs), tpot_ms: rounded(tpot),
              tokens_per_second: rounded(tps), decode_steps: steps.length, inference_ms: Math.round(inferenceMs), generated_token_ids: generated, text: answer, kv_flushes: s.flushes, package: url });
        if (!options.onText) chatAnswer.flush();
        setStatus("ready");
        return { text: answer, promptTokens: total, imageTokens: llm.stats.imageTokens ?? 0, generated: generated.length, prefillMs, visionMs: llm.stats.visionMs ?? 0, buildMs: imageInfo?.buildMs ?? 0, preprocessMs: imageInfo?.preprocessMs ?? 0,
            tpotMs: tpot, tps, inferenceMs, grid: imageInfo?.grid ?? null, pixels: imageInfo?.pixels ?? "", detail: imageInfo?.detail ?? "",
            // one entry per picture of the prompt (the runtime takes one today); the vision encode time is of the whole prompt's pictures
            images: imageInfo ? [{ name: imageInfo.name, detail: imageInfo.detail, grid: imageInfo.grid, pixels: imageInfo.pixels, tokens: images[0].tokens, preprocessMs: imageInfo.preprocessMs, buildMs: imageInfo.buildMs, visionMs: llm.stats.visionMs ?? 0 }] : [] };
    }

    return {
        getSession, currentSession: () => current, generate, sharedBuildMs: 0, pkg,
        countPromptTokens: (text) => promptIds(text).length,
    };
}

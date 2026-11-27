// The pipeline diagram of chat.html for a model PACKAGE: written from the package's own model.json / plan.json / weights manifest (layer types, KV sharing, block graph kinds, per-family host work,
// image tower), so that every family (Qwen3-like, Qwen3.5 hybrid, Gemma 4) is described with its own numbers. The element ids are the ones chat.js / chat-package.js use (phase-*, prefill-* / decode-*, time-*, stage-*).
// The hand-written Qwen3-4B path (model=js) keeps the static markup of the page.
//
//   renderPipeline(body, summaryLine, pkg)          draw the diagram of a (peeked) package
//   updatePipelineSession(body, pkg, session)       the parts that depend on the session options: blocks per graph, KV protocol (flush box), staging rows, chunk sizes / past length (optional)
//   cacheLayerCount(model)                          layers that have a KV cache of their own (the flush graph runs once per such layer)

const esc = (text) => String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const plural = (n, one, many = one + "s") => `${n} ${n === 1 ? one : many}`;

/** layers with a KV cache of their own: not a Gated DeltaNet layer (it carries a state instead) and not a layer that reads the cache of an earlier one (Gemma 4 kv_producer) */
export function cacheLayerCount(model) {
    let count = 0;
    for (let i = 0; i < model.num_layers; i++) {
        if (model.kv_producer && model.kv_producer[i] >= 0) continue;
        if (model.layer_types[i] === "linear_attention") continue;
        count++;
    }
    return count;
}

const KIND_NAMES = { full_attention: "full attention", sliding_attention: "sliding-window attention", linear_attention: "Gated DeltaNet (linear attention)" };
const KV_STYLES = {
    staged: "staging buffer + flush graph",
    onehot: "the whole cache is rewritten every step (one-hot)",
    scatterND: "in-graph scatterND of the new rows",
    tensorScatter: "in-graph tensorScatter of the new rows",
};

/** everything the diagram says, as plain data derived from the package */
export function describePackage(pkg) {
    const { model: m, plan: p, manifest } = pkg, kind = p.architecture_kind ?? "qwen3-like";
    const N = m.num_layers, hidden = m.hidden, vocab = m.vocab, vocabPadded = m.vocab_padded;
    const counts = new Map();
    for (const type of m.layer_types) counts.set(type, (counts.get(type) ?? 0) + 1);
    const shared = (m.kv_producer ?? []).filter((producer) => producer >= 0).length, own = cacheLayerCount(m), linear = counts.get("linear_attention") ?? 0;
    const census = [...counts].map(([type, n]) => `${n} × ${type === "sliding_attention" && m.sliding_window ? `sliding-window attention (${m.sliding_window} tokens)` : KIND_NAMES[type] ?? type}`);
    const firstLength = String(p.max_lengths[0]), table = p.graphs[p.default_variant].decode[firstLength] ?? {};
    const blockKinds = Object.keys(table.blocks ?? {});
    const quant = manifest?.quant ?? {}, embedding = quant.embedding ?? {};
    const embeddingText = embedding.bits === 8 ? `int8 row × row scale${embedding.asymmetric ? " (zero point per row)" : ""}` : "quantized row × scale";
    const ropeTypes = Object.keys(m.rope ?? {});
    const sampling = p.recommended?.sampling;
    const samplingText = sampling ? [sampling.temperature !== undefined && `temperature ${sampling.temperature}`, sampling.top_k && `top-k ${sampling.top_k}`, sampling.top_p && `top-p ${sampling.top_p}`, sampling.presence_penalty && `presence penalty ${sampling.presence_penalty}`].filter(Boolean).join(" · ") : "greedy argmax";
    const vision = p.vision && m.vision ? { plan: p.vision, spec: m.vision } : null;
    return { kind, N, hidden, vocab, vocabPadded, heads: m.num_heads, kvHeads: m.num_kv_heads, headDim: m.head_dim, ffn: m.ffn, census, own, shared, linear, blockKinds, quant, embeddingText,
        ropeTypes, samplingText, sampling: !!sampling, embedScale: m.embed_scale && m.embed_scale !== 1 ? m.embed_scale : 0, pleDim: m.ple_dim ?? 0, softcap: m.logit_softcap ?? 0, mrope: !!m.mrope,
        tiedHead: !!m.tie_embeddings, lmHeadFormat: p.profile?.lm_head_format, state: p.state ?? null, vision, kvStyles: [...new Set(p.variants.map((v) => v.kv_style))], model: m, plan: p };
}

/** the one-line summary above the diagram */
function summaryOf(d) {
    const parts = [`${plural(d.N, "layer")}: ${d.census.join(" + ")}`, `hidden ${d.hidden}`, `${d.heads} heads (${d.kvHeads} KV, head dim ${d.headDim})`, `FFN ${d.ffn}`, `vocab ${d.vocab}`];
    if (d.quant.bits) parts.push(`int${d.quant.bits} weights, group ${d.quant.group_size}`);
    for (const [type, dim] of Object.entries(d.model.layer_dims ?? {})) parts.push(`${type === "full_attention" ? "full-attention" : type.replace(/_/g, " ")} layers: ${dim.num_heads} heads (${dim.num_kv_heads} KV, head dim ${dim.head_dim})`);
    if (d.shared) parts.push(`${d.shared} layers read the K/V of an earlier layer`);
    if (d.vision) parts.push(`image tower: ${d.vision.spec.depth} blocks`);
    return parts.join(" · ");
}

const box = (cls, id, title, text, { time = "", tip = "" } = {}) =>
    `<div class="box ${cls}" id="${id}"${tip ? ` data-tip="${esc(tip)}"` : ""}>${title}<small>${text}</small>${time ? `<span class="time" id="${time}"></span>` : ""}</div>`;
const arrow = '<div class="arrow">→</div>';
const row = (boxes, id = "") => `<div class="row"${id ? ` id="${id}"` : ""}>${boxes.join(arrow)}</div>`;
const tipIcon = (tip) => `<i class="tip" data-tip="${esc(tip)}">i</i>`;

export function renderPipeline(body, summaryLine, pkg) {
    const d = describePackage(pkg), H = d.hidden, last = d.N - 1;
    if (summaryLine) summaryLine.textContent = summaryOf(d);
    const kindsText = d.blockKinds.length > 1 ? `one graph per kind (${d.blockKinds.map((k) => k.replace(/_/g, " ")).join(", ")}), weights rebound per layer` : "one graph, weights rebound per layer";
    const typeNames = d.ropeTypes.map((t) => t.replace(/_attention/, "")).join(", ");

    // ---- host steps
    const host = [
        box("host", "stage-tokenizer", "1. Tokenizer", "chat template → token ids"),
        box("host", "stage-embedding", "2. Embedding lookup", `${esc(d.embeddingText)}${d.embedScale ? ` × ${d.embedScale.toFixed(2)}` : ""} → f16 hidden [1,S,${H}]${d.vision ? "<br>(picture rows come from the image tower)" : ""}`,
            { tip: `Done in JavaScript: one row of the embedding table per token.${d.tiedHead ? " The same table is the lm_head weight (tied embeddings)." : ""}${d.embedScale ? ` Gemma scales the rows by sqrt(hidden) = ${d.embedScale.toFixed(4)}.` : ""}` }),
    ];
    if (d.pleDim) host.push(box("host", "stage-ple", "2b. Per-layer embeddings", `per prompt token a row of ${d.N} × ${d.pleDim} values<br>→ HEAD's ple_token input`,
        { tip: "Gemma 4 gives every layer its own small embedding of the token (PLE). The table is read on the host (int8 × row scale); HEAD turns it into ple_0 … ple_N-1." }));
    const inputs = [`rope cos/sin${d.mrope ? " (3 MRoPE axes with a picture)" : ""}`, `attention mask${typeNames && d.ropeTypes.length > 1 ? " per layer type" : ""}`, "KV selectors"];
    if (d.linear) inputs.push("valid_mask · conv_perm (Gated DeltaNet prefill)");
    host.push(box("host", "stage-inputs", "3. Per-step inputs", inputs.join(" · "),
        { tip: d.ropeTypes.length > 1 ? `The rope tables and masks are bound per layer type: ${typeNames}.` : "" }));

    // ---- phases
    const lmText = `hidden [1,1,${H}] → logits f16 [1,1,${d.vocabPadded}]${d.softcap ? `<br>logit softcap ${d.softcap}` : ""}`;
    const prefillTip = `Each chunk attends to the KV cache (rows written by earlier chunks) plus its own rows, then writes its rows into the cache.\nThe last chunk is left-padded; only it runs TAIL + LM HEAD and yields the first token.`
        + (d.linear ? "\nGated DeltaNet layers carry their state from chunk to chunk; the pad rows of the last chunk are masked out of the state." : "")
        + (d.shared ? "\nLayers that share K/V read the cache of their producer after this chunk's update." : "");
    const headText = `layer 0 attention half${d.pleDim ? "<br>+ per-layer-embedding prologue" : ""}`;
    const prefillBlock = `FFN of layer i + attention of layer i+1<br>(${kindsText})<span class="pl-prefill-groups"></span>`;
    const prefill = `<div class="phase" id="phase-prefill"><div class="title">PREFILL — chunks of S=<span id="chunkSizeLabel">?</span> tokens ${tipIcon(prefillTip)}</div>`
        + row([box("device", "prefill-head", "HEAD", headText, { time: "time-prefill-head" }),
            box("device", "prefill-block", `BLOCK × <span class="layerCountMinusOne">${last}</span>`, prefillBlock, { time: "time-prefill-block" }),
            box("device", "prefill-tail", "TAIL (last chunk)", "last FFN + final RMSNorm, last row", { time: "time-prefill-tail" }),
            box("device", "prefill-lm_head", "LM HEAD (last chunk)", lmText, { time: "time-lm_head", tip: d.tiedHead ? "The weight is the embedding table (tied embeddings)." : "" })]) + "</div>";
    const stateText = d.linear ? `<br>${d.linear} Gated DeltaNet layers update their state (conv + recurrent) instead of a cache` : "";
    const decodeTip = `S=1, past length = max length − 1, repeated until an end-of-turn token or the token limit.
The KV update protocol of the session is shown below the boxes.`
        + (d.shared ? `\n${d.shared} of ${d.N} layers read the cache of an earlier layer: they have no K/V of their own.` : "");
    const decode = `<div class="phase" id="phase-decode"><div class="title">DECODE — one token per step (past <span id="pastLengthLabel">?</span>) ${tipIcon(decodeTip)}</div>`
        + row([box("device", "decode-head", "HEAD", headText, { time: "time-decode-head" }),
            box("device", "decode-block", `BLOCK × <span class="layerCountMinusOne">${last}</span>`, `FFN of layer i + attention of layer i+1<br><span class="pl-decode-groups">one block per graph</span>${stateText}`, { time: "time-decode-block" }),
            box("device", "decode-tail", "TAIL", "last FFN + final RMSNorm", { time: "time-decode-tail" }),
            box("device", "decode-lm_head", "LM HEAD", `→ logits f16 [1,1,${d.vocabPadded}]${d.softcap ? `<br>logit softcap ${d.softcap}` : ""}`)])
        + `<div class="row" id="flushRow" style="display:none">${box("helper", "decode-kv_flush", `KV FLUSH × <span class="cacheLayerCount">${d.own}</span>`,
            `every <span id="stagingRowsLabel">?</span> steps: staging buffer → main cache${d.plan.architecture_kind === "gemma4" ? "<br>one flush graph per layer type" : ""}${d.linear ? "<br>(full attention layers only)" : ""}`, { time: "time-decode-kv_flush" })}</div><div class="hint pl-kv"></div></div>`;

    // ---- the image tower (Qwen3.5 with a vision plan): its own phase, before the prefill
    let tower = "";
    if (d.vision) {
        const v = d.vision.spec, presets = d.vision.plan.presets ?? [], tokens = presets.map((x) => x.tokens), patchVector = v.channels * v.temporal_patch_size * v.patch_size ** 2;
        const towerTip = `One set of graphs per grid preset (${presets.length} presets, ${Math.min(...tokens)}–${Math.max(...tokens)} image tokens); the graphs of a preset are built the first time a picture gets it.\nThe picture is resized to the grid, normalised and cut into patches (merge blocks of ${v.merge_size}×${v.merge_size}); the image tokens replace the <|image_pad|> embedding rows of the prompt in prefill.`;
        tower = `<div class="phase" id="phase-vision"><div class="title">IMAGE TOWER — per picture, before the prefill ${tipIcon(towerTip)}</div>`
            + row([box("host", "vision-preprocess", "Preprocess", `resize to a grid (patch ${v.patch_size}) → patches [N,${patchVector}]`),
                box("device", "vision-head", "HEAD", `patch embedding + position embedding of the grid → hidden [1,N,${v.hidden}]`),
                box("device", "vision-block", `BLOCK × ${v.depth}`, "one graph per grid, weights rebound per block", { tip: `${v.heads} heads, FFN ${v.ffn}, ${v.activation}` }),
                box("device", "vision-merger", "MERGER", `${v.merge_size}×${v.merge_size} merge → image_embeds [1,N/${v.merge_size ** 2},${v.out_hidden}]`),
                box("host", "vision-rows", "→ prefill", "image rows replace the <|image_pad|> embedding rows")]) + "</div>";
    }

    const sampling = `<div class="row">${box("host", "stage-readback", "4. Read back logits", `${d.vocab} values per step${d.vocabPadded !== d.vocab ? ` (of ${d.vocabPadded} padded)` : ""}`)}${arrow}`
        + `${box("host", "stage-sampling", "5. Sampling", `${esc(d.samplingText)} → next token → step 2`, { tip: d.sampling ? "The package's recommended sampling (model card). ?temperature=0 makes it greedy." : "" })}</div>`;
    body.innerHTML = `<div class="row" id="hostRow">${host.join(arrow)}</div>` + tower + prefill + decode + sampling;
}

/** the session-dependent labels: blocks per graph (decode / prefill), the flush box (staged KV only), the staging rows. session = { blocksPerGraph, prefillBlocksPerGraph, kvStyle, stagingRows } */
export function updatePipelineSession(body, pkg, session) {
    const last = pkg.model.num_layers - 1, set = (selector, text) => { const el = body.querySelector(selector); if (el) el.textContent = text; };
    const groups = (k) => (k > 1 ? `${k} blocks per graph → ${Math.ceil(last / k)} dispatches` : "one block per graph");
    set(".pl-decode-groups", groups(session.blocksPerGraph || 1));
    if (session.chunks) set("#chunkSizeLabel", session.chunks);   // before the first load: the chunk size(s) and past length the options on the page would give
    if (session.pastLength) set("#pastLengthLabel", String(session.pastLength));
    set(".pl-prefill-groups", session.prefillBlocksPerGraph > 1 ? ` · prefill: ${groups(session.prefillBlocksPerGraph)}` : "");
    const flush = body.querySelector("#flushRow");
    if (flush) flush.style.display = session.kvStyle === "staged" ? "" : "none";
    if (session.stagingRows) set("#stagingRowsLabel", String(session.stagingRows));
    set(".pl-kv", session.kvStyle ? `KV update of this session: ${KV_STYLES[session.kvStyle] ?? session.kvStyle}` : "");
}

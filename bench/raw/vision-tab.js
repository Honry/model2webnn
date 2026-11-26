// The "Vision" tab of chat.html (packages with an image tower): pick / drop / paste a picture, see which grid preset the runtime will give it (chosen AUTOMATICALLY from the size of the picture and the
// Detail level: runtime chooseGrid), ask a question, read the answer and the time split (preprocess, graph build on the first use of a grid, vision encode, prefill, TTFT, TPOT).
// The tab shares the model / session of the page with the Chat tab (one load of the weights): `host.generate` is the page's generate with an image.
//
// host = { byId, log, spec: model.json `vision`, presets: [[h, w], ...] (the candidates of the picture: every preset of the package, or the ?visionGrids= override), detail: initial detail level,
//          generate(question, maxNewTokens, { image: { source, name, detail }, onText }) -> result, isBusy() }
import { chooseGrid, describeGrid } from "../../lib/runtime/index.js";
import { answerView } from "./answer-view.js?v=1";

const QUESTIONS = {
    describe: "Describe this image in detail.",
    ocr: "Read all the text in this image. Reply with the text only.",
    count: "How many distinct objects are in this image? Count them and list them.",
    colors: "What are the main colors in this image, and where do they appear?",
};
const seconds = (ms) => (ms / 1000).toFixed(2) + " s";
const ms = (value) => Math.round(value) + " ms";

export function setupVisionTab(host) {
    const { byId, spec } = host;
    const state = { bitmap: null, name: "", detail: host.detail ?? "auto", result: null };
    const grids = host.presets, answer = answerView(byId("visionAnswer"));   // Markdown | Raw view of the answer (the returned result and the log keep the raw text)
    byId("detailSelect").value = state.detail;
    byId("visionGreedy").checked = !!host.greedy;

    /** what the runtime does with a picture of this size at this detail level (also drawn: the picture stretched to the pixels the model sees, with the image-token blocks) */
    function describePicture() {
        const view = byId("modelView"), caption = byId("visionCaption");
        if (!state.bitmap) { view.style.display = "none"; caption.textContent = "No picture yet."; return null; }
        const width = state.bitmap.width, height = state.bitmap.height, grid = chooseGrid(grids, width, height, spec.patch_size, state.detail);
        const pixelsW = grid[1] * spec.patch_size, pixelsH = grid[0] * spec.patch_size, tokens = (grid[0] * grid[1]) / (spec.merge_size ** 2);
        view.width = pixelsW; view.height = pixelsH; view.style.display = "";
        const context = view.getContext("2d");
        context.imageSmoothingQuality = "high";
        context.drawImage(state.bitmap, 0, 0, pixelsW, pixelsH);
        if (byId("showBlocks").checked) {   // one block = one image token (2 x 2 patches of 16 px)
            const block = spec.patch_size * spec.merge_size;
            context.strokeStyle = "rgba(108, 182, 255, 0.55)"; context.lineWidth = 1; context.beginPath();
            for (let x = block; x < pixelsW; x += block) { context.moveTo(x + 0.5, 0); context.lineTo(x + 0.5, pixelsH); }
            for (let y = block; y < pixelsH; y += block) { context.moveTo(0, y + 0.5); context.lineTo(pixelsW, y + 0.5); }
            context.stroke();
        }
        const squeezed = Math.abs(Math.log((pixelsW / pixelsH) / (width / height)));
        caption.innerHTML = `<b>${state.name || "picture"}</b>: ${describeGrid(grid, width, height, spec.patch_size, spec.merge_size, state.detail)}`
            + (squeezed > 0.08 ? `<br>The picture is stretched ${(Math.exp(squeezed)).toFixed(2)}x to fit the preset's aspect ratio.` : "")
            + (grids.length < 2 ? `<br>Only one preset is allowed (URL override): ${grids[0].join("x")}.` : "");
        return grid;
    }

    function setPicture(bitmap, name) {
        state.bitmap?.close?.();
        state.bitmap = bitmap; state.name = name;
        byId("visionRun").disabled = !bitmap || host.isBusy();
        describePicture();
    }
    async function fromFile(file) {
        if (!file || !file.type.startsWith("image/")) return;
        setPicture(await createImageBitmap(file), file.name);
    }
    async function fromUrl(url, name) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
        setPicture(await createImageBitmap(await response.blob()), name);
    }

    // picture sources: file picker, drop zone, paste, samples (the images of the vision suite)
    byId("imageInput").onchange = () => fromFile(byId("imageInput").files[0]);
    const zone = byId("dropZone");
    zone.ondragover = (event) => { event.preventDefault(); zone.classList.add("over"); };
    zone.ondragleave = () => zone.classList.remove("over");
    zone.ondrop = (event) => { event.preventDefault(); zone.classList.remove("over"); fromFile(event.dataTransfer.files[0]); };
    byId("imageInput").onclick = (event) => event.stopPropagation();   // the input sits inside the zone: its own click must not reach the zone
    zone.onclick = () => byId("imageInput").click();
    zone.onkeydown = (event) => { if (event.key === "Enter" || event.key === " ") byId("imageInput").click(); };
    document.addEventListener("paste", (event) => { if (byId("panel-vision").style.display !== "none") fromFile([...event.clipboardData.files][0]); });
    const samples = byId("sampleSelect");
    samples.add(new Option("(none)", ""));
    let sampleCases = [];
    fetch(new URL("vision-suite/cases.json", location.href)).then((r) => (r.ok ? r.json() : null)).then((json) => {
        sampleCases = json?.cases ?? [];
        for (const c of sampleCases) samples.add(new Option(c.id, c.id));
    }).catch(() => {});
    samples.onchange = async () => {
        const c = sampleCases.find((x) => x.id === samples.value);
        if (!c) return;
        await fromUrl(new URL(`vision-suite/${c.image}`, location.href), c.image);
        byId("visionQuestion").value = c.question;
    };

    byId("detailSelect").onchange = () => { state.detail = byId("detailSelect").value; describePicture(); host.onDetail?.(state.detail); };
    byId("showBlocks").onchange = describePicture;
    for (const button of document.querySelectorAll("[data-question]")) button.onclick = () => { byId("visionQuestion").value = QUESTIONS[button.dataset.question]; };
    byId("visionQuestion").value = QUESTIONS.describe;

    function showResult(result) {
        const full = result.promptTokens + result.generated >= result.maxLength - 1;
        if (full) answer.set(answer.text + `\n\n[stopped: the KV cache of this session holds ${result.maxLength} tokens (${result.promptTokens} prompt + ${result.generated} generated). Open the page with a longer max length that the package has.]`);
        answer.flush();
        const rows = [
            ["Preprocess (canvas resize + patchify)", ms(result.preprocessMs), "host"],
            ["Graph build (first use of this grid only)", result.buildMs ? ms(result.buildMs) : "0 (already built)", result.buildMs ? "host" : "dim"],
            [`Vision encode (${spec.depth} blocks)`, ms(result.visionMs), "device"],
            ["Prefill of the rest (text + image rows into the language model)", ms(Math.max(0, result.prefillMs - result.visionMs - result.buildMs)), "device"],
            ["TTFT (prefill incl. vision encode and graph build)", seconds(result.prefillMs), "total"],
            ["Time to first token from the click (+ preprocess)", seconds(result.prefillMs + result.preprocessMs), "total"],
            ["TPOT", `${result.tpotMs.toFixed(1)} ms/token (${result.tps.toFixed(1)} tokens/s)`, "host"],
            ["Tokens", `prompt ${result.promptTokens} (image ${result.imageTokens}), generated ${result.generated}`, "dim"],
            ["Session", `max length ${result.maxLength}: ${result.promptTokens + result.generated} of ${result.maxLength - 1} cache rows used${full ? " (FULL: the answer was cut)" : ""}`, full ? "host" : "dim"],
            ["Grid", `${result.grid.join("x")} = ${result.pixels} px, ${result.detail}`, "dim"],
        ];
        const table = byId("visionTiming");
        table.innerHTML = "<tr><th>step</th><th>time</th></tr>" + rows.map(([label, value, kind]) => `<tr class="v-${kind}"><td>${label}</td><td>${value}</td></tr>`).join("");
    }

    byId("visionRun").onclick = async () => {
        if (!state.bitmap) return;
        answer.set("");
        try {
            const result = await host.generate(byId("visionQuestion").value, +byId("visionMaxNew").value || 128,
                { image: { source: state.bitmap, name: state.name, detail: state.detail }, ...(byId("visionGreedy").checked ? { sampling: { temperature: 0 } } : {}), onText: (text) => answer.set(text) });
            answer.flush();
            state.result = result;
            if (result) showResult(result);
        } catch (error) {
            answer.set("FAILED: " + error); answer.flush();
            host.log("fatal", "Vision generation failed: " + String(error && error.stack ? error.stack : error).slice(0, 1500));
        }
    };
    describePicture();
    return {
        setSession: (text) => { byId("visionSession").textContent = text; },
        setBusy: (busy) => { byId("visionRun").disabled = busy || !state.bitmap; },
        loadUrl: fromUrl,
        hasPicture: () => !!state.bitmap,
        run: () => byId("visionRun").onclick(),
        setQuestion: (text) => { byId("visionQuestion").value = text; },
        get detail() { return state.detail; },
    };
}

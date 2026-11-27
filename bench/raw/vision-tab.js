// The "Vision" tab of chat.html (packages with an image tower): pick / drop / paste a picture, see which grid preset the runtime will give it (chosen AUTOMATICALLY from the size of the picture and the
// Detail level: runtime chooseGrid), ask a question. The pictures are a LIST of items (one block each; the runtime takes one per prompt today).
// The tab shares the model / session AND the results of the page with the Chat tab (answer card, metrics, Summary, decode table): `host.generate` is the page's generate with an image, and a picture only adds
// rows to the Summary (showImages: grid, image tokens, preprocess, graph build on the first use of a grid, vision encode).
//
// host = { byId, log, spec: model.json `vision`, presets: [[h, w], ...] (the candidates of the picture: every preset of the package, or the ?visionGrids= override), detail: initial detail level,
//          generate(question, maxNewTokens, { image: { source, name, detail }, sampling }) -> result, isBusy() }
import { chooseGrid, describeGrid } from "../../lib/runtime/index.js";
import { answerView } from "./answer-view.js?v=1";

const QUESTIONS = {
    describe: "Describe this image in detail.",
    ocr: "Read all the text in this image. Reply with the text only.",
    count: "How many distinct objects are in this image? Count them and list them.",
    colors: "What are the main colors in this image, and where do they appear?",
};
const ms = (value) => Math.round(value) + " ms";

export function setupVisionTab(host) {
    const { byId, spec } = host;
    const state = { items: [], detail: host.detail ?? "auto" };   // items: the pictures of the prompt, [{ bitmap, name }]; the runtime takes ONE picture per prompt today, the page is built for a list
    const grids = host.presets;
    byId("detailSelect").value = state.detail;
    byId("visionGreedy").checked = !!host.greedy;

    /** what the runtime does with each picture at this detail level: one block per item (also drawn: the picture stretched to the pixels the model sees, with the image-token blocks) */
    function describePictures() {
        const container = byId("imageItems");
        container.replaceChildren();
        if (!state.items.length) { const empty = document.createElement("div"); empty.className = "imageempty"; empty.textContent = "No picture yet."; container.append(empty); return; }
        state.items.forEach((item, index) => {
            const { bitmap, name } = item, width = bitmap.width, height = bitmap.height, grid = chooseGrid(grids, width, height, spec.patch_size, state.detail);
            const pixelsW = grid[1] * spec.patch_size, pixelsH = grid[0] * spec.patch_size;
            const block = document.createElement("div"), view = document.createElement("canvas"), caption = document.createElement("div");
            block.className = "imageitem"; caption.className = "imagecaption";
            view.title = "what the model sees: the picture resized to the pixels of the chosen grid; every block is one image token";
            view.width = pixelsW; view.height = pixelsH;
            const context = view.getContext("2d");
            context.imageSmoothingQuality = "high";
            context.drawImage(bitmap, 0, 0, pixelsW, pixelsH);
            if (byId("showBlocks").checked) {   // one block = one image token (2 x 2 patches of 16 px)
                const cell = spec.patch_size * spec.merge_size;
                context.strokeStyle = "rgba(108, 182, 255, 0.55)"; context.lineWidth = 1; context.beginPath();
                for (let x = cell; x < pixelsW; x += cell) { context.moveTo(x + 0.5, 0); context.lineTo(x + 0.5, pixelsH); }
                for (let y = cell; y < pixelsH; y += cell) { context.moveTo(0, y + 0.5); context.lineTo(pixelsW, y + 0.5); }
                context.stroke();
            }
            const squeezed = Math.abs(Math.log((pixelsW / pixelsH) / (width / height)));
            caption.innerHTML = `<b>${state.items.length > 1 ? `Image ${index + 1}: ` : ""}${name || "picture"}</b>: ${describeGrid(grid, width, height, spec.patch_size, spec.merge_size, state.detail)}`
                + (squeezed > 0.08 ? `<br>The picture is stretched ${(Math.exp(squeezed)).toFixed(2)}x to fit the preset's aspect ratio.` : "")
                + (grids.length < 2 ? `<br>Only one preset is allowed (URL override): ${grids[0].join("x")}.` : "");
            block.append(view, caption); container.append(block);
        });
    }

    function setPicture(bitmap, name) {   // today a new picture replaces the one of the prompt; several would be pushed onto state.items
        for (const item of state.items) item.bitmap.close?.();
        state.items = bitmap ? [{ bitmap, name }] : [];
        byId("visionRun").disabled = !state.items.length || host.isBusy();
        byId("visionHint").textContent = state.items.length ? "Loads the model first when it is not loaded yet." : "Add a picture first.";
        describePictures();
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

    byId("detailSelect").onchange = () => { state.detail = byId("detailSelect").value; describePictures(); host.onDetail?.(state.detail); };
    byId("showBlocks").onchange = describePictures;
    for (const button of document.querySelectorAll("[data-question]")) button.onclick = () => { byId("visionQuestion").value = QUESTIONS[button.dataset.question]; };
    byId("visionQuestion").value = QUESTIONS.describe;
    byId("visionQuestion").addEventListener("keydown", (event) => { if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !byId("visionRun").disabled) byId("visionRun").click(); });

    /** the image rows of the page's Summary (the table a text prompt fills too; a picture only adds these): one group per picture, [] clears them */
    function showImages(images) {
        const body = byId("sumImages");
        body.replaceChildren();
        const add = (label, value, className, tip = "") => {
            const row = body.insertRow(); row.className = className;
            const name = row.insertCell(); name.textContent = label; if (tip) { name.dataset.tip = tip; name.classList.add("has-tip"); }
            row.insertCell().textContent = value;
        };
        images.forEach((image, index) => {
            const head = body.insertRow(); head.className = "img-head"; const title = head.insertCell(); title.colSpan = 2; title.textContent = `Image ${index + 1}: ${image.name || "picture"}`;
            add("Grid", `${image.grid.join("x")} = ${image.pixels} px, ${image.detail}`, "img-row", "the grid preset the runtime chose from the size of the picture and the Detail level");
            add("Image tokens", String(image.tokens), "img-row", "tokens this picture adds to the prompt");
            add("Preprocess", ms(image.preprocessMs), "img-row host", "canvas resize + patchify (JavaScript)");
            add("Graph build", image.buildMs ? ms(image.buildMs) : "0 (already built)", "img-row host", "the first use of a grid builds its vision graphs; not part of the vision encode");
            add(`Vision encode (${spec.depth} blocks)`, ms(image.visionMs), "img-row device", "image tower on the device; part of the prefill time above");
        });
    }

    byId("visionRun").onclick = async () => {
        if (!state.items.length) return;
        const answer = answerView(byId("answer"));   // the answer card is the page's (the same as the Chat tab): the generation writes into it
        answer.set("");
        try {
            const [{ bitmap, name }] = state.items;   // one picture per prompt today
            await host.generate(byId("visionQuestion").value, +byId("visionMaxNew").value || 128,
                { image: { source: bitmap, name, detail: state.detail }, ...(byId("visionGreedy").checked ? { sampling: { temperature: 0 } } : {}) });
        } catch (error) {
            answer.set("FAILED: " + error); answer.flush();
            host.log("fatal", "Vision generation failed: " + String(error && error.stack ? error.stack : error).slice(0, 1500));
        }
    };
    describePictures();
    return {
        showImages,
        setBusy: (busy) => { byId("visionRun").disabled = busy || !state.items.length; },
        loadUrl: fromUrl,
        hasPicture: () => state.items.length > 0,
        run: () => byId("visionRun").onclick(),
        setQuestion: (text) => { byId("visionQuestion").value = text; },
        get detail() { return state.detail; },
    };
}

// The conversation and the composer of chat.html: the message list (user / assistant bubbles, notices about dropped turns) and the pictures attached to the message that is being written.
// No model code here: chat.js owns the turns and the generation; this file only draws and collects.
//
//   createConversation(list, { onView })      -> { addUser, addAssistant, addNotice, markDropped, clear, scrollToEnd }     onView(image, element): a picture of a message was clicked
//   createAttachments(host)                   -> { items, add, addFile, addUrl, remove, take, restore, clear, setDetail, view, refresh }
//   renderImageRows(body, images, spec)       the image rows of the Summary table: one group per picture of the last prompt
//        host = { byId, overlays (overlay.js), spec: model.json `vision`, presets: [[h, w], ...], detail, onChange(), setText(text), log }
import { chooseGrid, describeGrid, imageTokenCount } from "../../lib/runtime/index.js";
import { answerView } from "./answer-view.js?v=3";
import { createAutoScroll } from "./autoscroll.js?v=1";

const el = (tag, className, text) => { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; };
/** an inline SVG icon from the sprite of chat.html */
export function icon(name, className = "") {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"), use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    svg.setAttribute("class", `ic ${className}`.trim()); use.setAttribute("href", `#i-${name}`); svg.append(use);
    return svg;
}

/** a small canvas with the picture centre-cropped to a square; its size comes from CSS (--thumb), the pixels from the device pixel ratio; it never takes the pointer (the chip / bubble around it does) */
function thumbnail(bitmap, size) {
    const canvas = el("canvas", "thumb"), side = Math.min(bitmap.width, bitmap.height), pixels = Math.round(size * Math.min(window.devicePixelRatio || 1, 3));
    canvas.width = canvas.height = pixels; canvas.style.setProperty("--thumb", `${size}px`);
    canvas.getContext("2d").drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, pixels, pixels);
    return canvas;
}
export const imageSummary = (image) => `${image.tokens} image tokens (${image.grid.join("x")}, ${image.detail})`;

export function createConversation(list, { onView = null, jumpButton = null } = {}) {
    const empty = list.querySelector(".chat-empty") ?? el("div", "chat-empty");   // the empty state of the page: a title, the model name, suggestions
    if (!empty.parentNode) list.append(empty);
    // the list follows the bottom while it is pinned there (autoscroll.js watches the content: tokens, Markdown re-renders, bubbles, bars, chips); scrolling up unpins and shows the jump button
    const autoScroll = createAutoScroll(list, { button: jumpButton });
    const scrollToEnd = (force = false) => (force ? autoScroll.pin() : autoScroll.schedule());
    const views = [];
    const append = (node) => { empty.remove(); list.append(node); scrollToEnd(); return node; };
    return {
        /** turn: { text, images: [{ bitmap, name, ... }] } */
        addUser(turn) {
            const message = el("div", "msg user"), bubble = el("div", "bubble");
            for (const image of turn.images ?? []) {
                const item = el("div", "attached");
                item.append(thumbnail(image.bitmap, 56), el("span", "attached-text", `${image.name || "picture"} · ${imageSummary(image)}`));
                if (onView) { item.tabIndex = 0; item.setAttribute("role", "button"); item.title = "view the picture as the model sees it"; item.onclick = () => onView(image, item); item.onkeydown = (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); onView(image, item); } }; }
                bubble.append(item);
            }
            bubble.append(el("div", "text", turn.text));
            message.append(bubble);
            autoScroll.pin();   // sending a message always pins to the bottom again
            return append(message);
        },
        /** an assistant message: { element, box, view, setMeta(text, state) } (an avatar, the answer, a toolbar with the stats of the turn and the Markdown | Raw / Copy bar) */
        addAssistant() {
            const message = el("div", "msg assistant"), avatar = el("div", "avatar"), body = el("div", "body"), box = el("div", "answer-box"), toolbar = el("div", "toolbar"), meta = el("span", "meta");
            box.dataset.placeholder = "…";
            const gp = el("div", "gp"), gpBar = el("div", "gp-bar"), gpFill = el("i"), gpText = el("span", "gp-text");   // progress while the prompt is read (before the first token)
            gp.hidden = true; gpBar.append(gpFill); gp.append(gpBar, gpText);
            avatar.append(icon("zap")); toolbar.append(meta, el("span", "spacer")); body.append(gp, box, toolbar); message.append(avatar, body);
            append(message);
            const view = answerView(box, { barParent: toolbar });
            views.push(view);
            return { element: message, box, view,
                /** { label, fraction (0..1, null = indeterminate) } or null to hide the bar */
                setProgress(info) {
                    gp.hidden = !info;
                    if (!info) return;
                    gpText.textContent = info.label; gpFill.classList.toggle("pulse", info.fraction == null);
                    gpFill.style.width = info.fraction == null ? "100%" : `${Math.round(info.fraction * 100)}%`; scrollToEnd();
                },
                /** text: "stopped · TTFT 1.6 s · TPOT 58 ms/token · 27 + 64 tokens": one small chip per part */
                setMeta(text, state = "") {
                    meta.replaceChildren();
                    for (const part of text.split(" · ").filter(Boolean)) meta.append(el("span", "mchip" + (/^TTFT/.test(part) ? " ttft" : /^TPOT/.test(part) ? " tpot" : /tokens$/.test(part) ? " tok" : /stopped|full|cut/i.test(part) ? " warn" : ""), part));
                    message.dataset.state = state; scrollToEnd();
                } };
        },
        /** a dimmed line between messages; `after`: put it behind this message (where turns were dropped), default at the end; warn: amber */
        addNotice(text, { after = null, warn = false } = {}) {
            const notice = el("div", "notice" + (warn ? " warn" : ""), text);
            if (!after) return append(notice);
            empty.remove(); after.after(notice); return notice;
        },
        markDropped(elements, on = true) { for (const element of elements) element.classList.toggle("dropped", on); },
        clear() {
            for (const view of views.splice(0)) view.dispose();
            list.replaceChildren(empty);
            autoScroll.pin();
        },
        scrollToEnd,
    };
}

export function createAttachments(host) {
    const { byId, spec } = host;
    const state = { items: [], detail: host.detail ?? "auto", presets: host.presets };
    const chips = byId("composerImages");
    byId("detailSelect").value = state.detail;

    const describe = (bitmap, name, detail) => {
        const grid = chooseGrid(state.presets, bitmap.width, bitmap.height, spec.patch_size, detail);
        return { bitmap, name, detail, grid, tokens: imageTokenCount(grid, spec.merge_size), width: bitmap.width, height: bitmap.height, pixels: `${grid[1] * spec.patch_size}x${grid[0] * spec.patch_size}` };
    };
    // the picture viewer: a lightbox (modal, dimmed backdrop) with the picture as the model sees it (resized to the pixels of its grid) and the image-token blocks; it is open only on request
    const viewer = byId("imagePreview"), canvas = viewer.querySelector("canvas"), title = viewer.querySelector(".lb-title"), caption = viewer.querySelector(".lb-caption"), closeButton = viewer.querySelector(".lb-close");
    const blocksBox = viewer.querySelector(".lb-blocks input");
    let viewing = null, opener = null;
    function draw() {
        if (!viewing) return;
        const { grid } = viewing, w = grid[1] * spec.patch_size, h = grid[0] * spec.patch_size;
        const maxW = Math.max(160, window.innerWidth * 0.9 - 48), maxH = Math.max(120, window.innerHeight * 0.9 - 190);
        const scale = Math.min(maxW / w, maxH / h, 4), cssW = Math.round(w * scale), cssH = Math.round(h * scale), dpr = window.devicePixelRatio || 1;   // fits the window, never overflows
        canvas.width = Math.round(cssW * dpr); canvas.height = Math.round(cssH * dpr); canvas.style.width = `${cssW}px`; canvas.style.height = `${cssH}px`;
        const model = document.createElement("canvas"), modelContext = model.getContext("2d");   // the model's own pixels first, then scaled to the window
        model.width = w; model.height = h; modelContext.imageSmoothingQuality = "high"; modelContext.drawImage(viewing.bitmap, 0, 0, w, h);
        const context = canvas.getContext("2d");
        context.imageSmoothingEnabled = true; context.imageSmoothingQuality = "high"; context.drawImage(model, 0, 0, canvas.width, canvas.height);
        if (blocksBox.checked) {   // one block = one image token (merge x merge patches)
            const cell = spec.patch_size * spec.merge_size * scale * dpr;
            context.strokeStyle = "rgba(108, 182, 255, 0.7)"; context.lineWidth = Math.max(1, Math.round(dpr)); context.beginPath();
            for (let x = cell; x < canvas.width - 1; x += cell) { context.moveTo(Math.round(x) + 0.5, 0); context.lineTo(Math.round(x) + 0.5, canvas.height); }
            for (let y = cell; y < canvas.height - 1; y += cell) { context.moveTo(0, Math.round(y) + 0.5); context.lineTo(canvas.width, Math.round(y) + 0.5); }
            context.stroke();
        }
        const squeezed = Math.abs(Math.log((w / h) / (viewing.width / viewing.height)));
        title.textContent = viewing.name || "picture";
        caption.textContent = describeGrid(grid, viewing.width, viewing.height, spec.patch_size, spec.merge_size, viewing.detail)
            + (squeezed > 0.08 ? ` The picture is stretched ${Math.exp(squeezed).toFixed(2)}x to fit the preset's aspect ratio.` : "")
            + (state.presets.length < 2 ? ` Only one preset is allowed (URL override): ${state.presets[0].join("x")}.` : "");
    }
    function closeViewer() { viewer.hidden = true; viewing = null; host.overlays.release("lightbox"); opener?.focus?.(); opener = null; }
    function openViewer(image, from = null) {
        viewing = image; opener = from ?? document.activeElement;
        blocksBox.checked = byId("showBlocks").checked;
        host.overlays.open("lightbox", { close: closeViewer, contains: (target) => !!target.closest?.(".lb-card") });   // closes a popover / the drawer; Esc and a click on the backdrop close this
        viewer.hidden = false; draw(); closeButton.focus();
    }
    closeButton.onclick = closeViewer;
    viewer.querySelector(".lb-backdrop").onclick = closeViewer;
    blocksBox.onchange = () => { byId("showBlocks").checked = blocksBox.checked; draw(); };
    window.addEventListener("resize", draw);
    function render() {
        chips.replaceChildren();
        state.items.forEach((image, index) => {
            const chip = el("div", "chip"), text = el("span", "chip-text"), remove = el("button", "chip-x", "×");
            chip.title = "what the model sees (click)"; chip.tabIndex = 0; chip.setAttribute("role", "button"); remove.type = "button"; remove.title = "remove this picture";
            text.append(el("b", "", image.name || "picture"), el("small", "", `${image.grid.join("x")} · ${image.tokens} tokens`));
            chip.append(thumbnail(image.bitmap, 40), text, remove);
            chip.onclick = () => openViewer(image, chip);
            chip.onkeydown = (event) => { if (event.target === chip && (event.key === "Enter" || event.key === " ")) { event.preventDefault(); openViewer(image, chip); } };
            remove.onclick = (event) => { event.stopPropagation(); api.remove(index); };
            chips.append(chip);
        });
        chips.hidden = !state.items.length;
        const quick = byId("quickQuestions"); if (quick) quick.hidden = !state.items.length;   // the quick questions of a picture appear with the first picture
        host.onChange?.();
    }
    const MAX_IMAGES = 4;   // per message (the runtime takes any number; the cache is what runs out)
    const api = {
        get items() { return state.items; },
        add(bitmap, name) {
            if (state.items.length >= MAX_IMAGES) { bitmap.close?.(); host.log?.("warning", `at most ${MAX_IMAGES} pictures per message`); return; }
            state.items.push(describe(bitmap, name, state.detail));
            render();
        },
        async addFile(file) { if (file && file.type.startsWith("image/")) api.add(await createImageBitmap(file), file.name); },
        async addUrl(url, name) {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
            api.add(await createImageBitmap(await response.blob()), name);
        },
        remove(index) { const [gone] = state.items.splice(index, 1); if (viewing === gone) closeViewer(); gone?.bitmap.close?.(); render(); },
        /** hand the pictures over to a sent message (the bitmaps now belong to it) */
        take() { const taken = state.items.splice(0); render(); return taken; },
        /** give pictures back to the composer (a send that was stopped before the first token) */
        restore(images) { state.items.unshift(...images); render(); },
        clear() { for (const image of state.items.splice(0)) image.bitmap.close?.(); render(); },
        /** the lightbox for a picture (a pending one or the picture of a sent message) */
        view: openViewer,
        get detail() { return state.detail; },
        setDetail(detail) { state.detail = detail; byId("detailSelect").value = detail; state.items = state.items.map((image) => describe(image.bitmap, image.name, detail)); render(); },
        refresh: render,
    };

    // sources: file picker (the Attach button), drop on the composer, paste, the samples of the vision suite
    const input = byId("imageInput");
    input.onchange = async () => { for (const file of input.files) await api.addFile(file); input.value = ""; };
    byId("attachButton").onclick = () => input.click();
    const composer = byId("composer");
    composer.addEventListener("dragover", (event) => { event.preventDefault(); composer.classList.add("over"); });
    composer.addEventListener("dragleave", (event) => { if (!composer.contains(event.relatedTarget)) composer.classList.remove("over"); });
    composer.addEventListener("drop", async (event) => { event.preventDefault(); composer.classList.remove("over"); for (const file of event.dataTransfer.files) await api.addFile(file); });
    document.addEventListener("paste", async (event) => { for (const file of [...(event.clipboardData?.files ?? [])]) await api.addFile(file); });
    const samples = byId("sampleSelect");
    samples.add(new Option("", ""));
    let sampleCases = [];
    fetch(new URL("vision-suite/cases.json", location.href)).then((r) => (r.ok ? r.json() : null)).then((json) => {
        sampleCases = json?.cases ?? [];
        for (const c of sampleCases) samples.add(new Option(c.id, c.id));
    }).catch(() => {});
    samples.onchange = async () => {
        const c = sampleCases.find((x) => x.id === samples.value);
        samples.value = "";
        if (!c) return;
        await api.addUrl(new URL(`vision-suite/${c.image}`, location.href), c.image);
        host.setText(c.question);
    };
    byId("detailSelect").onchange = () => { api.setDetail(byId("detailSelect").value); host.onDetail?.(state.detail); };
    render();
    return api;
}

const ms = (value) => Math.round(value) + " ms";
/** the Summary of the page is the same for a text prompt and a prompt with pictures; pictures only add these rows (a group per picture, then the encode time of the tower) */
export function renderImageRows(body, images, spec, visionMs = 0) {
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
    });
    if (images.length) add(`Vision encode (${spec.depth} blocks)`, ms(visionMs), "img-row device", `image tower on the device, ${images.length === 1 ? "this picture" : `all ${images.length} pictures of the prompt`}; part of the prefill time above`);
}

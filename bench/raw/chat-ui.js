// The conversation and the composer of chat.html: the message list (user / assistant bubbles, notices about dropped turns) and the pictures attached to the message that is being written.
// No model code here: chat.js owns the turns and the generation; this file only draws and collects.
//
//   createConversation(list)                  -> { addUser, addAssistant, addNotice, markDropped, clear, scrollToEnd }
//   createAttachments(host)                   -> { items, add, addFile, addUrl, remove, take, restore, clear, setDetail, refresh }
//   renderImageRows(body, images, spec)       the image rows of the Summary table: one group per picture of the last prompt
//        host = { byId, spec: model.json `vision`, presets: [[h, w], ...], detail, onChange(), setText(text), log }
import { chooseGrid, describeGrid, imageTokenCount } from "../../lib/runtime/index.js";
import { answerView } from "./answer-view.js?v=3";

const el = (tag, className, text) => { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; };
/** an inline SVG icon from the sprite of chat.html */
export function icon(name, className = "") {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg"), use = document.createElementNS("http://www.w3.org/2000/svg", "use");
    svg.setAttribute("class", `ic ${className}`.trim()); use.setAttribute("href", `#i-${name}`); svg.append(use);
    return svg;
}

/** a small canvas with the picture centre-cropped to a square */
function thumbnail(bitmap, size) {
    const canvas = el("canvas", "thumb"), side = Math.min(bitmap.width, bitmap.height);
    canvas.width = canvas.height = size;
    canvas.getContext("2d").drawImage(bitmap, (bitmap.width - side) / 2, (bitmap.height - side) / 2, side, side, 0, 0, size, size);
    return canvas;
}
export const imageSummary = (image) => `${image.tokens} image tokens (${image.grid.join("x")}, ${image.detail})`;

export function createConversation(list) {
    const empty = list.querySelector(".chat-empty") ?? el("div", "chat-empty");   // the empty state of the page: a title, the model name, suggestions
    if (!empty.parentNode) list.append(empty);
    const follow = () => list.scrollHeight - list.scrollTop - list.clientHeight < 60;
    const scrollToEnd = (force = false) => { if (force || list.dataset.follow !== "off") list.scrollTop = list.scrollHeight; };
    list.addEventListener("scroll", () => { list.dataset.follow = follow() ? "on" : "off"; });
    const views = [];
    const append = (node) => { empty.remove(); list.append(node); scrollToEnd(); return node; };
    return {
        /** turn: { text, images: [{ bitmap, name, ... }] } */
        addUser(turn) {
            const message = el("div", "msg user"), bubble = el("div", "bubble");
            for (const image of turn.images ?? []) {
                const item = el("div", "attached");
                item.append(thumbnail(image.bitmap, 56), el("span", "attached-text", `${image.name || "picture"} · ${imageSummary(image)}`));
                bubble.append(item);
            }
            bubble.append(el("div", "text", turn.text));
            message.append(bubble);
            list.dataset.follow = "on";
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
            list.dataset.follow = "on";
        },
        scrollToEnd,
    };
}

export function createAttachments(host) {
    const { byId, spec } = host;
    const state = { items: [], detail: host.detail ?? "auto", presets: host.presets, selected: -1 };
    const chips = byId("composerImages"), preview = byId("imagePreview");
    byId("detailSelect").value = state.detail;

    const describe = (bitmap, name, detail) => {
        const grid = chooseGrid(state.presets, bitmap.width, bitmap.height, spec.patch_size, detail);
        return { bitmap, name, detail, grid, tokens: imageTokenCount(grid, spec.merge_size), width: bitmap.width, height: bitmap.height, pixels: `${grid[1] * spec.patch_size}x${grid[0] * spec.patch_size}` };
    };
    /** the picture as the model sees it (resized to the pixels of its grid), with the image-token blocks */
    function drawPreview() {
        preview.replaceChildren();
        const image = state.items[state.selected];
        preview.hidden = !image;
        if (!image) return;
        const { grid } = image, pixelsW = grid[1] * spec.patch_size, pixelsH = grid[0] * spec.patch_size, view = el("canvas");
        view.width = pixelsW; view.height = pixelsH; view.title = "what the model sees: the picture resized to the pixels of the chosen grid; every block is one image token";
        const context = view.getContext("2d");
        context.imageSmoothingQuality = "high";
        context.drawImage(image.bitmap, 0, 0, pixelsW, pixelsH);
        if (byId("showBlocks").checked) {   // one block = one image token (2 x 2 patches of 16 px)
            const cell = spec.patch_size * spec.merge_size;
            context.strokeStyle = "rgba(108, 182, 255, 0.55)"; context.lineWidth = 1; context.beginPath();
            for (let x = cell; x < pixelsW; x += cell) { context.moveTo(x + 0.5, 0); context.lineTo(x + 0.5, pixelsH); }
            for (let y = cell; y < pixelsH; y += cell) { context.moveTo(0, y + 0.5); context.lineTo(pixelsW, y + 0.5); }
            context.stroke();
        }
        const squeezed = Math.abs(Math.log((pixelsW / pixelsH) / (image.width / image.height))), caption = el("div", "imagecaption");
        caption.innerHTML = `<b></b>: ${describeGrid(grid, image.width, image.height, spec.patch_size, spec.merge_size, image.detail)}`
            + (squeezed > 0.08 ? `<br>The picture is stretched ${Math.exp(squeezed).toFixed(2)}x to fit the preset's aspect ratio.` : "")
            + (state.presets.length < 2 ? `<br>Only one preset is allowed (URL override): ${state.presets[0].join("x")}.` : "");
        caption.querySelector("b").textContent = image.name || "picture";
        const label = el("label", "blocks"), box = el("input"); box.type = "checkbox"; box.checked = byId("showBlocks").checked;
        box.onchange = () => { byId("showBlocks").checked = box.checked; drawPreview(); };
        label.append(box, " show the image-token blocks");
        const side = el("div"); side.append(caption, label);
        preview.append(view, side);
    }
    function render() {
        chips.replaceChildren();
        state.items.forEach((image, index) => {
            const chip = el("div", "chip" + (index === state.selected ? " on" : "")), text = el("span", "chip-text"), remove = el("button", "chip-x", "×");
            chip.title = "what the model sees (click)"; remove.type = "button"; remove.title = "remove this picture";
            text.append(el("b", "", image.name || "picture"), el("small", "", `${image.grid.join("x")} · ${image.tokens} tokens`));
            chip.append(thumbnail(image.bitmap, 40), text, remove);
            chip.onclick = () => { state.selected = state.selected === index ? -1 : index; render(); };
            remove.onclick = (event) => { event.stopPropagation(); api.remove(index); };
            chips.append(chip);
        });
        chips.hidden = !state.items.length;
        const quick = byId("quickQuestions"); if (quick) quick.hidden = !state.items.length;   // the quick questions of a picture appear with the first picture
        drawPreview();
        host.onChange?.();
    }
    const MAX_IMAGES = 4;   // per message (the runtime takes any number; the cache is what runs out)
    const api = {
        get items() { return state.items; },
        add(bitmap, name) {
            if (state.items.length >= MAX_IMAGES) { bitmap.close?.(); host.log?.("warning", `at most ${MAX_IMAGES} pictures per message`); return; }
            state.items.push(describe(bitmap, name, state.detail));
            state.selected = state.items.length - 1;
            render();
        },
        async addFile(file) { if (file && file.type.startsWith("image/")) api.add(await createImageBitmap(file), file.name); },
        async addUrl(url, name) {
            const response = await fetch(url);
            if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
            api.add(await createImageBitmap(await response.blob()), name);
        },
        remove(index) { const [gone] = state.items.splice(index, 1); gone?.bitmap.close?.(); state.selected = Math.min(state.selected, state.items.length - 1); render(); },
        /** hand the pictures over to a sent message (the bitmaps now belong to it) */
        take() { const taken = state.items.splice(0); state.selected = -1; render(); return taken; },
        /** give pictures back to the composer (a send that was stopped before the first token) */
        restore(images) { state.items.unshift(...images); state.selected = -1; render(); },
        clear() { for (const image of state.items.splice(0)) image.bitmap.close?.(); state.selected = -1; render(); },
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
    byId("showBlocks").onchange = drawPreview;
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

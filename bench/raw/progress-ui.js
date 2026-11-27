// Progress of loading a model, drawn from the runtime's progress events ({ stage, message, loaded, total, fromCache }, see Progress in src/runtime/package.ts):
// a stepper Download weights -> Build graphs -> Upload to device -> (Image tower) -> Ready with one slim bar per step, a caption with the numbers, and one slim line along the top bar.
// The steps overlap (the weights keep downloading while the graphs are built), so every step has its own bar. The same stepper is drawn into every root element (the session card, the empty chat).
//
//   const progress = createLoadProgress({ roots: [element, ...], topBar: element })
//   progress.begin(fresh)   fresh: the first load (download + upload); false: a reconfigure (only the graphs are rebuilt)
//   progress.event(p)       a runtime event       progress.fail(message)       progress.done()      progress.hide()
const el = (tag, className, text) => { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; };
const mb = (bytes) => (bytes / 2 ** 20 >= 100 ? (bytes / 2 ** 20).toFixed(0) : (bytes / 2 ** 20).toFixed(1));

const STEPS = [
    { key: "download", label: "Download", weight: 0.4 },
    { key: "build", label: "Build graphs", weight: 0.3 },
    { key: "upload", label: "Upload", weight: 0.3 },
];

export function createLoadProgress({ roots, topBar }) {
    let steps = [], dlText = "", actText = "", mode = "idle", fresh = true, downloadStart = 0, downloadBase = 0, hideTimer = null;   // mode: idle | running | error | done
    const views = roots.map((root) => { root.replaceChildren(); const list = el("div", "lp-steps"), text = el("div", "lp-caption"); root.append(list, text); return { root, list, text }; });
    const fill = (value) => `${Math.round(Math.max(0, Math.min(1, value)) * 100)}%`;
    const overall = () => {
        const used = steps.filter((s) => s.weight);
        const total = used.reduce((n, s) => n + s.weight, 0) || 1;
        return used.reduce((n, s) => n + s.weight * s.frac, 0) / total;
    };
    function render() {
        const active = steps.find((s) => s.frac < 1 && !s.kept);
        for (const { root, list, text } of views) {
            root.hidden = mode === "idle";
            root.dataset.state = mode;
            list.replaceChildren(...steps.map((step) => {
                const node = el("div", "lp-step"), bar = el("div", "lp-bar"), inner = el("i");
                node.dataset.step = step.key; node.classList.toggle("on", step === active && mode === "running"); node.classList.toggle("done", step.frac >= 1); node.classList.toggle("kept", !!step.kept);
                inner.style.width = step.indeterminate && step.frac < 1 ? "100%" : fill(step.frac); inner.classList.toggle("pulse", !!step.indeterminate && step.frac < 1);
                bar.append(inner); node.append(el("span", "lp-label", step.label), bar);
                return node;
            }));
            text.textContent = [dlText, actText].filter(Boolean).join("\n");   // the download keeps its own line while the graphs are built
        }
        if (topBar) {
            topBar.hidden = mode === "idle";
            topBar.dataset.state = mode;
            topBar.firstElementChild.style.width = mode === "done" ? "100%" : fill(overall());
        }
    }
    const step = (key) => steps.find((s) => s.key === key);
    const ensure = (key, label, weight = 0) => step(key) ?? (steps.splice(steps.length - 1, 0, { key, label, weight, frac: 0 }), step(key));   // an extra step before Ready
    const api = {
        begin(isFresh = true) {
            clearTimeout(hideTimer); fresh = isFresh; mode = "running"; dlText = ""; actText = "Starting…"; downloadStart = 0;
            steps = STEPS.map((s) => ({ ...s, frac: 0, kept: !isFresh && s.key !== "build" }));
            for (const s of steps) if (s.kept) s.frac = 1;
            steps.push({ key: "ready", label: "Ready", weight: 0, frac: 0 });
            render();
        },
        event(p) {
            if (mode === "done" && p.stage === "vision") { mode = "running"; clearTimeout(hideTimer); const ready = step("ready"); if (ready) ready.frac = 0; }   // the image tower loads after the language model said "ready"
            else if (mode !== "running") api.begin(true);
            const downloadStep = step("download"), build = step("build"), upload = step("upload");
            switch (p.stage) {
                case "package": actText = "Opening the package…"; break;
                case "download": {
                    const now = performance.now();
                    if (!downloadStart && p.loaded > (p.fromCache ?? 0)) { downloadStart = now; downloadBase = p.loaded; }
                    downloadStep.frac = p.total ? p.loaded / p.total : 0; downloadStep.indeterminate = false;
                    const cached = p.fromCache && p.fromCache >= p.loaded * 0.98;
                    const speed = downloadStart && now - downloadStart > 500 ? (p.loaded - downloadBase) / 2 ** 20 / ((now - downloadStart) / 1000) : 0;
                    dlText = downloadStep.frac >= 1 && !cached ? `Weights downloaded · ${mb(p.total)} MB` : cached ? `Weights from the local cache · ${mb(p.loaded)} / ${mb(p.total)} MB`
                        : `Downloading weights · ${mb(p.loaded)} / ${mb(p.total)} MB (${Math.round(downloadStep.frac * 100)}%)${speed >= 0.5 ? ` · ${speed.toFixed(speed >= 10 ? 0 : 1)} MB/s` : ""}${p.fromCache ? " · part from cache" : ""}`;
                    break;
                }
                case "embedding": actText = "Reading the embedding table…"; break;
                case "build":
                    if (/^build vision/.test(p.message ?? "")) break;   // the first use of a picture grid is shown with the message, not here
                    if (p.total) { build.frac = Math.min((p.loaded + 0.5) / p.total, 0.98); build.indeterminate = false;   // p.loaded = graphs finished: the one in progress counts half; the step is full only at "state" / "ready" actText = `Building graph ${Math.min(p.loaded + 1, p.total)} / ${p.total} · ${p.message ?? ""}`; }
                    else { build.indeterminate = true; actText = `Building graph · ${p.message ?? ""}`; }
                    break;
                case "weights":
                    upload.frac = p.total ? (p.loaded + 1) / p.total : 0; actText = `Uploading weights to the device · layer ${p.loaded + 1} / ${p.total}`;
                    break;
                case "vision": {
                    const tower = ensure("vision", "Image tower", 0.1);
                    tower.frac = p.total ? (p.loaded + 1) / p.total : 0; actText = `Image tower · ${p.message ?? ""}`;
                    break;
                }
                case "state": build.frac = 1; upload.frac = 1; actText = "Allocating the KV cache…"; break;
                case "ready": api.done(); return;
                default: break;
            }
            render();
        },
        /** files of the hand-written path (counts, no bytes) */
        files(done, total) { if (mode !== "running") api.begin(true); step("download").frac = total ? done / total : 0; dlText = `Weight files ${done} / ${total}`; render(); },
        fail(message) { if (mode === "idle") api.begin(true); clearTimeout(hideTimer); mode = "error"; dlText = ""; actText = message; render(); },
        done() {
            for (const s of steps) { s.frac = 1; s.indeterminate = false; }
            mode = "done"; dlText = ""; actText = "Ready"; render();
            clearTimeout(hideTimer); hideTimer = setTimeout(() => api.hide(), 3500);
        },
        hide() { mode = "idle"; render(); },
    };
    return api;
}

// The answer box of the chat page: shows the generated text as formatted Markdown (markdown.js) or as the RAW text, with a "Markdown | Raw" toggle and a "Copy raw" button.
// The raw text is the source of truth: it is what the log POST and the copy button use; the HTML is only a view of it. While tokens stream in, set(text) re-renders at most every `interval` ms
// (the last text is always rendered by the timer), flush() renders at once (end of a generation). The box keeps following the bottom while the user has not scrolled up.
// The view of an element is also available as element.answerView (the smoke test uses it).
import { markdownToHtml } from "./markdown.js?v=1";

const MODE_KEY = "m2w.answerMode";
const readMode = () => { try { return localStorage.getItem(MODE_KEY) === "raw" ? "raw" : "markdown"; } catch { return "markdown"; } };

export function answerView(box, { interval = 100 } = {}) {
    if (box.answerView) return box.answerView;
    let raw = "", mode = readMode(), timer = null, lastRender = -Infinity;
    const bar = document.createElement("div");
    bar.className = "answer-bar";
    bar.innerHTML = '<span class="seg"><button type="button" data-mode="markdown">Markdown</button><button type="button" data-mode="raw">Raw</button></span><button type="button" class="copy" title="copy the raw text (Markdown source) of the answer">Copy raw</button>';
    box.parentNode.insertBefore(bar, box);
    box.classList.add("answer-box");

    function render() {
        clearTimeout(timer); timer = null; lastRender = performance.now();
        const follow = box.scrollHeight - box.scrollTop - box.clientHeight < 40;   // at the bottom (or not scrollable): keep following the new text
        box.classList.toggle("md", mode === "markdown");
        if (mode === "markdown") box.innerHTML = markdownToHtml(raw); else box.textContent = raw;
        for (const button of bar.querySelectorAll("[data-mode]")) button.classList.toggle("on", button.dataset.mode === mode);
        if (follow) box.scrollTop = box.scrollHeight;
    }
    const view = {
        /** the text so far (call as often as tokens arrive) */
        set(text) {
            raw = String(text ?? "");
            if (!raw || timer) { if (!raw) render(); return; }
            const wait = interval - (performance.now() - lastRender);
            if (wait <= 0) render(); else timer = setTimeout(render, wait);
        },
        /** render the current text now (the end of a generation) */
        flush() { render(); },
        get text() { return raw; },
        get mode() { return mode; },
        setMode(next) { mode = next === "raw" ? "raw" : "markdown"; try { localStorage.setItem(MODE_KEY, mode); } catch { /* private mode */ } render(); },
    };
    for (const button of bar.querySelectorAll("[data-mode]")) button.onclick = () => view.setMode(button.dataset.mode);
    bar.querySelector(".copy").onclick = async (event) => {
        const button = event.currentTarget;
        try { await navigator.clipboard.writeText(raw); button.textContent = "Copied"; } catch { button.textContent = "Copy failed"; }
        setTimeout(() => { button.textContent = "Copy raw"; }, 1200);
    };
    box.answerView = view;
    render();
    return view;
}

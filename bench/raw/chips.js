// Chip groups for the <select> elements of chat.html. The page's scripts keep reading select.value, listening to "change" and filling the options (chat-package.js refills them whenever the max length
// changes); the select stays in the DOM, hidden, as the source of truth. chipGroup() draws a group of buttons next to it and keeps both in sync in BOTH directions: a click sets the value and fires "change",
// a script that sets .value, adds / replaces options or toggles .disabled is seen (value setter + MutationObserver).
//
//   chipGroup(select, { variant: "chips" | "segmented" | "rows", format(option) -> { main, sub?, tags?, title? } | null, label })   a disabled <option> is drawn dimmed and ignores clicks
//     chips: wrapping buttons   segmented: one joined control   rows: a vertical list (the model list), optgroups become small headings
const el = (tag, className, text) => { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; };
const valueSetter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, "value");

/** a default text for an option: the part before " (" is the label, the rest (and "(recommended)") is explanation */
export const defaultFormat = (option) => {
    if (option.value === "" && !option.textContent.trim()) return null;
    const text = option.textContent.trim(), [main, ...rest] = text.split(/\s+\(/);
    const star = /recommended/.test(text);
    return { main: main.replace(/\s+tokens$/, ""), tags: star ? ["recommended"] : [], title: rest.length ? text : "" };
};

export function chipGroup(select, { variant = "chips", format = defaultFormat, label = "" } = {}) {
    const group = el("div", `cg cg-${variant}`);
    group.setAttribute("role", "radiogroup"); if (label) group.setAttribute("aria-label", label);
    select.after(group);
    let pending = false;
    const render = () => {
        pending = false;
        group.replaceChildren();
        let heading = null;
        for (const option of select.options) {
            const parent = option.parentElement;
            if (variant === "rows" && parent.tagName === "OPTGROUP" && parent !== heading) { heading = parent; group.append(el("div", "cg-heading", parent.label)); }
            const shown = format(option);
            if (!shown) continue;
            const button = el("button", "cg-item"); button.type = "button"; button.setAttribute("role", "radio");
            const on = option.value === select.value;
            button.classList.toggle("on", on); button.setAttribute("aria-checked", String(on));
            if (variant === "rows") button.append(el("span", "cg-dot"));
            const text = el("span", "cg-text"); text.append(el("span", "cg-main", shown.main));
            if (shown.sub) text.append(el("span", "cg-sub", shown.sub));
            button.append(text);
            for (const tag of shown.tags ?? []) button.append(el("span", `cg-tag${tag === "recommended" ? " star" : ""}`, tag === "recommended" ? "★" : tag));
            if (shown.title) button.title = shown.title;
            button.disabled = select.disabled;
            if (option.disabled) { button.classList.add("dim"); button.setAttribute("aria-disabled", "true"); }   // dimmed (it keeps its tooltip, a click does nothing)
            button.onclick = () => { if (select.disabled || option.disabled || select.value === option.value) return; valueSetter.set.call(select, option.value); select.dispatchEvent(new Event("change", { bubbles: true })); render(); };
            group.append(button);
        }
        group.classList.toggle("disabled", select.disabled);
    };
    const schedule = () => { if (!pending) { pending = true; queueMicrotask(render); } };
    new MutationObserver(schedule).observe(select, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ["disabled", "label", "data-memory", "title"] });
    Object.defineProperty(select, "value", { configurable: true, get() { return valueSetter.get.call(this); }, set(next) { valueSetter.set.call(this, next); schedule(); } });
    select.addEventListener("change", schedule);
    render();
    return { render };
}

// formats of the selects of the page
export const formats = {
    /** "1024 tokens  (KV cache 64 MB)" -> 1024 and the memory this length is expected to need (data-memory, else the KV size) */
    maxLength: (option) => { const m = /^(\d+)\s*tokens\s*\(KV cache (\d+) MB\)/.exec(option.textContent.trim()); return m ? { main: m[1], sub: option.dataset.memory || `${m[2]} MB`, title: option.title || option.textContent.trim() } : defaultFormat(option); },
    /** "qwen: Qwen3.5-4B · text + image" -> a name and its tags */
    model: (option) => {
        const [name, ...tags] = option.textContent.trim().split(/\s+·\s+/);
        if (option.value === "js") return { main: "Qwen3-4B", tags: ["hand-written", "text"] };
        return { main: name, tags, title: option.textContent.trim() };
    },
};

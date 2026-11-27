// Everything that floats above the page, in one place: ONE transient overlay at a time (a popover, the picture lightbox, the drawer of the left column), a viewport-aware placement for popovers and
// one shared tooltip (no CSS ::after tooltips: they are clipped by scrolling containers and run out of the window).
// The z-index scale is defined once, as CSS variables in chat.html (--z-sticky < --z-progress < --z-drawer-backdrop < --z-drawer < --z-popover < --z-tooltip < --z-modal): use those, never a number.
//
//   const overlays = createOverlays()
//   overlays.open(id, { close, contains })   make `id` the open overlay (the previous one is closed); contains(target): is the click inside it (or on its button)? An outside click and Esc close it.
//   overlays.release(id)                     the overlay closed itself
//   overlays.closeAll()                      e.g. when a load starts or a message is sent
//   placePopover(panel, { anchor, avoid, bounds, top, width, margin })   position a fixed panel next to its button inside the window
//   createTooltips()                         the tooltip of every element with a data-tip attribute
export function createOverlays() {
    let current = null;
    const api = {
        open(id, { close, contains }) { if (current && current.id !== id) { const previous = current; current = null; previous.close(); } current = { id, close, contains }; },
        release(id) { if (current?.id === id) current = null; },
        closeAll() { const previous = current; current = null; previous?.close(); },
        get current() { return current?.id ?? null; },
    };
    document.addEventListener("keydown", (event) => { if (event.key === "Escape" && current) { event.preventDefault(); api.closeAll(); } });
    document.addEventListener("pointerdown", (event) => { if (current && !current.contains(event.target)) api.closeAll(); }, true);
    return api;
}

/**
 * Place a `position: fixed` panel beside its button without leaving the window: above `avoid` (the composer, so that Send / Stop stay free) or above the button, else below it, the right edge on the button,
 * shifted back into `bounds` (the main column: the left column is never covered). Max height with scroll. Call again on resize / scroll while it is open.
 */
export function placePopover(panel, { anchor, avoid = null, bounds = null, top = 0, width = 880, margin = 12 }) {
    const vw = document.documentElement.clientWidth, vh = window.innerHeight, a = anchor.getBoundingClientRect();
    const left0 = Math.max(margin, bounds ? bounds.getBoundingClientRect().left + margin : margin), right0 = vw - margin;
    const w = Math.max(240, Math.min(width, right0 - left0));
    const left = Math.min(Math.max(left0, a.right - w), right0 - w);
    const edge = (avoid ? avoid.getBoundingClientRect().top : a.top) - 8, above = edge - Math.max(top, margin), below = vh - (a.bottom + 8) - margin;
    panel.style.position = "fixed"; panel.style.width = `${w}px`; panel.style.left = `${left}px`; panel.style.right = "auto";
    if (above >= Math.min(below, 220) || above >= below) { panel.style.top = "auto"; panel.style.bottom = `${vh - edge}px`; panel.style.maxHeight = `${Math.max(120, above)}px`; panel.dataset.side = "above"; }
    else { panel.style.bottom = "auto"; panel.style.top = `${a.bottom + 8}px`; panel.style.maxHeight = `${Math.max(120, below)}px`; panel.dataset.side = "below"; }
}

export function createTooltips() {
    const tip = document.createElement("div");
    tip.className = "tooltip"; tip.setAttribute("role", "tooltip"); tip.hidden = true;
    document.body.append(tip);
    let target = null;
    const hide = () => { target = null; tip.hidden = true; };
    const show = (element) => {
        const text = element.dataset.tip;
        if (!text) return hide();
        target = element; tip.textContent = text; tip.hidden = false;
        const margin = 8, vw = document.documentElement.clientWidth, vh = window.innerHeight, a = element.getBoundingClientRect();
        tip.style.maxWidth = `${Math.min(360, vw - 2 * margin)}px`; tip.style.left = "0px"; tip.style.top = "0px";
        const t = tip.getBoundingClientRect();
        const left = Math.min(Math.max(margin, a.left + a.width / 2 - t.width / 2), vw - t.width - margin);
        const top = a.top - t.height - 8 >= margin ? a.top - t.height - 8 : Math.min(a.bottom + 8, vh - t.height - margin);   // above when it fits, else below
        tip.style.left = `${left}px`; tip.style.top = `${Math.max(margin, top)}px`;
    };
    document.addEventListener("pointerover", (event) => { const element = event.target.closest?.("[data-tip]"); if (element && element !== target) show(element); else if (!element && target) hide(); });
    document.addEventListener("pointerdown", hide, true);
    document.addEventListener("scroll", hide, true);
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") hide(); });
    window.addEventListener("blur", hide);
    return { hide };
}

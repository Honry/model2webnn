// Chat-style auto-scroll for the message list (the element with overflow: auto, not the page): the list follows the bottom while content grows (streamed tokens, a re-rendered Markdown answer, a new
// bubble, a progress bar, stat chips, a composer that grows and shrinks the list) as long as the reader is "pinned" to the bottom. Scrolling UP unpins; a small "Jump to latest" button then appears
// and re-pins. One scroll write per animation frame, instant (not smooth: smooth scrolling lags behind a stream).
//
//   isPinned({ scrollHeight, scrollTop, clientHeight }, threshold)    pure: is the view within `threshold` px of the bottom?
//   const scroll = createAutoScroll(container, { threshold: 80, button })
//   scroll.pin()        pin and scroll down (sending a message, a cleared chat, the button)       scroll.schedule()    follow the bottom if pinned       scroll.pinned
export const isPinned = (metrics, threshold = 80) => metrics.scrollHeight - metrics.scrollTop - metrics.clientHeight <= threshold;

export function createAutoScroll(container, { threshold = 80, button = null } = {}) {
    let pinned = true, frame = 0, lastTop = container.scrollTop;
    const show = () => { if (button) button.hidden = pinned; };
    const apply = () => {
        frame = 0;
        if (pinned) container.scrollTop = container.scrollHeight;
        lastTop = container.scrollTop;
    };
    const schedule = () => { if (!frame) frame = requestAnimationFrame(apply); };   // any number of changes in a frame: one scroll
    container.addEventListener("scroll", () => {
        const up = container.scrollTop < lastTop;   // content that grows does not move scrollTop: only the reader scrolling up unpins
        lastTop = container.scrollTop;
        if (up && !isPinned(container, threshold)) pinned = false;
        else if (isPinned(container, threshold)) pinned = true;
        show();
    }, { passive: true });
    const resize = typeof ResizeObserver === "function" ? new ResizeObserver(schedule) : null;   // the list itself (the composer grows) and every message (an image, a code block ...)
    resize?.observe(container);
    for (const child of container.children) resize?.observe(child);
    new MutationObserver((records) => {
        for (const record of records) for (const node of record.addedNodes) if (node.nodeType === 1 && node.parentNode === container) resize?.observe(node);
        schedule();
    }).observe(container, { childList: true, subtree: true, characterData: true });
    if (button) button.onclick = () => api.pin();
    const api = {
        pin() { pinned = true; show(); schedule(); },
        schedule,
        get pinned() { return pinned; },
    };
    show();
    return api;
}

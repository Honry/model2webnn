// The capacity / context arithmetic of the chat page in one place (no DOM): how many KV rows a session of a max length has, how much of it a prompt and a streamed answer use, the room kept free for the answer
// when the history is trimmed, the automatic length of an answer, and the speed so far of a running answer.
//
// capacity = the KV rows of a session: (max length - 1), or the whole max length for a protocol that attends over the updated cache (the hand-written path with attendInCache)
export const cacheCapacity = (maxLength, attendInCache = false) => (attendInCache ? maxLength : maxLength - 1);
/** The room kept free for the answer when the history is trimmed: min(512, a quarter of the cache). */
export const answerReserve = (capacity) => Math.min(512, Math.floor(capacity / 4));
/** The reserve of the page: an explicit maxNewTokens (URL) is reserved as it is, else answerReserve. */
export const reserveRows = (capacity, explicit = 0) => (explicit > 0 ? explicit : answerReserve(capacity));
/** The automatic length of an answer: every row that is left after the prompt (at least one). */
export const autoMaxNewTokens = (capacity, promptTokens) => Math.max(1, capacity - promptTokens);
/** The rows a turn uses (prompt + generated so far), clamped to the capacity, and whether the cache is full. */
export const rowsUsed = (promptTokens, generated, capacity) => { const total = promptTokens + generated; return { used: Math.min(total, capacity), full: total >= capacity }; };
export const contextText = (used, capacity) => `Context ${used ?? "–"} / ${capacity}`;
export const cacheText = (used, capacity, full) => `${Math.min(used, capacity)} / ${capacity}${full ? " · FULL" : ""}`;
/** How full the cache is, as a share of its capacity: from CONTEXT_WARN_SHARE the level is "warn" (amber), from CONTEXT_FULL_SHARE "full" (red). The thresholds live here, in one place. */
export const CONTEXT_WARN_SHARE = 0.8;
export const CONTEXT_FULL_SHARE = 0.95;
export function contextLevel(used, capacity) {
    if (!(capacity > 0) || used == null) return "ok";
    const share = used / capacity;
    return share >= CONTEXT_FULL_SHARE ? "full" : share >= CONTEXT_WARN_SHARE ? "warn" : "ok";
}
/** What the colours of the context numbers mean (the tooltip). */
export const CONTEXT_FULL_HINT = "When it is full, older turns are dropped before the next message is sent, and an answer is cut when the cache is full. A larger Max length is applied at Load.";
/** The lines in a message: dimmed while the answer streams towards the end of the cache, red when it was cut. */
export const CACHE_ALMOST_FULL_NOTE = "Cache almost full: the answer may be cut";
export const CACHE_CUT_NOTE = "Cut: the KV cache is full. Choose a larger Max length (applied at Load) or start a New chat";

/** The speed of a running answer from the time since its first token: null until there are two tokens. */
export function liveSpeed(generated, elapsedMs) {
    if (generated < 2 || !(elapsedMs > 0)) return null;
    const tpotMs = elapsedMs / (generated - 1);
    return { tpotMs, tps: 1000 / tpotMs };
}

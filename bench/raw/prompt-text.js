// Prompts of an exact length for the chat page (1k / 2k / 4k / 8k tokens): a long, deterministic, varied source text (numbered sections of generated sentences, not one repeated sentence) that is cut by a
// binary search on characters until the tokenizer counts the whole prompt (chat format included, through `count`) within 1% of the target, followed by a short question.
//
//   sourceText(minChars, seed)                    the deterministic source text
//   buildPromptOfTokens(target, count, options)   -> { text, tokens }      count(text) = the number of prompt tokens of that message
//   minLengthForPrompt(tokens, lengths)           the smallest max length (of `lengths`, else the exact bound) whose cache holds this prompt and the room kept free for the answer
import { answerReserve } from "./context-math.js?v=2";
export { answerReserve };
export const INSTRUCTION = "\n\nSummarize the text above in 3 short bullet points.";

const SUBJECTS = ["The early engineers", "A small research laboratory", "The regional archive", "Several independent teams", "The museum curator", "A group of students", "The harbour authority", "The old observatory",
    "Local farmers", "The city council", "A travelling surveyor", "The workshop foreman", "An international committee", "The village library", "The night shift", "A retired navigator", "The weather service",
    "The textile mill", "Two neighbouring schools", "The railway company", "A careful translator", "The orchestra", "The mountain rescue team", "A young apprentice"];
const VERBS = ["recorded", "compared", "rebuilt", "questioned", "measured", "described", "reorganised", "documented", "tested", "restored", "mapped", "reviewed", "simplified", "calibrated", "catalogued",
    "announced", "postponed", "repaired", "traced", "balanced", "sorted", "revisited", "outlined", "checked"];
const OBJECTS = ["the oldest set of maps", "a handwritten ledger of deliveries", "the schedule for the coming winter", "three competing designs for the bridge", "a collection of weather reports",
    "the rules of the annual market", "a stack of unsigned letters", "the water levels of the river", "an unfinished catalogue of seeds", "the route of the first telegraph line", "a dozen sample recipes",
    "the quarterly budget of the hospital", "a forgotten set of tuning forks", "the inventory of the east warehouse", "records of the lighthouse keeper", "the proposal for a new clock tower",
    "a table of tides and currents", "the original plans of the market hall", "reports from the northern valley", "the list of spare parts", "a box of glass negatives", "the minutes of the first meeting",
    "a short history of the canal", "the notes of a visiting physician"];
const CONNECTORS = ["because", "although", "while", "after", "whenever", "so that", "even though", "before", "since", "until", "as soon as", "in case"];
const CLAUSES = ["the winter had been longer than anyone expected", "nobody could agree on the units", "the funding arrived late that year", "a storm had damaged the main road", "the numbers did not add up at first",
    "the new method promised faster results", "the old method had always worked well enough", "visitors kept asking for more detail", "the pages were stored in the wrong order", "the instruments needed a fresh calibration",
    "a neighbouring town had tried something similar", "the committee wanted a second opinion", "the harvest was unusually early", "the translation left several questions open", "the lamp in the hall kept flickering",
    "everybody was tired after the long journey", "the records were incomplete", "the weather turned warm again", "the supplier changed the specification", "a clerk found a mistake in the first draft"];
const TOPICS = ["Maps and measurements", "The market and its rules", "Water, tides and weather", "Letters and records", "Machines and workshops", "Schools and libraries", "Roads, rails and bridges", "Seeds and harvests",
    "Music and instruments", "Hospitals and budgets", "Lighthouses and harbours", "Clocks and calendars"];

/** a small deterministic generator (the same text on every page load) */
function random(seed) { let state = seed >>> 0; return () => (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 2 ** 32; }

/** numbered sections of generated sentences, at least `minChars` long */
export function sourceText(minChars = 80000, seed = 7) {
    const next = random(seed), pick = (list) => list[Math.floor(next() * list.length)];
    const parts = [];
    let length = 0, section = 1;
    while (length < minChars) {
        const sentences = Array.from({ length: 4 + Math.floor(next() * 3) }, () => `${pick(SUBJECTS)} ${pick(VERBS)} ${pick(OBJECTS)} ${pick(CONNECTORS)} ${pick(CLAUSES)}.`);
        const block = `Section ${section}: ${pick(TOPICS)}\n${sentences.join(" ")}\n\n`;
        parts.push(block); length += block.length; section++;
    }
    return parts.join("");
}

let cachedSource = null;
/** The prompt (source text cut to fit + the question) whose token count, as `count` says, is within `tolerance` of `target`. */
export function buildPromptOfTokens(target, count, { instruction = INSTRUCTION, tolerance = 0.01, source = (cachedSource ??= sourceText()) } = {}) {
    const build = (chars) => `${source.slice(0, chars).replace(/\s+\S*$/, "")}${instruction}`;
    let low = 0, high = source.length, best = { text: build(0), tokens: count(build(0)) };
    for (let step = 0; step < 40 && low <= high; step++) {
        const middle = Math.floor((low + high) / 2), text = build(middle), tokens = count(text);
        if (Math.abs(tokens - target) < Math.abs(best.tokens - target)) best = { text, tokens };
        if (Math.abs(tokens - target) <= target * tolerance) return { text, tokens };
        if (tokens < target) low = middle + 1; else high = middle - 1;
    }
    return best;
}

/** The smallest max length of `lengths` whose cache (max length - 1 rows) holds `tokens` plus the reserve; when none does, the exact bound. */
export function minLengthForPrompt(tokens, lengths = []) {
    const fits = (length) => tokens + answerReserve(length - 1) <= length - 1;
    const fitting = [...lengths].sort((a, b) => a - b).find(fits);
    if (fitting) return fitting;
    let length = tokens + 1;
    while (!fits(length)) length++;
    return length;
}

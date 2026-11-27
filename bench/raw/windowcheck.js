// Window caches vs full caches at the LOGITS level, on the same device: the sampled text of two packages cannot say whether a difference is a bug or f16 noise, the logits can.
// ONE package is loaded at a time (the weights of Gemma 4 are 2.6 GB each): A is loaded, its logits are recorded, it is destroyed, then B is loaded and fed the SAME tokens (teacher forcing:
// the greedy tokens of A), and the logits of every step are compared. Noise floor: the same comparison of the full package with itself at another prefill chunk size.
// URL: windowcheck.html?full=../packages/gemma4-e4b-pkg/&window=../packages/gemma4-e4b-pkg-window/&maxLength=2048&chunk=256&promptFile=long-prompt.txt&promptChars=4000&steps=64&device=npu
//      chunkFull / chunkWindow   prefill chunk size of A / B (default: chunk)      variant / computeType / bpg   session options of both (default: the package's)
//      noise=1                   B = the FULL package again with chunk size chunkB (default 64): the noise floor of the device (the window package is not loaded)
//      baseline=<chunk>          after A vs B also run the noise floor (A vs the full package with this chunk size, e.g. baseline=64) and print the verdict
//      noiseMaxDiff=<x>          a known noise floor (mean of the max |logit difference| per step) for the verdict instead of running it
//      prompt=<text>             the message instead of the file       cache=0   no OPFS cache for the weights
import { loadModel, buildPrompt, tokenizerFromTransformersJs } from "../../lib/runtime/index.js?v=1";

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js";
const query = new URLSearchParams(location.search);
const param = (name, fallback) => query.get(name) ?? fallback;
const byId = (id) => document.getElementById(id);
const start = performance.now();
const log = (message) => { byId("out").textContent += `[${((performance.now() - start) / 1000).toFixed(1).padStart(6)} s] ${message}\n`; console.log(message); fetch("/log", { method: "POST", body: JSON.stringify({ experiment: "windowcheck", message }) }).catch(() => {}); };
const url = (name) => new URL(param(name, ""), location.href).href;

const device = param("device", "npu"), maxLength = +param("maxLength", 2048), steps = +param("steps", 64), chunk = +param("chunk", 256);
const noise = param("noise", "0") === "1";
const chunkA = +param("chunkFull", chunk), chunkB = noise ? +param("chunkB", param("chunkWindow", 64)) : +param("chunkWindow", chunk);

/** index of the largest entry, its logit and the second largest logit */
function top2(l) {
    let a = 0, second = -Infinity;
    for (let i = 1; i < l.length; i++) if (l[i] > l[a]) { second = l[a]; a = i; } else if (l[i] > second) second = l[i];
    return { index: a, top: l[a], second };
}

async function loadIds(llm) {
    const text = query.get("prompt") ?? `${(await (await fetch(param("promptFile", "long-prompt.txt"))).text()).slice(0, +param("promptChars", 0) || undefined)}\n\nSummarize the text above in two sentences.`;
    const lib = await import(TRANSFORMERS_URL), tokenizer = await tokenizerFromTransformersJs(lib, llm.pkg), format = llm.pkg.chatFormat;
    return tokenizer.encode(format ? buildPrompt(format, [{ user: text }], undefined, new Date("2026-01-01T12:00:00")) : text);
}

/** Load a package, run the prompt and `steps` teacher-forced (or, without `forced`, greedy) decode steps, return Float32 copies of the real-vocab logits of every position, destroy the model. */
async function record(label, packageUrl, chunkSize, ids, forced) {
    log(`=== ${label}: ${packageUrl} (chunk ${chunkSize}, max length ${maxLength}, ${forced ? "teacher forced" : "greedy"})`);
    let last = "";
    const options = { device, maxLength, chunkSize, prefillBlocksPerGraph: 1, cache: param("cache", "1") !== "0",
        ...(query.get("variant") ? { variant: query.get("variant") } : {}), ...(query.get("computeType") ? { computeType: query.get("computeType") } : {}), ...(query.get("bpg") ? { blocksPerGraph: +query.get("bpg") } : {}),
        onProgress: (p) => { if (p.stage !== last && p.stage !== "download" && p.stage !== "build" && p.stage !== "weights") log(`${label}: ${p.stage}${p.message ? ` ${p.message}` : ""}`); last = p.stage; } };
    const llm = await loadModel(packageUrl, options);
    const vocab = llm.pkg.model.vocab, ring = llm.pkg.plan.profile?.sliding_cache === "window" ? llm.pkg.plan.profile.sliding_ring : 0;
    log(`${label}: ${llm.pkg.model.architecture}, variant ${llm.variant.id}, ${ring ? `window caches (ring ${ring} rows)` : "full caches"}, graphs built in ${(llm.stats.buildMs / 1000).toFixed(1)} s`);
    const prompt = ids ?? await loadIds(llm);
    if (prompt.length + steps >= maxLength - 1) throw new Error(`the prompt has ${prompt.length} tokens, plus ${steps} steps: needs a larger maxLength (${maxLength})`);
    const copy = (half) => Float32Array.from(half.subarray(0, vocab));
    const logits = [], tokens = [];
    const t0 = performance.now();
    logits.push(copy(await llm.prefill(prompt)));   // the logits of the last prompt token (the runtime's own prefill, then its own decode step: exactly what generate() runs)
    log(`${label}: prefill of ${prompt.length} tokens in ${llm.stats.prefillPlan.length} chunk(s) (${llm.stats.prefillPlan.join("+")}) ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    for (let i = 0; i < steps; i++) {
        const token = forced ? forced[i] : top2(logits[i]).index;
        tokens.push(token);
        logits.push(copy(await llm.step(token)));
    }
    log(`${label}: ${steps} decode steps recorded`);
    llm.destroy();
    await new Promise((resolve) => setTimeout(resolve, 500));   // let the device free the weights before the next package is loaded
    return { logits, tokens, ids: prompt, vocab };
}

/** per position: cosine, max |diff|, argmax agreement, A's top-2 margin, rank of A's argmax in B */
function compare(a, b) {
    const rows = [];
    for (let s = 0; s < a.logits.length; s++) {
        const x = a.logits[s], y = b.logits[s];
        let dot = 0, nx = 0, ny = 0, maxDiff = 0;
        for (let i = 0; i < x.length; i++) { dot += x[i] * y[i]; nx += x[i] * x[i]; ny += y[i] * y[i]; maxDiff = Math.max(maxDiff, Math.abs(x[i] - y[i])); }
        const ta = top2(x), tb = top2(y);
        let rank = 0;
        for (let i = 0; i < y.length; i++) if (y[i] > y[ta.index]) rank++;
        rows.push({ step: s, cosine: dot / Math.sqrt(nx * ny), maxDiff, agree: ta.index === tb.index, margin: ta.top - ta.second, rank, a: ta.index, b: tb.index });
    }
    return rows;
}

function summarize(name, rows) {
    const bad = rows.filter((r) => !r.agree), mean = (f) => rows.reduce((s, r) => s + f(r), 0) / rows.length;
    const stats = { name, steps: rows.length, disagreements: bad.length, minCosine: Math.min(...rows.map((r) => r.cosine)), meanCosine: mean((r) => r.cosine), meanMaxDiff: mean((r) => r.maxDiff), maxMaxDiff: Math.max(...rows.map((r) => r.maxDiff)), first: bad[0]?.step ?? -1 };
    log(`--- ${name}: ${rows.length} positions (0 = the last prompt token, k = after k teacher-forced tokens)`);
    log(`step  cosine     max|diff|  argmax A / B   A margin (top1 - top2)  rank of A's argmax in B`);
    for (const r of rows) if (!r.agree || r.step % 8 === 0 || r.step === rows.length - 1) log(`${String(r.step).padStart(4)}  ${r.cosine.toFixed(6)}  ${r.maxDiff.toFixed(4).padStart(9)}  ${String(r.a).padStart(6)} / ${String(r.b).padEnd(6)} ${r.margin.toFixed(3).padStart(12)} ${String(r.rank).padStart(24)}${r.agree ? "" : "   <== ARGMAX DIFFERS"}`);
    log(`SUMMARY ${name}: argmax differs at ${bad.length}/${rows.length} steps${bad.length ? ` (steps ${bad.map((r) => r.step).join(",")}; margins of A ${bad.map((r) => r.margin.toFixed(2)).join(",")}; ranks in B ${bad.map((r) => r.rank).join(",")})` : ""}; first argmax difference: ${stats.first < 0 ? "none" : `step ${stats.first}`}`);
    log(`SUMMARY ${name}: min cosine ${stats.minCosine.toFixed(6)}, mean cosine ${stats.meanCosine.toFixed(6)}, max|diff| mean ${stats.meanMaxDiff.toFixed(4)} max ${stats.maxMaxDiff.toFixed(4)}`);
    return stats;
}

function verdict(window, floor) {
    const ratio = window.meanMaxDiff / floor.meanMaxDiff, extra = window.disagreements - floor.disagreements;
    const same = ratio < 2 && extra <= 1 && window.minCosine > floor.minCosine - 1e-4;
    log(`VERDICT: ${window.name} differs from full ${same ? "by about the noise floor" : "MORE than the noise floor"} (mean max|diff| ${window.meanMaxDiff.toFixed(4)} vs ${floor.meanMaxDiff.toFixed(4)} = x${ratio.toFixed(2)}; argmax differences ${window.disagreements} vs ${floor.disagreements}; min cosine ${window.minCosine.toFixed(6)} vs ${floor.minCosine.toFixed(6)})`);
}

async function main() {
    log(`windowcheck: device ${device}, max length ${maxLength}, ${steps} teacher-forced steps, chunks A ${chunkA} / B ${chunkB}${noise ? " (noise floor: full vs full)" : ""}`);
    const a = await record("A (full)", url("full"), chunkA, null, null);
    log(`prompt ${a.ids.length} tokens; A's greedy continuation: ${JSON.stringify(a.tokens)}`);
    const b = await record(noise ? "B (full, other chunk)" : "B (window)", noise ? url("full") : url("window"), chunkB, a.ids, a.tokens);
    const result = summarize(noise ? `noise floor (full c${chunkA} vs full c${chunkB})` : `window c${chunkB} vs full c${chunkA}`, compare(a, b));
    let floor = null;
    if (noise) floor = result;
    else if (query.get("baseline")) {
        const c = await record("C (full, baseline chunk)", url("full"), +param("baseline", 64), a.ids, a.tokens);
        floor = summarize(`noise floor (full c${chunkA} vs full c${param("baseline", 64)})`, compare(a, c));
    } else if (query.get("noiseMaxDiff")) floor = { meanMaxDiff: +param("noiseMaxDiff", 0), disagreements: 0, minCosine: 1 };
    if (floor && !noise) verdict(result, floor);
    else if (!noise) log("no noise floor in this run: add baseline=64 (runs full c64 as the reference noise) or compare with the noise=1 URL");
    log("DONE");
    byId("status").textContent = "done";
}
main().catch((e) => { log(`ERROR ${e?.stack || e}`); byId("status").textContent = "failed"; });

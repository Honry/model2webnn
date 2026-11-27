// Browser test page for the model2webnn runtime (src/runtime, built to lib/runtime by `npm run build:lib`): loads a package written by `m2w compile`, checks it against the tokens the
// Python reference produced (verification.json), and chats. URL parameters pre-fill the form: ?package=..&device=npu&maxLength=1024&chunk=64&bpg=1&variant=onehot-float32&pbpg=1&auto=check|chat
// (chunk may list several sizes: 64,128; empty variant / chunk = what verification.json was made with, else the package defaults)
import { loadModel, UrlSource, CachedSource, openPackage, chat, tokenizerFromTransformersJs } from "../../lib/runtime/index.js";

const query = new URLSearchParams(location.search);
const byId = (id) => document.getElementById(id);
for (const [id, name] of [["pkg", "package"], ["device", "device"], ["maxLength", "maxLength"], ["chunk", "chunk"], ["bpg", "bpg"], ["pbpg", "pbpg"], ["variant", "variant"]]) if (query.get(name)) byId(id).value = query.get(name);
const start = performance.now();
const log = (message) => {
    byId("out").textContent += `[${((performance.now() - start) / 1000).toFixed(1).padStart(6)} s] ${message}\n`;
    console.log(message);
    fetch("/log", { method: "POST", body: JSON.stringify({ experiment: "m2wllm", message }) }).catch(() => {});
};
const setStatus = (text) => (byId("status").textContent = text);
const packageUrl = () => new URL(byId("pkg").value, location.href).href;

let llm = null, tokenizer = null;

async function load() {
    llm?.destroy(); llm = null; tokenizer = null;
    byId("check").disabled = byId("send").disabled = byId("bench").disabled = true;
    const options = { device: byId("device").value, blocksPerGraph: +byId("bpg").value || 1, prefillBlocksPerGraph: +byId("pbpg").value || 0 };
    let recorded = {};   // the variant / chunk sizes the Python reference was run with: the default of the form, so that "check" compares like with like
    try { recorded = await (await fetch(new URL("verification.json", packageUrl()).href, { cache: "no-store" })).json(); } catch { /* no verification.json */ }
    const variant = byId("variant").value || recorded.variant, chunks = byId("chunk").value ? byId("chunk").value.split(",").map(Number) : recorded.chunks;
    if (byId("maxLength").value) options.maxLength = +byId("maxLength").value;
    else if (recorded.max_length) options.maxLength = recorded.max_length;
    if (variant) options.variant = variant;
    if (chunks?.length) options.chunkSizes = chunks;
    let lastStage = "", lastDownload = 0;
    options.onProgress = (p) => {
        setStatus(`${p.stage}${p.message ? ": " + p.message : ""}`);
        if (p.stage !== lastStage && p.stage !== "download") log(`${p.stage}${p.message ? ": " + p.message : ""}`);
        if (p.stage === "download" && p.loaded - lastDownload > 128 << 20) { lastDownload = p.loaded; log(`downloaded ${Math.round(p.loaded / 2 ** 20)} / ${Math.round(p.total / 2 ** 20)} MB`); }
        lastStage = p.stage;
    };
    log(`loading ${packageUrl()} on ${options.device}`);
    llm = await loadModel(packageUrl(), options);
    const m = llm.pkg.model;
    log(`READY: ${m.architecture}, ${m.num_layers} layers, max length ${llm.maxLength}, chunks ${llm.chunkSizes.join('+')}, variant ${llm.variant.id}; load ${(llm.stats.loadMs / 1000).toFixed(1)} s, graph build ${(llm.stats.buildMs / 1000).toFixed(1)} s`);
    setStatus("ready");
    byId("check").disabled = false; byId("send").disabled = false; byId("bench").disabled = false;
}

async function check() {
    setStatus("checking");
    const url = new URL("verification.json", packageUrl()).href;
    const expected = await (await fetch(url, { cache: "no-store" })).json();
    if (expected.max_length !== llm.maxLength) log(`note: verification.json was made with max length ${expected.max_length}, this model runs ${llm.maxLength} (results may differ slightly)`);
    const generated = [];
    for await (const token of llm.generate(expected.prompt_ids, { maxNewTokens: expected.generated_ids.length, stopTokens: [], temperature: 0 })) generated.push(token);
    const same = generated.filter((t, i) => t === expected.generated_ids[i]).length, firstSame = generated[0] === expected.generated_ids[0];
    const s = llm.stats;
    log(`expected ${JSON.stringify(expected.generated_ids)}`);
    log(`got      ${JSON.stringify(generated)}`);
    if (expected.text) log(`reference text: ${JSON.stringify(expected.text)}`);
    log(`RESULT check: ${same}/${expected.generated_ids.length} tokens equal to the Python reference, first token ${firstSame ? "equal" : "DIFFERENT"}`);
    log(`prompt ${s.promptTokens} tokens, prefill (TTFT) ${s.prefillMs.toFixed(0)} ms, decode ${s.decodeSteps} steps, TPOT ${s.tpotMs.toFixed(1)} ms`);
    reportSteps("check");
    byId("status").textContent = same === generated.length ? "check OK" : "check: differences";
    return same === generated.length;
}

async function send() {
    setStatus("tokenizer");
    if (!tokenizer) {
        const lib = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2");
        tokenizer = await tokenizerFromTransformersJs(lib, llm.pkg);
        log("tokenizer loaded (transformers.js)");
    }
    byId("answer").textContent = "";
    setStatus("generating");
    const t0 = performance.now();
    let pieces = 0;
    for await (const piece of chat(llm, tokenizer, [{ user: byId("prompt").value }], { maxNewTokens: +byId("maxNew").value || 128 })) { byId("answer").textContent += piece; pieces++; }
    const s = llm.stats;
    log(`answer: ${JSON.stringify(byId("answer").textContent)}`);
    reportSteps("chat");
    log(`RESULT chat: prompt ${s.promptTokens} tokens, TTFT ${s.prefillMs.toFixed(0)} ms, ${s.decodeSteps} decode steps, TPOT ${s.tpotMs.toFixed(1)} ms (${(1000 / s.tpotMs).toFixed(1)} tok/s), total ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    setStatus("ready");
}

// per-step timings: the first steps of a freshly loaded model include one-time work on the device
function reportSteps(label) {
    const ms = llm.stats.stepMs, sorted = (a) => [...a].sort((x, y) => x - y), median = (a) => (a.length ? sorted(a)[Math.floor(a.length / 2)] : 0);
    const steady = ms.slice(3), mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
    log(`${label} steps (ms): first 6 = [${ms.slice(0, 6).map((x) => x.toFixed(0)).join(", ")}]; steady state (after 3): median ${median(steady).toFixed(1)}, mean ${mean(steady).toFixed(1)}, min ${Math.min(...steady, Infinity).toFixed(1)}, n=${steady.length}`);
}
async function bench() {
    setStatus("bench");
    const expected = await (await fetch(new URL("verification.json", packageUrl()).href, { cache: "no-store" })).json();
    const tokens = +byId("maxNew").value || 64;
    for (const run of [1, 2]) {   // the second run shows what a warm model does
        const generated = [];
        for await (const token of llm.generate(expected.prompt_ids, { maxNewTokens: tokens, stopTokens: [], temperature: 0 })) generated.push(token);
        const s = llm.stats;
        log(`RESULT bench run ${run}: ${tokens} tokens, TTFT ${s.prefillMs.toFixed(0)} ms, TPOT ${s.tpotMs.toFixed(1)} ms (mean of all ${s.decodeSteps} steps)`);
        reportSteps(`run ${run}`);
    }
    setStatus("ready");
}

const guarded = (fn) => async () => { try { await fn(); } catch (error) { log(`ERROR ${error?.stack || error}`); setStatus("failed"); } };
byId("load").onclick = guarded(load);
byId("check").onclick = guarded(check);
byId("send").onclick = guarded(send);
byId("bench").onclick = guarded(bench);
byId("clear").onclick = guarded(async () => { await new CachedSource(new UrlSource(packageUrl())).clear(); log("cache cleared"); });
if (query.get("auto")) guarded(async () => { await load(); if (query.get("auto") === "check") await check(); else if (query.get("auto") === "bench") await bench(); else await send(); })();
void openPackage;

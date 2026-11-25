// Runs a Qwen3.5 package on the device for the prompt of a reference trace (python tools/qwen35_trace.py -> bench/data/qwen35-trace/) and compares the hidden state entering every layer, after the
// last prefill chunk and after the first decode step, with the numpy interpreter's: where does the device first deviate?  (f16 graphs are checked HERE: the NPU's compiler returned silently wrong
// results for some operator patterns, see bench/raw/gdn.html.)
// URL: qwen35check.html?package=../packages/qwen35-4b-pkg/&trace=../data/qwen35-trace/&device=npu&maxLength=1152&computeType=float32&threshold=0.05&chunk=64&stepwise=1
//      (chunk: the prefill chunk size to run, any size of the package: the prompt is one chunk of any of them; stepwise=0: queue everything like the real runtime)
import { Qwen35LLM } from "../../lib/runtime/index.js";

const query = new URLSearchParams(location.search);
const param = (name, fallback) => query.get(name) ?? fallback;
const packageUrl = new URL(param("package", "../packages/qwen35-4b-pkg/"), location.href).href, traceUrl = new URL(param("trace", "../data/qwen35-trace/"), location.href).href;
const threshold = +param("threshold", 0.05);
const byId = (id) => document.getElementById(id);
const start = performance.now();
const log = (message) => { byId("out").textContent += `[${((performance.now() - start) / 1000).toFixed(1).padStart(6)} s] ${message}\n`; console.log(message); fetch("/log", { method: "POST", body: JSON.stringify({ experiment: "qwen35check", message }) }).catch(() => {}); };

function compare(got, want) {
    let maxAbs = 0, errSq = 0, refSq = 0, bad = 0;
    for (let i = 0; i < want.length; i++) {
        const g = got[i];
        if (!Number.isFinite(g)) { bad++; continue; }
        const d = g - want[i]; maxAbs = Math.max(maxAbs, Math.abs(d)); errSq += d * d; refSq += want[i] * want[i];
    }
    return { maxAbs, rel: Math.sqrt(errSq / (refSq + 1e-30)), bad };
}
const fmt = (c) => `rel ${c.rel.toExponential(2)}  max abs ${c.maxAbs.toExponential(2)}${c.bad ? `  NOT FINITE x${c.bad}` : ""}`;

async function main() {
    const trace = await (await fetch(`${traceUrl}trace.json`)).json(), blob = await (await fetch(`${traceUrl}trace.bin`)).arrayBuffer();
    const reference = (entry) => new Float16Array(blob, entry.offset, entry.bytes / 2);
    log(`prompt ${trace.tokens.length} tokens, plan ${trace.plan}, decode token ${trace.decode_token}; reference variant ${trace.variant}`);
    Qwen35LLM.debugReadback = true;
    const options = { device: param("device", "npu"), maxLength: +param("maxLength", 1152), chunkSize: +param("chunk", trace.chunk), computeType: param("computeType", "float32"), cache: param("cache", "1") !== "0",
        onProgress: (p) => { if (p.stage === "build" || p.stage === "ready") log(`${p.stage} ${p.message ?? ""}`); } };
    const llm = await Qwen35LLM.load(packageUrl, options);
    log(`loaded: variant ${llm.variant.id}, graphs built in ${(llm.stats.buildMs / 1000).toFixed(1)} s`);
    const stepwise = param("stepwise", "1") !== "0";   // every dispatch is awaited and named in the log first: after a crash / lost context the last line names the graph
    const result = await llm.debugTrace(trace.tokens, trace.decode_token, { stepwise, onDispatch: stepwise ? (label) => log(`dispatch ${label}`) : undefined });
    const kinds = llm.pkg.model.layer_types, hidden = trace.hidden;
    for (const phase of ["prefill", "decode"]) {
        const got = result[phase], entries = trace.entries[phase];
        const rows = phase === "prefill" ? (got[0].length / hidden) : 1, pad = phase === "prefill" ? rows - trace.last_chunk_valid : 0;
        log(`--- ${phase} (${phase === "prefill" ? `last chunk of ${rows} rows, ${trace.last_chunk_valid} real` : "1 row"}); hidden state entering each layer, real rows only`);
        let first = -1;
        for (let l = 0; l < got.length; l++) {
            const want = reference(entries[l]), have = got[l];
            const c = l === got.length - 1 ? compare(have, want) : compare(have.subarray(pad * hidden), want.subarray(pad * hidden));
            const name = l === got.length - 1 ? "final hidden (after the last layer + final norm)" : `layer ${String(l).padStart(2)} (${kinds[l] === "linear_attention" ? "linear" : "full  "})`;
            const flag = c.rel > threshold || c.bad ? "  <== DEVIATES" : "";
            if (flag && first < 0) first = l;
            log(`${name.padEnd(48)} ${fmt(c)}${flag}`);
        }
        log(first < 0 ? `${phase}: every layer within ${threshold}` : `${phase}: the first layer that deviates more than ${threshold}: ${first < got.length - 1 ? `layer ${first} (its input is still fine only when the line above is)` : "the final hidden state"}`);
    }
    const top = (l) => [...Array(llm.pkg.model.vocab).keys()].sort((a, b) => l[b] - l[a]).slice(0, 5);
    log(`prefill logits top-5: device ${top(result.prefillLogits)}  reference ${trace.prefill_top}  ${top(result.prefillLogits)[0] === trace.prefill_top[0] ? "same argmax" : "DIFFERENT ARGMAX"}`);
    log(`decode  logits top-5: device ${top(result.decodeLogits)}  reference ${trace.decode_top}  ${top(result.decodeLogits)[0] === trace.decode_top[0] ? "same argmax" : "DIFFERENT ARGMAX"}`);
    log("DONE");
    byId("status").textContent = "done";
}
main().catch((e) => { log(`ERROR ${e?.stack || e}`); byId("status").textContent = "failed"; });

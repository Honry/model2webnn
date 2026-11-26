// Runs the image tower of a Qwen3.5 package (HEAD, the vision blocks, MERGER) on the device on the patches of a reference trace (python tools/qwen35_vision_check.py --export -> bench/data/qwen35_vision/)
// and compares the hidden state after HEAD and after every block, and the image tokens, with the numpy interpreter's: where does the device first deviate? (the activations of this ViT leave the
// comfortable float16 range: a few register tokens carry values of 100s from block 10 on, the last block is in the 1000s: look at the relative error AND at "NOT FINITE").
// URL: qwen35vision.html?package=../packages/qwen35-4b-vl-pkg/&trace=../data/qwen35_vision/&device=npu&threshold=0.05&stepwise=1&repeat=1
//      (stepwise=1: every dispatch is awaited and named in the log first; repeat=N: encode N times and report the timings)
import { openPackage, VisionTower } from "../../lib/runtime/index.js";

const query = new URLSearchParams(location.search);
const param = (name, fallback) => query.get(name) ?? fallback;
const packageUrl = new URL(param("package", "../packages/qwen35-4b-vl-pkg/"), location.href).href, traceUrl = new URL(param("trace", "../data/qwen35_vision/"), location.href).href;
const threshold = +param("threshold", 0.05);
const byId = (id) => document.getElementById(id);
const start = performance.now();
const log = (message) => { byId("out").textContent += `[${((performance.now() - start) / 1000).toFixed(1).padStart(6)} s] ${message}\n`; console.log(message); fetch("/log", { method: "POST", body: JSON.stringify({ experiment: "qwen35vision", message }) }).catch(() => {}); };

function compare(got, want) {
    let maxAbs = 0, errSq = 0, refSq = 0, bad = 0, refMax = 0;
    for (let i = 0; i < want.length; i++) {
        const g = got[i];
        refMax = Math.max(refMax, Math.abs(want[i]));
        if (!Number.isFinite(g)) { bad++; continue; }
        const d = g - want[i]; maxAbs = Math.max(maxAbs, Math.abs(d)); errSq += d * d; refSq += want[i] * want[i];
    }
    return { maxAbs, refMax, rel: Math.sqrt(errSq / (refSq + 1e-30)), bad };
}
const fmt = (c) => `rel ${c.rel.toExponential(2)}  max abs ${c.maxAbs.toExponential(2)} (max |ref| ${c.refMax.toPrecision(3)})${c.bad ? `  NOT FINITE x${c.bad}` : ""}`;
const cosine = (a, b, width) => {   // per image token (row)
    let min = 1, mean = 0, rows = a.length / width;
    for (let r = 0; r < rows; r++) {
        let ab = 0, aa = 0, bb = 0;
        for (let i = r * width; i < (r + 1) * width; i++) { ab += a[i] * b[i]; aa += a[i] * a[i]; bb += b[i] * b[i]; }
        const c = ab / Math.sqrt(aa * bb + 1e-30); min = Math.min(min, c); mean += c / rows;
    }
    return { min, mean };
};

async function main() {
    const trace = await (await fetch(`${traceUrl}trace.json`)).json(), blob = await (await fetch(`${traceUrl}trace.bin`)).arrayBuffer();
    const reference = (name) => { const e = trace.entries[name]; return new Float16Array(blob, e.offset, e.bytes / 2); };
    log(`image ${trace.image}: grid ${trace.grid.join("x")}, weights ${trace.weights}, reference compute type ${trace.compute_type}, ${trace.depth} blocks`);
    VisionTower.debugReadback = true;
    const pkg = await openPackage(packageUrl, param("cache", "1") !== "0");
    if (!pkg.plan.vision) throw new Error("this package has no image tower");
    log(`package vision: weights ${pkg.plan.vision.weights}, graph compute type ${pkg.plan.vision.compute_type}, presets ${pkg.plan.vision.presets.map((p) => p.grid.join("x")).join(", ")}`);
    const context = await navigator.ml.createContext({ deviceType: param("device", "npu") });
    const tower = await VisionTower.create(context, pkg, { grids: [trace.grid], preload: [trace.grid] }, (p) => { if (p.stage === "build" || p.stage === "vision") log(`${p.stage} ${p.message ?? ""}`); });   // the graphs of a preset are built on first use: preload builds this one now
    log(`loaded: weights ${(tower.weightsMs / 1000).toFixed(1)} s, graphs built in ${(tower.buildMs / 1000).toFixed(1)} s`);
    const stepwise = param("stepwise", "1") !== "0";
    if (stepwise) tower.onDispatch = (label) => log(`dispatch ${label}`);
    const patches = reference("patches"), tokens = (trace.grid[0] * trace.grid[1]) / 4;
    const image = { patches, grid: trace.grid, tokens, width: 0, height: 0 };
    let rows = null;
    for (let run = 0; run < +param("repeat", 1); run++) {
        rows = await tower.encode(image);
        log(`encode ${run + 1}: host ${tower.lastEncode.hostMs.toFixed(0)} ms (queueing), device ${tower.lastEncode.deviceMs.toFixed(0)} ms (wait + readback of ${trace.depth + 1} hidden states in this debug mode)`);
        if (!stepwise && +param("repeat", 1) === 1) break;
    }
    let first = -1;
    for (let i = 0; i <= trace.depth; i++) {
        const c = compare(tower.trace[i], reference(`hidden_${i}`)), flag = c.rel > threshold || c.bad ? "  <== DEVIATES" : "";
        if (flag && first < 0) first = i;
        log(`${(i === 0 ? "after HEAD" : `after block ${String(i - 1).padStart(2)}`).padEnd(16)} ${fmt(c)}${flag}`);
    }
    log(first < 0 ? `every hidden state within ${threshold}` : `the first hidden state that deviates more than ${threshold}: ${first === 0 ? "HEAD (the patch embedding + position table)" : `block ${first - 1} (its input is fine only if the line above is)`}`);
    const embeds = reference("image_embeds"), c = compare(rows, embeds), k = cosine(rows, embeds, trace.out_hidden);
    log(`image tokens (MERGER)  ${fmt(c)}${c.rel > threshold || c.bad ? "  <== DEVIATES" : ""}`);
    log(`cosine per image token: min ${k.min.toFixed(5)} mean ${k.mean.toFixed(5)}  ${k.min > 0.99 ? "OK" : "LOW"}`);
    log("DONE");
    byId("status").textContent = "done";
}
main().catch((e) => { log(`ERROR ${e?.stack || e}`); byId("status").textContent = "failed"; });

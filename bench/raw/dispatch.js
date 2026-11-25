// Fixed cost of one WebNN dispatch, separated from the cost of the graph itself.
// A tiny graph y = x + t_1 + ... + t_n  (all [1,1,16] f16) is dispatched `chain` times back to back (one readback at the end, like chat.html's decode chain).
//   cases = "n:m:big,..."   n = number of small input tensors bound per dispatch   m = number of output tensors (extra outputs are scaled copies of y)
//                           big = number of 8 MB f16 inputs [1,1,4M] of which only a slice is used (does the SIZE of a bound tensor matter?)
// Result: per_dispatch_ms for each case; js_enqueue_ms_per_dispatch is the JavaScript time of the dispatch() calls alone.
const P = new URLSearchParams(location.search);
const DEVICE = P.get("device") || "npu", CHAIN = +(P.get("chain") || 200), ITERS = +(P.get("iters") || 5), TAG = P.get("tag") || "";
const CASES = (P.get("cases") || "0:1:0,8:1:0,16:1:0,32:1:0,64:1:0,0:8:0,0:1:4,0:1:16").split(",").map((c) => c.split(":").map(Number));
const statusEl = document.getElementById("status"), outEl = document.getElementById("out");
function emit(obj) {
    const s = JSON.stringify({ exp: "dispatch", tag: TAG, device: DEVICE, ...obj });
    console.log(s); outEl.textContent += s + "\n";
    fetch("/log", { method: "POST", body: s }).catch(() => {});
}
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };

async function runCase(context, [n, m, big]) {
    const builder = new MLGraphBuilder(context);
    const x = builder.input("x", { dataType: "float16", shape: [1, 1, 16] });
    const inputs = { x };
    let y = builder.add(x, builder.constant({ dataType: "float16", shape: [1] }, new Float16Array([0])));   // (a graph input cannot be a graph output itself)
    for (let i = 0; i < n; i++) { const t = builder.input(`t${i}`, { dataType: "float16", shape: [1, 1, 16] }); inputs[`t${i}`] = t; y = builder.add(y, t); }
    const bigWords = 4 * 1024 * 1024;
    for (let i = 0; i < big; i++) {
        const b = builder.input(`big${i}`, { dataType: "float16", shape: [1, 1, bigWords] }); inputs[`big${i}`] = b;
        y = builder.add(y, builder.slice(b, [0, 0, 0], [1, 1, 16]));
    }
    const outputs = { y };
    for (let i = 1; i < m; i++) outputs[`y${i}`] = builder.mul(y, builder.constant({ dataType: "float16", shape: [1] }, new Float16Array([i])));
    const t0 = performance.now(), graph = await builder.build(outputs), buildMs = Math.round(performance.now() - t0);
    const inputTensors = {}, outputTensors = {};
    for (const [name, operand] of Object.entries(inputs)) {
        inputTensors[name] = await context.createTensor({ dataType: "float16", shape: operand.shape, writable: true });
        context.writeTensor(inputTensors[name], new Float16Array(operand.shape.reduce((a, b) => a * b, 1)).fill(0.5));
    }
    for (const [name, operand] of Object.entries(outputs)) outputTensors[name] = await context.createTensor({ dataType: "float16", shape: operand.shape, readable: true });
    const readBuffer = new Float16Array(16), times = [];
    for (let i = 0; i < 2 + ITERS; i++) {
        const start = performance.now();
        for (let c = 0; c < CHAIN; c++) context.dispatch(graph, inputTensors, outputTensors);
        const enqueued = performance.now();
        await context.readTensor(outputTensors.y, readBuffer);
        if (i >= 2) times.push([(performance.now() - start) / CHAIN, (enqueued - start) / CHAIN]);
    }
    emit({ event: "case", inputs: n, outputs: m, bigInputs: big, buildMs, per_dispatch_ms: Math.round(median(times.map((t) => t[0])) * 1000) / 1000, js_enqueue_ms_per_dispatch: Math.round(median(times.map((t) => t[1])) * 1000) / 1000 });
}
(async () => {
    try {
        const context = await navigator.ml.createContext({ deviceType: DEVICE });
        emit({ event: "ctx", ua: navigator.userAgent, chain: CHAIN });
        for (const c of CASES) { statusEl.textContent = `case ${c.join(":")}`; try { await runCase(context, c); } catch (e) { emit({ event: "case_failed", case: c.join(":"), err: String(e).slice(0, 200) }); } }
        statusEl.textContent = "done";
    } catch (error) { emit({ event: "fatal", err: String(error && error.stack ? error.stack : error).slice(0, 1500) }); statusEl.textContent = "FAILED: " + error; }
    emit({ event: "done" });
})();

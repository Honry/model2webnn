// Numerical probe of WebNN scatterND on the chosen device: is the result of
//     updated = scatterND(past [1,8,T,128] f16, indices [1,8,S,3] int32 = (0, head, row), updates [1,8,S,128] f16)
// the same as the host reference?  This is exactly the KV-cache update of chat.html?kvStyle=scatterND.
//
// Why a separate page: chat.html with a short prompt cannot tell a wrong scatterND from a right one on the first token (the cache the
// prefill chunk writes is only READ by the next step), so a device whose scatterND is wrong only shows "!!!!" in the decode steps.
//
// URL params:  device=npu|gpu|cpu   T=1151 (cache rows)   S=1,64 (comma list: new rows per call; 1 = decode, 64 = prefill chunk)
//              steps=3 (calls per case: the output of call i is the cache of call i+1 = decode's double buffer swap)
//              start=100 (first row written)   tag=
// Per case it reports: updatedRowsBad (rows that should hold the new data), untouchedRowsBad (rows that should be unchanged), nanCount,
// and the first mismatch (head, row, column, got, expected).
const P = new URLSearchParams(location.search);
const DEVICE = P.get("device") || "npu", T = +(P.get("T") || 1151), STEPS = +(P.get("steps") || 3), START = +(P.get("start") || 100), TAG = P.get("tag") || "";
const SIZES = (P.get("S") || "1,64").split(",").map(Number);
const HEADS = 8, DIM = 128;
const statusEl = document.getElementById("status"), outEl = document.getElementById("out");
function emit(obj) {
    const s = JSON.stringify({ exp: "scatter", tag: TAG, device: DEVICE, ...obj });
    console.log(s); outEl.textContent += s + "\n";
    fetch("/log", { method: "POST", body: s }).catch(() => {});
}
const setStatus = (s) => (statusEl.textContent = s);

function buildScatterGraph(builder, rows, newRows) {
    const past = builder.input("past", { dataType: "float16", shape: [1, HEADS, rows, DIM] });
    const indices = builder.input("indices", { dataType: "int32", shape: [1, HEADS, newRows, 3] });
    const updates = builder.input("updates", { dataType: "float16", shape: [1, HEADS, newRows, DIM] });
    return { updated: builder.scatterND(past, indices, updates) };
}

async function runCase(context, newRows) {
    const builder = new MLGraphBuilder(context);
    const t0 = performance.now();
    const graph = await builder.build(buildScatterGraph(builder, T, newRows));
    const buildMs = Math.round(performance.now() - t0);
    const cacheSize = HEADS * T * DIM, updatesSize = HEADS * newRows * DIM;
    let reference = new Float16Array(cacheSize); for (let i = 0; i < cacheSize; i++) reference[i] = (Math.random() - 0.5) * 4;
    const makeTensor = (dataType, shape, extra) => context.createTensor({ dataType, shape, ...extra });
    let cacheA = await makeTensor("float16", [1, HEADS, T, DIM], { writable: true, readable: true });
    let cacheB = await makeTensor("float16", [1, HEADS, T, DIM], { writable: true, readable: true });
    const indicesTensor = await makeTensor("int32", [1, HEADS, newRows, 3], { writable: true });
    const updatesTensor = await makeTensor("float16", [1, HEADS, newRows, DIM], { writable: true });
    context.writeTensor(cacheA, reference);
    context.writeTensor(cacheB, new Float16Array(cacheSize).fill(NaN));   // the spare buffer starts as garbage, like the real double buffer
    const result = [];
    for (let step = 0; step < STEPS; step++) {
        const firstRow = START + step * newRows;
        const indices = new Int32Array(HEADS * newRows * 3), updates = new Float16Array(updatesSize);
        for (let h = 0; h < HEADS; h++) for (let r = 0; r < newRows; r++) {
            const at = (h * newRows + r) * 3; indices[at] = 0; indices[at + 1] = h; indices[at + 2] = firstRow + r;
            for (let c = 0; c < DIM; c++) updates[(h * newRows + r) * DIM + c] = (Math.random() - 0.5) * 4;
        }
        context.writeTensor(indicesTensor, indices); context.writeTensor(updatesTensor, updates);
        const expected = Float16Array.from(reference);
        for (let h = 0; h < HEADS; h++) for (let r = 0; r < newRows; r++) for (let c = 0; c < DIM; c++)
            expected[(h * T + firstRow + r) * DIM + c] = updates[(h * newRows + r) * DIM + c];
        context.dispatch(graph, { past: cacheA, indices: indicesTensor, updates: updatesTensor }, { updated: cacheB });
        const got = new Float16Array(cacheSize);
        await context.readTensor(cacheB, got);
        let updatedRowsBad = 0, untouchedRowsBad = 0, nanCount = 0, first = null;
        for (let h = 0; h < HEADS; h++) for (let row = 0; row < T; row++) {
            const isUpdated = row >= firstRow && row < firstRow + newRows;
            let rowBad = false;
            for (let c = 0; c < DIM; c++) {
                const at = (h * T + row) * DIM + c, g = got[at], e = expected[at];
                if (Number.isNaN(g)) nanCount++;
                if (!(Math.abs(g - e) <= 1e-3 * Math.max(1, Math.abs(e)))) { if (!rowBad && !first) first = { head: h, row, column: c, got: g, expected: e }; rowBad = true; }
            }
            if (rowBad) { if (isUpdated) updatedRowsBad++; else untouchedRowsBad++; }
        }
        result.push({ step, firstRow, updatedRowsBad, updatedRowsTotal: HEADS * newRows, untouchedRowsBad, nanCount, first });
        reference = expected; [cacheA, cacheB] = [cacheB, cacheA];   // the output becomes the next input
    }
    const ok = result.every((r) => r.updatedRowsBad === 0 && r.untouchedRowsBad === 0 && r.nanCount === 0);
    emit({ event: "case", newRows, rows: T, buildMs, ok, steps: result });
    return ok;
}

(async () => {
    try {
        const context = await navigator.ml.createContext({ deviceType: DEVICE });
        emit({ event: "ctx", ua: navigator.userAgent });
        for (const newRows of SIZES) { setStatus(`scatterND S=${newRows}`); await runCase(context, newRows); }
        setStatus("done");
    } catch (error) { emit({ event: "fatal", err: String(error && error.stack ? error.stack : error).slice(0, 1500) }); setStatus("FAILED: " + error); }
    emit({ event: "done" });
})();

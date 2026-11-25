// Spike A of the model2webnn compiler: is the BLOCK graph that the Python compiler writes (m2w-graph JSON, loaded by ir-loader.js) the SAME graph as the
// hand-written one in qwen3_subgraphs.js?   Build both, bind the same random inputs, compare the outputs, then time N back-to-back dispatches of each (interleaved).
//
// URL: irblock.html?device=npu&S=1&P=1151&chain=35&ir=../ir/decode_block.json   (decode, staged KV, merged projections: the production block)
//      irblock.html?device=npu&S=64&P=1151&kvStyle=onehot&ir=../ir/prefill_block_c64.json  (prefill chunk 64)
// Regenerate the JSON with:  python -m m2w.cli graphs --preset qwen3-4b --out <dir> --max-length 1152 --chunk 64   then copy to bench/ir/
import { halfFill, ropeRows } from "./layer.js?v=20261119x";
import { buildBlockGraph, setMergedProjections } from "./qwen3_subgraphs.js?v=20261124a";
import { loadGraph } from "../../lib/runtime/ir-loader.js";

const query = new URLSearchParams(location.search);
const param = (name, fallback) => query.get(name) ?? fallback;
const deviceType = param("device", "npu"), S = +param("S", 1), P = +param("P", 1151), chainLength = +param("chain", 35), iterations = +param("iters", 7);
const kvStyle = param("kvStyle", S === 1 ? "staged" : "onehot"), stagingRows = +param("stagingRows", 64), computeType = param("computeType", "float32");
const irUrl = param("ir", S === 1 ? "../ir/decode_block.json" : `../ir/prefill_block_c${S}.json`);
const byId = (id) => document.getElementById(id);
const startTime = performance.now();
const log = (message) => { byId("out").textContent += `[${((performance.now() - startTime) / 1000).toFixed(1).padStart(6)} s] ${message}\n`; console.log(message); fetch("/log", { method: "POST", body: JSON.stringify({ experiment: "irblock", message }) }).catch(() => {}); };
const elementCount = (shape) => shape.reduce((a, b) => a * b, 1);
const median = (values) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

let seed = 20261124;
const random = () => { seed ^= seed << 13; seed ^= seed >>> 17; seed ^= seed << 5; return ((seed >>> 0) / 4294967296); };
const normal = () => Math.sqrt(-2 * Math.log(random() + 1e-12)) * Math.cos(2 * Math.PI * random());
const typedArrayFor = (dataType, count) => (dataType === "float16" ? new Float16Array(count) : dataType === "float32" ? new Float32Array(count) : dataType === "int32" ? new Int32Array(count) : new Uint8Array(count));

function inputData(name, operand, position) {
    const count = elementCount(operand.shape);
    if (operand.dataType === "int4") { const data = new Uint8Array(Math.ceil(count / 2)); for (let i = 0; i < data.length; i++) data[i] = (random() * 256) | 0; return data; }
    if (name === "rope_cos" || name === "rope_sin") return ropeRows(position, S)[name === "rope_cos" ? "cos" : "sin"];
    if (name === "attention_mask") {   // the first `position` columns of the main cache and the newest rows are visible
        const total = operand.shape[3], mask = new Float16Array(S * total).fill(-65504);
        for (let row = 0; row < S; row++) { mask.fill(0, row * total, row * total + Math.min(position, P)); if (kvStyle === "staged") mask.fill(0, row * total + P, row * total + P + 3); }
        return mask;
    }
    if (name.endsWith("row_selector")) { const data = new Float16Array(count); for (let s = 0; s < operand.shape[3]; s++) data[(s + 2) * operand.shape[3] + s] = 1; return data; }
    if (name.endsWith("rows_to_keep")) { const data = new Float16Array(count).fill(1); for (let r = 2; r < 2 + S && r < count; r++) data[r] = 0; return data; }
    if (name.endsWith("_scale")) return Float16Array.from({ length: count }, () => 0.002 + 0.002 * random());
    if (name.endsWith("_gamma")) return Float16Array.from({ length: count }, () => 1 + 0.1 * normal());
    return Float16Array.from({ length: count }, () => 0.1 * normal());   // activations, KV
}

async function main() {
    setMergedProjections(true);
    const options = { matmulStyle: "gemm", zeroPoint: "const", dequantStyle: "dequantizeLinear", computeType, fuseKvUpdate: true, kvStyle, valueLayout: "natural", stagingRows };
    const mlContext = await navigator.ml.createContext({ deviceType });
    log(`WebNN context (device=${deviceType}), S=${S}, P=${P}, kvStyle=${kvStyle}, computeType=${computeType}`);

    // 1. build both graphs
    let t = performance.now();
    const handBuilder = new MLGraphBuilder(mlContext), hand = buildBlockGraph(handBuilder, { sequenceLength: S, pastLength: P, maskDataType: "float16", options });
    const handGraph = await handBuilder.build(hand.outputs), handBuild = Math.round(performance.now() - t);
    const doc = await (await fetch(irUrl)).json();
    t = performance.now();
    const irBuilder = new MLGraphBuilder(mlContext), ir = loadGraph(irBuilder, doc);
    const irGraph = await irBuilder.build(ir.outputs), irBuild = Math.round(performance.now() - t);
    log(`built hand-written graph in ${handBuild} ms, IR graph "${doc.name}" in ${irBuild} ms`);

    // 2. same interface?
    const describe = (operands) => Object.entries(operands).map(([name, o]) => `${name}:${o.dataType}[${o.shape.join(",")}]`).sort();
    const sameIn = JSON.stringify(describe(hand.inputs)) === JSON.stringify(describe(ir.inputs)), sameOut = JSON.stringify(describe(hand.outputs)) === JSON.stringify(describe(ir.outputs));
    log(`inputs  ${sameIn ? "IDENTICAL" : "DIFFER"} (${Object.keys(hand.inputs).length} hand / ${Object.keys(ir.inputs).length} IR)`);
    log(`outputs ${sameOut ? "IDENTICAL" : "DIFFER"} (${Object.keys(hand.outputs).length} hand / ${Object.keys(ir.outputs).length} IR)`);
    if (!sameIn || !sameOut) {
        log(`hand inputs only: ${describe(hand.inputs).filter((x) => !describe(ir.inputs).includes(x)).join(" ")}`); log(`IR inputs only: ${describe(ir.inputs).filter((x) => !describe(hand.inputs).includes(x)).join(" ")}`);
        log(`hand outputs only: ${describe(hand.outputs).filter((x) => !describe(ir.outputs).includes(x)).join(" ")}`); log(`IR outputs only: ${describe(ir.outputs).filter((x) => !describe(hand.outputs).includes(x)).join(" ")}`);
        return;
    }

    // 3. one set of tensors, shared by both graphs
    const position = 100, inputs = {}, outputsHand = {}, outputsIr = {};
    for (const [name, operand] of Object.entries(hand.inputs)) { inputs[name] = await mlContext.createTensor({ dataType: operand.dataType, shape: operand.shape, writable: true }); mlContext.writeTensor(inputs[name], inputData(name, operand, position)); }
    for (const [name, operand] of Object.entries(hand.outputs)) {
        outputsHand[name] = await mlContext.createTensor({ dataType: operand.dataType, shape: operand.shape, readable: true });
        outputsIr[name] = await mlContext.createTensor({ dataType: operand.dataType, shape: operand.shape, readable: true });
    }

    // 4. numerics
    mlContext.dispatch(handGraph, inputs, outputsHand); mlContext.dispatch(irGraph, inputs, outputsIr);
    let allEqual = true;
    for (const [name, operand] of Object.entries(hand.outputs)) {
        const a = typedArrayFor(operand.dataType, elementCount(operand.shape)), b = typedArrayFor(operand.dataType, elementCount(operand.shape));
        await mlContext.readTensor(outputsHand[name], a); await mlContext.readTensor(outputsIr[name], b);
        let maxDiff = 0, maxAbs = 0, different = 0, finite = true;
        for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > 0) different++; maxDiff = Math.max(maxDiff, d); maxAbs = Math.max(maxAbs, Math.abs(a[i])); if (!Number.isFinite(a[i]) || !Number.isFinite(b[i])) finite = false; }
        if (different) allEqual = false;
        log(`  ${name.padEnd(22)} max|hand|=${maxAbs.toFixed(3)}  max|hand-IR|=${maxDiff.toExponential(2)}  differing elements=${different}/${a.length}  finite=${finite}`);
    }
    log(allEqual ? "RESULT numerics: BIT-IDENTICAL outputs" : "RESULT numerics: outputs differ (see above)");

    // 5. timing: `chain` back-to-back dispatches, one readback; interleaved A B A B
    const firstName = Object.keys(hand.outputs)[0], firstOperand = hand.outputs[firstName], sink = typedArrayFor(firstOperand.dataType, elementCount(firstOperand.shape));
    const run = async (graph, outputs) => { const s = performance.now(); for (let c = 0; c < chainLength; c++) mlContext.dispatch(graph, inputs, outputs); await mlContext.readTensor(outputs[firstName], sink); return performance.now() - s; };
    const handTimes = [], irTimes = [];
    for (let i = 0; i < 3 + iterations; i++) {
        const a = await run(handGraph, outputsHand), b = await run(irGraph, outputsIr), c = await run(irGraph, outputsIr), d = await run(handGraph, outputsHand);
        if (i >= 3) { handTimes.push((a + d) / 2); irTimes.push((b + c) / 2); }
    }
    const handMs = median(handTimes), irMs = median(irTimes);
    log(`RESULT timing: chain=${chainLength}  hand-written ${handMs.toFixed(1)} ms (${(handMs / chainLength).toFixed(3)} ms/dispatch)   IR ${irMs.toFixed(1)} ms (${(irMs / chainLength).toFixed(3)} ms/dispatch)   IR/hand = ${(irMs / handMs).toFixed(3)}`);
    byId("status").textContent = "done";
}
main().catch((error) => { log(`ERROR ${error && error.stack || error}`); byId("status").textContent = "failed"; });

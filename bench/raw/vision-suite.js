// The image test suite of the Qwen3.5 image input, on the device: every case of vision-suite/cases.json (python/tools/make_vision_suite.py: a picture, a question, the substrings the answer must
// contain) is run through the REAL path of the runtime (llm.prepareImage -> grid preset chosen from the picture -> VisionTower -> prefill with image rows -> greedy decode) and scored PASS / FAIL.
// The result lines go to the page and to the /log mechanism of the other bench pages (POST /log, one JSON object per line); the last lines are the summary (passed / total, timing per grid preset).
//
// URL: vision-suite.html?model=qwen35-4b-vl-pkg&device=npu&detail=auto&maxNewTokens=24
//      model=<id of bench/packages/models.json> | package=<url of a package directory>      (default model qwen35-4b-vl-pkg)
//      detail=low|auto|high   how much detail the automatic grid choice allows (auto: up to 24x24 = 144 image tokens; low 16x16 = 64; high 32x32 = 256)
//      grids=24x24,16x24      restrict the presets a picture may get (default: every preset of the package); a single one forces it
//      case=ocr-small         run one case (or a comma separated list)      repeat=2   run the list N times (first-use graph builds only count once)
//      maxNewTokens=24  maxLength=1024  chunk=64  suite=vision-suite/  cache=0 (no OPFS cache of the weights)
import { loadModel, buildPrompt, tokenizerFromTransformersJs } from "../../lib/runtime/index.js";

const TRANSFORMERS_URL = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js";
const query = new URLSearchParams(location.search);
const param = (name, fallback) => query.get(name) ?? fallback;
const byId = (id) => document.getElementById(id);
const start = performance.now();
const deviceType = param("device", "npu"), detail = param("detail", "auto"), maxNewTokens = +param("maxNewTokens", 24), repeat = Math.max(1, +param("repeat", 1));
const modelId = param("model", "qwen35-4b-vl-pkg");
const packageUrl = new URL(query.get("package") ?? `../packages/${modelId}/`, location.href).href.replace(/\/?$/, "/");
const suiteUrl = new URL(param("suite", "vision-suite/"), location.href).href.replace(/\/?$/, "/");
const only = query.get("case") ? query.get("case").split(",") : null;
const grids = query.get("grids") ? query.get("grids").split(/[\s,]+/).filter(Boolean).map((g) => g.toLowerCase().split("x").map(Number)) : undefined;
const seconds = (ms) => (ms / 1000).toFixed(2) + " s", round = (v) => Math.round(v * 10) / 10;

function log(event, message, fields = {}) {
    byId("out").textContent += `[${((performance.now() - start) / 1000).toFixed(1).padStart(6)} s] ${message}\n`;
    console.log(message);
    fetch("/log", { method: "POST", body: JSON.stringify({ experiment: "vision-suite", event, device: deviceType, package: packageUrl, detail, message, ...fields }) }).catch(() => {});
}
const setStatus = (text) => (byId("status").textContent = text);

/** the same rules as python/tools/qwen35_vision_quality.py score(): every expect_all substring, one of expect_any (when given), every expect_regex; case-insensitive */
export function scoreAnswer(testCase, answer) {
    const text = answer.toLowerCase();
    return (testCase.expect_all ?? []).every((s) => text.includes(s.toLowerCase()))
        && (!testCase.expect_any?.length || testCase.expect_any.some((s) => text.includes(s.toLowerCase())))
        && (testCase.expect_regex ?? []).every((r) => new RegExp(r, "i").test(answer));
}

async function run() {
    const { cases: all } = await (await fetch(new URL("cases.json", suiteUrl), { cache: "no-store" })).json();
    const cases = all.filter((c) => !only || only.includes(c.id));
    if (!cases.length) throw new Error(`no case matches ${query.get("case")} (the suite has ${all.map((c) => c.id).join(", ")})`);
    for (const c of cases) {   // thumbnails
        const figure = Object.assign(document.createElement("figure"), { id: `case-${c.id}` });
        figure.innerHTML = `<img src="${new URL(c.image, suiteUrl).href}" alt=""><figcaption>${c.id}</figcaption>`;
        byId("cases").append(figure);
    }
    const session = { ...(query.get("maxLength") ? { maxLength: +query.get("maxLength") } : {}), ...(query.get("chunk") ? { chunkSizes: [+query.get("chunk")] } : {}) };
    setStatus("loading the model");
    const t0 = performance.now();
    const llm = await loadModel(packageUrl, { device: deviceType, cache: param("cache", "1") !== "0", ...session, vision: grids ? { grids } : true,
        onProgress: (p) => setStatus(p.stage === "download" ? `downloading ${Math.round(p.loaded / 2 ** 20)} / ${Math.round(p.total / 2 ** 20)} MB` : `${p.stage}${p.message ? ": " + p.message : ""}`) });
    if (!llm.vision) throw new Error("this package has no image tower");
    const pkg = llm.pkg, tokenizer = await tokenizerFromTransformersJs(await import(TRANSFORMERS_URL), pkg), stop = new Set(pkg.model.eos_token_ids);
    llm.vision.progress = (p) => { if (p.stage === "build") setStatus(p.message); };
    log("loaded", `loaded ${pkg.model.architecture} in ${seconds(performance.now() - t0)}: image tower weights ${pkg.plan.vision.weights}, candidate presets ${llm.vision.grids.map((g) => g.join("x")).join(" ")}, detail ${detail}, max new tokens ${maxNewTokens}, max length ${llm.maxLength}`,
        { weights: pkg.plan.vision.weights, presets: llm.vision.grids, max_length: llm.maxLength, load_ms: Math.round(performance.now() - t0) });

    const results = [];
    for (let pass = 1; pass <= repeat; pass++) {
        for (const c of cases) {
            const id = repeat > 1 ? `${c.id}#${pass}` : c.id;
            try {
                setStatus(`${id}: preprocess`);
                const blob = await (await fetch(new URL(c.image, suiteUrl))).blob(), bitmap = await createImageBitmap(blob);
                const t1 = performance.now(), image = llm.prepareImage(bitmap, undefined, detail), preprocessMs = performance.now() - t1;
                const width = bitmap.width, height = bitmap.height;
                bitmap.close?.();
                const buildMs = await llm.vision.ensure(image.grid);   // graphs of a preset are built on its first use: do it here so that TTFT / vision ms are clean (0 when the preset was built before)
                setStatus(`${id}: generating`);
                const ids = tokenizer.encode(buildPrompt(pkg.chatFormat, [{ user: c.question, images: 1 }], undefined, new Date(), pkg.model.vision.image_prompt));
                const generated = [];
                for await (const token of llm.generate(ids, { maxNewTokens, images: [image], temperature: 0 })) { if (stop.has(token)) break; generated.push(token); }
                const answer = tokenizer.decode(generated).trim(), s = llm.stats, steps = s.steps;
                const tpot = steps.length ? steps.reduce((sum, step) => sum + step.totalMs, 0) / steps.length : 0;
                const passedExpectations = scoreAnswer(c, answer), expectedFail = (c.xfail_at ?? []).includes(detail), ok = passedExpectations;   // xfail_at: a FAIL at this detail level is expected (resolution), reported as XFAIL and not counted
                const r = { id, case: c.id, ok, expectedFail, grid: image.grid.join("x"), pixels: `${image.grid[1] * pkg.model.vision.patch_size}x${image.grid[0] * pkg.model.vision.patch_size}`, imageTokens: s.imageTokens, visionMs: s.visionMs ?? 0, buildMs,
                    preprocessMs, ttftMs: s.prefillMs, tpotMs: tpot, answer };
                results.push(r);
                byId(`case-${c.id}`).style.outline = `3px solid ${ok ? "#7fcf8a" : expectedFail ? "#ffb454" : "#ff7a7a"}`;
                log("case", `${ok ? (expectedFail ? "XPASS" : "PASS") : expectedFail ? "XFAIL" : "FAIL"} ${id}: ${width}x${height} -> grid ${r.grid} (${r.pixels} px), ${r.imageTokens} image tokens, preprocess ${round(preprocessMs)} ms, graph build ${round(buildMs)} ms, vision encode ${round(r.visionMs)} ms, TTFT ${seconds(r.ttftMs)}, TPOT ${round(tpot)} ms | answer: ${JSON.stringify(answer)}`,
                    { case: c.id, pass: ok, grid: r.grid, width, height, image_tokens: r.imageTokens, preprocess_ms: round(preprocessMs), build_ms: Math.round(buildMs), vision_ms: Math.round(r.visionMs), ttft_ms: Math.round(r.ttftMs),
                      tpot_ms: round(tpot), prompt_tokens: s.promptTokens, answer, question: c.question });
            } catch (error) {
                results.push({ id, case: c.id, ok: false, error: String(error) });
                log("case", `FAIL ${id}: ${String(error && error.stack ? error.stack : error).slice(0, 600)}`, { case: c.id, pass: false, error: String(error) });
            }
        }
    }
    const counted = results.filter((r) => !(r.expectedFail && !r.ok)), passed = counted.filter((r) => r.ok).length, xfail = results.length - counted.length;   // an expected failure (xfail_at) is neither a pass nor a failure
    log("summary", `SUMMARY: ${passed}/${counted.length} passed (detail ${detail}, weights ${pkg.plan.vision.weights})${xfail ? `, ${xfail} expected failure(s) at this detail level not counted` : ""}; failed: ${counted.filter((r) => !r.ok).map((r) => r.id).join(", ") || "none"}`, { passed, total: counted.length, xfail, detail, weights: pkg.plan.vision.weights });
    const perGrid = new Map();
    for (const r of results.filter((x) => x.grid)) (perGrid.get(r.grid) ?? perGrid.set(r.grid, []).get(r.grid)).push(r);
    for (const [grid, list] of perGrid) {
        const mean = (key) => list.reduce((sum, r) => sum + r[key], 0) / list.length, built = list.reduce((sum, r) => sum + r.buildMs, 0);
        log("grid", `grid ${grid}: ${list.length} run(s), ${list[0].imageTokens} image tokens, vision encode ${round(mean("visionMs"))} ms, TTFT ${seconds(mean("ttftMs"))}, TPOT ${round(mean("tpotMs"))} ms, graph build ${round(built)} ms (first use), ${list.filter((r) => r.ok).length}/${list.length} passed`,
            { grid, runs: list.length, image_tokens: list[0].imageTokens, vision_ms: Math.round(mean("visionMs")), ttft_ms: Math.round(mean("ttftMs")), tpot_ms: round(mean("tpotMs")), build_ms: Math.round(built), passed: list.filter((r) => r.ok).length });
    }
    setStatus(`done: ${passed}/${counted.length} passed`);
    log("done", "Done");
}
run().catch((error) => { log("fatal", "Failed: " + String(error && error.stack ? error.stack : error).slice(0, 1500)); setStatus("FAILED: " + error); });

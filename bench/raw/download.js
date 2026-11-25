// Download throughput of a model package: the same weight files fetched with different numbers of files in flight (windows) and byte ranges per big file (segments).
// URL: download.html?package=../packages/qwen3-4b-pkg-i8/&windows=1,2,4,8&segments=1,4&limitMB=600&repeat=1
// The runtime keeps a window of files ahead of the consumer (LoadOptions.parallelDownloads) and splits big files into ranges (LoadOptions.downloadSegments); this page finds good defaults
// for a given server / network. Nothing is cached: UrlSource fetches with cache: "no-store"; a server or CDN cache can still make the first combination slower than the later ones (repeat=2).
import { UrlSource, openPackage } from "../../lib/runtime/index.js";

const query = new URLSearchParams(location.search), byId = (id) => document.getElementById(id);
const list = (name, fallback) => (query.get(name) ?? fallback).split(",").map(Number).filter((n) => n > 0);
const packageUrl = new URL(query.get("package") ?? "../packages/qwen3-4b-pkg-i8/", location.href).href;
const windows = list("windows", "1,2,4,8"), segmentCounts = list("segments", "1,4"), limitBytes = +(query.get("limitMB") ?? 600) * 2 ** 20, repeat = +(query.get("repeat") ?? 1);
const start = performance.now();
const log = (message) => {
    byId("out").textContent += `[${((performance.now() - start) / 1000).toFixed(1).padStart(6)} s] ${message}\n`;
    console.log(message);
    fetch("/log", { method: "POST", body: JSON.stringify({ experiment: "download", message }) }).catch(() => {});
};

/** download `files` with at most `window` in flight; resolves with the elapsed ms */
async function download(source, files, window) {
    const t0 = performance.now();
    let next = 0;
    const worker = async () => {
        while (next < files.length) {
            const file = files[next++];
            const buffer = await source.bytes(`weights/${file.file}`, undefined, file.bytes);
            if (buffer.byteLength !== file.bytes) throw new Error(`${file.file}: ${buffer.byteLength} of ${file.bytes} bytes`);
        }
    };
    await Promise.all(Array.from({ length: Math.min(window, files.length) }, worker));
    return performance.now() - t0;
}

async function main() {
    const pkg = await openPackage(packageUrl, false);
    const order = (role) => pkg.manifest.files.filter((f) => f.role === role);
    const all = [...order("embedding"), ...order("final_norm"), ...order("layer").sort((a, b) => a.layer - b.layer)], files = [];
    let bytes = 0;
    for (const file of all) { if (files.length && bytes + file.bytes > limitBytes) break; files.push(file); bytes += file.bytes; }
    log(`${packageUrl}: ${files.length} files, ${(bytes / 2 ** 20).toFixed(0)} MB (of ${all.length} files); windows ${windows}, segments ${segmentCounts}, repeat ${repeat}`);
    const results = [];
    for (let round = 0; round < repeat; round++) {
        for (const segments of segmentCounts) {
            for (const window of windows) {
                byId("status").textContent = `window ${window}, segments ${segments}`;
                const ms = await download(new UrlSource(packageUrl, { segments, segmentMinBytes: 48 * 2 ** 20 }), files, window);
                const rate = bytes / 2 ** 20 / (ms / 1000);
                results.push({ window, segments, ms, rate });
                log(`RESULT window=${window} segments=${segments}: ${(ms / 1000).toFixed(2)} s, ${rate.toFixed(0)} MB/s${repeat > 1 ? ` (round ${round + 1})` : ""}`);
            }
        }
    }
    const best = results.reduce((a, b) => (b.rate > a.rate ? b : a));
    log(`BEST window=${best.window} segments=${best.segments}: ${best.rate.toFixed(0)} MB/s`);
    byId("status").textContent = "done";
}
main().catch((error) => { log("FAILED " + String(error && error.stack ? error.stack : error)); byId("status").textContent = "FAILED"; });

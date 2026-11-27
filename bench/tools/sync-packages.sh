#!/usr/bin/env bash
# Prepare the benchmark pages for the model packages built by `m2w compile` (python/):
#   bench/packages/<name>    -> a junction to python/_out/<name> (no copy of the 1.3 GB), tiny-pkg -> tests/golden/tiny-package
#   bench/packages/models.json   the model list of chat.html (label from the package's own files; tiny-pkg only with ?dev=1)
#   bench/ir/*.json              (with --ir) the Qwen3-4B graphs for raw/irblock.html, regenerated with the Python compiler
# The pages import the runtime from lib/ (npm run build:lib), so build it first.
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
bench="$here/bench"
if [ "${1:-}" = "--ir" ]; then
  (cd "$here/python" && PYTHONPATH=. python -m m2w.cli graphs --preset qwen3-4b --out _out/qwen3-4b --max-length 1152 --chunk 64 --blocks-per-graph 5 >/dev/null)
  mkdir -p "$bench/ir" && cp "$here"/python/_out/qwen3-4b/*.json "$bench/ir/" && echo "bench/ir: $(ls "$bench/ir" | wc -l) graphs"
fi
mkdir -p "$bench/packages"
for name in tiny-pkg tiny-qwen35-pkg aion-2.3b-i8 qwen3-4b-pkg-i8 gemma4-e4b-pkg qwen35-4b-pkg qwen35-4b-pkg-gqa qwen35-4b-pkg-gqa16 qwen35-4b-pkg-split qwen35-4b-vl-pkg qwen35-4b-vl-int8g128-pkg qwen35-4b-vl-float16-pkg qwen35-4b-vl-int4-pkg; do
  src="$here/python/_out/$name"
  [ "$name" = tiny-pkg ] && src="$here/tests/golden/tiny-package"
  [ "$name" = tiny-qwen35-pkg ] && src="$here/tests/golden/tiny-qwen35-package"
  if [ -d "$src" ] && [ ! -e "$bench/packages/$name" ]; then
    cmd //c mklink //J "$(cygpath -w "$bench/packages/$name")" "$(cygpath -w "$src")" >/dev/null && echo "linked packages/$name"
  fi
done
echo "packages: $(ls "$bench/packages" | xargs)"
# models.json
node -e '
const fs = require("fs"), path = require("path"), dir = process.argv[1];
const entries = fs.readdirSync(dir, { withFileTypes: true }).filter((d) => d.isDirectory() || d.isSymbolicLink()).map((d) => d.name).filter((n) => fs.existsSync(path.join(dir, n, "m2w.json"))).map((id) => {
  const index = JSON.parse(fs.readFileSync(path.join(dir, id, "m2w.json"), "utf8")), plan = JSON.parse(fs.readFileSync(path.join(dir, id, "plan.json"), "utf8"));
  // listed by default: one entry per model (text, or text + image); the test models and the experiment siblings (attention variants, other vision weight formats, the same package under a second name) only with ?dev=1
  const experiment = /-(gqa|gqa16|split)$|^qwen35-4b-vl-(int8g128|float16|int4)-pkg$/.test(id), dev = id.startsWith("tiny-") || experiment;
  const name = id === "tiny-pkg" ? "tiny hybrid test model" : id === "tiny-qwen35-pkg" ? "tiny Qwen3.5 test model" : String(index.source || id).split("/").pop().replace(/-Instruct$|-it$/, "");
  const kind = plan.vision ? "text + image" : "text";
  return { id, label: dev ? `${name} · ${kind}${experiment ? ` · ${id}` : ""}` : `${name} · ${kind}`, url: `${id}/`, dev };
}).sort((x, y) => x.dev - y.dev || x.label.localeCompare(y.label));
fs.writeFileSync(path.join(dir, "models.json"), JSON.stringify(entries, null, 1));
console.log("models.json: " + entries.map((e) => e.id).join(", "));
' "$bench/packages"

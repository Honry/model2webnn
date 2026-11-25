#!/bin/bash
# usage: run_variants.sh <prefix> <name=query> ...   runs raw/npuw.html?device=npu&only=block&S=1&P=${P:-1151}&chain=35&<query> for each variant, prints ms per dispatch
cd "$(dirname "$0")"
prefix=$1; shift
for v in "$@"; do
  name=${v%%=*}; q=${v#*=}; [ "$q" = "$v" ] && q=""
  run=${prefix}_$name
  powershell -ExecutionPolicy Bypass -File run_bench.ps1 -Page raw/npuw.html -Build npuflag -Tag ort-ov-npu-qwen2-1102-RelWithDebInfo -ExtraArgs '--webnn-ort-ovep-npuw=NPU_COMPILER_DYNAMIC_QUANTIZATION=YES' -Query "device=npu&only=block&S=1&P=${P:-1151}&iters=5&warm=2&chain=35&tag=$run&$q" -Run $run -TimeoutMin 10 >/dev/null 2>&1
  r=$(grep -o '"per_dispatch_ms":[0-9.]*' results/$run.jsonl | head -1)
  e=$(grep -o '"event":"fatal"[^}]*\|Failed[^"]*' results/$run.jsonl | head -1 | cut -c1-160)
  echo "$name [$q]: ${r:-no result} $e"
done

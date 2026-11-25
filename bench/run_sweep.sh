#!/bin/bash
# usage: run_sweep.sh <prefix> <chain> <name=query> ...   block graph at the given query (S etc. inside), prints ms per dispatch
cd "$(dirname "$0")"
prefix=$1; chain=$2; shift; shift
for v in "$@"; do
  name=${v%%=*}; q=${v#*=}
  run=${prefix}_$name
  powershell -ExecutionPolicy Bypass -File run_bench.ps1 -Page raw/npuw.html -Build npuflag -Tag ort-ov-npu-qwen2-1102-RelWithDebInfo -ExtraArgs '--webnn-ort-ovep-npuw=NPU_COMPILER_DYNAMIC_QUANTIZATION=YES' -Query "device=npu&only=block&P=1151&iters=3&warm=1&chain=$chain&tag=$run&$q" -Run $run -TimeoutMin 20 >/dev/null 2>&1
  r=$(grep -o '"per_dispatch_ms":[0-9.]*' results/$run.jsonl | head -1); b=$(grep -o '"build_ms":[0-9]*' results/$run.jsonl | head -1)
  e=$(grep -o '"event":"fatal"[^}]*\|Failed[^"]*' results/$run.jsonl | head -1 | cut -c1-120)
  echo "$name [$q]: ${r:-no result} $b $e"
done

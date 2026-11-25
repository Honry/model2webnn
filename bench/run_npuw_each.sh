#!/bin/bash
# build + dispatch each NPUW-style subgraph alone on the NPU (decode shapes unless S/P given)
cd "$(dirname "$0")"
S=${1:-1}; P=${2:-1151}; PH=${3:-decode}
for o in head block tail lmhead; do
  powershell -ExecutionPolicy Bypass -File run_bench.ps1 -Page raw/npuw.html -Query "device=npu&only=$o&S=$S&P=$P&iters=10&tag=npu_${PH}_$o" -Run npuw_npu_${PH}_$o -TimeoutMin 40 2>&1 | grep -E "TIMEOUT"
done

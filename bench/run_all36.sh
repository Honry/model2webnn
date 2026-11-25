#!/bin/bash
# NPU: all 36 layers in ONE graph. const vs hybrid (int4 input, scale+gamma const), S = 1 / 8 / 64.
cd "$(dirname "$0")"
run() { tag=$1; q=$2; powershell -NoProfile -ExecutionPolicy Bypass -File run_bench.ps1 -Page raw/minimal.html -Query "$q&device=npu&K=36&T=512&tag=$tag" -Run r_$tag -TimeoutMin 40 > /dev/null 2>&1; }
run all36_const_S1  "mode=const&S=1&iters=30&warm=5"
run all36_const_S8  "mode=const&S=8&iters=20&warm=3"
run all36_const_S64 "mode=const&S=64&iters=4&warm=1"
run all36_hyb_S1    "mode=input&constScale=1&constNorm=1&S=1&iters=15&warm=3"
run all36_hyb_S64   "mode=input&constScale=1&constNorm=1&S=64&iters=4&warm=1"
echo ALL_DONE > results/all36.done

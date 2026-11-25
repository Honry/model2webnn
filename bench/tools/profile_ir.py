"""Per-layer NPU profile of an OpenVINO IR (e.g. the OVEP dump of one of our WebNN subgraphs, or an NPUW dump xml).

usage:  profile_ir.py <model.xml> [iterations=20] [KEY=VALUE ...]      (extra KEY=VALUE are NPU compile properties)
Compiles the model for the NPU with PERF_COUNT, runs it with random inputs and prints the per-layer real time (microseconds, averaged over the
iterations) of the COMPILED graph: layer names, exec types and how the compiler fused things are visible there.
"""
import sys, time
import numpy as np
import openvino as ov

path = sys.argv[1]
iterations = int(sys.argv[2]) if len(sys.argv) > 2 and "=" not in sys.argv[2] else 20
extra = dict(a.split("=", 1) for a in sys.argv[2:] if "=" in a)
core = ov.Core()
model = core.read_model(path)
config = {"PERF_COUNT": True, "NPU_COMPILER_DYNAMIC_QUANTIZATION": "YES", **extra}
t0 = time.time()
compiled = core.compile_model(model, "NPU", config)
print(f"compiled in {time.time() - t0:.1f} s, properties {config}")
request = compiled.create_infer_request()
rng = np.random.default_rng(1)
keep = []   # ov.Tensor shares the numpy memory: the arrays must stay alive
for i, port in enumerate(compiled.inputs):
    shape = list(port.get_shape()); et = port.get_element_type(); count = int(np.prod(shape))
    name = port.get_any_name()
    if et == ov.Type.i4 or et == ov.Type.u4:
        raw = rng.integers(0, 256, size=(count + 1) // 2, dtype=np.uint8); keep.append(raw)
        request.set_input_tensor(i, ov.Tensor(raw, shape, et))
    elif et == ov.Type.f16 or et == ov.Type.f32:
        npt = np.float16 if et == ov.Type.f16 else np.float32
        value = 1.0 if "gamma" in name else 0.002 if "scale" in name else 0.05
        data = np.full(shape, value, dtype=npt)
        if "mask" in name: data[...] = 0
        keep.append(data)
        request.set_input_tensor(i, ov.Tensor(data))
    elif et == ov.Type.i32 or et == ov.Type.i64:
        request.set_input_tensor(i, ov.Tensor(np.zeros(shape, dtype=np.int32 if et == ov.Type.i32 else np.int64)))
    else:
        request.set_input_tensor(i, ov.Tensor(rng.integers(0, 2, size=shape).astype(np.uint8), shape, et) if et == ov.Type.u8 else ov.Tensor(np.zeros(shape, dtype=np.uint8)))
for _ in range(3): request.infer()
totals, order, meta = {}, [], {}
wall = []
for _ in range(iterations):
    t = time.perf_counter(); request.infer(); wall.append((time.perf_counter() - t) * 1000)
    for p in request.get_profiling_info():
        key = p.node_name
        if key not in totals: totals[key] = 0.0; order.append(key); meta[key] = (p.node_type, p.exec_type)
        totals[key] += p.real_time.total_seconds() * 1e6
print(f"wall time per infer: median {np.median(wall):.3f} ms, min {min(wall):.3f} ms ({iterations} runs)")
rows = [(totals[k] / iterations, k) for k in order]
print(f"sum of layer real_time: {sum(r[0] for r in rows) / 1000:.3f} ms over {len(rows)} layers")
print(f"{'us':>9}  {'type':<22} {'exec':<24} name")
import os
for us, key in (rows if os.environ.get("ORDER") else sorted(rows, reverse=True)[:60]):
    print(f"{us:9.1f}  {meta[key][0]:<22.22} {meta[key][1]:<24.24} {key[:110]}")

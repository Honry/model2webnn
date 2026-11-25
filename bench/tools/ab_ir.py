"""A/B timing of several OpenVINO IRs on the NPU under identical conditions (same harness, interleaved rounds to cancel drift).

usage:  ab_ir.py <rounds> <iterations> name=model.xml [name=model.xml ...] [KEY=VALUE properties via env NPU_PROPS="K=V;K=V"]
Every model gets random-but-finite inputs of the same kind (int4 weights random, scales 0.002, gammas 1, masks 0, other f16/f32 0.05) set ONCE;
each infer() therefore includes the host->device copy of the inputs, identically for all models that have the same input sizes.
Prints the median ms per infer of every model per round and overall.
"""
import os, sys, time
import numpy as np
import openvino as ov

rounds, iterations = int(sys.argv[1]), int(sys.argv[2])
models = [a.split("=", 1) for a in sys.argv[3:]]
props = {"NPU_COMPILER_DYNAMIC_QUANTIZATION": "YES"}
for kv in filter(None, os.environ.get("NPU_PROPS", "").split(";")):
    k, v = kv.split("=", 1); props[k] = v
core = ov.Core()
rng = np.random.default_rng(1)
requests, keep = {}, []
for name, path in models:
    compiled = core.compile_model(core.read_model(path), "NPU", props)
    request = compiled.create_infer_request()
    for i, port in enumerate(compiled.inputs):
        shape = list(port.get_shape()); et = port.get_element_type(); count = int(np.prod(shape))
        pname = port.get_any_name()
        if et in (ov.Type.i4, ov.Type.u4):
            raw = rng.integers(0, 256, size=(count + 1) // 2, dtype=np.uint8); keep.append(raw); request.set_input_tensor(i, ov.Tensor(raw, shape, et))
        elif et in (ov.Type.f16, ov.Type.f32):
            npt = np.float16 if et == ov.Type.f16 else np.float32
            value = 1.0 if "gamma" in pname else 0.002 if "scale" in pname else 0.05
            data = np.full(shape, value, dtype=npt)
            if "mask" in pname: data[...] = 0
            keep.append(data); request.set_input_tensor(i, ov.Tensor(data))
        else:
            data = np.zeros(shape, dtype=np.int32); keep.append(data); request.set_input_tensor(i, ov.Tensor(data))
    for _ in range(5): request.infer()
    requests[name] = request
    print(f"compiled {name}: {len(compiled.inputs)} inputs, {len(compiled.outputs)} outputs", flush=True)
results = {name: [] for name, _ in models}
for r in range(rounds):
    line = []
    for name, _ in models:
        ts = []
        for _ in range(iterations):
            t = time.perf_counter(); requests[name].infer(); ts.append((time.perf_counter() - t) * 1000)
        results[name].append(float(np.median(ts))); line.append(f"{name} {results[name][-1]:.3f}")
    print(f"round {r + 1}: " + " | ".join(line), flush=True)
print("overall median of round medians (ms): " + " | ".join(f"{n} {np.median(v):.3f}" for n, v in results.items()))

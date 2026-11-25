"""Does a zero point break the NPU compiler's int4 dynamic-dequantization absorption?

Builds one decode-shaped int4 projection  y = x @ dequant(W)^T  (W int4 [G,N,128] and scale f16 [G,N,1] are graph inputs, as in our block graphs)
in several zero-point forms and saves each as an OpenVINO IR:
  sym          Convert(i4->f16) -> Multiply(scale)                                  (no zero point at all)
  zp_zero      Convert(i4->f16) -> Subtract(Convert(i4 CONST, all zeros)) -> Multiply(scale)
  zp_splat     same with a constant zero point of 3 everywhere
  zp_varying   same with a constant zero point that differs per (group, row)
  zp_param     zero point is a graph input (dynamic)
usage:  zp_probe.py <output dir> [N=4096] [K=2560]
"""
import sys, os
import numpy as np
import openvino as ov
from openvino import opset13 as ops, Type

out_dir = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4096
K = int(sys.argv[3]) if len(sys.argv) > 3 else 2560
G = K // 128
os.makedirs(out_dir, exist_ok=True)
rng = np.random.default_rng(0)

def int4_constant(values):  # values: int array in [-8, 7], shape [G,N,1]
    packed = np.zeros((values.size + 1) // 2, dtype=np.uint8)
    flat = (values.reshape(-1) & 0xF).astype(np.uint8)
    packed[: (flat.size + 1) // 2] = flat[0::2] | (np.pad(flat[1::2], (0, flat.size % 2)) << 4)
    tensor = ov.Tensor(packed, list(values.shape), Type.i4)
    return ops.constant(tensor)

def build(name, zero_point):
    x = ops.parameter([1, 1, K], Type.f16, name="x")
    w = ops.parameter([G, N, 128], Type.i4, name="weight")
    scale = ops.parameter([G, N, 1], Type.f16, name="scale")
    params = [x, w, scale]
    wf = ops.convert(w, Type.f16)
    if zero_point is not None:
        if zero_point == "param":
            zp = ops.parameter([G, N, 1], Type.i4, name="zero_point"); params.append(zp)
        else:
            values = {"zero": np.zeros((G, N, 1), np.int8), "splat": np.full((G, N, 1), 3, np.int8),
                      "varying": rng.integers(-8, 8, size=(G, N, 1)).astype(np.int8)}[zero_point]
            zp = int4_constant(values)
        wf = ops.subtract(wf, ops.convert(zp, Type.f16))
    wf = ops.multiply(wf, scale)
    wf = ops.reshape(ops.transpose(wf, ops.constant(np.array([1, 0, 2], np.int64))), ops.constant(np.array([N, K], np.int64)), special_zero=False)
    wf = ops.convert(wf, Type.f32)
    xr = ops.reshape(ops.convert(x, Type.f32), ops.constant(np.array([1, K], np.int64)), special_zero=False)
    y = ops.matmul(xr, wf, transpose_a=False, transpose_b=True)
    model = ov.Model([ops.convert(y, Type.f16)], params, name)
    path = os.path.join(out_dir, name + ".xml")
    ov.save_model(model, path)
    return path

for name, zp in [("sym", None), ("zp_zero", "zero"), ("zp_splat", "splat"), ("zp_varying", "varying"), ("zp_param", "param")]:
    print(build(name, zp))

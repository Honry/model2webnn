"""Extract the Qwen3-4B weights of an OpenVINO IR (optimum-intel int4 export) into the NPUW closure layout used by the WebNN demo.

  python -I extract_ov_weights.py <openvino_model.xml> <out_dir>

Per decoder layer N a file l<N>.bin with, in this order (see manifest.json for dtype/shape/offset):
  query_weight query_scale query_norm_gamma key_weight key_scale key_norm_gamma value_weight value_scale
  output_projection_weight output_projection_scale gate_projection_* up_projection_* down_projection_*
  input_layernorm_gamma post_attention_layernorm_gamma
  *_weight : int4 packed, laid out [G, N, 128] (group-major, as NPUW's closure after its permute); OV stores [N, G, 128]
  *_scale  : f16 scales [G, N, 1]
  *_gamma  : f32 RMSNorm gammas ([1,1,E] or [1,1,1,128])
Plus embed.bin (u8 [V,E]), embed_zp.bin (u8 [V,1]), embed_s.bin (f16 [V,1]) (tied lm_head), final_norm.bin (f32 [1,1,E]).
"""
import json, mmap, os, re, sys
import numpy as np
import xml.etree.ElementTree as ET

xml_path, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
root = ET.parse(xml_path).getroot()
layers = {l.get("id"): l for l in root.find("layers")}
src = {}  # (to_layer, to_port) -> from_layer
for e in root.find("edges"):
    src[(e.get("to-layer"), e.get("to-port"))] = e.get("from-layer")
by_name = {l.get("name"): l for l in layers.values()}
f = open(os.path.splitext(xml_path)[0] + ".bin", "rb")
mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
NP = {"f32": np.float32, "f16": np.float16, "u8": np.uint8, "i8": np.int8}


def const(l):
    d = l.find("data")
    shape = [int(x) for x in d.get("shape").split(",")] if d.get("shape") else []
    et = d.get("element_type")
    off, size = int(d.get("offset")), int(d.get("size"))
    raw = mm[off:off + size]
    return et, shape, raw


def resolve_const(layer_id):
    """follow Convert chains back to the Constant"""
    l = layers[layer_id]
    while l.get("type") == "Convert":
        l = layers[src[(l.get("id"), "0")]]
    assert l.get("type") == "Const", (l.get("name"), l.get("type"))
    return l


def f32_tensor(l):
    et, shape, raw = const(l)
    return np.frombuffer(raw, dtype=NP[et]).astype(np.float32).reshape(shape)


def gamma(norm_prefix):
    """norm_prefix e.g. '__module.model.layers.0.input_layernorm' -> the Multiply_1(gamma, normalized) constant input"""
    m = by_name[norm_prefix + "/aten::mul/Multiply_1"]
    for port in ("0", "1"):
        try:
            l = resolve_const(src[(m.get("id"), port)])
            return f32_tensor(l)
        except AssertionError:
            continue
    raise RuntimeError("no const input for " + norm_prefix)


def i4_group_major(raw, shape):
    """OV i4 const [N,G,128] (low nibble first) -> packed [G,N,128] bytes"""
    N, G, B = shape
    b = np.frombuffer(raw, dtype=np.uint8)
    nib = np.empty(b.size * 2, dtype=np.uint8)
    nib[0::2] = b & 15
    nib[1::2] = b >> 4
    nib = nib.reshape(N, G, B).transpose(1, 0, 2).reshape(-1)
    return ((nib[0::2]) | (nib[1::2] << 4)).astype(np.uint8), [G, N, B]


def proj(layer, mod, short):
    w = by_name[f"self.model.layers.{layer}.{mod}.weight"]
    s = by_name[f"self.model.layers.{layer}.{mod}.weight/scale"]
    et, shape, raw = const(w)
    assert et == "i4" and len(shape) == 3, (w.get("name"), et, shape)
    packed, gs = i4_group_major(raw, shape)
    set_, sshape, sraw = const(s)
    assert set_ == "f16" and sshape[2] == 1
    sc = np.frombuffer(sraw, dtype=np.float16).reshape(sshape).transpose(1, 0, 2)  # [N,G,1] -> [G,N,1]
    return [(f"{short}_weight", "int4", gs, packed.tobytes()), (f"{short}_scale", "float16", list(sc.shape), np.ascontiguousarray(sc).tobytes())]


def g32(name, arr, shape):
    return (name, "float32", shape, np.ascontiguousarray(arr, dtype=np.float32).tobytes())


manifest = {"layers": [], "model": os.path.basename(os.path.dirname(xml_path))}
nl = max(int(m.group(1)) for n in by_name for m in [re.match(r"self\.model\.layers\.(\d+)\.", n)] if m) + 1
for n in range(nl):
    pre = f"__module.model.layers.{n}"
    t = []
    t += proj(n, "self_attn.q_proj", "query") + [g32("query_norm_gamma", gamma(pre + ".self_attn.q_norm"), [1, 1, 1, 128])]
    t += proj(n, "self_attn.k_proj", "key") + [g32("key_norm_gamma", gamma(pre + ".self_attn.k_norm"), [1, 1, 1, 128])]
    t += proj(n, "self_attn.v_proj", "value") + proj(n, "self_attn.o_proj", "output_projection")
    t += proj(n, "mlp.gate_proj", "gate_projection") + proj(n, "mlp.up_proj", "up_projection") + proj(n, "mlp.down_proj", "down_projection")
    t += [g32("input_layernorm_gamma", gamma(pre + ".input_layernorm"), [1, 1, 2560]), g32("post_attention_layernorm_gamma", gamma(pre + ".post_attention_layernorm"), [1, 1, 2560])]
    off, meta = 0, []
    with open(os.path.join(out, f"l{n}.bin"), "wb") as fo:
        for name, dt, shape, data in t:
            meta.append({"name": name, "dtype": dt, "shape": shape, "offset": off, "bytes": len(data)})
            fo.write(data)
            off += len(data)
    manifest["layers"].append({"file": f"l{n}.bin", "tensors": meta, "bytes": off})
    print("layer", n, off // 2**20, "MiB", flush=True)

for fn, name in (("embed.bin", "self.model.embed_tokens.weight"), ("embed_zp.bin", "self.model.embed_tokens.weight/zero_point"), ("embed_s.bin", "self.model.embed_tokens.weight/scale")):
    et, shape, raw = const(by_name[name])
    open(os.path.join(out, fn), "wb").write(raw)
    manifest[fn] = {"dtype": et, "shape": shape}
m = by_name["__module.model.norm/aten::mul/Multiply_1"]
fn_arr = None
for port in ("0", "1"):
    try:
        fn_arr = f32_tensor(resolve_const(src[(m.get("id"), port)])); break
    except AssertionError:
        pass
open(os.path.join(out, "final_norm.bin"), "wb").write(fn_arr.astype(np.float32).tobytes())
manifest["final_norm.bin"] = {"dtype": "float32", "shape": [1, 1, 2560]}
json.dump(manifest, open(os.path.join(out, "manifest.json"), "w"), indent=1)
print("done", nl, "layers")

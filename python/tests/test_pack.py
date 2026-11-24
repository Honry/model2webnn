import json

import numpy as np

from m2w.frontends.hf_llm import pack
from m2w.frontends.hf_llm.arch import spec_from_config
from m2w.frontends.hf_llm.builders import decoder as D
from m2w.frontends.hf_llm.safetensors_io import TensorStore, write_safetensors
from m2w.quantize import dequantize_i8_symmetric_rowwise, dequantize_int4, dequantize_u8_rowwise, unpack_int4

from tiny_model import CONFIG, write_tiny_model


def test_safetensors_reader_handles_bf16_f16_f32_and_row_slices(tmp_path):
    a = np.random.default_rng(0).standard_normal((6, 4)).astype(np.float32)
    for dtype, tol in (("F32", 0), ("F16", 1e-3), ("BF16", 1e-2)):
        path = tmp_path / f"t_{dtype}.safetensors"
        write_safetensors(path, {"w": a}, dtype)
        store = TensorStore([path])
        assert store.shape("w") == (6, 4) and store.dtype("w") == dtype
        np.testing.assert_allclose(store.read("w"), a, atol=tol * 4 + 1e-12)
        np.testing.assert_array_equal(store.read("w", rows=(2, 5)), store.read("w")[2:5])


def read_tensor(shard_dir, manifest_file, name):
    entry = next(t for t in manifest_file["tensors"] if t["name"] == name)
    raw = (shard_dir / manifest_file["file"]).read_bytes()[entry["offset"]: entry["offset"] + entry["bytes"]]
    shape = tuple(entry["shape"])
    if entry["dtype"] == "int4":
        return unpack_int4(np.frombuffer(raw, np.uint8), shape)
    return np.frombuffer(raw, {"float16": np.float16, "uint8": np.uint8, "int8": np.int8}[entry["dtype"]]).reshape(shape)


def test_pack_model_writes_graph_named_shards_that_dequantize_to_the_original(tmp_path):
    tensors = write_tiny_model(tmp_path / "hf", dtype="F32")
    spec = spec_from_config(CONFIG)
    store = TensorStore.open(tmp_path / "hf")
    out = tmp_path / "pkg"
    manifest = pack.pack_model(spec, store, out, workers=2)
    assert json.loads((out / "manifest.json").read_text()) == manifest
    roles = [f["role"] for f in manifest["files"]]
    assert roles.count("layer") == 3 and roles.count("final_norm") == 1 and roles.count("embedding") == 2

    dims, opt = spec.decoder_dims(), D.GraphOptions()
    expected = {n for n, _, _ in D.attention_weight_specs(dims, opt) + D.feed_forward_weight_specs(dims, opt)}
    layer1 = next(f for f in manifest["files"] if f.get("layer") == 1)
    assert {t["name"] for t in layer1["tensors"]} == expected                      # exactly the graph input names of one block
    assert layer1["bytes"] == (out / layer1["file"]).stat().st_size

    # the merged qkv projection is q, k, v stacked along the output channels and dequantizes back to them
    q = read_tensor(out, layer1, "qkv_projection_weight")
    s = read_tensor(out, layer1, "qkv_projection_scale")
    original = np.concatenate([tensors[f"model.layers.1.self_attn.{n}_proj.weight"] for n in "qkv"], axis=0)
    rec = dequantize_int4(q, s)
    assert rec.shape == original.shape
    assert np.square(rec - original).sum() / np.square(original).sum() < 0.02
    np.testing.assert_allclose(read_tensor(out, layer1, "input_layernorm_gamma").astype(np.float32).reshape(-1), tensors["model.layers.1.input_layernorm.weight"], atol=1e-3)

    # embedding: padded vocabulary, pad rows dequantize to exactly zero
    emb = [f for f in manifest["files"] if f["role"] == "embedding"]
    byname = {t["name"]: (f, t) for f in emb for t in f["tensors"]}
    w = read_tensor(out, *[byname["lm_head_weight"][0]], "lm_head_weight")
    sc = read_tensor(out, byname["lm_head_scale"][0], "lm_head_scale")
    assert "lm_head_zero_point" not in byname          # the default format, int8 symmetric, has no zero point
    assert w.dtype == np.int8 and w.shape == (384, 256) and not dequantize_i8_symmetric_rowwise(w, sc)[300:].any()
    assert (sc.astype(np.float32) > 0).all()        # pad rows included: a zero scale can make the NPU lm_head dramatically slower
    table = tensors["model.embed_tokens.weight"]
    assert np.abs(dequantize_i8_symmetric_rowwise(w, sc)[:300] - table).max() < 0.02

    # the OpenVINO-style uint8 asymmetric table is still available
    out2 = tmp_path / "pkg_asym"
    manifest2 = pack.pack_model(spec, store, out2, D.GraphOptions(lm_head_format="uint8-asym"), workers=2)
    by2 = {t["name"]: (f, t) for f in manifest2["files"] if f["role"] == "embedding" for t in f["tensors"]}
    w2 = read_tensor(out2, by2["lm_head_weight"][0], "lm_head_weight")
    zp2 = read_tensor(out2, by2["lm_head_zero_point"][0], "lm_head_zero_point")
    sc2 = read_tensor(out2, by2["lm_head_scale"][0], "lm_head_scale")
    assert w2.dtype == np.uint8 and not dequantize_u8_rowwise(w2, zp2, sc2)[300:].any() and (sc2.astype(np.float32) > 0).all()
    assert np.abs(dequantize_u8_rowwise(w2, zp2, sc2)[:300] - table).max() < 0.02

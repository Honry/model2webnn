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


def test_shards_are_standard_safetensors_files_and_the_manifest_offsets_point_into_them(tmp_path):
    """Any safetensors reader opens a shard (int4 as U8 with the last axis halved); the manifest's absolute offsets give the same bytes as the header."""
    import struct

    from safetensors import safe_open

    write_tiny_model(tmp_path / "hf", dtype="F32")
    spec = spec_from_config(CONFIG)
    out = tmp_path / "pkg"
    manifest = pack.pack_model(spec, TensorStore.open(tmp_path / "hf"), out, workers=2)
    assert manifest["container"] == "safetensors" and not list(out.glob("*.bin"))
    layer = next(f for f in manifest["files"] if f["role"] == "layer" and f["layer"] == 1)
    assert layer["file"] == "layer_001.safetensors" and layer["bytes"] == (out / layer["file"]).stat().st_size
    raw = (out / layer["file"]).read_bytes()
    (header_size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8: 8 + header_size])
    assert header_size % 8 == 0
    meta = header.pop("__metadata__")
    assert meta["format"] == "m2w" and json.loads(meta["dtypes"])["qkv_projection_weight"] == "int4"
    with safe_open(str(out / layer["file"]), framework="np") as f:
        assert set(f.keys()) == {t["name"] for t in layer["tensors"]}
        for t in layer["tensors"]:
            info = header[t["name"]]
            assert t["offset"] == 8 + header_size + info["data_offsets"][0] and t["bytes"] == info["data_offsets"][1] - info["data_offsets"][0]
            stored = f.get_tensor(t["name"])
            assert stored.tobytes() == raw[t["offset"]: t["offset"] + t["bytes"]]
            if t["dtype"] == "int4":
                assert info["dtype"] == "U8" and list(stored.shape) == t["shape"][:-1] + [t["shape"][-1] // 2]
            else:
                assert list(stored.shape) == t["shape"]


def test_repack_converts_an_older_bin_package_without_changing_a_tensor(tmp_path):
    """`m2w repack`: *.bin shards (concatenated tensors) -> .safetensors; every tensor dequantizes to the same values as before."""
    import hashlib

    from m2w import compile as C

    write_tiny_model(tmp_path / "hf", dtype="BF16")
    out = C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(64,), chunks=(8,), staging_rows=(4,), workers=2, progress=lambda m: None)
    weights = out / "weights"
    new = json.loads((weights / "manifest.json").read_text())
    before = {(f["file"], t["name"]): read_tensor(weights, f, t["name"]).tobytes() for f in new["files"] for t in f["tensors"]}
    # make an "older" package: the same tensors as concatenated raw shards
    old_files = []
    for f in new["files"]:
        blob = b"".join(read_raw := [(weights / f["file"]).read_bytes()[t["offset"]: t["offset"] + t["bytes"]] for t in f["tensors"]])
        name = f["file"].replace(".safetensors", ".bin")
        (weights / name).write_bytes(blob)
        (weights / f["file"]).unlink()
        at, tensors = 0, []
        for t, raw in zip(f["tensors"], read_raw):
            tensors.append({**t, "offset": at})
            at += len(raw)
        old_files.append({**f, "file": name, "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(), "tensors": tensors})
    legacy = {k: v for k, v in new.items() if k != "container"}
    (weights / "manifest.json").write_text(json.dumps({**legacy, "files": old_files}))
    C.repack_package(out, progress=lambda m: None)
    repacked = json.loads((weights / "manifest.json").read_text())
    assert repacked["container"] == "safetensors" and not list(weights.glob("*.bin"))
    after = {(f["file"], t["name"]): read_tensor(weights, f, t["name"]).tobytes() for f in repacked["files"] for t in f["tensors"]}
    assert after == before
    index = json.loads((out / "m2w.json").read_text())
    assert {f["path"] for f in index["files"] if f["path"].startswith("weights/")} == {"weights/manifest.json", *(f"weights/{f['file']}" for f in repacked["files"])}
    assert C.repack_package(out, progress=lambda m: None) == out   # idempotent

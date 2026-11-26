"""Qwen3.5 image tower: the compiled vision graphs (HEAD / BLOCK x depth / MERGER), run by the numpy interpreter on the packed weights, against transformers' Qwen3_5VisionModel fed with the
SAME (dequantized) package weights: the image embeddings and the hidden state after every block, for every grid preset of a tiny model; the per-grid tables against transformers' own."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers", minversion="5.3")
sys.path.insert(0, str(Path(__file__).parent))
import tiny_qwen35 as T  # noqa: E402

from m2w.compile import compile_model  # noqa: E402
from m2w.frontends.hf_llm import arch, vision as V  # noqa: E402
from m2w.reference.qwen35_hf import Qwen35VisionHF  # noqa: E402
from m2w.reference.qwen35_vision import Qwen35VisionRunner  # noqa: E402

PATCH = T.VISION["patch_size"]


def build(tmp_path_factory, weights: str):
    root = tmp_path_factory.mktemp("q35v_" + weights)
    T.write_tiny_qwen35(root / "hf")
    out = compile_model(str(root / "hf"), root / "pkg", max_lengths=(64,), chunks=(16,), staging_rows=(8,), vision_weights=weights, vision_grids=T.GRIDS, progress=lambda m: None)
    return out, arch.ModelSpec.from_dict(json.loads((out / "model.json").read_text()))


@pytest.fixture(scope="module")
def package(tmp_path_factory):
    return build(tmp_path_factory, "int8")


def pixels(grid, seed=0):
    """a smooth-ish random image in [-1, 1] (what the preprocessing hands to patchify)"""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((3, grid[0] * PATCH // 4 + 1, grid[1] * PATCH // 4 + 1))
    up = np.kron(base, np.ones((1, 4, 4)))[:, :grid[0] * PATCH, :grid[1] * PATCH]
    return np.clip(up * 0.5 + rng.standard_normal(up.shape) * 0.05, -1, 1).astype(np.float32)


def test_spec_and_plan(package):
    out, spec = package
    assert spec.vision["depth"] == 2 and spec.vision["merge_size"] == 2 and spec.vision["image_token_id"] == 290 and spec.mrope == {"section": [1, 1, 2], "interleaved": True}
    plan = json.loads((out / "plan.json").read_text())["vision"]
    assert plan["weights"] == "int8" and [p["grid"] for p in plan["presets"]] == [list(g) for g in T.GRIDS] and [p["tokens"] for p in plan["presets"]] == [4, 6, 6]
    manifest = json.loads((out / "weights" / "manifest.json").read_text())
    assert manifest["quant"]["vision"]["weights"] == "int8"
    roles = [f["role"] for f in manifest["files"]]
    assert roles.count("vision_block") == 2 and roles.count("vision_grid") == 3 and {"vision_head", "vision_merger", "vision_pos_embed"} <= set(roles)
    blocks = next(f for f in manifest["files"] if f["role"] == "vision_block")
    qkv = next(t for t in blocks["tensors"] if t["name"] == "qkv_weight")
    assert qkv["dtype"] == "int8" and qkv["shape"] == [384, 128] and next(t for t in blocks["tensors"] if t["name"] == "qkv_scale")["shape"] == [384, 1]


def test_tables_equal_transformers(package):
    """the interpolated position embedding and the rotary cos / sin of a grid (merge-block order) against the model's own code"""
    out, spec = package
    model = Qwen35VisionHF(T.VISION, Qwen35VisionRunner(out, spec).reader.vision_hf_state_dict()).model
    table = model.pos_embed.weight.detach().numpy()
    for grid in [*T.GRIDS, (8, 6), (2, 2)]:
        thw = torch.tensor([[1, *grid]])
        with torch.no_grad():
            want_pos, want_rot = model.fast_pos_embed_interpolate(thw).numpy(), model.rot_pos_emb(thw)
        np.testing.assert_allclose(V.interpolated_pos_embed(table, grid[0], grid[1], 2), want_pos, atol=1e-5)
        emb = torch.cat((want_rot, want_rot), dim=-1)
        cos, sin = V.rotary_tables(T.VISION["hidden_size"] // T.VISION["num_heads"], grid[0], grid[1], 2)
        np.testing.assert_allclose(cos, emb.cos().numpy(), atol=1e-6)
        np.testing.assert_allclose(sin, emb.sin().numpy(), atol=1e-6)


@pytest.mark.parametrize("grid", T.GRIDS)
def test_image_embeddings_match_transformers(package, grid):
    out, spec = package
    runner = Qwen35VisionRunner(out, spec)
    reference = Qwen35VisionHF(T.VISION, runner.reader.vision_hf_state_dict())
    patches = V.patchify(pixels(grid), grid[0], grid[1], PATCH, 2, 2).astype(np.float16)   # the graph boundary is float16: both sides see the same numbers
    got = runner.encode(patches, grid).astype(np.float32)
    want, _ = reference.forward(patches.astype(np.float32), grid)
    assert got.shape == want.shape == (grid[0] * grid[1] // 4, T.VISION["out_hidden_size"])
    assert np.abs(got - want).max() < 0.02 * np.abs(want).max() + 1e-3, (np.abs(got - want).max(), np.abs(want).max())
    for i, (a, b) in enumerate(zip(runner.trace[1:], reference.block_outputs)):   # the hidden state after every block
        assert np.abs(a.astype(np.float32) - b).max() < 0.01 * np.abs(b).max() + 1e-3, (i, np.abs(a.astype(np.float32) - b).max(), np.abs(b).max())


@pytest.mark.parametrize("weights", ["float16", "int4"])
def test_other_weight_formats(tmp_path_factory, weights):
    """--vision-weights float16 / int4: the same graphs with the other weight inputs, equal to the reference on the dequantized weights"""
    out, spec = build(tmp_path_factory, weights)
    runner = Qwen35VisionRunner(out, spec)
    reference = Qwen35VisionHF(T.VISION, runner.reader.vision_hf_state_dict())
    grid = T.GRIDS[1]
    patches = V.patchify(pixels(grid, 3), grid[0], grid[1], PATCH, 2, 2).astype(np.float16)
    got, (want, _) = runner.encode(patches, grid).astype(np.float32), reference.forward(patches.astype(np.float32), grid)
    assert np.abs(got - want).max() < 0.02 * np.abs(want).max() + 1e-3


def test_float16_compute_type_stays_close(tmp_path_factory):
    """the float16 compute type (what the NPU does for float32 graphs anyway) against the float32 one: same graphs, only the interior type differs"""
    root = tmp_path_factory.mktemp("q35v_f16")
    T.write_tiny_qwen35(root / "hf")
    results = {}
    for ct in ("float32", "float16"):
        out = compile_model(str(root / "hf"), root / f"pkg_{ct}", max_lengths=(64,), chunks=(16,), staging_rows=(8,), vision_grids=T.GRIDS[:1], vision_compute_type=ct, progress=lambda m: None)
        spec = arch.ModelSpec.from_dict(json.loads((out / "model.json").read_text()))
        patches = V.patchify(pixels(T.GRIDS[0], 5), 4, 4, PATCH, 2, 2).astype(np.float16)
        results[ct] = Qwen35VisionRunner(out, spec).encode(patches, T.GRIDS[0]).astype(np.float32)
    assert np.abs(results["float32"] - results["float16"]).max() < 0.05 * np.abs(results["float32"]).max()


def test_patchify_equals_the_image_processor():
    """patchify (merge-block order, (channel, temporal, kh, kw) vectors) against transformers' Qwen2VLImageProcessor on an image whose size needs no resizing"""
    pytest.importorskip("PIL")
    from PIL import Image
    from transformers import Qwen2VLImageProcessor

    grid = (16, 16)   # 256 x 256 = min_pixels: no resize
    rng = np.random.default_rng(0)
    image = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8))
    processor = Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3, min_pixels=65536, max_pixels=16777216)
    out = processor(images=[image], return_tensors="np")
    assert out["image_grid_thw"].tolist() == [[1, 16, 16]]
    mine = V.patchify((np.asarray(image, np.float32).transpose(2, 0, 1) / 255 - 0.5) / 0.5, grid[0], grid[1], 16, 2, 2)
    np.testing.assert_allclose(mine, out["pixel_values"], atol=1e-5)


def test_choose_grid():
    g = V.DEFAULT_GRIDS   # (grid_h, grid_w): the closest aspect ratio first, then the closest pixel count
    assert V.choose_grid(g, 1000, 1000) == (32, 32) and V.choose_grid(g, 400, 400) == (24, 24) and V.choose_grid(g, 200, 200) == (16, 16)
    assert V.choose_grid(g, 1824, 2358) == (32, 24) and V.choose_grid(g, 1500, 1000) == (16, 24) and V.choose_grid(g, 1000, 1500) == (24, 16) and V.choose_grid(g, 1280, 960) == (24, 32)


def test_add_vision_to_a_text_only_package_and_rebuild_graphs(tmp_path_factory):
    """`m2w add-vision` adds the tower to a package compiled without it (text files untouched); `rebuild-graphs` rewrites the tower's graphs with the package's options"""
    from m2w.compile import add_vision, rebuild_graphs

    root = tmp_path_factory.mktemp("q35v_add")
    T.write_tiny_qwen35(root / "hf")
    out = compile_model(str(root / "hf"), root / "pkg", max_lengths=(64,), chunks=(16,), staging_rows=(8,), vision=False, progress=lambda m: None)
    assert "vision" not in json.loads((out / "plan.json").read_text()) and "vision" not in json.loads((out / "model.json").read_text())
    text_files = {p.name: p.read_bytes() for p in (out / "weights").glob("layer_*.safetensors")}
    add_vision(out, str(root / "hf"), vision_weights="int8", vision_grids=T.GRIDS[:2], progress=lambda m: None)
    assert {p.name: p.read_bytes() for p in (out / "weights").glob("layer_*.safetensors")} == text_files
    index = json.loads((out / "m2w.json").read_text())
    assert any(f["path"] == "weights/vision_head.safetensors" for f in index["files"]) and "clamp" in index["requires"]["webnn"]["ops"] and "int8" in index["requires"]["webnn"]["dataTypes"]
    rebuild_graphs(out, progress=lambda m: None)
    spec = arch.ModelSpec.from_dict(json.loads((out / "model.json").read_text()))
    plan = json.loads((out / "plan.json").read_text())["vision"]
    assert [p["grid"] for p in plan["presets"]] == [list(g) for g in T.GRIDS[:2]] and all((out / p).exists() for preset in plan["presets"] for p in preset["graphs"].values())
    runner, grid = Qwen35VisionRunner(out, spec), T.GRIDS[1]
    patches = V.patchify(pixels(grid, 7), grid[0], grid[1], PATCH, 2, 2).astype(np.float16)
    want, _ = Qwen35VisionHF(T.VISION, runner.reader.vision_hf_state_dict()).forward(patches.astype(np.float32), grid)
    assert np.abs(runner.encode(patches, grid).astype(np.float32) - want).max() < 0.02 * np.abs(want).max() + 1e-3
    with pytest.raises(ValueError, match="no preset"):
        runner.encode(patches, (6, 4))

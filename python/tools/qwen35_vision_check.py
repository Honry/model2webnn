"""The Qwen3.5 image tower of a package on a real image: the compiled graphs (numpy interpreter) against transformers' Qwen3_5VisionModel fed with (a) the dequantized package weights (graphs +
packing must agree to float16 noise) and (b) the original checkpoint weights (the cost of the int8 / int4 quantization), with the range of the activations (the float16 range is the NPU's problem).

    python tools/qwen35_vision_check.py <package> <hf checkpoint dir> <image> [--grid 24x16] [--out ../bench/data/qwen35_vision] [--export]

--export writes the patches, the grid and the hidden state after HEAD and after every block (float16) of the interpreter into --out (trace.json + trace.bin): what bench/raw/qwen35vision.html compares the device with.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2w.frontends.hf_llm import arch, vision as V  # noqa: E402
from m2w.frontends.hf_llm.safetensors_io import TensorStore  # noqa: E402
from m2w.reference import protocol  # noqa: E402
from m2w.reference.qwen35_hf import Qwen35VisionHF  # noqa: E402
from m2w.reference.qwen35_vision import Qwen35VisionRunner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("source")
    ap.add_argument("image")
    ap.add_argument("--grid", help="preset HxW (default: chosen from the image)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "bench" / "data" / "qwen35_vision"))
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--skip-original", action="store_true")
    ap.add_argument("--compute-type", default="", help="build the graphs here with this type (float16: the interior type the NPU uses; shows overflows) instead of reading the package's")
    args = ap.parse_args()
    from PIL import Image

    package = Path(args.package)
    spec = arch.ModelSpec.from_dict(json.loads((package / "model.json").read_text(encoding="utf-8")))
    runner = Qwen35VisionRunner(package, spec, args.compute_type)
    image = Image.open(args.image).convert("RGB")
    patches, grid = protocol.prepare_image(image, spec, runner.grids, V.parse_grid(args.grid) if args.grid else None)
    print(f"image {image.width}x{image.height} -> grid {V.grid_name(grid)} ({patches.shape[0]} patches, {protocol.image_token_count(spec, grid)} image tokens), weights {runner.plan['weights']}")
    patches16 = patches.astype(np.float16)   # the graph boundary: both sides get the same numbers
    t = time.time()
    got = runner.encode(patches16, grid).astype(np.float32)
    print(f"graphs (interpreter, {args.compute_type or runner.plan['compute_type']}): {time.time() - t:.1f} s, non-finite values in the image embeds: {int((~np.isfinite(got)).sum())}")
    vision_config = {"depth": spec.vision["depth"], "hidden_size": spec.vision["hidden"], "num_heads": spec.vision["heads"], "intermediate_size": spec.vision["ffn"], "patch_size": spec.vision["patch_size"],
                     "temporal_patch_size": spec.vision["temporal_patch_size"], "spatial_merge_size": spec.vision["merge_size"], "out_hidden_size": spec.vision["out_hidden"],
                     "num_position_embeddings": spec.vision["pos_grid"] ** 2, "hidden_act": "gelu_pytorch_tanh", "in_channels": spec.vision["channels"], "deepstack_visual_indexes": []}
    sources = [("dequantized package weights", runner.reader.vision_hf_state_dict())]
    if not args.skip_original:
        store = TensorStore.open(args.source)
        sources.append(("original checkpoint weights", {n: store.read(n) for n in store.names() if n.startswith("model.visual.")}))
    for label, tensors in sources:
        t = time.time()
        reference = Qwen35VisionHF(vision_config, tensors)
        want, _ = reference.forward(patches16.astype(np.float32), grid)
        print(f"\nreference: transformers, {label} ({time.time() - t:.1f} s)")
        rel = lambda a, b: float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-9))  # noqa: E731
        print(f"  image embeds: max abs diff {np.abs(got - want).max():.4f} (max |ref| {np.abs(want).max():.3f}, relative {rel(got, want):.4f}), rms ratio {np.sqrt(np.mean((got - want) ** 2)) / np.sqrt(np.mean(want ** 2)):.4f}")
        cos = (got * want).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(want, axis=1))
        print(f"  cosine per image token: min {cos.min():.5f} mean {cos.mean():.5f}")
        if label.startswith("dequantized"):
            print("  per block: max |hidden| (reference), max abs diff, relative, max |centered x|, rms of the centered x")
            for i, (a, b) in enumerate(zip(runner.trace[1:], reference.block_outputs)):
                a32 = a.astype(np.float32)
                centered = b - b.mean(axis=-1, keepdims=True)
                print(f"   block {i:2d}: {np.abs(b).max():8.2f}  {np.abs(a32 - b).max():8.4f}  {rel(a32, b):.5f}   {np.abs(centered).max():8.2f}  {np.sqrt((centered ** 2).mean()):6.2f}")
    if args.export:   # the reference of bench/raw/qwen35vision.html: the patches the device is fed with and the hidden state after HEAD and after every block (float16), in one blob + a json index
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        blob, entries = bytearray(), {}

        def add(name, array):
            raw = np.ascontiguousarray(array, np.float16).tobytes()
            entries[name] = {"shape": list(array.shape), "offset": len(blob), "bytes": len(raw)}
            blob.extend(raw)

        add("patches", patches16)
        for i, h in enumerate(runner.trace):
            add(f"hidden_{i}", h)
        add("image_embeds", got)
        (out / "trace.bin").write_bytes(bytes(blob))
        (out / "trace.json").write_text(json.dumps({"grid": list(grid), "depth": spec.vision["depth"], "hidden": spec.vision["hidden"], "out_hidden": spec.vision["out_hidden"], "weights": runner.plan["weights"],
                                                    "compute_type": args.compute_type or runner.plan["compute_type"], "image": Path(args.image).name, "entries": entries}), encoding="utf-8", newline="\n")
        print(f"\nwritten to {out}: {len(blob) / 2**20:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

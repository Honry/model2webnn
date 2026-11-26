"""Qwen3.5 with an image, end to end on the CPU: the compiled graphs (numpy interpreter: image tower + injection + MRoPE + prefill + decode) against transformers' Qwen3_5ForConditionalGeneration
(a) on the DEQUANTIZED package weights: the graphs + protocol must agree to float16 noise, (b) on the ORIGINAL checkpoint (bfloat16): what the quantization costs in the answer.

    python tools/qwen35_vl_check.py <package> <hf checkpoint dir> <image> [--question "..."] [--grid 24x16] [--steps 12] [--chunk 64] [--original]

Slow (the interpreter runs a 4B model): minutes for the prefill of a few hundred tokens. The ONE preprocessing (protocol.prepare_image) feeds all of them the same patches.
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2w.frontends.hf_llm import arch, chat_format, vision as V  # noqa: E402
from m2w.reference import protocol  # noqa: E402
from m2w.reference.qwen35_runner import Qwen35PackageReader, Qwen35Runner  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("package")
    ap.add_argument("source")
    ap.add_argument("image")
    ap.add_argument("--question", default="Describe this image in two sentences.")
    ap.add_argument("--grid")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=0)
    ap.add_argument("--original", action="store_true", help="also generate with the original bfloat16 checkpoint (transformers) and print its answer")
    ap.add_argument("--skip-reference", action="store_true", help="only run the compiled graphs")
    args = ap.parse_args()
    import torch
    from PIL import Image
    from tokenizers import Tokenizer
    from transformers import Qwen3_5ForConditionalGeneration

    package = Path(args.package)
    spec = arch.ModelSpec.from_dict(json.loads((package / "model.json").read_text(encoding="utf-8")))
    tokenizer = Tokenizer.from_file(str(package / "tokenizer" / "tokenizer.json"))
    fmt = json.loads((package / "chat_format.json").read_text(encoding="utf-8"))
    text = chat_format.fill_placeholders(chat_format.compose(fmt, [(spec.vision["image_prompt"] + args.question, None)]), datetime.datetime.now())
    tokens = tokenizer.encode(text, add_special_tokens=False).ids
    assert tokens.count(spec.vision["image_token_id"]) == 1, "the template must give exactly one <|image_pad|>"

    runner = Qwen35Runner.from_package(package, spec, chunk_sizes=[args.chunk], max_length=args.max_length)
    patches, grid = protocol.prepare_image(Image.open(args.image).convert("RGB"), spec, runner.vision.grids, V.parse_grid(args.grid) if args.grid else None)
    patches = patches.astype(np.float16)
    print(f"prompt {len(tokens)} tokens + image {V.grid_name(grid)} ({protocol.image_token_count(spec, grid)} tokens), chunk {args.chunk}")
    t = time.time()
    generated, logits = runner.generate(tokens, args.steps, eos=spec.eos_token_ids, images=[(patches, grid)])
    print(f"compiled graphs on the CPU: {len(generated)} tokens in {time.time() - t:.0f} s (prefill plan {runner.plan(len(runner.prompt))}, rope_delta {runner.rope_delta})")
    print("  answer:", repr(tokenizer.decode(generated, skip_special_tokens=False)))
    if args.skip_reference:
        return 0

    reader = Qwen35PackageReader(spec, package / "weights")
    config = json.loads((Path(args.source) / "config.json").read_text(encoding="utf-8"))
    from transformers import Qwen3_5Config

    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**config)).eval()
    state = {**reader.hf_state_dict(prefix="model.language_model."), **reader.vision_hf_state_dict()}
    print("loading the dequantized weights into transformers", model.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in state.items()}, strict=False).missing_keys)
    del state
    expanded = list(runner.prompt)
    pixel = torch.from_numpy(patches.astype(np.float32))
    thw = torch.tensor([[1, *grid]])

    def reference_logits(ids):
        t = torch.tensor([ids])
        with torch.no_grad():
            return model(input_ids=t, pixel_values=pixel, image_grid_thw=thw, mm_token_type_ids=(t == spec.vision["image_token_id"]).int()).logits[0, -1].float().numpy()

    cosines, agree = [], 0
    for k in range(len(generated)):
        want = reference_logits(expanded + generated[:k])
        got = logits[k]
        cosines.append(float(got @ want / (np.linalg.norm(got) * np.linalg.norm(want))))
        agree += int(np.argmax(want)) == generated[k]
        print(f"  step {k}: cosine {cosines[-1]:.5f}, max |diff| {np.abs(got - want).max():.3f}, reference picks {int(np.argmax(want))} (graphs {generated[k]})")
    print(f"vs transformers on the dequantized weights: logits cosine min {min(cosines):.5f} mean {np.mean(cosines):.5f}; the same token at {agree}/{len(generated)} steps")
    del model
    if args.original:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(args.source, dtype=torch.bfloat16).eval()
        t = torch.tensor([expanded])
        with torch.no_grad():
            out = model.generate(input_ids=t, pixel_values=pixel.to(torch.bfloat16), image_grid_thw=thw, mm_token_type_ids=(t == spec.vision["image_token_id"]).int(), max_new_tokens=args.steps, do_sample=False)
        print("original checkpoint (bfloat16) answer:", repr(tokenizer.decode(out[0, t.shape[1]:].tolist(), skip_special_tokens=False)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run the numpy reference of Gemma 4 on a prompt:  python tools/gemma4_ref.py <model dir> "What is the capital of France?" [--steps 8]"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m2w.frontends.hf_llm.safetensors_io import TensorStore  # noqa: E402
from m2w.reference.gemma4_ref import PREFIX, Gemma4Reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompt")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()
    from jinja2 import Environment
    from tokenizers import Tokenizer

    model_dir = Path(args.model)
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    env = Environment()
    env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
    template = env.from_string((model_dir / "chat_template.jinja").read_text(encoding="utf-8"))
    text = template.render(messages=[{"role": "user", "content": args.prompt}], add_generation_prompt=True, enable_thinking=False, bos_token="<bos>", tools=None)
    print(repr(text))
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    print("prompt ids", ids)
    store = TensorStore.open(model_dir)
    ref = Gemma4Reference(json.loads((model_dir / "config.json").read_text(encoding="utf-8"))["text_config"], lambda name, rows=None: store.read(PREFIX + name, rows=rows), verbose=True)
    out = []
    for _ in range(args.steps):
        logits = ref.logits(ids + out)[0]
        token = int(np.argmax(logits))
        out.append(token)
        print("token", token, repr(tokenizer.decode(out)), flush=True)
        if token in (1, 106, 50):
            break


if __name__ == "__main__":
    main()

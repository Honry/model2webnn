#!/usr/bin/env python
"""Tokenize a Qwen3 chat prompt to exactly N tokens -> qwen3_prompt_<N>.json (list of ids).

  python prep_prompt.py 192
  python prep_prompt.py --decode 1 2 3        # ids -> text
"""
import json
import os
import sys

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
QDIR = os.environ.get("BENCH_QWEN_DIR") or sys.exit("set BENCH_QWEN_DIR to the folder that holds the Qwen3 tokenizer.json")
tok = Tokenizer.from_file(os.path.join(QDIR, "tokenizer.json"))
BASE = "Explain, in plain language, how a neural processing unit differs from a GPU and why running a large language model on it needs static tensor shapes."
SYS = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n"


def build(words, pad=0):
    return SYS + " ".join(words) + " now" * pad + TAIL


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--decode":
        print(tok.decode([int(x) for x in sys.argv[2:]], skip_special_tokens=False))
        return
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 192
    words = BASE.split()
    k = 1
    while len(tok.encode(build(words * k)).ids) < n:
        k += 1
    words = words * k
    while len(tok.encode(build(words)).ids) > n:  # drop leading words until we are at or under n
        words.pop(0)
    pad = 0
    while len(tok.encode(build(words, pad)).ids) < n:  # top up with single-token filler
        pad += 1
    ids = tok.encode(build(words, pad)).ids
    assert len(ids) == n, len(ids)
    json.dump(ids, open(os.path.join(HERE, f"qwen3_prompt_{n}.json"), "w"))
    print(n, "tokens ->", f"qwen3_prompt_{n}.json")


if __name__ == "__main__":
    main()

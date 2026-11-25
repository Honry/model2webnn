#!/usr/bin/env python
"""Generate the synthetic block models an experiment needs (skips ones that already exist).

  python prep.py chain --layers 36 --splits 1,3,12 --q 1 --t 512 --w dq_const
  python prep.py blocks --ranges 0:1,0:12 --q 1,16,64 --t 512 --w dq_const,matmulnbits
  python prep.py m0
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "models")


def gen(a, b, q, t, w):
    name = f"blk_{w}_L{a}-{b}_q{q}_t{t}"
    if os.path.exists(os.path.join(OUT, name + ".json")):
        print("have", name)
        return
    print("gen ", name, flush=True)
    subprocess.check_call([sys.executable, os.path.join(HERE, "gen_blocks.py"), "--layers", f"{a}:{b}", "--query", str(q), "--kv", str(t), "--weights", w, "--out", OUT])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["chain", "blocks", "m0"])
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--splits", default="1")
    ap.add_argument("--ranges", default="0:1")
    ap.add_argument("--q", default="1")
    ap.add_argument("--t", default="512")
    ap.add_argument("--w", default="dq_const")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.what == "m0":
        if not os.path.exists(os.path.join(OUT, "tiny.json")):
            subprocess.check_call([sys.executable, os.path.join(HERE, "gen_blocks.py"), "--kind", "tiny", "--out", OUT])
        for w in ("dq_const", "matmulnbits", "dq_input"):
            gen(0, 1, 1, 512, w)
    elif a.what == "chain":
        for q in a.q.split(","):
            for t in a.t.split(","):
                for w in a.w.split(","):
                    for s in map(int, a.splits.split(",")):
                        per = a.layers // s
                        for i in range(s):
                            gen(i * per, (i + 1) * per, q, t, w)
    else:
        for q in a.q.split(","):
            for t in a.t.split(","):
                for w in a.w.split(","):
                    for r in a.ranges.split(","):
                        x, y = map(int, r.split(":"))
                        gen(x, y, q, t, w)


if __name__ == "__main__":
    main()

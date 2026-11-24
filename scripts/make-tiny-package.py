"""A tiny random hybrid-layer model compiled into a package (tests/golden/tiny-package, committed) with a verification record: the fast way to test the browser runtime
(scheduler, bindings, masks) on a WebNN device before loading a real model.   python scripts/make-tiny-package.py [max_length] [chunk] [staging_rows]"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "python"
sys.path[:0] = [str(root), str(root / "tests")]
import numpy as np  # noqa: E402

from m2w.compile import compile_model  # noqa: E402
from m2w.verify import verify_package  # noqa: E402
from tiny_model import write_tiny_model  # noqa: E402

max_length, chunk, staging = (int(a) for a in (sys.argv[1:4] + ["128", "16", "8"][len(sys.argv[1:4]):]))
out = root / "_out"
golden = root.parent / "tests" / "golden"
write_tiny_model(out / "tiny-hf", dtype="BF16")
pkg = compile_model(str(out / "tiny-hf"), golden / "tiny-package", max_lengths=(max_length,), chunks=(chunk,), staging_rows=staging, workers=2, progress=lambda m: None)
ids = [int(t) for t in np.random.default_rng(0).integers(3, 300, 45)]
sys.exit(verify_package(pkg, str(out / "tiny-hf"), prompt="", steps=12, token_ids=ids))

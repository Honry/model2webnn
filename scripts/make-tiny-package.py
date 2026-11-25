"""A tiny random hybrid-layer model compiled into a package (tests/golden/tiny-package, committed) with a verification record: the fast way to test the browser runtime
(scheduler, bindings, masks, every KV protocol / compute type / chunk plan) on a WebNN device before loading a real model.   python scripts/make-tiny-package.py [max_length] [chunk] [staging_rows]
Also writes tests/golden/tiny-gemma4-package, the same for a tiny Gemma 4 style model (per-layer head dims, KV sharing, per-layer embeddings),
and tests/golden/tiny-qwen35-package for a tiny Qwen3.5 (Gated DeltaNet layers + gated full attention; needs torch + transformers)."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "python"
sys.path[:0] = [str(root), str(root / "tests")]
import numpy as np  # noqa: E402

from m2w.compile import compile_model  # noqa: E402
from m2w.verify import verify_package  # noqa: E402
from tiny_gemma4 import write_tiny_gemma4  # noqa: E402
from tiny_model import write_tiny_model  # noqa: E402

max_length, chunk, staging = (int(a) for a in (sys.argv[1:4] + ["128", "16", "8"][len(sys.argv[1:4]):]))
small_chunk = max(chunk // 2, 1)
out = root / "_out"
golden = root.parent / "tests" / "golden"
write_tiny_model(out / "tiny-hf", dtype="BF16")
pkg = compile_model(str(out / "tiny-hf"), golden / "tiny-package", max_lengths=(max_length,), chunks=(chunk, small_chunk),
                    kv_styles=("staged", "onehot", "tensorScatter", "scatterND"), staging_rows=(staging,), compute_types=("float32", "float16"), blocks_per_graph=(1, 2), prefill_blocks_per_graph=(1, 2), workers=2, progress=lambda m: None)
ids = [int(t) for t in np.random.default_rng(0).integers(3, 300, 45)]
status = verify_package(pkg, str(out / "tiny-hf"), prompt="", steps=12, token_ids=ids, variants=["all"], chunks=[chunk, small_chunk])

write_tiny_gemma4(out / "tiny-gemma4-hf", dtype="BF16")
gemma = compile_model(str(out / "tiny-gemma4-hf"), golden / "tiny-gemma4-package", max_lengths=(max_length,), chunks=(chunk, small_chunk), staging_rows=(staging,),
                      compute_types=("float32", "float16"), workers=2, progress=lambda m: None)
status |= verify_package(gemma, str(out / "tiny-gemma4-hf"), prompt="", steps=12, token_ids=ids, variants=["all"], chunks=[chunk, small_chunk])

try:   # a tiny Qwen3.5 (Gated DeltaNet hybrid) needs torch + transformers >= 5.3 to create its random weights
    from tiny_qwen35 import write_tiny_qwen35
except ImportError:
    print("skipping tests/golden/tiny-qwen35-package: torch / transformers are not installed")
else:
    write_tiny_qwen35(out / "tiny-qwen35-hf")
    qwen = compile_model(str(out / "tiny-qwen35-hf"), golden / "tiny-qwen35-package", max_lengths=(max_length,), chunks=(chunk, 2 * chunk), staging_rows=(staging,),
                         compute_types=("float32", "float16"), workers=2, progress=lambda m: None)
    status |= verify_package(qwen, str(out / "tiny-qwen35-hf"), prompt="", steps=12, token_ids=ids, variants=["all"], chunks=[chunk, 2 * chunk])
sys.exit(status)

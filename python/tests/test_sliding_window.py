"""Sliding window KV caches (sliding_cache = "window": the sliding layers keep a ring of rows instead of max_length - 1):
  * the host protocol (slots, masks, selectors) simulated on position ids: every query sees exactly the positions of its window, whatever the chunk / flush / wrap-around pattern;
  * the compiled graphs of the tiny hybrid model and the tiny Gemma 4 run by the numpy interpreter: window == full logits and tokens over prompts and generations that wrap the ring;
  * the compiler: plan.json, graph tables, the CLI option, refused combinations."""
import json

import numpy as np
import pytest

from m2w import compile as C
from m2w.frontends.hf_llm import arch, pack, pack_gemma4
from m2w.frontends.hf_llm.arch import spec_from_config
from m2w.frontends.hf_llm.builders import decoder as D
from m2w.frontends.hf_llm.safetensors_io import TensorStore
from m2w.reference import protocol
from m2w.reference.gemma4_runner import Gemma4Runner
from m2w.reference.graph_runner import PackageRunner
from m2w.verify import verify_package

from tiny_gemma4 import CONFIG as GEMMA4_CONFIG, write_tiny_gemma4
from tiny_model import CONFIG, write_tiny_model

W = 16   # the window of both tiny models


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------------------------------------------------
# the protocol on position ids
# ---------------------------------------------------------------------------------------------------------------------
def test_slot_positions_and_ring_rows():
    assert slot_list(4, 0) == [-1] * 4 and slot_list(4, 3) == [0, 1, 2, -1] and slot_list(4, 4) == [0, 1, 2, 3] and slot_list(4, 6) == [4, 5, 2, 3] and slot_list(4, 9) == [8, 5, 6, 7]
    assert D.sliding_ring_rows(512, 256, 64, True) == 768 and D.sliding_ring_rows(2048, 256, 128, False) == 2048 and D.sliding_ring_rows(512, 64, 64, False) == 512
    assert D.sliding_ring_rows(16, 8, 4, False, multiple=8) == 16 and D.sliding_ring_rows(16, 8, 4, True, multiple=8) == 24 and D.sliding_ring_rows(4, 64, 8, False, multiple=8) == 64
    assert protocol.cache_rows("sliding_attention", 127, 64) == 64 and protocol.cache_rows("full_attention", 127, 64) == 127 and protocol.cache_rows("sliding_attention", 63, 64) == 63
    assert protocol.cache_rows("sliding_attention", 127, 0) == 127


def slot_list(rows, written):
    return protocol.slot_positions(rows, written).tolist()


class Ring:
    """What the cache of a sliding layer holds: the position id of every slot (-1 = nothing), updated with the SAME selectors the graphs get."""

    def __init__(self, rows):
        self.pos = np.full(rows, -1.0)

    def update(self, keep, selector, new_positions):   # past * keep + selector @ new
        self.pos = self.pos * keep.reshape(-1) + selector.reshape(len(self.pos), -1) @ np.asarray(new_positions, np.float64)


def visible_positions(mask_row, held, rows, extra_positions):
    """The positions a mask row selects: the cache columns (slots holding `held` positions) and the extra columns (chunk / staging rows) after them."""
    seen = {int(held[j]) for j in range(rows) if mask_row[j] == 0 and held[j] >= 0}
    seen |= {int(p) for c, p in enumerate(extra_positions) if p >= 0 and mask_row[rows + c] == 0}
    return seen


@pytest.mark.parametrize("chunk,staging", [(8, 4), (8, 8), (16, 4), (16, 16), (16, 8)])
@pytest.mark.parametrize("shared_reader", [False, True])
def test_every_query_sees_exactly_its_window_through_prefill_chunks_flushes_and_wrap_arounds(chunk, staging, shared_reader):
    """Simulates a whole generation on position ids (the ring holds ids, written with the selectors of the protocol): a query at q sees exactly the positions q - W + 1 .. q."""
    spec = spec_from_config(CONFIG)
    P = 127
    ring_rows = D.sliding_ring_rows(W, chunk, staging, shared_reader, multiple=1)   # the smallest ring that is allowed
    for prompt in (1, W - 1, W, W + 1, 2 * W + 3, 40):
        ring, full = Ring(ring_rows), Ring(P)
        start = 0
        for index, C_ in enumerate(protocol.plan_prefill([chunk], prompt)):
            valid = min(C_, prompt - start)
            pad = C_ - valid
            own = protocol.prefill_masks(spec, P, C_, start, valid, ring_rows)["sliding_attention"][0, 0]
            held = protocol.slot_positions(ring_rows, start)
            chunk_positions = [-1] * pad + list(range(start, start + valid))
            incache = protocol.incache_prefill_masks(spec, P, C_, start, valid, ring_rows)["sliding_attention"][0, 0]
            keep, selector = protocol.prefill_selectors(ring_rows, C_, start, valid, wrap=True)
            ring.update(keep, selector, chunk_positions)
            for r in range(pad, C_):
                q = start + r - pad
                window = set(range(max(0, q - W + 1), q + 1))
                assert visible_positions(own[r], held, ring_rows, chunk_positions) == window, (prompt, start, r)
                if shared_reader:   # reads the ring AFTER the update of the chunk
                    assert {int(p) for j, p in enumerate(ring.pos) if incache[r, j] == 0} == window, ("shared reader", prompt, start, r)
            start += valid
        # decode: tokens at positions prompt .. prompt + 70 (the ring wraps several times), the staging buffer is flushed every `staging` steps
        base, filled, staged = prompt, 0, []
        for pos in range(prompt, prompt + 70):
            if filled == staging:
                keep, selector = protocol.flush_selectors(ring_rows, staging, base, wrap=True)
                ring.update(keep, selector, staged)
                base, filled, staged = base + staging, 0, []
            mask = protocol.decode_masks(spec, P, staging, pos, base, filled, ring_rows)["sliding_attention"][0, 0, 0]
            staged.append(pos)
            held = protocol.slot_positions(ring_rows, base)
            assert np.array_equal(held, ring.pos), (prompt, pos)   # the slots hold what the host believes they hold
            seen = visible_positions(mask, held, ring_rows, staged + [-1] * (staging - len(staged)))
            assert seen == set(range(max(0, pos - W + 1), pos + 1)), (prompt, pos)
            filled += 1


def test_a_full_layer_and_a_ring_that_is_not_smaller_than_the_cache_are_unchanged():
    spec = spec_from_config(CONFIG)
    P = 63
    for ring in (0, 63, 64, 100):   # no ring, or one that holds the whole cache
        for start, valid in ((0, 8), (16, 5), (40, 8)):
            a, b = protocol.prefill_masks(spec, P, 8, start, valid), protocol.prefill_masks(spec, P, 8, start, valid, ring)
            assert all(np.array_equal(a[t], b[t]) for t in a)
            a, b = protocol.incache_prefill_masks(spec, P, 8, start, valid), protocol.incache_prefill_masks(spec, P, 8, start, valid, ring)
            assert all(np.array_equal(a[t], b[t]) for t in a)
        for pos, base, filled in ((19, 16, 3), (50, 48, 2)):
            a, b = protocol.decode_masks(spec, P, 4, pos, base, filled), protocol.decode_masks(spec, P, 4, pos, base, filled, ring)
            assert all(np.array_equal(a[t], b[t]) for t in a)
    full = protocol.prefill_masks(spec, P, 8, 16, 5, 24)["full_attention"]
    assert full.shape == (1, 1, 8, P + 8)   # the full-attention layers keep all rows
    assert protocol.prefill_masks(spec, P, 8, 16, 5, 24)["sliding_attention"].shape == (1, 1, 8, 24 + 8)


# ---------------------------------------------------------------------------------------------------------------------
# graphs + weights: window == full
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny_window")
    write_tiny_model(root / "hf", dtype="F32")
    spec = spec_from_config(CONFIG)
    pack.pack_model(spec, TensorStore.open(root / "hf"), root / "pkg", workers=2)
    return spec, root / "pkg"


@pytest.fixture(scope="module")
def tiny_gemma4(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny_gemma4_window")
    write_tiny_gemma4(root / "hf", dtype="F32")
    spec = arch.spec_from_config(GEMMA4_CONFIG)
    pack_gemma4.pack_model(spec, TensorStore.open(root / "hf"), root / "pkg" / "weights", workers=2)
    return spec, root / "pkg" / "weights"


def generic(tiny, window, ring=0, *, chunks=(8,), staging=4, max_length=128, group=1):
    spec, pkg = tiny
    opt = D.GraphOptions(kv_style="staged", staging_rows=staging, **({"sliding_cache": "window", "sliding_ring": ring} if window else {}))
    return PackageRunner(spec, pkg, max_length=max_length, chunk_sizes=chunks, opt=opt, decode_group=group)


def gemma(tiny_gemma4, window, ring=0, *, chunks=(8,), staging=4, max_length=128, group=1):
    spec, weights = tiny_gemma4
    opt = D.GraphOptions(kv_style="staged", staging_rows=staging, **({"sliding_cache": "window", "sliding_ring": ring} if window else {}))
    return Gemma4Runner(spec, weights, max_length=max_length, chunk_sizes=chunks, opt=opt, blocks_per_graph=group)


def same_run(full, window, tokens, steps):
    """Greedy generation with both runners: the same logits (cosine ~ 1) and tokens at every step; returns the number of steps compared."""
    lf, lw = full.prefill(tokens), window.prefill(tokens)
    for step in range(steps + 1):
        assert cosine(lf, lw) > 0.99999, (len(tokens), step, cosine(lf, lw))
        assert np.abs(lf - lw).max() < 0.01 * max(1.0, np.abs(lf).max()), (len(tokens), step)
        token = int(np.argmax(lf))
        assert token == int(np.argmax(lw)), (len(tokens), step)
        if step == steps:
            break
        lf, lw = full.step(token), window.step(token)
    return steps


def prompt_of(n, seed):
    return [int(t) for t in np.random.default_rng(seed).integers(3, 300, n)]


# prompt lengths around the window: shorter than W, W - 1, W, W + 1, several windows; the generation wraps the ring (16 / 24 rows) many times
@pytest.mark.parametrize("n", [5, W - 1, W, W + 1, 2 * W + 5, 45])
def test_generic_window_cache_equals_the_full_cache(tiny, n):
    assert same_run(generic(tiny, False), generic(tiny, True, 16), prompt_of(n, n), 40) == 40


@pytest.mark.parametrize("chunk,staging,ring", [(8, 4, 16), (16, 8, 16), ((8, 16), 4, 24), (16, 16, 24), (8, 8, 32)])
def test_generic_chunk_and_staging_sizes_with_the_smallest_rings(tiny, chunk, staging, ring):
    chunks = chunk if isinstance(chunk, tuple) else (chunk,)
    tokens = prompt_of(37, 3)
    same_run(generic(tiny, False, chunks=chunks, staging=staging), generic(tiny, True, ring, chunks=chunks, staging=staging), tokens, 50)


def test_generic_window_cache_with_decode_block_groups_and_the_tiny_cache_shapes(tiny):
    full, window = generic(tiny, False, group=2), generic(tiny, True, 16, group=2)
    same_run(full, window, prompt_of(30, 5), 30)
    types = tiny[0].layer_types
    assert [window.k[i].shape[2] for i in range(3)] == [16 if t == "sliding_attention" else 127 for t in types]
    assert [window.v[i].shape[2] for i in range(3)] == [window.k[i].shape[2] for i in range(3)]
    assert generic(tiny, True, 200).k[0].shape[2] == 127   # a ring that is not smaller than the cache is the full cache


@pytest.mark.parametrize("n", [5, W - 1, W, W + 1, 2 * W + 5, 45])
def test_gemma4_window_cache_equals_the_full_cache(tiny_gemma4, n):
    assert same_run(gemma(tiny_gemma4, False), gemma(tiny_gemma4, True, 24), prompt_of(n, n + 100), 40) == 40


@pytest.mark.parametrize("chunk,staging,ring", [(8, 4, 24), (16, 8, 32), ((8, 16), 4, 32), (16, 16, 32)])
def test_gemma4_chunk_and_staging_sizes_with_the_smallest_rings(tiny_gemma4, chunk, staging, ring):
    chunks = chunk if isinstance(chunk, tuple) else (chunk,)
    assert ring == D.sliding_ring_rows(W, max(chunks), staging, True, multiple=8) or ring > D.sliding_ring_rows(W, max(chunks), staging, True, multiple=8)
    same_run(gemma(tiny_gemma4, False, chunks=chunks, staging=staging), gemma(tiny_gemma4, True, ring, chunks=chunks, staging=staging), prompt_of(37, 4), 50)


def test_gemma4_window_cache_with_block_groups_and_shapes(tiny_gemma4):
    full, window = gemma(tiny_gemma4, False, group=3), gemma(tiny_gemma4, True, 24, group=3)
    same_run(full, window, prompt_of(30, 6), 30)
    spec = tiny_gemma4[0]
    assert {i: window.k[i].shape[2] for i in window.k} == {i: 24 if spec.layer_types[i] == "sliding_attention" else 127 for i in window.k}
    assert sorted(window.k) == [i for i in range(spec.num_layers) if spec.kv_producer[i] < 0]   # the shared layers have no cache


def test_a_ring_of_one_row_less_loses_a_position_of_the_first_row_of_a_chunk_for_a_shared_layer():
    """The minimum for a layer that reads the cache after the update of a chunk is W - 1 + chunk rows: the first row of a full chunk needs W - 1 older positions that the chunk rows have not
    overwritten (in the model only the LAST row of the prompt reaches the logits, so a smaller ring would not show in them, but the masks would no longer be what the specification says)."""
    spec = spec_from_config(CONFIG)
    P, chunk, start = 127, 8, 32
    for ring, ok in ((W - 1 + chunk, True), (W - 2 + chunk, False)):
        ids = Ring(ring)
        for first in range(0, start + chunk, chunk):   # the prompt so far (full chunks), then the chunk at `start`
            keep, selector = protocol.prefill_selectors(ring, chunk, first, chunk, wrap=True)
            ids.update(keep, selector, list(range(first, first + chunk)))
        mask = protocol.incache_prefill_masks(spec, P, chunk, start, chunk, ring)["sliding_attention"][0, 0]
        seen = {int(p) for j, p in enumerate(ids.pos) if mask[0, j] == 0}
        assert (seen == set(range(start - W + 1, start + 1))) == ok, (ring, sorted(seen))


# ---------------------------------------------------------------------------------------------------------------------
# the compiler
# ---------------------------------------------------------------------------------------------------------------------
def test_compile_window_package_of_the_generic_model(tmp_path, capsys):
    write_tiny_model(tmp_path / "hf", dtype="BF16")
    out = C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(64, 128), chunks=(8, 16), staging_rows=(4, 8), compute_types=("float32", "float16"), blocks_per_graph=(1, 2),
                          sliding_cache="window", workers=2, progress=lambda m: None)
    plan = json.loads((out / "plan.json").read_text())
    assert plan["profile"]["sliding_cache"] == "window" and plan["profile"]["sliding_ring"] == 64   # roundup64(max(W - 1, 16, 8))
    assert "sliding_window_cache" in plan["kv_protocols"]
    table = plan["graphs"]["staged-r4-float32"]
    decode = table["decode"]["128"]
    assert set(decode["blocks"]) == {"sliding_attention", "full_attention"} and set(decode["flush"]) == {"sliding_attention", "full_attention"} and "block" not in decode
    assert decode["blocks"]["sliding_attention"] != decode["blocks"]["full_attention"]
    assert decode["flush"]["sliding_attention"] != decode["flush"]["full_attention"]
    assert [(e["first_block"], e["blocks"], e["key"]) for e in decode["block_groups"]["2"]] == [(0, 2, "f-s")]
    # the 64 row ring is not smaller than the cache of max length 64 (63 rows): both types share one graph file there
    short = table["decode"]["64"]
    assert short["blocks"]["sliding_attention"] == short["blocks"]["full_attention"] and short["flush"]["sliding_attention"] == short["flush"]["full_attention"]
    assert set(table["prefill"]["128"]["16"]["blocks"]) == {"sliding_attention", "full_attention"}
    ids = [int(t) for t in np.random.default_rng(1).integers(3, 300, 45)]
    for max_length in (64, 128):
        assert verify_package(out, str(tmp_path / "hf"), prompt="", steps=30 if max_length == 128 else 10, token_ids=ids[:19] if max_length == 64 else ids, max_length=max_length,
                              variants=["staged-r4-float32", "staged-r8-float32"], chunks=[8, 16], decode_group=2 if max_length == 128 else 1, skip_float_reference=True) == 0
    assert capsys.readouterr().out.count("RESULT: pipeline OK") == 2


def test_compile_window_package_of_gemma4(tmp_path, capsys):
    write_tiny_gemma4(tmp_path / "hf", dtype="BF16")
    out = C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(128,), chunks=(8, 16), staging_rows=(8,), blocks_per_graph=(1, 3), sliding_cache="window", workers=2,
                          progress=lambda m: None)
    plan = json.loads((out / "plan.json").read_text())
    assert plan["profile"]["sliding_cache"] == "window" and plan["profile"]["sliding_ring"] == 64   # roundup64(W - 1 + 16)
    assert set(plan["graphs"]["staged-r8-float32"]["decode"]["128"]["flush"]) == {"sliding_attention", "full_attention"}
    ids = [int(t) for t in np.random.default_rng(2).integers(3, 300, 45)]
    assert verify_package(out, str(tmp_path / "hf"), prompt="", steps=30, token_ids=ids, variants=["all"], chunks=[8, 16], decode_group=3, skip_float_reference=True) == 0
    assert capsys.readouterr().out.count("RESULT: pipeline OK") == 1


def test_full_stays_the_default_and_writes_nothing_new_into_the_plan(tmp_path):
    write_tiny_model(tmp_path / "hf", dtype="BF16")
    out = C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(64,), chunks=(8,), staging_rows=(4,), workers=2, progress=lambda m: None)
    plan = json.loads((out / "plan.json").read_text())
    assert "sliding_cache" not in plan["profile"] and "sliding_ring" not in plan["profile"] and "block" in plan["graphs"]["staged-r4-float32"]["decode"]["64"]
    # rebuild the graphs of this package as a window package (weights untouched), and back
    C.rebuild_graphs(out, sliding_cache="window", progress=lambda m: None)
    assert json.loads((out / "plan.json").read_text())["profile"]["sliding_cache"] == "window"
    C.rebuild_graphs(out, sliding_cache="full", progress=lambda m: None)
    assert "sliding_cache" not in json.loads((out / "plan.json").read_text())["profile"]


def test_refused_combinations(tmp_path):
    write_tiny_model(tmp_path / "hf", dtype="BF16")
    with pytest.raises(ValueError, match="staged KV protocol only"):
        C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(64,), chunks=(8,), kv_styles=("staged", "onehot"), sliding_cache="window", workers=2, progress=lambda m: None)
    with pytest.raises(ValueError, match="prefill block groups"):
        C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(64,), chunks=(8,), prefill_blocks_per_graph=(1, 2), sliding_cache="window", workers=2, progress=lambda m: None)
    with pytest.raises(ValueError, match="unknown sliding cache"):
        C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(64,), chunks=(8,), sliding_cache="ring", workers=2, progress=lambda m: None)
    spec = spec_from_config(CONFIG)
    spec.layer_types, spec.sliding_window = ["full_attention"] * 3, None
    with pytest.raises(ValueError, match="no sliding-attention layers"):
        C.resolve_sliding_cache(spec, D.GraphOptions(sliding_cache="window"), [{"kv_style": "staged", "staging_rows": 4}], (8,))


@pytest.mark.parametrize("name,write,groups", [("tiny-window-package", write_tiny_model, (1, 2)), ("tiny-gemma4-window-package", write_tiny_gemma4, (1, 2, 3))])
def test_committed_window_packages_are_what_the_compiler_writes_today(tmp_path, name, write, groups):
    """tests/golden/tiny-window-package / tiny-gemma4-window-package are fixtures of the TS runtime tests: regenerate with `python scripts/make-tiny-package.py --window-only`."""
    from pathlib import Path

    committed = Path(__file__).resolve().parents[2] / "tests" / "golden" / name
    write(tmp_path / "hf", dtype="BF16")
    out = C.compile_model(str(tmp_path / "hf"), tmp_path / "pkg", max_lengths=(64, 128), chunks=(16, 8), staging_rows=(8,), compute_types=("float32", "float16"), blocks_per_graph=groups,
                          sliding_cache="window", workers=2, progress=lambda m: None)
    fresh = {f["path"]: f["sha256"] for f in json.loads((out / "m2w.json").read_text())["files"]}
    old = {f["path"]: f["sha256"] for f in json.loads((committed / "m2w.json").read_text())["files"]}
    assert fresh == old, f"tests/golden/{name} is stale: run `python scripts/make-tiny-package.py --window-only`"

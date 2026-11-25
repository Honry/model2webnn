"""Gated DeltaNet as static graphs: the chunkwise (loop-free) form reproduces the token-by-token recurrence."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from gdn_experiment import DK, DV, error, graph_feeds, make_inputs  # noqa: E402

from m2w.frontends.hf_llm.builders.gated_delta import build_core_graph, build_recurrent_graph  # noqa: E402
from m2w.ir import interpreter  # noqa: E402
from m2w.reference.gated_delta import recurrent  # noqa: E402


@pytest.mark.parametrize("regime", ["mixed", "mixed-corr"])
@pytest.mark.parametrize("sub,base", [(16, 16), (32, 8), (64, 8)])
def test_chunkwise_matches_the_token_loop(regime, sub, base):
    rng = np.random.default_rng(3)
    heads, tokens = 2, 64
    inputs = make_inputs(rng, heads, tokens, regime)
    state0 = rng.standard_normal((heads, DK, DV)) * 0.1
    want_out, want_state = recurrent(*inputs, state0)
    graph = build_core_graph(heads, DK, DV, tokens, sub, "float32", base=base)
    got = interpreter.run(graph, graph_feeds(inputs, state0, "float32"))
    assert error(got["out"][0], want_out)[1] < 1e-4
    assert error(got["new_state"][0], want_state)[1] < 1e-4


def test_pad_tokens_leave_the_state_alone():
    rng = np.random.default_rng(5)
    heads, tokens, pad = 2, 32, 8
    inputs = list(make_inputs(rng, heads, tokens, "mixed"))
    state0 = rng.standard_normal((heads, DK, DV)) * 0.1
    inputs[3][:, :pad] = 0.0   # g
    inputs[4][:, :pad] = 0.0   # beta
    graph = build_core_graph(heads, DK, DV, tokens, 32, "float32", base=8)
    got = interpreter.run(graph, graph_feeds(inputs, state0, "float32"))
    want_out, want_state = recurrent(*(x[:, pad:] for x in inputs), state0)
    assert error(got["new_state"][0], want_state)[1] < 1e-5
    assert error(got["out"][0][:, pad:], want_out)[1] < 1e-5


def test_decode_step_is_the_recurrence():
    rng = np.random.default_rng(9)
    heads = 4
    inputs = make_inputs(rng, heads, 1, "mixed")
    state0 = rng.standard_normal((heads, DK, DV)) * 0.1
    want_out, want_state = recurrent(*inputs, state0)
    got = interpreter.run(build_recurrent_graph(heads, DK, DV, "float32"), graph_feeds(inputs, state0, "float32"))
    assert error(got["out"][0], want_out)[1] < 1e-5 and error(got["new_state"][0], want_state)[1] < 1e-5

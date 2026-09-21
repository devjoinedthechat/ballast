"""The benchmark is code too, and it makes claims.

Its scenarios are exercised at a size that runs in a moment, so a change that
breaks the measurement is caught here rather than the next time someone quotes
a number from it.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import benchmark

TINY = {"layers": 2, "hidden": 64, "rank": 4, "model_layers": 2, "model_hidden": 64}


def test_the_adapter_fixture_has_the_shape_a_lora_has():
    tensors = benchmark.lora(TINY["layers"], TINY["hidden"], TINY["rank"])
    assert len(tensors) == TINY["layers"] * 4 * 2  # four projections, A and B
    a = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    b = tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    assert a.shape == (TINY["rank"], TINY["hidden"])
    assert b.shape == (TINY["hidden"], TINY["rank"])
    assert a.dtype == benchmark.BF16


def test_the_commit_scenario_measures_what_it_says(tmp_path):
    out = benchmark.scenario_commit(tmp_path, TINY)
    assert out["tensors"] == 16
    # The claim: a retrain of two tensors adds two blocks, not the adapter.
    assert out["blocks_added"] == 2
    assert out["bytes_added_mb"] == pytest.approx(out["touched_mb"], rel=0.01)
    assert out["bytes_added_mb"] < out["adapter_mb"]
    assert out["commit_mbs"] > 0
    assert out["dedup"] > 1


def test_apply_streamed_and_whole_produce_the_same_model(tmp_path):
    import numpy as np

    from ballast import tensors as st

    facts = benchmark.prepare_apply(tmp_path, TINY)
    (tmp_path / "facts.json").write_text(json.dumps(facts))
    streamed = benchmark.scenario_apply(tmp_path, TINY)
    whole = benchmark.scenario_apply_whole(tmp_path, TINY)

    assert streamed["mode"] == "streamed"
    assert whole["mode"] == "whole"
    one = st.load_dir(tmp_path / "out")
    two = st.load_dir(tmp_path / "out-whole")
    assert sorted(one) == sorted(two)
    for name, value in one.items():
        assert np.array_equal(value.view(np.uint8), two[name].view(np.uint8)), name


def test_merge_streamed_and_whole_agree(tmp_path):
    import numpy as np

    from ballast import Store

    facts = benchmark.prepare_merge(tmp_path, TINY)
    (tmp_path / "facts.json").write_text(json.dumps(facts))
    streamed = benchmark.scenario_merge_streamed(tmp_path, TINY)
    whole = benchmark.scenario_merge_whole(tmp_path, TINY)
    assert streamed["tensors"] == whole["tensors"] == 16

    store = Store(tmp_path / "store", cache_views=False)
    a = dict(store.checkout_stream("bench", "v"))
    b = store.checkout("bench", "v")
    for name, value in a.items():
        assert np.array_equal(value, b[name]), name


def test_every_size_is_a_complete_specification():
    for name, spec in benchmark.SIZES.items():
        assert set(spec) == {"layers", "hidden", "rank", "model_layers", "model_hidden"}, name
        assert all(value > 0 for value in spec.values()), name


def test_peak_rss_reports_something_plausible():
    assert 1 < benchmark.peak_rss_mb() < 100_000

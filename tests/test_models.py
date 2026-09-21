"""Full models in, full models out."""

import json

import numpy as np
import pytest
from conftest import adapter

from ballast import models
from ballast import tensors as st
from ballast.models import apply, delta


def model_dir(path, tensors, name="org/base"):
    path.mkdir(parents=True)
    st.save(path / "model.safetensors", tensors, {"format": "pt"})
    (path / "config.json").write_text(json.dumps({"_name_or_path": name, "hidden_size": 64}))
    (path / "tokenizer.json").write_text("{}")
    return path


def test_delta_is_model_minus_base_and_omits_the_unchanged(tmp_path, rng):
    base = adapter(rng)
    tuned = dict(base)
    tuned["layers.0.lora_A.weight"] = (base["layers.0.lora_A.weight"].astype(np.float32) * 1.5).astype(
        base["layers.0.lora_A.weight"].dtype
    )
    model_dir(tmp_path / "base", base)
    model_dir(tmp_path / "tuned", tuned)

    d, report = delta(tmp_path / "tuned", tmp_path / "base")
    assert report.changed == ["layers.0.lora_A.weight"]
    assert len(report.unchanged) == len(base) - 1
    assert set(d) == {"layers.0.lora_A.weight"}
    assert d["layers.0.lora_A.weight"].dtype == base["layers.0.lora_A.weight"].dtype


def test_apply_reconstructs_the_model_and_carries_the_config(tmp_path, rng):
    base = {k: v.astype(np.float32) for k, v in adapter(rng).items()}
    tuned = dict(base)
    tuned["layers.2.lora_B.weight"] = base["layers.2.lora_B.weight"] + 0.25
    model_dir(tmp_path / "base", base)
    model_dir(tmp_path / "tuned", tuned)

    d, _ = delta(tmp_path / "tuned", tmp_path / "base")
    out = apply(d, tmp_path / "base", tmp_path / "rebuilt")
    rebuilt = st.load_dir(out)
    for name, expected in tuned.items():
        assert np.array_equal(rebuilt[name], expected), name
    assert json.loads((out / "config.json").read_text())["_name_or_path"] == "org/base"
    assert (out / "tokenizer.json").exists()


def test_shape_mismatches_and_extra_tensors_are_reported_not_guessed(tmp_path, rng):
    base = adapter(rng)
    tuned = dict(base)
    tuned["layers.0.lora_A.weight"] = rng.standard_normal((4, 4)).astype(np.float32)
    tuned["brand_new"] = np.ones(3, dtype=np.float32)
    del tuned["layers.3.lora_B.weight"]
    model_dir(tmp_path / "base", base)
    model_dir(tmp_path / "tuned", tuned)
    _, report = delta(tmp_path / "tuned", tmp_path / "base")
    assert report.mismatched == ["layers.0.lora_A.weight"]
    assert report.only_in_model == ["brand_new"]
    assert report.only_in_base == ["layers.3.lora_B.weight"]


def test_apply_refuses_a_delta_naming_tensors_the_base_lacks(tmp_path, rng):
    model_dir(tmp_path / "base", adapter(rng))
    with pytest.raises(KeyError, match="base does not have"):
        apply({"nope": np.zeros(2, dtype=np.float32)}, tmp_path / "base", tmp_path / "out")


def test_commit_delta_and_apply_commit_round_trip_through_the_store(store, tmp_path, rng):
    base = {k: v.astype(np.float32) for k, v in adapter(rng).items()}
    tuned = dict(base)
    tuned["layers.1.lora_A.weight"] = base["layers.1.lora_A.weight"] * 3
    model_dir(tmp_path / "base", base)
    model_dir(tmp_path / "tuned", tuned)

    commit, report = models.commit_delta(
        store, "t", tmp_path / "tuned", tmp_path / "base", message="ft run 7"
    )
    assert report.changed == ["layers.1.lora_A.weight"]
    manifest = store.manifest("t", commit.manifest_id)
    assert manifest.base_model == "org/base"
    assert manifest.config["delta_of"] == "org/base"

    out = models.apply_commit(store, "t", commit.id, tmp_path / "base", tmp_path / "rebuilt")
    rebuilt = st.load_dir(out)
    assert all(np.array_equal(rebuilt[k], tuned[k]) for k in tuned)


def test_identical_model_and_base_is_refused(store, tmp_path, rng):
    base = adapter(rng)
    model_dir(tmp_path / "base", base)
    model_dir(tmp_path / "same", base)
    with pytest.raises(ValueError, match="identical"):
        models.commit_delta(store, "t", tmp_path / "same", tmp_path / "base", message="x")

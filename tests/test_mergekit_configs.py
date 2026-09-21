"""Configurations in the shapes people actually publish.

Every config here is a form that appears on real model cards: TIES with a base
and per-model density, DARE with rescale, SLERP with a layer gradient, a
`slices` passthrough, and a linear merge with normalisation turned off. The
point is not that ballast can resolve all of them — it cannot, and says so —
but that it reads them without guessing and records what it will not act on.
"""

import numpy as np
import pytest
import yaml
from conftest import adapter

from ballast import merge as merging
from ballast import mergekit
from ballast.mergekit import SliceMerge

TIES = """
models:
  - model: mistralai/Mistral-7B-Instruct-v0.2
    parameters:
      density: 0.5
      weight: 0.5
  - model: BioMistral/BioMistral-7B
    parameters:
      density: 0.5
      weight: 0.5
merge_method: ties
base_model: mistralai/Mistral-7B-v0.1
parameters:
  normalize: true
  int8_mask: true
dtype: bfloat16
"""

DARE_TIES = """
models:
  - model: teknium/OpenHermes-2.5-Mistral-7B
    parameters:
      density: 0.53
      weight: 0.4
  - model: Intel/neural-chat-7b-v3-1
    parameters:
      density: 0.53
      weight: 0.3
merge_method: dare_ties
base_model: mistralai/Mistral-7B-v0.1
parameters:
  int8_mask: true
  rescale: true
dtype: bfloat16
tokenizer_source: union
"""

LINEAR_UNNORMALISED = """
models:
  - model: org/a
    parameters:
      weight: 1.0
  - model: org/b
    parameters:
      weight: 0.5
merge_method: linear
parameters:
  normalize: false
dtype: float16
"""

TIES_LAMBDA = """
models:
  - model: org/a
    parameters: {weight: 0.6, density: 0.4}
  - model: org/b
    parameters: {weight: 0.4, density: 0.4}
merge_method: ties
base_model: org/base
parameters:
  normalize: false
  lambda: 1.1
"""

SLERP_GRADIENT = """
slices:
  - sources:
      - model: org/a
        layer_range: [0, 32]
      - model: org/b
        layer_range: [0, 32]
merge_method: slerp
base_model: org/a
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - value: 0.5
dtype: bfloat16
"""

PASSTHROUGH_SLICES = """
slices:
  - sources:
      - model: org/a
        layer_range: [0, 24]
  - sources:
      - model: org/b
        layer_range: [8, 32]
merge_method: passthrough
dtype: float16
"""

MIXED_DENSITY = """
models:
  - model: org/a
    parameters: {weight: 0.5, density: 0.3}
  - model: org/b
    parameters: {weight: 0.5, density: 0.7}
merge_method: ties
base_model: org/base
"""


@pytest.fixture
def two(store, rng):
    store.commit("t", adapter(rng), message="a", ref="a", base_model="base")
    store.commit("t", adapter(rng), message="b", ref="b", base_model="base")
    return {
        "mistralai/Mistral-7B-Instruct-v0.2": "a",
        "BioMistral/BioMistral-7B": "b",
        "teknium/OpenHermes-2.5-Mistral-7B": "a",
        "Intel/neural-chat-7b-v3-1": "b",
        "org/a": "a",
        "org/b": "b",
    }


def write(tmp_path, text, name="merge.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_published_ties_config_resolves_with_its_density_and_base(store, tmp_path, two):
    commit = mergekit.import_config(store, "t", write(tmp_path, TIES), refs=two)
    config = store.manifest("t", commit.manifest_id).config
    assert config["method"] == "ties"
    assert config["density"] == 0.5
    assert config["normalize"] is True
    assert config["provenance"]["base_model"] == "mistralai/Mistral-7B-v0.1"
    assert config["provenance"]["dtype"] == "bfloat16"
    # int8_mask changes nothing here, so it is recorded rather than acted on.
    assert config["provenance"]["parameters"] == {"int8_mask": True}
    assert store.checkout("t", commit.id)


def test_a_published_dare_config_gets_a_seed_and_keeps_its_extras(store, tmp_path, two):
    commit = mergekit.import_config(store, "t", write(tmp_path, DARE_TIES), refs=two)
    config = store.manifest("t", commit.manifest_id).config
    assert config["method"] == "dare_ties"
    assert config["density"] == 0.53
    assert isinstance(config["seed"], int)
    assert config["provenance"]["tokenizer_source"] == "union"
    assert set(config["provenance"]["parameters"]) == {"int8_mask", "rescale"}
    assert store.checkout("t", commit.id)


def test_normalize_false_is_carried_into_the_result(store, tmp_path, two):
    """A recipe that says not to normalise must not be normalised.

    TIES defaults to normalising, so a config setting it false resolves to
    different weights. Ignoring the flag would give a view that reads like the
    recipe and computes something else.
    """
    commit = mergekit.import_config(store, "t", write(tmp_path, TIES_LAMBDA), refs=two)
    config = store.manifest("t", commit.manifest_id).config
    assert config["normalize"] is False
    assert config["lambda"] == 1.1

    a, b = store.checkout("t", "a"), store.checkout("t", "b")
    expected = merging.ties([a, b], [0.6, 0.4], density=0.4, normalize=False, lambda_=1.1)
    got = store.checkout("t", commit.id)
    name = next(iter(expected))
    assert np.allclose(got[name].astype(np.float32), expected[name].astype(np.float32), rtol=1e-3)
    # And it genuinely differs from the normalised form.
    other = merging.ties([a, b], [0.6, 0.4], density=0.4, normalize=True, lambda_=1.1)
    assert not np.allclose(got[name].astype(np.float32), other[name].astype(np.float32))


def test_a_linear_config_with_normalize_false_is_recorded(store, tmp_path, two):
    commit = mergekit.import_config(store, "t", write(tmp_path, LINEAR_UNNORMALISED), refs=two)
    config = store.manifest("t", commit.manifest_id).config
    assert config["method"] == "linear"
    assert config["normalize"] is False
    assert store.checkout("t", commit.id)


def test_a_slice_config_is_refused_by_default(store, tmp_path, two):
    with pytest.raises(SliceMerge, match="layer ranges"):
        mergekit.import_config(store, "t", write(tmp_path, PASSTHROUGH_SLICES), refs=two)


def test_a_slice_config_can_be_recorded_but_never_resolves(store, tmp_path, two):
    commit = mergekit.import_config(
        store, "t", write(tmp_path, PASSTHROUGH_SLICES), refs=two, allow_slices=True
    )
    config = store.manifest("t", commit.manifest_id).config
    assert config["method"] == "passthrough:slices"
    assert config["provenance"]["slices"]
    assert "unresolvable" in config["provenance"]
    with pytest.raises(NotImplementedError, match="slice configuration"):
        store.checkout("t", commit.id)


def test_a_slerp_gradient_is_read_without_being_acted_on(store, tmp_path, two):
    commit = mergekit.import_config(store, "t", write(tmp_path, SLERP_GRADIENT), refs=two, allow_slices=True)
    config = store.manifest("t", commit.manifest_id).config
    assert config["method"] == "slerp:slices"
    assert config["provenance"]["parameters"]["t"][0]["filter"] == "self_attn"
    with pytest.raises(NotImplementedError):
        store.checkout("t", commit.id)


def test_per_model_densities_that_differ_are_refused_not_averaged(store, tmp_path, two):
    with pytest.raises(ValueError, match="densities differ"):
        mergekit.import_config(store, "t", write(tmp_path, MIXED_DENSITY), refs=two)


def test_every_config_here_is_valid_yaml_mergekit_would_accept():
    """Guards the fixtures themselves against drifting into something unreal."""
    for text in (TIES, DARE_TIES, LINEAR_UNNORMALISED, TIES_LAMBDA, SLERP_GRADIENT, PASSTHROUGH_SLICES):
        parsed = yaml.safe_load(text)
        assert "merge_method" in parsed
        assert parsed.get("models") or parsed.get("slices")

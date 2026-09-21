"""The hand-off to a serving runtime.

vLLM itself is not installed in CI — it is CUDA-first and does not build on the
machines these tests run on — so what is checked here is everything up to the
request object: the layout, the identity, and that it does not rewrite work it
has already done.
"""

import json

import pytest
from conftest import adapter, scaled

from ballast import peft as peft_io
from ballast import serving
from ballast.cli import main


def test_an_export_has_the_layout_a_server_loads(store, rng, tmp_path):
    store.commit("t", adapter(rng), message="v1", base_model="org/base", config={"r": 8})
    item = serving.export(store, "t", "main", tmp_path / "loras")

    assert (item.path / peft_io.WEIGHTS).exists()
    assert (item.path / peft_io.CONFIG).exists()
    assert json.loads((item.path / peft_io.CONFIG).read_text())["r"] == 8
    assert item.base_model == "org/base"
    assert not item.reused


def test_the_integer_id_is_stable_and_in_range(store, rng, tmp_path):
    """A server keyed by number must see the same number every time.

    Two processes exporting the same commit have to agree, or the server's
    adapter cache holds one delta twice under different ids.
    """
    c = store.commit("t", adapter(rng), message="v1")
    first = serving.export(store, "t", "main", tmp_path / "a")
    second = serving.export(store, "t", c.id, tmp_path / "b")
    assert first.int_id == second.int_id == serving.int_id(c.id)
    assert 0 < first.int_id <= serving.MAX_INT_ID


def test_different_commits_get_different_ids(store, rng, tmp_path):
    a = adapter(rng)
    c1 = store.commit("t", a, message="v1")
    c2 = store.commit("t", scaled(a, "layers.0.lora_A.weight", 2.0), message="v2")
    assert serving.int_id(c1.id) != serving.int_id(c2.id)


def test_exporting_twice_reuses_the_directory(store, rng, tmp_path):
    store.commit("t", adapter(rng), message="v1")
    root = tmp_path / "loras"
    first = serving.export(store, "t", "main", root)
    written = (first.path / peft_io.WEIGHTS).stat().st_mtime_ns

    second = serving.export(store, "t", "main", root)
    assert second.reused
    assert (second.path / peft_io.WEIGHTS).stat().st_mtime_ns == written

    third = serving.export(store, "t", "main", root, refresh=True)
    assert not third.reused


def test_a_view_exports_with_its_recipe(store, rng, tmp_path):
    c1 = store.commit("t", adapter(rng), message="a", base_model="org/base")
    c2 = store.commit("t", adapter(rng), message="b", base_model="org/base")
    m = store.merge("t", "ties", [(c1.id, 0.6), (c2.id, 0.4)], density=0.5, message="blend", ref="blend")

    item = serving.export(store, "t", "blend", tmp_path / "loras")
    provenance = json.loads((item.path / peft_io.PROVENANCE).read_text())
    assert provenance["kind"] == "composite"
    assert provenance["recipe"]["method"] == "ties"
    assert [i["weight"] for i in provenance["recipe"]["inputs"]] == [0.6, 0.4]
    assert item.manifest == m.manifest_id


def test_tenants_export_into_separate_directories(store, rng, tmp_path):
    store.commit("alpha", adapter(rng), message="v1")
    store.commit("beta", adapter(rng), message="v1")
    root = tmp_path / "loras"
    a = serving.export(store, "alpha", "main", root)
    b = serving.export(store, "beta", "main", root)
    assert a.path.parent.name == "alpha"
    assert b.path.parent.name == "beta"
    assert a.name != b.name


def test_a_manifest_file_lists_what_a_server_should_load(store, rng, tmp_path):
    store.commit("t", adapter(rng), message="v1", base_model="org/base")
    store.commit("u", adapter(rng), message="v1", base_model="org/base")
    exports = [serving.export(store, t, "main", tmp_path / "loras") for t in ("t", "u")]
    path = serving.manifest_json(exports, tmp_path / "loras.json")

    entries = json.loads(path.read_text())
    assert len(entries) == 2
    assert {e["tenant"] for e in entries} == {"t", "u"}
    assert all(e["int_id"] > 0 for e in entries)


def test_lora_request_needs_vllm_and_says_so(store, rng, tmp_path):
    store.commit("t", adapter(rng), message="v1")
    item = serving.export(store, "t", "main", tmp_path / "loras")
    try:
        import vllm  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            serving.lora_request(item)
    else:  # pragma: no cover - only where vLLM is installed
        assert serving.lora_request(item).lora_int_id == item.int_id


def test_a_broken_view_is_skipped_and_named_not_fatal(store, rng, tmp_path, capsys):
    """One lost input must not stop the rest of a fleet from being served."""
    c1 = store.commit("t", adapter(rng), message="a", base_model="b", ref="a")
    store.commit("t", adapter(rng), message="b", base_model="b", ref="b")
    store.merge("t", "linear", [("a", 0.5), ("b", 0.5)], message="blend", ref="blend")
    store.forget_commit("t", c1.id, "withdrawn")
    store.close()

    code = main(["--root", str(store.root), "--tenant", "t", "serve-export", "-o", str(tmp_path / "loras")])
    out = capsys.readouterr().out
    assert code == 4
    assert "skipped" in out
    assert "blend" in out
    assert (tmp_path / "loras" / "t").exists()  # the healthy refs still exported

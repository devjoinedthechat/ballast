import json
import threading

import pytest
from conftest import adapter, scaled

from ballast import Store
from ballast import peft as peft_io
from ballast.cli import main


@pytest.mark.parametrize("bad", ["../escape", ".hidden", "a/b", "", "x" * 65, "sp ace", "-lead"])
def test_tenant_names_that_could_escape_the_store_are_refused(store, rng, bad):
    with pytest.raises(ValueError, match="invalid tenant"):
        store.commit(bad, adapter(rng), message="x")


def test_ref_names_are_validated(store, rng):
    with pytest.raises(ValueError, match="invalid ref"):
        store.commit("t", adapter(rng), message="x", ref="feature/branch")


def test_a_spec_with_wildcards_is_not_a_prefix_match(store, rng):
    store.commit("t", adapter(rng), message="v1")
    with pytest.raises(LookupError):
        store.resolve("t", "%")
    with pytest.raises(LookupError):
        store.resolve("t", "_")


def test_commit_metadata_is_kept_and_returned(store, rng):
    meta = {"run": "train-2026-09-21-41", "dataset_sha": "abc123", "approved_by": "ops"}
    c = store.commit("t", adapter(rng), message="v1", metadata=meta)
    assert store.resolve("t", c.id).metadata == meta
    assert store.log("t")[0].metadata == meta


def test_probe_sets_are_stored_so_a_fingerprint_id_can_be_explained(store, rng):
    store.commit("t", adapter(rng), message="v1")
    pid = store.record_fingerprint("t", "main", ["a?", "b?"], ["x", "y"])
    assert store.probe_set(pid) == ["a?", "b?"]
    assert store.fingerprint_of("t", "main", pid) == ["x", "y"]


def test_a_proof_survives_the_process_and_verifies_by_attestation(store, rng, tmp_path):
    store.commit("t", adapter(rng), message="v1")
    proof = store.forget("t", "erasure")
    store.close()

    reopened = Store(store.root)
    assert reopened.verify(proof.attestation) == []
    assert reopened.proof(proof.attestation).reason == "erasure"


def test_fsck_catches_a_chunk_whose_bytes_changed_on_disk(store, rng):
    a = adapter(rng)
    store.commit("t", a, message="v1")
    assert store.fsck("t") == []
    row = store.db.execute("SELECT hash FROM chunks WHERE tenant = 't' LIMIT 1").fetchone()
    path = store.chunks.path("t", row["hash"])
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(data)
    problems = store.fsck("t")
    assert any("does not hash to its name" in p for p in problems)
    assert store.fsck("t", verify_bytes=False) == []  # the graph is still fine


def test_fsck_catches_a_chunk_missing_from_disk(store, rng):
    store.commit("t", adapter(rng), message="v1")
    row = store.db.execute("SELECT hash FROM chunks WHERE tenant = 't' LIMIT 1").fetchone()
    store.chunks.path("t", row["hash"]).unlink()
    assert any("missing from the backend" in p for p in store.fsck("t"))


def test_fsck_reports_a_broken_view(store, rng):
    c1 = store.commit("t", adapter(rng), message="a")
    c2 = store.commit("t", adapter(rng), message="b")
    store.merge("t", "linear", [(c1.id, 1.0), (c2.id, 1.0)], message="avg")
    store.forget_commit("t", c1.id, "gone")
    assert any("broken view" in p for p in store.fsck("t"))


def test_concurrent_commits_from_threads_do_not_corrupt_the_store(store, rng):
    base = adapter(rng)
    errors: list[Exception] = []

    def work(i: int) -> None:
        try:
            own = Store(store.root)
            own.commit(
                "t", scaled(base, "layers.0.lora_A.weight", 1.0 + i / 10), message=f"w{i}", ref=f"r{i}"
            )
            own.close()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert store.stats("t").commits == 8
    assert store.fsck("t") == []


def test_checkout_writes_provenance_next_to_the_adapter(store, rng, tmp_path):
    peft_io.export(tmp_path / "in", adapter(rng), {"r": 8, "base_model_name_or_path": "org/base"})
    root = str(tmp_path / "store")
    assert (
        main(
            [
                "--root",
                root,
                "--tenant",
                "t",
                "commit",
                str(tmp_path / "in"),
                "-m",
                "v1",
                "--metadata",
                '{"run": "41"}',
            ]
        )
        == 0
    )
    assert main(["--root", root, "--tenant", "t", "checkout", "main", "-o", str(tmp_path / "out")]) == 0
    prov = json.loads((tmp_path / "out" / "ballast.json").read_text())
    assert prov["tenant"] == "t"
    assert prov["metadata"] == {"run": "41"}
    assert prov["kind"] == "leaf"
    assert len(prov["commit"]) == 64


def test_json_output_is_parseable(store, rng, tmp_path, capsys):
    peft_io.export(tmp_path / "in", adapter(rng), {"r": 8})
    root = str(tmp_path / "store")
    main(["--root", root, "--tenant", "t", "--json", "commit", str(tmp_path / "in"), "-m", "v1"])
    out = json.loads(capsys.readouterr().out)
    assert out["message"] == "v1"
    main(["--root", root, "--tenant", "t", "--json", "stats"])
    assert json.loads(capsys.readouterr().out)["commits"] == 1

import numpy as np
from conftest import adapter

from ballast import mergekit
from ballast import peft as peft_io
from ballast.cli import main

CONFIG = """
merge_method: ties
base_model: org/base-7b
models:
  - model: org/finance-lora
    parameters:
      weight: 0.6
      density: 0.5
  - model: org/support-lora
    parameters:
      weight: 0.4
dtype: bfloat16
"""


def test_mergekit_config_becomes_a_composite_with_its_parameters(store, rng, tmp_path):
    store.commit("t", adapter(rng), message="finance", ref="finance", base_model="org/base-7b")
    store.commit("t", adapter(rng), message="support", ref="support", base_model="org/base-7b")
    cfg = tmp_path / "merge.yaml"
    cfg.write_text(CONFIG)

    commit = mergekit.import_config(
        store, "t", cfg, refs={"org/finance-lora": "finance", "org/support-lora": "support"}
    )
    manifest = store.manifest("t", commit.manifest_id)
    assert manifest.kind == "composite"
    assert manifest.config == {"method": "ties", "density": 0.5}
    rows = store.db.execute(
        "SELECT weight FROM manifest_inputs WHERE manifest_id = ? ORDER BY position", (commit.manifest_id,)
    ).fetchall()
    assert [r["weight"] for r in rows] == [0.6, 0.4]
    assert store.checkout("t", commit.id)


def test_peft_directory_round_trip(tmp_path, rng):
    arrays = adapter(rng)
    config = {"r": 8, "lora_alpha": 16, "base_model_name_or_path": "org/base-7b"}
    peft_io.export(tmp_path / "adapter", arrays, config)
    loaded, cfg, base = peft_io.load(tmp_path / "adapter")
    assert cfg == config
    assert base == "org/base-7b"
    assert all(np.array_equal(loaded[k].view(np.uint8), arrays[k].view(np.uint8)) for k in arrays)


def test_cli_commit_log_diff_checkout_forget(tmp_path, rng, capsys):
    root = str(tmp_path / "store")
    a = adapter(rng)
    peft_io.export(tmp_path / "v1", a, {"r": 8, "base_model_name_or_path": "org/base"})
    b = dict(a)
    layer = a["layers.1.lora_A.weight"]
    b["layers.1.lora_A.weight"] = (layer.astype(np.float32) * 3).astype(layer.dtype)
    peft_io.export(tmp_path / "v2", b, {"r": 8, "base_model_name_or_path": "org/base"})

    assert main(["--root", root, "--tenant", "t", "commit", str(tmp_path / "v1"), "-m", "v1"]) == 0
    assert main(["--root", root, "--tenant", "t", "commit", str(tmp_path / "v2"), "-m", "v2"]) == 0
    assert main(["--root", root, "--tenant", "t", "log"]) == 0
    out = capsys.readouterr().out
    assert "v2" in out
    assert "v1" in out

    ids = [line.split()[0] for line in out.strip().splitlines() if "leaf" in line]
    assert main(["--root", root, "--tenant", "t", "diff", ids[1], ids[0]]) == 0
    assert "1 changed of 8" in capsys.readouterr().out

    assert main(["--root", root, "--tenant", "t", "checkout", ids[0], "-o", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out" / "adapter_model.safetensors").exists()

    assert main(["--root", root, "--tenant", "t", "stats"]) == 0
    assert "9 chunks" in capsys.readouterr().out

    assert main(["--root", root, "--tenant", "t", "forget", "--reason", "test"]) == 0
    assert "verified" in capsys.readouterr().out

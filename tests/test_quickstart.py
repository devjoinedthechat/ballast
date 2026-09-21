"""The flow the README documents, run exactly as written.

Every other test calls the library. This one goes through `main()` with the
arguments a reader would type, because the first thing anyone does with a
project is copy its quickstart, and a flag renamed in one place and not the
other breaks that without breaking anything else.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from ballast import tensors as st
from ballast.cli import main


def model_dir(path: Path, tensors: dict, name: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    st.save(path / "model.safetensors", tensors, {"format": "pt"})
    (path / "config.json").write_text(json.dumps({"_name_or_path": name}))
    (path / "tokenizer.json").write_text("{}")
    return path


@pytest.fixture
def models_on_disk(tmp_path, rng):
    base = {
        f"model.layers.{i}.self_attn.q_proj.weight": rng.standard_normal((32, 32)).astype(np.float32)
        for i in range(6)
    }
    model_dir(tmp_path / "llama-8b", base, "meta-llama/Llama-3.1-8B")

    support = dict(base)
    support["model.layers.1.self_attn.q_proj.weight"] = base["model.layers.1.self_attn.q_proj.weight"] * 1.3
    model_dir(tmp_path / "support-ft", support, "meta-llama/Llama-3.1-8B")

    # A different layer, which is the normal case and the one a strict merge
    # would refuse: two fine-tunes of a base rarely touch the same weights.
    finance = dict(base)
    finance["model.layers.4.self_attn.q_proj.weight"] = base["model.layers.4.self_attn.q_proj.weight"] * 0.8
    model_dir(tmp_path / "finance-ft", finance, "meta-llama/Llama-3.1-8B")
    return tmp_path, base


def test_the_readme_quickstart_runs_as_written(models_on_disk, capsys):
    root, base = models_on_disk
    store = str(root / ".ballast")

    def ballast(*args):
        code = main(["--root", store, "--tenant", "acme", *args])
        assert code == 0, f"ballast {' '.join(args)} exited {code}"
        return capsys.readouterr().out

    ballast(
        "delta",
        str(root / "support-ft"),
        "--base",
        str(root / "llama-8b"),
        "-m",
        "support, run 41",
        "--ref",
        "support",
    )
    ballast(
        "delta",
        str(root / "finance-ft"),
        "--base",
        str(root / "llama-8b"),
        "-m",
        "finance",
        "--ref",
        "finance",
    )
    ballast(
        "merge",
        "support@0.6",
        "finance@0.4",
        "--method",
        "ties",
        "--density",
        "0.5",
        "-m",
        "blend",
        "--ref",
        "blend",
    )
    ballast("apply", "blend", "--base", str(root / "llama-8b"), "-o", str(root / "blend-model"))

    rebuilt = st.load_dir(root / "blend-model")
    assert sorted(rebuilt) == sorted(base)
    assert (root / "blend-model" / "config.json").exists()
    assert (root / "blend-model" / "tokenizer.json").exists()
    # The two layers the fine-tunes touched moved; the rest came through untouched.
    changed = [k for k in base if not np.array_equal(rebuilt[k], base[k])]
    assert len(changed) == 2


def test_every_documented_command_parses(models_on_disk, capsys):
    """A flag renamed in the parser but not the handler fails only when run."""
    root, _ = models_on_disk
    store = str(root / ".ballast")
    main(
        [
            "--root",
            store,
            "--tenant",
            "acme",
            "delta",
            str(root / "support-ft"),
            "--base",
            str(root / "llama-8b"),
            "-m",
            "a",
            "--ref",
            "a",
        ]
    )
    main(
        [
            "--root",
            store,
            "--tenant",
            "acme",
            "delta",
            str(root / "finance-ft"),
            "--base",
            str(root / "llama-8b"),
            "-m",
            "b",
            "--ref",
            "b",
        ]
    )
    capsys.readouterr()

    for args in (
        ["log", "a"],
        ["reflog", "a"],
        ["merge", "a@0.5", "b@0.5", "-m", "m", "--ref", "m1"],
        ["merge", "a@0.5", "b@0.5", "--method", "ties", "--density", "0.5", "-m", "m", "--ref", "m2"],
        ["merge", "a@0.5", "b@0.5", "--strict", "-m", "m", "--ref", "m3"],
        ["merge", "a@0.6", "b@0.4", "--method", "slerp", "--t", "0.5", "-m", "m", "--ref", "m4"],
        ["checkout", "a", "-o", str(root / "out-a")],
        ["apply", "m1", "--base", str(root / "llama-8b"), "-o", str(root / "out-m")],
        ["diff", "a", "b"],
        ["stats"],
        ["grants"],
        ["fsck", "--fast"],
        ["gc"],
        ["serve-export", "-o", str(root / "loras")],
        ["reset", "a", "b"],
    ):
        code = main(["--root", store, "--tenant", "acme", *args])
        assert code in (0, 4), f"ballast {' '.join(args)} exited {code}"
        capsys.readouterr()


def test_the_strict_flag_refuses_what_the_default_merges(models_on_disk, capsys):
    root, _ = models_on_disk
    store = str(root / ".ballast")
    for ref, model in (("a", "support-ft"), ("b", "finance-ft")):
        main(
            [
                "--root",
                store,
                "--tenant",
                "acme",
                "delta",
                str(root / model),
                "--base",
                str(root / "llama-8b"),
                "-m",
                ref,
                "--ref",
                ref,
            ]
        )
    main(
        ["--root", store, "--tenant", "acme", "merge", "a@0.5", "b@0.5", "--strict", "-m", "s", "--ref", "s"]
    )
    capsys.readouterr()

    # The strict view refuses; the default one merges the same inputs.
    assert main(["--root", store, "--tenant", "acme", "checkout", "s", "-o", str(root / "nope")]) == 2
    main(["--root", store, "--tenant", "acme", "merge", "a@0.5", "b@0.5", "-m", "u", "--ref", "u"])
    assert main(["--root", store, "--tenant", "acme", "checkout", "u", "-o", str(root / "yes")]) == 0


def test_an_unknown_ref_is_a_message_not_a_traceback(models_on_disk, capsys):
    """The most common mistake at the keyboard deserves one line, not a stack."""
    root, _ = models_on_disk
    code = main(["--root", str(root / ".ballast"), "--tenant", "acme", "log", "nosuchref"])
    assert code == 2
    assert "nosuchref" in capsys.readouterr().err


def test_every_resolvable_method_is_reachable_from_the_command_line(models_on_disk, capsys):
    """A method the library resolves but the CLI cannot express is half-shipped.

    `slerp` was exactly that: implemented, tested, and missing its `--t`.
    """
    from ballast import merge as merging

    root, _ = models_on_disk
    store = str(root / ".ballast")
    for ref, model in (("a", "support-ft"), ("b", "finance-ft")):
        main(
            [
                "--root",
                store,
                "--tenant",
                "acme",
                "delta",
                str(root / model),
                "--base",
                str(root / "llama-8b"),
                "-m",
                ref,
                "--ref",
                ref,
            ]
        )
    capsys.readouterr()

    needs_density = {
        "ties",
        "breadcrumbs",
        "breadcrumbs_ties",
        "dare_ties",
        "dare_linear",
        "della",
        "della_linear",
    }
    for method in merging.RESOLVABLE:
        inputs = ["a@1.0"] if method == "passthrough" else ["a@0.6", "b@0.4"]
        extra: list[str] = ["--density", "0.5"] if method in needs_density else []
        if method == "slerp":
            extra = ["--t", "0.5"]
        ref = f"c-{method}"

        assert (
            main(
                [
                    "--root",
                    store,
                    "--tenant",
                    "acme",
                    "merge",
                    *inputs,
                    "--method",
                    method,
                    *extra,
                    "-m",
                    method,
                    "--ref",
                    ref,
                ]
            )
            == 0
        ), method
        assert (
            main(["--root", store, "--tenant", "acme", "checkout", ref, "-o", str(root / "out" / method)])
            == 0
        ), method
        capsys.readouterr()
        assert (root / "out" / method / "adapter_model.safetensors").exists(), method

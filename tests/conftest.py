import ml_dtypes
import numpy as np
import pytest

from ballast import Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "store")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def adapter(rng, layers=4, rank=8, width=64, dtype=ml_dtypes.bfloat16):
    out = {}
    for i in range(layers):
        out[f"layers.{i}.lora_A.weight"] = rng.standard_normal((rank, width)).astype(dtype)
        out[f"layers.{i}.lora_B.weight"] = rng.standard_normal((width, rank)).astype(np.float16)
    return out


def scaled(tensors, name, factor):
    out = dict(tensors)
    t = tensors[name]
    out[name] = (t.astype(np.float32) * factor).astype(t.dtype)
    return out

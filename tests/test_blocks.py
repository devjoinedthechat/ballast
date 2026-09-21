"""Sub-tensor blocks: the property that makes this a delta store."""

import ml_dtypes
import numpy as np
import pytest

from ballast import Store
from ballast.backends import LocalBackend
from ballast.chunks import ChunkStore


@pytest.fixture
def small_blocks(tmp_path):
    """A store with 4 KiB blocks, so ordinary test tensors span many."""
    return Store(tmp_path / "store", block_size=4096)


def test_a_tensor_spans_blocks_and_round_trips(small_blocks, rng):
    big = rng.standard_normal((512, 64)).astype(np.float32)  # 128 KiB -> 32 blocks
    c = small_blocks.commit("t", {"w": big}, message="big")
    assert small_blocks.stats("t").chunks == 32
    out = small_blocks.checkout("t", c.id)["w"]
    assert out.dtype == big.dtype
    assert np.array_equal(out, big)


def test_changing_one_value_in_a_large_tensor_stores_one_block(small_blocks, rng):
    big = rng.standard_normal((512, 64)).astype(np.float32)
    small_blocks.commit("t", {"w": big}, message="v1")
    edited = big.copy()
    edited[300, 7] += 1.0
    small_blocks.commit("t", {"w": edited}, message="v2")
    s = small_blocks.stats("t")
    assert s.chunks == 33
    assert s.dedup_ratio == pytest.approx(2 * big.nbytes / (33 * 4096), rel=1e-3)


def test_the_diff_fast_path_skips_tensors_whose_blocks_match(small_blocks, rng):
    a = {
        "x": rng.standard_normal((512, 64)).astype(np.float32),
        "y": rng.standard_normal((16, 16)).astype(np.float32),
    }
    c1 = small_blocks.commit("t", a, message="v1")
    b = dict(a)
    b["y"] = a["y"] * 2
    c2 = small_blocks.commit("t", b, message="v2")
    d = small_blocks.diff("t", c1.id, c2.id)
    assert d.changed == ("y",)
    assert d.max_tensor_change == pytest.approx(1.0)


def test_block_boundary_that_does_not_divide_the_tensor(small_blocks, rng):
    odd = rng.standard_normal((1000, 3)).astype(np.float16)  # 6000 bytes: one full block and a 1904-byte tail
    c = small_blocks.commit("t", {"w": odd}, message="odd")
    assert small_blocks.stats("t").chunks == 2
    assert np.array_equal(small_blocks.checkout("t", c.id)["w"].view(np.uint8), odd.view(np.uint8))


def test_empty_tensor_has_no_blocks_and_comes_back_empty(store):
    c = store.commit("t", {"e": np.zeros((0, 8), dtype=np.float32)}, message="empty")
    out = store.checkout("t", c.id)["e"]
    assert out.shape == (0, 8)
    assert store.stats("t").chunks == 0


def test_low_entropy_blocks_compress_and_random_ones_are_kept_raw(tmp_path, rng):
    store = Store(tmp_path / "s")
    zeros = np.zeros((1024, 1024), dtype=np.float32)
    noise = rng.standard_normal((256, 256)).astype(np.float32)
    store.commit("t", {"zeros": zeros, "noise": noise}, message="mix")
    rows = {
        r["encoding"]: r["stored_bytes"]
        for r in store.db.execute("SELECT encoding, stored_bytes FROM chunks")
    }
    assert "zstd" in rows
    assert rows["zstd"] < zeros.nbytes // 100
    assert store.stats("t").compression_ratio > 10


def test_a_raw_store_reads_single_block_tensors_by_memory_map(tmp_path, rng):
    store = Store(tmp_path / "raw", compression="raw")
    w = rng.standard_normal((64, 64)).astype(np.float32)
    c = store.commit("t", {"w": w}, message="v1")
    out = store.checkout("t", c.id)["w"]
    assert isinstance(out.base, np.memmap)


def test_block_size_and_compression_are_fixed_at_creation(tmp_path, rng):
    Store(tmp_path / "s", block_size=4096, compression="raw").commit(
        "t", {"w": rng.standard_normal(8)}, message="v1"
    )
    reopened = Store(tmp_path / "s", block_size=1 << 30, compression="zstd")
    assert reopened.chunks.block_size == 4096
    assert reopened.chunks.compression == "raw"


def test_block_address_does_not_depend_on_encoding(tmp_path):
    data = memoryview(bytes(range(256)) * 64)
    raw = ChunkStore(LocalBackend(tmp_path / "raw"), compression="raw")
    zstd = ChunkStore(LocalBackend(tmp_path / "zstd"), compression="zstd")
    assert raw.put_block("t", data).digest == zstd.put_block("t", data).digest


def test_bfloat16_survives_blocks(small_blocks, rng):
    w = rng.standard_normal((300, 300)).astype(ml_dtypes.bfloat16)
    c = small_blocks.commit("t", {"w": w}, message="bf16")
    out = small_blocks.checkout("t", c.id)["w"]
    assert out.dtype == ml_dtypes.bfloat16
    assert np.array_equal(out.view(np.uint8), w.view(np.uint8))

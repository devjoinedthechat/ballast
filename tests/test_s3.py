"""The S3 backend, against moto's in-process mock."""

import boto3
import numpy as np
import pytest
from conftest import adapter, scaled
from moto import mock_aws

from ballast import Store
from ballast.backends import S3Backend, from_url


@pytest.fixture
def s3_store(tmp_path):
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="deltas")
        yield Store(tmp_path / "meta", backend=S3Backend("deltas", "prod", client=client)), client


def test_commit_checkout_and_dedup_over_s3(s3_store, rng):
    store, client = s3_store
    a = adapter(rng)
    c1 = store.commit("t", a, message="v1")
    store.commit("t", scaled(a, "layers.1.lora_A.weight", 1.5), message="v2")
    keys = [o["Key"] for o in client.list_objects_v2(Bucket="deltas", Prefix="prod/t/")["Contents"]]
    assert len(keys) == len(a) + 1
    out = store.checkout("t", c1.id)
    assert all(np.array_equal(out[k].view(np.uint8), a[k].view(np.uint8)) for k in a)


def test_forget_removes_every_object_and_the_proof_verifies(s3_store, rng):
    store, client = s3_store
    store.commit("gone", adapter(rng), message="v1")
    store.commit("stays", adapter(rng), message="v1")
    proof = store.forget("gone", "erasure")
    assert store.verify(proof) == []
    assert "Contents" not in client.list_objects_v2(Bucket="deltas", Prefix="prod/gone/")
    assert client.list_objects_v2(Bucket="deltas", Prefix="prod/stays/")["KeyCount"] == len(adapter(rng))


def test_fsck_catches_an_object_deleted_behind_the_stores_back(s3_store, rng):
    store, client = s3_store
    store.commit("t", adapter(rng), message="v1")
    key = client.list_objects_v2(Bucket="deltas", Prefix="prod/t/")["Contents"][0]["Key"]
    client.delete_object(Bucket="deltas", Key=key)
    assert any("missing from the backend" in p for p in store.fsck("t"))


def test_no_path_for_a_remote_backend(s3_store):
    store, _ = s3_store
    with pytest.raises(TypeError, match="no filesystem paths"):
        store.chunks.path("t", "ab" * 32)


def test_backend_url_parsing(tmp_path):
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="bucket")
        backend = from_url("s3://bucket/some/prefix", tmp_path)
        assert isinstance(backend, S3Backend)
        assert backend.describe() == "s3://bucket/some/prefix"
    with pytest.raises(ValueError, match="unsupported backend"):
        from_url("gs://nope", tmp_path)

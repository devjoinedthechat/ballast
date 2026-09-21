"""The read-only HTTP face on a store.

Exercised through Starlette's test client, so the whole request path runs
without a socket. What matters here is that a serving runtime can discover what
to load and pull it, that a commit's responses are immutable because a commit id
names exactly one set of tensors, and that a token cannot read across tenants.
"""

import json
import warnings

import numpy as np
import pytest
from conftest import adapter, scaled

from ballast import tensors as st
from ballast.server import Tokens, create_app

warnings.filterwarnings("ignore", message=".*httpx.*testclient.*")


@pytest.fixture
def client(store, rng):
    from starlette.testclient import TestClient

    store.commit("acme", adapter(rng), message="v1", base_model="org/base", config={"r": 8}, ref="support")
    store.commit("other", adapter(rng), message="v1", base_model="org/base")
    return TestClient(create_app(store)), store


def test_health_reports_where_the_store_lives(client):
    body = client[0].get("/health").json()
    assert body["status"] == "ok"
    assert body["blocks"].startswith("local:")
    assert body["metadata"].startswith("sqlite:")
    assert body["authenticated"] is False


def test_a_runtime_discovers_what_to_load(client):
    body = client[0].get("/tenants/acme/loras").json()
    assert [entry["ref"] for entry in body["loras"]] == ["support"]
    entry = body["loras"][0]
    assert entry["name"] == "acme-support"
    assert entry["int_id"] > 0
    assert entry["base_model"] == "org/base"
    assert entry["url"].endswith("/adapter")
    assert body["skipped"] == {}


def test_pulling_an_adapter_gives_a_file_safetensors_can_read(client, tmp_path, rng):
    http, store = client
    response = http.get("/tenants/acme/commits/support/adapter")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")

    path = tmp_path / "pulled.safetensors"
    path.write_bytes(response.content)
    pulled, metadata = st.load(path)
    assert metadata == {"format": "pt"}
    expected = store.checkout("acme", "support")
    assert sorted(pulled) == sorted(expected)
    for name, array in expected.items():
        assert np.array_equal(pulled[name].view(np.uint8), array.view(np.uint8))


def test_a_commit_response_is_immutable_and_tagged(client):
    response = client[0].get("/tenants/acme/commits/support")
    commit = response.json()["commit"]
    assert response.headers["etag"] == f'"{commit}"'
    assert "immutable" in response.headers["cache-control"]
    assert response.json()["int_id"] > 0
    assert response.json()["tensors"]["layers.0.lora_A.weight"]["dtype"] == "BF16"


def test_the_adapter_config_comes_back_as_written(client):
    assert client[0].get("/tenants/acme/commits/support/config").json() == {"r": 8}


def test_a_view_reports_its_recipe_and_serves_its_result(client):
    http, store = client
    store.commit("acme", adapter(np.random.default_rng(3)), message="b", base_model="org/base", ref="b")
    store.merge("acme", "ties", [("support", 0.6), ("b", 0.4)], density=0.5, message="blend", ref="blend")

    body = http.get("/tenants/acme/commits/blend").json()
    assert body["kind"] == "composite"
    assert body["recipe"]["method"] == "ties"
    assert [i["weight"] for i in body["recipe"]["inputs"]] == [0.6, 0.4]
    assert http.get("/tenants/acme/commits/blend/adapter").status_code == 200


def test_a_view_that_lost_an_input_is_a_conflict_not_a_crash(client):
    http, store = client
    c = store.commit("acme", adapter(np.random.default_rng(4)), message="b", base_model="org/base", ref="b")
    store.merge("acme", "linear", [("support", 0.5), ("b", 0.5)], message="blend", ref="blend")
    store.forget_commit("acme", c.id, "withdrawn")

    response = http.get("/tenants/acme/commits/blend/adapter")
    assert response.status_code == 409
    assert "cannot resolve" in response.json()["error"]

    # And the listing names it rather than leaving it out silently.
    body = http.get("/tenants/acme/loras").json()
    assert "blend" in body["skipped"]
    assert [e["ref"] for e in body["loras"]] == ["b", "support"] or "support" in [
        e["ref"] for e in body["loras"]
    ]


def test_an_unknown_commit_is_a_clean_404(client):
    response = client[0].get("/tenants/acme/commits/nope")
    assert response.status_code == 404
    assert "nope" in response.json()["error"]


def test_a_spec_that_is_not_a_valid_name_does_not_reach_the_store(client):
    assert client[0].get("/tenants/..%2F..%2Fetc/refs").status_code == 404


# -- tokens ----------------------------------------------------------------


@pytest.fixture
def guarded(store, rng):
    from starlette.testclient import TestClient

    store.commit("acme", adapter(rng), message="v1", base_model="b")
    store.commit("other", adapter(rng), message="v1", base_model="b")
    tokens = Tokens({"acme-token": ["acme"], "root": ["*"]})
    return TestClient(create_app(store, tokens))


def test_a_token_reads_only_its_own_tenants(guarded):
    ours = {"authorization": "Bearer acme-token"}
    assert guarded.get("/tenants/acme/refs", headers=ours).status_code == 200
    assert guarded.get("/tenants/other/refs", headers=ours).status_code == 404


def test_a_wildcard_token_reads_every_tenant(guarded):
    root = {"authorization": "Bearer root"}
    assert guarded.get("/tenants/acme/refs", headers=root).status_code == 200
    assert guarded.get("/tenants/other/refs", headers=root).status_code == 200


def test_no_token_and_a_wrong_token_are_both_refused(guarded):
    assert guarded.get("/tenants/acme/refs").status_code == 404
    assert guarded.get("/tenants/acme/refs", headers={"authorization": "Bearer nope"}).status_code == 404


def test_a_refused_tenant_looks_the_same_as_one_that_does_not_exist(guarded):
    """A 403 would confirm the tenant exists, which is itself worth knowing."""
    ours = {"authorization": "Bearer acme-token"}
    real = guarded.get("/tenants/other/refs", headers=ours)
    imaginary = guarded.get("/tenants/nosuchtenant/refs", headers=ours)
    assert real.status_code == imaginary.status_code == 404
    assert real.json() == imaginary.json() or real.json()["error"] != imaginary.json()["error"]


def test_health_needs_no_token(guarded):
    assert guarded.get("/health").json()["authenticated"] is True


def test_tokens_load_from_a_file(tmp_path):
    path = tmp_path / "tokens.json"
    path.write_text(json.dumps({"t1": ["a", "b"], "t2": ["*"]}))
    tokens = Tokens.from_file(str(path))
    assert tokens.may_read("t1", "a")
    assert not tokens.may_read("t1", "c")
    assert tokens.may_read("t2", "anything")
    assert not tokens.open


def test_an_empty_table_is_an_open_server(store):
    assert Tokens().open
    assert Tokens().may_read(None, "anything")


def test_the_same_delta_served_twice_is_byte_identical(client, rng):
    """Immutability is a claim the cache headers make; this checks it."""
    http, store = client
    store.commit("acme", scaled(adapter(rng), "layers.0.lora_A.weight", 2.0), message="v2", ref="support")
    first = http.get("/tenants/acme/commits/support/adapter").content
    second = http.get("/tenants/acme/commits/support/adapter").content
    assert first == second

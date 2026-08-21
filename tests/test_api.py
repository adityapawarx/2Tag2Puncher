from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tagpuncher.serve.api import create_app
from tagpuncher.serve.inference import TaggingService


@pytest.fixture(scope="module")
def client(tiny_model):
    with TestClient(create_app(TaggingService(tiny_model))) as test_client:
        yield test_client


def test_healthz_reports_the_loaded_model(client):
    payload = client.get("/v1/healthz").json()
    assert payload == {"status": "ok", "model_version": "test", "n_labels": 3}


def test_tag_returns_the_topic_the_text_is_about(client):
    response = client.post(
        "/v1/tag",
        json={
            "text": "The observatory telescope tracked a comet orbiting a distant star.",
            "top_k": 3,
        },
    )
    assert response.status_code == 200

    body = response.json()
    assert body["model_version"] == "test"
    assert body["tags"][0]["tag"] == "Astronomy"
    assert body["tags"][0]["score"] > 0
    assert body["tags"][0]["chunks"] == [0]


def test_tag_chunks_long_documents_and_reports_supporting_chunks(client):
    document = "\n\n".join(
        ["the orchestra performed a symphony " * 80, "the acid neutralised the base " * 80]
    )
    body = client.post("/v1/tag", json={"text": document, "top_k": 5}).json()

    assert len(body["chunks"]) == 2
    tags = {tag["tag"] for tag in body["tags"]}
    assert {"Music", "Chemistry"} <= tags
    assert all(tag["chunks"] for tag in body["tags"])


def test_min_score_filters_weak_tags(client):
    body = client.post(
        "/v1/tag",
        json={"text": "a choir sang the chorus", "top_k": 10, "min_score": 0.99},
    ).json()
    assert body["tags"] == []


def test_empty_text_is_rejected(client):
    assert client.post("/v1/tag", json={"text": ""}).status_code == 422


def test_tag_document_accepts_plain_text_uploads(client):
    files = {"file": ("paper.txt", b"chemists measured the enthalpy of the reaction", "text/plain")}
    body = client.post("/v1/tag/document", files=files, data={"top_k": 1}).json()
    assert body["tags"][0]["tag"] == "Chemistry"

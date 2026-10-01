"""Tests for the GNOMEdata Loading Dock web interface."""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_loading_dock_page_exists() -> None:
    response = client.get("/dock")

    assert response.status_code == 200
    assert "GNOMEdata" in response.text
    assert "Loading Dock" in response.text


def test_ingest_rejects_empty_text() -> None:
    response = client.post(
        "/api/ingest",
        json={
            "collection_id": "default",
            "source_name": "manual.txt",
            "text": "",
        },
    )

    assert response.status_code == 422


def test_ingest_rejects_empty_source_name() -> None:
    response = client.post(
        "/api/ingest",
        json={
            "collection_id": "default",
            "source_name": "",
            "text": "Hydraulic pump maintenance instructions.",
        },
    )

    assert response.status_code == 422


def test_ingest_requires_configured_mill() -> None:
    """A valid load cannot be milled without application resources."""

    import pytest

    with pytest.raises(
        RuntimeError,
        match="Lumber Yard is not configured",
    ):
        client.post(
            "/api/ingest",
            json={
                "collection_id": "default",
                "source_name": "manual.txt",
                "text": (
                    "Inspect hydraulic hoses "
                    "before operation."
                ),
            },
        )


def test_ingest_actually_mills_and_stores_lumber(
    tmp_path,
) -> None:
    """Accepted dock loads must reach the real Lumber Yard."""

    from app.rag.embeddings import StaticEmbedder
    from app.rag.yard import LumberYard

    text = (
        "Inspect hydraulic hoses before operation and "
        "service the hydraulic pump every five hundred hours."
    )

    embedder = StaticEmbedder(
        {
            text: [1.0, 0.0, 0.0],
        },
        model_name="dock-test-model",
    )

    database = tmp_path / "dock.db"

    app.state.yard = LumberYard(database)
    app.state.embedder = embedder

    try:
        response = client.post(
            "/api/ingest",
            json={
                "collection_id": "default",
                "source_name": "hydraulic-manual.txt",
                "text": text,
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert body["status"] == "milled"
        assert body["source_name"] == "hydraulic-manual.txt"
        assert body["collection_id"] == "default"
        assert body["total"] == 1
        assert body["accepted"] == 1
        assert body["embedded"] == 1

        assert app.state.yard.count() == 1

    finally:
        app.state.yard.close()

        del app.state.yard
        del app.state.embedder

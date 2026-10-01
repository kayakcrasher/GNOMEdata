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


def test_application_resources_can_power_real_ingestion(
    tmp_path,
) -> None:
    """Application-owned resources can power the Loading Dock."""

    from app.rag.embeddings import StaticEmbedder
    from app.rag.yard import LumberYard

    text = (
        "Inspect hydraulic hoses before operation and "
        "service the hydraulic pump regularly."
    )

    yard = LumberYard(tmp_path / "lifespan.db")

    embedder = StaticEmbedder(
        {
            text: [1.0, 0.0, 0.0],
        },
        model_name="lifespan-test-model",
    )

    app.state.yard = yard
    app.state.embedder = embedder

    try:
        response = client.post(
            "/api/ingest",
            json={
                "collection_id": "default",
                "source_name": "lifespan-manual.txt",
                "text": text,
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "milled"
        assert yard.count() == 1

    finally:
        yard.close()
        del app.state.yard
        del app.state.embedder


def test_collection_workspace_lists_stored_lumber() -> None:
    with TestClient(app) as workspace_client:
        ingest_response = workspace_client.post(
            "/api/ingest",
            json={
                "collection_id": "default",
                "source_name": "OakvilleHistory",
                "text": (
                    "Oakville is a small town in Northern Kentucky. "
                    "Residents developed a long tradition involving pies "
                    "after a severe winter disrupted normal food supplies."
                ),
            },
        )

        assert ingest_response.status_code == 200

        response = workspace_client.get(
            "/api/collections/default/boards"
        )

        assert response.status_code == 200

        body = response.json()

        assert body["collection_id"] == "default"
        assert body["total"] >= 1

        assert any(
            board["source_name"] == "OakvilleHistory"
            for board in body["boards"]
        )


def test_collection_workspace_rejects_unknown_collection() -> None:
    with TestClient(app) as workspace_client:
        response = workspace_client.get(
            "/api/collections/does-not-exist/boards"
        )

        assert response.status_code == 404


def test_local_query_api_returns_evidence() -> None:
    with TestClient(app) as query_client:
        ingest = query_client.post(
            "/api/ingest",
            json={
                "collection_id": "default",
                "source_name": "RandyRecords",
                "text": (
                    "Randy sells pies from his home in Oakville. "
                    "His dog is named Tom."
                ),
            },
        )

        assert ingest.status_code == 200

        response = query_client.post(
            "/api/query",
            json={
                "question": "Where does Randy sell pies?",
                "collection_id": "default",
                "allow_online": False,
            },
        )

        assert response.status_code == 200

        body = response.json()

        assert body["online_used"] is False
        assert body["evidence"]

        assert any(
            item["source_name"] == "RandyRecords"
            for item in body["evidence"]
        )

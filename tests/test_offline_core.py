"""Tests for GNOMEdata's portable offline core."""

from app.core.workspaces import WorkspaceStore
from app.rag.local_embedder import LocalHashEmbedder


def test_workspace_can_be_created_and_reopened(
    tmp_path,
) -> None:
    database = tmp_path / "workspaces.db"

    with WorkspaceStore(database) as store:
        workspace = store.create(
            "Oakville Archives",
            "Research into suspicious pie activity.",
        )

        assert workspace.workspace_id == (
            "oakville-archives"
        )

    with WorkspaceStore(database) as store:
        workspace = store.get(
            "oakville-archives"
        )

        assert workspace is not None
        assert workspace.name == "Oakville Archives"


def test_multiple_workspaces_are_isolated(
    tmp_path,
) -> None:
    with WorkspaceStore(
        tmp_path / "workspaces.db"
    ) as store:
        store.create("Company Financials")
        store.create("Oakville Research")

        names = {
            workspace.name
            for workspace in store.list()
        }

        assert names == {
            "Company Financials",
            "Oakville Research",
        }


def test_local_embedder_is_deterministic() -> None:
    embedder = LocalHashEmbedder()

    first = embedder.embed(
        "Randy sells pies in Oakville."
    )

    second = embedder.embed(
        "Randy sells pies in Oakville."
    )

    assert first.vector == second.vector
    assert first.model == second.model
    assert first.dimensions == 512


def test_local_embedder_requires_no_api_configuration(
    monkeypatch,
) -> None:
    monkeypatch.delenv(
        "HUGGINGFACE_TOKEN",
        raising=False,
    )

    embedder = LocalHashEmbedder()

    embedding = embedder.embed(
        "GNOMEdata works without the internet."
    )

    assert embedding.dimensions == 512


def test_local_embedder_runs_the_real_mill(
    tmp_path,
) -> None:
    from app.rag.boards import Log
    from app.rag.pipeline import MillPipeline
    from app.rag.yard import LumberYard

    with LumberYard(
        tmp_path / "yard.db"
    ) as yard:
        mill = MillPipeline(
            yard=yard,
            embedder=LocalHashEmbedder(),
        )

        report = mill.process(
            Log(
                log_id="oakville-log",
                source_name="OakvilleHistory",
                text=(
                    "Oakville contains several historical "
                    "records describing local businesses "
                    "and unusual pie traditions."
                ),
            )
        )

        assert report.total >= 1
        assert report.embedded >= 1

        query = LocalHashEmbedder().embed(
            "Oakville pie traditions"
        )

        results = yard.vector_search(
            query,
            limit=5,
        )

        assert results
        assert (
            results[0].board.source_name
            == "OakvilleHistory"
        )

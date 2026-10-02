
"""GNOMEdata API application."""

from contextlib import asynccontextmanager
import hashlib
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.rag.local_embedder import LocalHashEmbedder
from app.rag.yard import LumberYard


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own GNOMEdata's long-lived application resources."""

    load_dotenv()

    yard = LumberYard("data/gnomedata.db")
    embedder = LocalHashEmbedder()

    app.state.yard = yard
    app.state.embedder = embedder
    app.state.llm = MicroLLM()

    try:
        yield
    finally:
        yard.close()


app = FastAPI(
    title="GNOMEdata API",
    description=(
        "A document-grounded knowledge assistant "
        "for small businesses."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    """Return basic service information."""
    return {
        "name": "GNOMEdata",
        "status": "prototype",
        "message": "Welcome to the GNOMEdata API.",
    }


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Check whether the API is responding."""
    return {"status": "ok"}


app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static",
)

# ---------------------------------------------------------------------------
# Loading Dock
# ---------------------------------------------------------------------------

from typing import Annotated

from fastapi.responses import HTMLResponse
from pydantic import BaseModel, StringConstraints

from uuid import uuid4

from app.rag.boards import Log
from app.rag.pipeline import MillPipeline
from app.core.query import LocalQueryEngine
from app.llm import MicroLLM


NonEmptyText = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
    ),
]


class IngestRequest(BaseModel):
    """Raw information delivered to the GNOMEdata mill."""

    collection_id: NonEmptyText
    source_name: NonEmptyText
    text: NonEmptyText




class ChatRequest(BaseModel):
    """Message sent to the local GNOME."""

    message: NonEmptyText
    collection_id: NonEmptyText = "default"
    limit: int = 4


class QueryRequest(BaseModel):
    """A question sent to the local GNOMEdata brain."""

    question: NonEmptyText
    collection_id: NonEmptyText = "default"
    board_ids: list[str] | None = None
    allow_online: bool = False


@app.get(
    "/dock",
    response_class=HTMLResponse,
    tags=["loading-dock"],
)
def loading_dock() -> str:
    """Serve the human-facing GNOMEdata Loading Dock."""

    return Path("app/templates/index.html").read_text(encoding="utf-8")

@app.post(
    "/api/ingest",
    tags=["loading-dock"],
)
def ingest(
    request: IngestRequest,
) -> dict[str, str | int]:
    """Mill one load delivered through the Loading Dock."""

    yard = getattr(app.state, "yard", None)
    embedder = getattr(app.state, "embedder", None)

    if yard is None:
        raise RuntimeError(
            "GNOMEdata Lumber Yard is not configured."
        )

    if embedder is None:
        raise RuntimeError(
            "GNOMEdata embedder is not configured."
        )

    fingerprint = hashlib.sha256(
        request.text.encode("utf-8")
    ).hexdigest()

    if yard.has_ingestion(
        request.collection_id,
        fingerprint,
    ):
        return {
            "status": "duplicate",
            "source_name": request.source_name,
            "collection_id": request.collection_id,
            "log_id": "",
            "total": 0,
            "accepted": 0,
            "review": 0,
            "rejected": 0,
            "embedded": 0,
        }

    mill = MillPipeline(
        yard=yard,
        embedder=embedder,
        collection_id=request.collection_id,
    )

    log = Log(
        log_id=f"dock-{uuid4().hex}",
        source_name=request.source_name,
        text=request.text,
    )

    report = mill.process(log)

    yard.record_ingestion(
        request.collection_id,
        fingerprint,
        report.log_id,
        request.source_name,
    )

    return {
        "status": "milled",
        "source_name": report.source_name,
        "collection_id": request.collection_id,
        "log_id": report.log_id,
        "total": report.total,
        "accepted": report.accepted,
        "review": report.review,
        "rejected": report.rejected,
        "embedded": report.embedded,
    }


@app.delete(
    "/api/collections/{collection_id}/sources/{source_name}",
    tags=["collections"],
)
def delete_collection_source(
    collection_id: str,
    source_name: str,
) -> dict[str, str | int]:
    """Delete one source and all of its lumber from a collection."""

    yard = getattr(app.state, "yard", None)

    if yard is None:
        raise RuntimeError(
            "GNOMEdata Lumber Yard is not configured."
        )

    if yard.get_collection(collection_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Collection not found.",
        )

    deleted = yard.delete_source(
        source_name,
        collection_id=collection_id,
    )

    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail="Source not found.",
        )

    return {
        "status": "deleted",
        "collection_id": collection_id,
        "source_name": source_name,
        "boards_deleted": deleted,
    }


@app.get(
    "/api/collections/{collection_id}/boards",
    tags=["collections"],
)
def collection_boards(
    collection_id: str,
) -> dict:
    """Return the lumber stored in one collection."""

    yard = getattr(app.state, "yard", None)

    if yard is None:
        raise RuntimeError(
            "GNOMEdata Lumber Yard is not configured."
        )

    collection = yard.get_collection(collection_id)

    if collection is None:
        raise HTTPException(
            status_code=404,
            detail="Collection not found.",
        )

    boards = yard.list_boards(collection_id)

    return {
        "collection_id": collection.collection_id,
        "name": collection.name,
        "total": len(boards),
        "boards": [
            {
                "board_id": board.board_id,
                "log_id": board.log_id,
                "source_name": board.source_name,
                "grade": board.grade.value,
                "text": board.text,
                "start_offset": board.start_offset,
                "end_offset": board.end_offset,
            }
            for board in boards
        ],
    }


@app.post(
    "/api/chat",
    tags=["gnome"],
)
def chat(request: ChatRequest) -> dict:
    """Talk to the local GNOME using Lumber Yard context."""

    yard = getattr(app.state, "yard", None)
    llm = getattr(app.state, "llm", None)

    if yard is None:
        raise RuntimeError(
            "GNOMEdata Lumber Yard is not configured."
        )

    if llm is None:
        raise RuntimeError(
            "GNOMEdata local model is not configured."
        )

    if yard.get_collection(request.collection_id) is None:
        raise HTTPException(
            status_code=404,
            detail="Collection not found.",
        )

    limit = max(1, min(request.limit, 8))

    matches = yard.search(
        request.message,
        limit=limit,
    )

    context = tuple(
        match.board.text
        for match in matches
    )

    response = llm.generate(
        request.message,
        context=context,
    )

    return {
        "reply": response.text,
        "model": response.model,
        "context_used": response.context_used,
        "sources": [
            {
                "board_id": match.board.board_id,
                "source_name": match.board.source_name,
                "score": match.score,
            }
            for match in matches
        ],
    }


@app.post(
    "/api/query",
    tags=["query"],
)
def query_yard(
    request: QueryRequest,
) -> dict:
    """Search GNOMEdata locally and return ranked evidence."""

    yard = getattr(app.state, "yard", None)

    if yard is None:
        raise RuntimeError(
            "GNOMEdata Lumber Yard is not configured."
        )

    # Local is always the default query path.
    embedder = LocalHashEmbedder()

    engine = LocalQueryEngine(
        yard=yard,
        embedder=embedder,
    )

    try:
        result = engine.query(
            question=request.question,
            collection_id=request.collection_id,
            board_ids=(
                set(request.board_ids)
                if request.board_ids is not None
                else None
            ),
            allow_online=request.allow_online,
        )
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail=str(error),
        ) from error

    return {
        "question": result.question,
        "collection_id": result.collection_id,
        "online_used": result.online_used,
        "answer": result.answer,
        "facts": [
            {
                "subject": fact.subject,
                "relation": fact.relation,
                "value": fact.value,
                "source_name": fact.source_name,
                "board_id": fact.board_id,
            }
            for fact in result.facts
        ],
        "evidence": [
            {
                "board_id": item.board_id,
                "source_name": item.source_name,
                "text": item.text,
                "score": item.score,
            }
            for item in result.evidence
        ],
    }

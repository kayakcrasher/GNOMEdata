
"""GNOMEdata API application."""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.rag.huggingface_embedder import (
    HuggingFaceConfig,
    HuggingFaceEmbedder,
)
from app.rag.yard import LumberYard


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Own GNOMEdata's long-lived application resources."""

    load_dotenv()

    yard = LumberYard("data/gnomedata.db")
    config = HuggingFaceConfig.from_env()
    embedder = HuggingFaceEmbedder(config)

    app.state.yard = yard
    app.state.embedder = embedder

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


# ---------------------------------------------------------------------------
# Loading Dock
# ---------------------------------------------------------------------------

from typing import Annotated

from fastapi.responses import HTMLResponse
from pydantic import BaseModel, StringConstraints

from uuid import uuid4

from app.rag.boards import Log
from app.rag.pipeline import MillPipeline


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


@app.get(
    "/dock",
    response_class=HTMLResponse,
    tags=["loading-dock"],
)
def loading_dock() -> str:
    """Serve the human-facing GNOMEdata Loading Dock."""

    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>GNOMEdata Loading Dock</title>

    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            min-height: 100vh;
            font-family: system-ui, sans-serif;
            background: #111813;
            color: #edf4ed;
        }

        main {
            width: min(900px, 92%);
            margin: auto;
            padding: 48px 0;
        }

        .panel {
            padding: 28px;
            border: 1px solid #34463a;
            border-radius: 16px;
            background: #1b251e;
        }

        h1 {
            margin-top: 0;
        }

        .subtitle {
            color: #b8c7bb;
        }

        label {
            display: block;
            margin: 18px 0 6px;
            font-weight: 700;
        }

        input,
        textarea,
        button {
            width: 100%;
            font: inherit;
            border-radius: 8px;
        }

        input,
        textarea {
            padding: 12px;
            border: 1px solid #526557;
            background: #0e1510;
            color: #edf4ed;
        }

        textarea {
            min-height: 220px;
            resize: vertical;
        }

        button {
            margin-top: 20px;
            padding: 14px;
            border: 0;
            cursor: pointer;
            font-weight: 800;
        }

        button:disabled {
            cursor: wait;
            opacity: 0.65;
        }

        .mill {
            display: none;
            margin-top: 30px;
            padding: 22px;
            border: 1px solid #34463a;
            border-radius: 14px;
            background: #101712;
        }

        .mill.running,
        .mill.finished,
        .mill.failed {
            display: block;
        }

        .document {
            width: 54px;
            height: 66px;
            margin: 0 auto 18px;
            padding-top: 13px;
            border: 2px solid #dce7dd;
            border-radius: 5px;
            text-align: center;
            font-size: 26px;
            background: #e8efe9;
            color: #172019;
            transition:
                transform 0.35s ease,
                opacity 0.35s ease;
        }

        .document.moving {
            transform: translateY(8px) rotate(2deg);
        }

        .stations {
            display: grid;
            gap: 9px;
        }

        .station {
            padding: 12px 14px;
            border: 1px solid #34463a;
            border-radius: 8px;
            color: #829087;
            transition:
                transform 0.25s ease,
                background 0.25s ease,
                border-color 0.25s ease,
                color 0.25s ease;
        }

        .station.active {
            transform: scale(1.02);
            border-color: #d7e6da;
            background: #263329;
            color: #ffffff;
        }

        .station.done {
            border-color: #526b59;
            color: #bcd0c0;
        }

        .station.done::after {
            content: "  ✓";
        }

        .progress-shell {
            height: 12px;
            margin-top: 18px;
            overflow: hidden;
            border-radius: 999px;
            background: #29342c;
        }

        .progress-bar {
            width: 0%;
            height: 100%;
            background: #dce7dd;
            transition: width 0.35s ease;
        }

        .mill-status {
            margin-top: 12px;
            text-align: center;
            color: #b8c7bb;
        }

        #result {
            display: none;
            margin-top: 22px;
            padding: 18px;
            border: 1px solid #415247;
            border-radius: 10px;
            background: #111813;
        }

        #result.visible {
            display: block;
        }

        .result-title {
            margin: 0 0 16px;
        }

        .metrics {
            display: grid;
            grid-template-columns:
                repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin: 16px 0;
        }

        .metric {
            padding: 12px;
            border: 1px solid #34463a;
            border-radius: 8px;
        }

        .metric strong {
            display: block;
            font-size: 1.5rem;
        }

        .meta {
            overflow-wrap: anywhere;
            color: #b8c7bb;
        }

        .error {
            color: #ffd0d0;
        }
    </style>
</head>

<body>
<main>
    <section class="panel">
        <h1>GNOMEdata</h1>

        <p class="subtitle">
            Loading Dock — deliver information to the mill.
        </p>

        <form id="ingest-form">
            <label for="collection">
                Collection
            </label>

            <input
                id="collection"
                value="default"
                required
            >

            <label for="source">
                Source name
            </label>

            <input
                id="source"
                placeholder="manual.txt"
                required
            >

            <label for="text">
                Information
            </label>

            <textarea
                id="text"
                placeholder="Paste information here..."
                required
            ></textarea>

            <button id="mill-button" type="submit">
                MILL THIS
            </button>
        </form>

        <section id="mill" class="mill">
            <div id="document" class="document">📄</div>

            <div class="stations">
                <div class="station" data-stage="0">
                    🌲 Receiving Log
                </div>

                <div class="station" data-stage="1">
                    🪚 Cutting Boards
                </div>

                <div class="station" data-stage="2">
                    🔎 Grading
                </div>

                <div class="station" data-stage="3">
                    🪵 Stacking
                </div>

                <div class="station" data-stage="4">
                    🧠 Embedding
                </div>

                <div class="station" data-stage="5">
                    💾 Lumber Yard
                </div>
            </div>

            <div class="progress-shell">
                <div id="progress" class="progress-bar"></div>
            </div>

            <div id="mill-status" class="mill-status">
                Mill standing by.
            </div>
        </section>

        <section id="result"></section>
    </section>
</main>

<script>
const form = document.getElementById("ingest-form");
const mill = document.getElementById("mill");
const button = document.getElementById("mill-button");
const documentCard = document.getElementById("document");
const progress = document.getElementById("progress");
const millStatus = document.getElementById("mill-status");
const result = document.getElementById("result");

const stations = Array.from(
    document.querySelectorAll(".station")
);

const stageNames = [
    "Receiving source log...",
    "Cutting contextual boards...",
    "Inspecting board quality...",
    "Sorting finished lumber...",
    "Generating embedding vectors...",
    "Delivering lumber to the yard..."
];

let animationTimer = null;
let currentStage = 0;

function resetMill() {
    clearInterval(animationTimer);

    currentStage = 0;

    mill.className = "mill running";
    result.className = "";
    result.innerHTML = "";

    progress.style.width = "0%";

    for (const station of stations) {
        station.className = "station";
    }

    documentCard.className = "document";
}

function showStage(index) {
    stations.forEach((station, stationIndex) => {
        if (stationIndex < index) {
            station.className = "station done";
        } else if (stationIndex === index) {
            station.className = "station active";
        } else {
            station.className = "station";
        }
    });

    millStatus.textContent = stageNames[index];

    const percent = ((index + 1) / stations.length) * 90;
    progress.style.width = percent + "%";

    documentCard.classList.add("moving");

    setTimeout(() => {
        documentCard.classList.remove("moving");
    }, 220);
}

function startMillAnimation() {
    resetMill();
    showStage(0);

    animationTimer = setInterval(() => {
        if (currentStage < stations.length - 1) {
            currentStage += 1;
            showStage(currentStage);
        }
    }, 550);
}

function finishMill(body) {
    clearInterval(animationTimer);

    stations.forEach((station) => {
        station.className = "station done";
    });

    progress.style.width = "100%";
    millStatus.textContent = "Load stored successfully.";
    mill.className = "mill finished";

    result.className = "visible";

    result.innerHTML = `
        <h2 class="result-title">✓ LOAD MILLED</h2>

        <div class="metrics">
            <div class="metric">
                <strong>${body.total}</strong>
                Boards
            </div>

            <div class="metric">
                <strong>${body.accepted}</strong>
                Accepted
            </div>

            <div class="metric">
                <strong>${body.review}</strong>
                Review
            </div>

            <div class="metric">
                <strong>${body.rejected}</strong>
                Rejected
            </div>

            <div class="metric">
                <strong>${body.embedded}</strong>
                Embedded
            </div>
        </div>

        <div class="meta">
            <strong>Source:</strong>
            ${escapeHtml(body.source_name)}
            <br>

            <strong>Collection:</strong>
            ${escapeHtml(body.collection_id)}
            <br>

            <strong>Log:</strong>
            ${escapeHtml(body.log_id)}
        </div>
    `;
}

function failMill(body) {
    clearInterval(animationTimer);

    mill.className = "mill failed";
    progress.style.width = "0%";
    millStatus.textContent = "Mill stopped.";

    result.className = "visible error";
    result.textContent =
        "Mill rejected the load:\\n" +
        JSON.stringify(body, null, 2);
}

function escapeHtml(value) {
    const element = document.createElement("div");
    element.textContent = String(value ?? "");
    return element.innerHTML;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    button.disabled = true;
    button.textContent = "MILL RUNNING...";

    startMillAnimation();

    try {
        const response = await fetch("/api/ingest", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                collection_id:
                    document.getElementById("collection").value,
                source_name:
                    document.getElementById("source").value,
                text:
                    document.getElementById("text").value
            })
        });

        const body = await response.json();

        if (!response.ok) {
            failMill(body);
            return;
        }

        finishMill(body);
    } catch (error) {
        failMill({
            error: "Unable to reach GNOMEdata.",
            detail: String(error)
        });
    } finally {
        button.disabled = false;
        button.textContent = "MILL THIS";
    }
});
</script>
</body>
</html>
"""

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

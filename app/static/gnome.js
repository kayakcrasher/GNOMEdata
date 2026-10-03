document.body.setAttribute(
    "data-gnomedata-js",
    "alive"
);

const probe = document.createElement("div");
probe.id = "js-probe";
probe.textContent = "⚡ GNOMEDATA JAVASCRIPT IS RUNNING";
probe.style.cssText =
    "position:fixed;" +
    "top:10px;" +
    "left:10px;" +
    "right:10px;" +
    "z-index:99999;" +
    "padding:12px;" +
    "background:white;" +
    "color:black;" +
    "font-weight:bold;" +
    "text-align:center;";

document.body.appendChild(probe);

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
    loadWorkspace();
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


const workspaceSummary =
    document.getElementById("workspace-summary");

const boardGrid =
    document.getElementById("board-grid");

const refreshWorkspace =
    document.getElementById("refresh-workspace");


async function loadWorkspace() {
    const collectionId =
        document.getElementById("collection").value.trim();

    if (!collectionId) {
        return;
    }

    workspaceSummary.textContent =
        "Opening " + collectionId + "...";

    try {
        const response = await fetch(
            "/api/collections/" +
            encodeURIComponent(collectionId) +
            "/boards"
        );

        const body = await response.json();

        if (!response.ok) {
            workspaceSummary.textContent =
                "Collection unavailable.";

            boardGrid.innerHTML = "";
            return;
        }

        workspaceSummary.textContent =
            body.name +
            " — " +
            body.total +
            (body.total === 1 ? " board" : " boards");

        boardGrid.innerHTML = "";

        // ----------------------------------------------------
        // Human-facing document library
        // ----------------------------------------------------

        const sourceLibrary =
            document.getElementById("source-library");

        if (sourceLibrary) {
            sourceLibrary.innerHTML = "";

            const sources = new Map();

            for (const board of body.boards) {
                const sourceName =
                    board.source_name || "Unknown source";

                sources.set(
                    sourceName,
                    (sources.get(sourceName) || 0) + 1
                );
            }

            if (sources.size === 0) {
                const empty =
                    document.createElement("p");

                empty.className = "subtitle";
                empty.textContent =
                    "No documents stored yet.";

                sourceLibrary.appendChild(empty);
            }

            for (const [sourceName, count] of sources) {
                const card =
                    document.createElement("article");

                card.className = "source-card";

                const info =
                    document.createElement("div");

                const title =
                    document.createElement("h3");

                title.textContent =
                    "📄 " + sourceName;

                const meta =
                    document.createElement("p");

                meta.textContent =
                    count +
                    (count === 1
                        ? " searchable chunk"
                        : " searchable chunks");

                info.appendChild(title);
                info.appendChild(meta);

                const remove =
                    document.createElement("button");

                remove.type = "button";
                remove.className = "delete-source";
                remove.textContent = "DELETE";

                remove.addEventListener(
                    "click",
                    async () => {
                        const confirmed =
                            window.confirm(
                                'Remove "' +
                                sourceName +
                                '" from GNOMEdata?'
                            );

                        if (!confirmed) {
                            return;
                        }

                        remove.disabled = true;
                        remove.textContent =
                            "REMOVING...";

                        try {
                            const deleteResponse =
                                await fetch(
                                    "/api/collections/" +
                                    encodeURIComponent(
                                        collectionId
                                    ) +
                                    "/sources/" +
                                    encodeURIComponent(
                                        sourceName
                                    ),
                                    {
                                        method: "DELETE"
                                    }
                                );

                            const deleteBody =
                                await deleteResponse.json();

                            if (!deleteResponse.ok) {
                                throw new Error(
                                    deleteBody.detail ||
                                    "Delete failed."
                                );
                            }

                            await loadWorkspace();

                        } catch (error) {
                            console.error(
                                "GNOMEdata delete error:",
                                error
                            );

                            window.alert(
                                "Could not remove document.\n\n" +
                                String(error)
                            );

                            remove.disabled = false;
                            remove.textContent = "DELETE";
                        }
                    }
                );

                card.appendChild(info);
                card.appendChild(remove);

                sourceLibrary.appendChild(card);
            }
        }

        if (body.boards.length === 0) {
            boardGrid.innerHTML =
                '<p class="subtitle">' +
                'No lumber stored yet.' +
                '</p>';

            return;
        }

        for (const board of body.boards) {
            const card = document.createElement("article");
            card.className = "board-card selected";

            const preview =
                board.text.length > 180
                    ? board.text.slice(0, 180) + "..."
                    : board.text;

            card.innerHTML = `
                <div class="board-top">
                    <div>
                        <h3 class="board-source">
                            📄 ${escapeHtml(board.source_name)}
                        </h3>

                        <span class="board-grade">
                            ${escapeHtml(board.grade)}
                        </span>
                    </div>

                    <input
                        class="board-select"
                        type="checkbox"
                        checked
                        data-board-id="${escapeHtml(board.board_id)}"
                        aria-label="Select board"
                    >
                </div>

                <p class="board-preview">
                    ${escapeHtml(preview)}
                </p>
            `;

            const checkbox =
                card.querySelector(".board-select");

            checkbox.addEventListener("change", () => {
                card.classList.toggle(
                    "selected",
                    checkbox.checked
                );
            });

            boardGrid.appendChild(card);
        }
    } catch (error) {
        workspaceSummary.textContent =
            "Unable to load Lumber Yard: " +
            String(error);

        boardGrid.innerHTML = "";
        console.error(
            "GNOMEdata workspace error:",
            error
        );
    }
}


refreshWorkspace.addEventListener(
    "click",
    loadWorkspace
);


const queryButton =
    document.getElementById("query-button");

const queryResult =
    document.getElementById("query-result");

if (queryButton && queryResult) {
    queryResult.textContent =
        "⚡ QUERY CONTROLS ONLINE";
}


queryButton.addEventListener("click", async () => {
    const question =
        document.getElementById("query-question")
            .value
            .trim();

    if (!question) {
        queryResult.textContent =
            "Give the Yard a question first.";
        return;
    }

    const selectedBoards = Array.from(
        document.querySelectorAll(
            ".board-select:checked"
        )
    ).map(
        checkbox => checkbox.dataset.boardId
    );

    const collectionId =
        document.getElementById("collection")
            .value
            .trim();

    queryResult.textContent =
        "Searching local lumber...";

    let response;
    let body;

    try {
        response = await fetch(
            "/api/query",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    question: question,
                    collection_id: collectionId,
                    board_ids:
                        selectedBoards.length > 0
                            ? selectedBoards
                            : null,
                    allow_online:
                        document.getElementById(
                            "allow-online"
                        ).checked
                })
            }
        );

        body = await response.json();
    } catch (error) {
        queryResult.textContent =
            "Unable to reach the GNOMEdata query engine.\n" +
            String(error);

        return;
    }

    if (!response.ok) {
        queryResult.textContent =
            "Query failed:\n" +
            JSON.stringify(body, null, 2);
        return;
    }

    queryResult.innerHTML = "";

    const status = document.createElement("p");

    status.textContent =
        body.online_used
            ? "Online assistance used."
            : "LOCAL ONLY • No online assistance used.";

    queryResult.appendChild(status);

    if (!body.evidence.length) {
        const empty = document.createElement("p");
        empty.textContent =
            "No matching evidence found.";
        queryResult.appendChild(empty);
        return;
    }

    for (const evidence of body.evidence) {
        const card = document.createElement("article");
        card.className = "board-card";

        const title = document.createElement("h3");
        title.textContent =
            "📄 " + evidence.source_name;

        const score = document.createElement("p");
        score.textContent =
            "Relevance: " +
            evidence.score.toFixed(3);

        const text = document.createElement("p");
        text.textContent = evidence.text;

        card.appendChild(title);
        card.appendChild(score);
        card.appendChild(text);

        queryResult.appendChild(card);
    }
});

// ------------------------------------------------------------
// GNOMEdata UI startup
// ------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
    console.log("GNOMEdata UI online.");

    loadWorkspace();

    if (queryResult) {
        queryResult.textContent =
            "⚡ QUERY CONTROLS ONLINE";
    }
});

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


// ------------------------------------------------------------
// GNOME — local micro-LLM document assistant
// ------------------------------------------------------------

document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("gnome-form");
    const input = document.getElementById("gnome-input");
    const chat = document.getElementById("gnome-chat");
    const send = document.getElementById("gnome-send");
    const state = document.getElementById("gnome-state");
    const model = document.getElementById("gnome-model");
    const sourceList =
        document.getElementById("gnome-source-list");

    if (
        !form ||
        !input ||
        !chat ||
        !send ||
        !state ||
        !sourceList
    ) {
        console.error("GNOME UI controls missing.");
        return;
    }

    function setState(name) {
        state.textContent = name;
        state.className =
            "gnome-state " + name.toLowerCase();
    }

    function addMessage(speaker, text, type) {
        const message = document.createElement("div");

        message.className =
            "gnome-message gnome-message-" + type;

        const label = document.createElement("div");
        label.className = "message-speaker";
        label.textContent = speaker;

        const body = document.createElement("div");
        body.className = "message-body";
        body.textContent = text;

        message.appendChild(label);
        message.appendChild(body);

        chat.appendChild(message);
        chat.scrollTop = chat.scrollHeight;
    }

    function showSources(sources) {
        sourceList.innerHTML = "";

        if (!sources || sources.length === 0) {
            sourceList.textContent =
                "GNOME didn't need any stored boards.";
            return;
        }

        for (const source of sources) {
            const card = document.createElement("div");
            card.className = "gnome-source-card";

            const score =
                Number(source.score).toFixed(3);

            card.textContent =
                source.source_name +
                " • relevance " +
                score;

            sourceList.appendChild(card);
        }
    }

    form.addEventListener("submit", async event => {
        event.preventDefault();

        const message = input.value.trim();

        if (!message) {
            return;
        }

        const collection =
            document.getElementById("collection");

        const collectionId =
            collection && collection.value.trim()
                ? collection.value.trim()
                : "default";

        addMessage("YOU", message, "user");

        input.value = "";
        send.disabled = true;

        setState("SEARCHING");

        try {
            const response = await fetch(
                "/api/chat",
                {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        message: message,
                        collection_id: collectionId,
                        limit: 4
                    })
                }
            );

            setState("READING");

            const body = await response.json();

            if (!response.ok) {
                throw new Error(
                    body.detail ||
                    "GNOME could not answer."
                );
            }

            setState("THINKING");

            await new Promise(resolve =>
                setTimeout(resolve, 180)
            );

            addMessage(
                "GNOME",
                body.reply,
                "gnome"
            );

            showSources(body.sources);

            if (model && body.model) {
                model.textContent =
                    "LOCAL • " +
                    body.model +
                    " • " +
                    body.context_used +
                    " boards";
            }

            setState("READY");
        } catch (error) {
            addMessage(
                "GNOME",
                "Something jammed in the mill.\n" +
                String(error),
                "gnome"
            );

            setState("ERROR");
        } finally {
            send.disabled = false;
            input.focus();
        }
    });

    input.addEventListener("keydown", event => {
        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {
            event.preventDefault();
            form.requestSubmit();
        }
    });

    setState("READY");
});

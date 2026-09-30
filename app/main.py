
"""GNOMEdata API application."""

from fastapi import FastAPI

app = FastAPI(
    title="GNOMEdata API",
    description=(
        "A document-grounded knowledge assistant "
        "for small businesses."
    ),
    version="0.1.0",
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

"""Config-aware FastAPI entrypoint for the MindBridge platform."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI

from backend.routes import ingest_router
from core.config import load_active_config
from core.validation import validate_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and validate the active use-case profile before serving requests."""
    load_dotenv(REPO_ROOT / ".env")
    config = load_active_config()
    validate_config(config, base_dir=REPO_ROOT)
    app.state.config = config
    yield


app = FastAPI(
    title="MindBridge Platform API",
    description="Config-driven RAG platform — profile selected via USE_CASE_CONFIG",
    lifespan=lifespan,
)
app.include_router(ingest_router)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check; only reachable if config validation passed at startup."""
    config = app.state.config
    return {
        "status": "ok",
        "profile": config.name,
        "display_name": config.display_name,
    }

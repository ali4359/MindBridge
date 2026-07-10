"""MindBridge platform FastAPI app — config-driven lifespan and generic endpoints.

The active use case is selected by ``USE_CASE_CONFIG``. All behaviour is read from
the loaded YAML profile at startup so the same endpoints serve any domain.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI

from backend.routes.entity import router as entity_router
from backend.routes.ingest import router as ingest_router
from backend.routes.output import router as output_router
from backend.routes.platform import router as platform_router
from backend.routes.query import router as query_router
from core.agent import build_agent
from core.chain import build_chain
from core.config import load_active_config, resolve_config_path
from core.embeddings import get_embeddings
from core.graph import get_graph_driver
from core.llm import build_llm
from core.retriever import build_retriever
from core.router import route_query
from core.tools.cache import HybridCache
from core.tools.entity_tools import ENTITY_TOOLS, init_tools
from core.tools.source_system import SourceSystemInterface
from core.validation import validate_config
from core.vectorstore import documents_from_vectorstore, load_vectorstore

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
LIGHT_STARTUP_ENV = "MINDBRIDGE_LIGHT_STARTUP"


def _connect_neo4j() -> Optional[Any]:
    """Open a Neo4j driver; return ``None`` when the graph is unreachable."""
    try:
        driver = get_graph_driver()
        driver.verify_connectivity()
        return driver
    except Exception as exc:  # noqa: BLE001 — allow boot without a live graph
        logger.warning("Neo4j unavailable at startup: %s", exc)
        return None


def _light_startup(app: FastAPI) -> None:
    """Config-only boot used by unit tests (skip embeddings / Chroma / Neo4j)."""
    config_path = resolve_config_path()
    config = load_active_config()
    validate_config(config, base_dir=REPO_ROOT)
    app.state.config = config
    app.state.config_path = config_path
    app.state.embeddings = None
    app.state.vectorstore = None
    app.state.chunks = []
    app.state.chunk_count = 0
    app.state.llm = None
    app.state.retriever = None
    app.state.chain = None
    app.state.graph_driver = None
    app.state.ragas_scores = None
    app.state.source_system = None
    app.state.cache = None
    app.state.agent = None
    app.state.router = route_query
    app.state.started_at = time.time()
    logger.info(
        "MindBridge platform loaded — light startup [%s → %s]",
        config_path,
        config.display_name,
    )


def initialize_platform(app: FastAPI) -> int:
    """Wire the full RAG stack into ``app.state``. Returns indexed chunk count."""
    config_path = resolve_config_path()
    config = load_active_config()
    validate_config(config, base_dir=REPO_ROOT)

    embeddings = get_embeddings(config)
    vectorstore = load_vectorstore(config, base_dir=REPO_ROOT)
    chunk_count = vectorstore._collection.count()
    chunks = documents_from_vectorstore(vectorstore)

    llm = build_llm(config)
    retriever = build_retriever(config, vectorstore, chunks=chunks, llm=llm)
    chain = build_chain(config, retriever, llm)
    graph_driver = _connect_neo4j()

    # Vendor adapter injection lands in a later stage; inject None for now.
    source_system = SourceSystemInterface(None)
    cache = HybridCache(vectorstore=vectorstore)
    init_tools(source_system, cache, retriever)
    agent = build_agent(config, llm, source_system, cache, retriever=retriever)

    app.state.config = config
    app.state.config_path = config_path
    app.state.embeddings = embeddings
    app.state.vectorstore = vectorstore
    app.state.chunks = chunks
    app.state.chunk_count = chunk_count
    app.state.llm = llm
    app.state.retriever = retriever
    app.state.chain = chain
    app.state.graph_driver = graph_driver
    app.state.ragas_scores = None
    app.state.source_system = source_system
    app.state.cache = cache
    app.state.agent = agent
    app.state.router = route_query
    app.state.started_at = time.time()

    logger.info(
        "MindBridge platform loaded — %d chunks indexed. [%s → %s]",
        chunk_count,
        config_path,
        config.display_name,
    )
    logger.info("Agent initialised with %d tools", len(ENTITY_TOOLS))
    return chunk_count


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config, wire RAG stack into ``app.state``, tear down on shutdown."""
    load_dotenv(REPO_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if os.environ.get(LIGHT_STARTUP_ENV) == "1":
        _light_startup(app)
    else:
        initialize_platform(app)

    try:
        yield
    finally:
        driver = getattr(app.state, "graph_driver", None)
        if driver is not None:
            driver.close()
            app.state.graph_driver = None


def create_app() -> FastAPI:
    """Build the platform FastAPI application with all generic routes."""
    application = FastAPI(
        title="MindBridge Platform API",
        description=(
            "Config-driven RAG platform — active profile selected via USE_CASE_CONFIG. "
            "All endpoints are domain-agnostic."
        ),
        lifespan=lifespan,
    )
    application.include_router(query_router)
    application.include_router(output_router)
    application.include_router(ingest_router)
    application.include_router(entity_router)
    application.include_router(platform_router)
    return application


app = create_app()

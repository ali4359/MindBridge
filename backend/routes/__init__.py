"""FastAPI route modules."""

from backend.routes.entity import router as entity_router
from backend.routes.ingest import router as ingest_router
from backend.routes.output import router as output_router
from backend.routes.platform import router as platform_router
from backend.routes.query import router as query_router

__all__ = [
    "entity_router",
    "ingest_router",
    "output_router",
    "platform_router",
    "query_router",
]

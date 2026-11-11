"""FastAPI application factory."""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from utils.version import __version__
from web.backend.config import DEV_ORIGINS
from web.backend.routers import capabilities, dataset, graph, labels, perturbation
from web.backend.services import upload_service

app = FastAPI(
    title="Absynthe Web API",
    description=(
        "Interactive web interface for the Absynthe GNN explainability "
        "benchmarking framework."
    ),
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=DEV_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(capabilities.router, prefix="/api", tags=["capabilities"])
app.include_router(graph.router, prefix="/api/graph", tags=["graph"])
app.include_router(labels.router, prefix="/api/labels", tags=["labels"])
app.include_router(perturbation.router, prefix="/api/perturbation", tags=["perturbation"])
app.include_router(dataset.router, prefix="/api/dataset", tags=["dataset"])


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.on_event("shutdown")
async def _cleanup_temp_dirs() -> None:
    upload_service.cleanup_all()


# ── Serve built frontend in production ──────────────────────────────────────
_FRONTEND_DIST = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
)

if os.path.isdir(_FRONTEND_DIST):
    # Serve compiled assets directory
    _assets = os.path.join(_FRONTEND_DIST, "assets")
    if os.path.isdir(_assets):
        app.mount("/assets", StaticFiles(directory=_assets), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _spa(full_path: str) -> FileResponse:
        candidate = os.path.join(_FRONTEND_DIST, full_path)
        if os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))

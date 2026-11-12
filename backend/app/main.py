"""SmartYield AI - FastAPI application.

Run (from the project root):
    python -m uvicorn backend.app.main:app --reload --port 8010
Interactive API docs: http://localhost:8010/docs
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from . import db  # noqa: E402
from .routes import analyses, auth_routes, fields, insights  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="SmartYield AI API", version="1.0.0", lifespan=lifespan,
              description="Drone-based crop monitoring and yield estimation (computer vision + machine learning).")
app.add_middleware(CORSMiddleware,
                   allow_origins=os.environ.get("SMARTYIELD_CORS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
for r in (auth_routes.router, fields.router, analyses.router, insights.router):
    app.include_router(r)

# Serve the built React app (frontend/dist) when present -> single-process deployment.
DIST = ROOT / "frontend" / "dist"
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    _DIST_ROOT = DIST.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str):
        if path.startswith("api/") or path == "api":
            raise HTTPException(404, "Not found")
        f = (DIST / path).resolve()
        # only files inside frontend/dist may be served (blocks ../ traversal)
        if path and f.is_file() and f.is_relative_to(_DIST_ROOT):
            return FileResponse(f)
        return FileResponse(_DIST_ROOT / "index.html")

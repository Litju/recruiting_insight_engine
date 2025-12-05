"""
Main FastAPI App — Recruiting Insight Engine (Phase 3 UI)
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import logging
from pathlib import Path

from app.api.routes import router as api_router

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
logger = logging.getLogger("insight_engine_errors")
logger.setLevel(logging.ERROR)
handler = logging.FileHandler("errors.log")
handler.setLevel(logging.ERROR)
logger.addHandler(handler)

# ---------------------------------------------------------
# FASTAPI CONFIG
# ---------------------------------------------------------
app = FastAPI(
    title="Recruiting Insight Engine API",
    description="Core REST API for salary prediction and HR insight generation.",
    version="1.0.0",
)

# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# STATIC UI
# ---------------------------------------------------------
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/ui", response_class=HTMLResponse, summary="Analyst Console UI")
def serve_ui():
    ui_file = STATIC_DIR / "ui.html"
    if not ui_file.exists():
        return HTMLResponse("<h1>UI file not found</h1>", status_code=404)
    return ui_file.read_text(encoding="utf-8")

# ---------------------------------------------------------
# CUSTOM GLOBAL EXCEPTION HANDLER
# ---------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled error", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

# ---------------------------------------------------------
# ROUTERS
# ---------------------------------------------------------
app.include_router(api_router, prefix="/api")

# ---------------------------------------------------------
# ROOT ENDPOINT
# ---------------------------------------------------------
@app.get("/", summary="Health Check")
def root():
    return {"status": "ok", "service": "Recruiting Insight Engine API"}
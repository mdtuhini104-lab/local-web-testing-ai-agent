"""
FastAPI Server Entry Point (app/main.py).
Configures routes, WebSocket handlers, static asset serving, and database initialization on startup.
"""

from contextlib import asynccontextmanager
import sys
import os
import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.db import database
from app.api.routes import router as api_router
from app.api.websocket import ws_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite tables on startup
    await database.init_db()
    try:
        from agent_brain import preload_global_model
        asyncio.create_task(asyncio.to_thread(preload_global_model))
    except Exception as err:
        print(f"⚠️ Startup model pre-warm note: {err}")
    yield


app = FastAPI(
    title="Local Autonomous Web Testing AI Agent",
    description="100% Local AI-Powered Web App QA, Bug & UX Auditor",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST API
app.include_router(api_router)

# Mount Storage Directory (for serving screenshot artifacts)
os.makedirs(settings.STORAGE_DIR, exist_ok=True)
app.mount("/storage", StaticFiles(directory=settings.STORAGE_DIR), name="storage")

# Mount Dashboard Static Files
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def get_dashboard():
    """Serves the main frontend dashboard HTML."""
    index_file = os.path.join(static_dir, "index.html")
    return FileResponse(index_file, headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.websocket("/ws")
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str = None):
    """WebSocket endpoint for real-time live browser streaming, log streaming, and session switching."""
    await ws_manager.connect(websocket, session_id=session_id)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, session_id=session_id)


if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)

"""FastAPI application entrypoint."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

from app.api import routes_admin, routes_auth, routes_batches, routes_reports, routes_twilio
from app.config import get_settings
from app.db import Mongo
from app.logging_config import log, setup_logging
from app.services.dialer import Dialer
from app.services.llm import GroqLLM
from app.services.telephony import Telephony

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    await Mongo.connect(settings)
    # Shared, long-lived clients (connection pools are reused across calls)
    app.state.settings = settings
    app.state.llm = GroqLLM(settings)
    app.state.telephony = Telephony(settings)
    app.state.dial_semaphore = asyncio.Semaphore(settings.max_concurrent_outbound_calls)
    app.state.dialer = Dialer(settings, app.state.telephony)
    app.state.dialer.start()  # works through queued calls from CSV uploads
    log(logger, logging.INFO, "startup_complete", env=settings.environment)
    yield
    await app.state.dialer.stop()
    await app.state.llm.close()
    await Mongo.close()


app = FastAPI(
    title="Multilingual Voice Sales Agent",
    version="1.0.0",
    description="Outbound AI sales calls: Twilio + Sarvam STT/TTS + Groq + MongoDB",
    lifespan=lifespan,
)
app.include_router(routes_admin.router)
app.include_router(routes_twilio.router)
app.include_router(routes_reports.api_router)
app.include_router(routes_reports.public_router)
app.include_router(routes_batches.router)
app.include_router(routes_auth.router)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
async def upload_page():
    """The daily CSV upload dashboard."""
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok"}


@app.get("/ready", tags=["ops"])
async def ready():
    try:
        await Mongo.client.admin.command("ping")
        return {"status": "ready"}
    except Exception as exc:
        return JSONResponse({"status": "not_ready", "error": str(exc)}, status_code=503)

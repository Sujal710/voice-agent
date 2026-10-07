"""Twilio webhooks and the bidirectional Media Streams WebSocket."""

import asyncio
import json
import logging

from fastapi import APIRouter, Depends, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app import repositories as repo
from app.logging_config import log
from app.pipeline.call_session import CallSession, run_in_background
from app.pipeline.report import create_report_for_call
from app.security import verify_twilio_request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/twilio", tags=["twilio"])

TERMINAL = {"completed", "busy", "no-answer", "failed", "canceled"}


@router.post("/voice")
async def voice_webhook(request: Request, call_id: str = Query(...), form: dict = Depends(verify_twilio_request)):
    """Called by Twilio when the customer answers. Returns TwiML that opens the media stream."""
    telephony = request.app.state.telephony
    if form.get("AnsweredBy", "").startswith("machine"):
        await repo.update_call(call_id, {"status": "completed", "answered_by": form["AnsweredBy"]})
        return Response(
            '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>', media_type="application/xml"
        )
    return Response(telephony.stream_twiml(call_id), media_type="application/xml")


async def _ensure_report(llm, call_id: str, delay_s: float = 20) -> None:
    await asyncio.sleep(delay_s)  # give the media-stream report (if any) time to finish first
    call = await repo.get_call(call_id)
    if call and not call.get("report_ready"):
        await create_report_for_call(llm, call_id)


@router.post("/status")
async def status_webhook(request: Request, call_id: str = Query(...), form: dict = Depends(verify_twilio_request)):
    """Twilio call status callbacks (ringing, answered, completed, ...)."""
    call_status = form.get("CallStatus", "")
    fields = {"status": call_status}
    if form.get("CallDuration"):
        fields["duration_s"] = int(form["CallDuration"])
    if call_status in TERMINAL:
        fields["ended_at"] = repo.now()
    await repo.update_call(call_id, fields)
    if call_status in TERMINAL:
        # Answered calls get their report when the media stream closes. For calls that
        # never connected (no answer, busy, voicemail) save a report here as well.
        run_in_background(_ensure_report(request.app.state.llm, call_id))
    log(logger, logging.INFO, "call_status", call_id=call_id, status=call_status)
    return Response(status_code=204)


@router.websocket("/media-stream")
async def media_stream(ws: WebSocket):
    """Bidirectional audio with Twilio. One WebSocket == one live phone call."""
    await ws.accept()
    app = ws.app
    session = CallSession(ws=ws, settings=app.state.settings, llm=app.state.llm, telephony=app.state.telephony)
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            event = msg.get("event")
            if event == "media":
                # Only the caller's audio is forwarded to STT
                if msg["media"].get("track", "inbound") == "inbound":
                    await session.on_media(msg["media"]["payload"])
            elif event == "start":
                if not await session.on_start(msg["start"]):
                    await ws.close(code=1008)
                    return
            elif event == "mark":
                await session.on_mark(msg.get("mark", {}).get("name", ""))
            elif event == "stop":
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        log(logger, logging.ERROR, "media_stream_error", call_id=session.call_id, error=repr(exc))
    finally:
        await session.close()

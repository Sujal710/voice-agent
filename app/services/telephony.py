"""Twilio REST helpers.

The Twilio Python SDK is synchronous (blocking HTTP). Calling it directly
inside `async def` would freeze the event loop and stall every live call,
so each call is pushed to a worker thread with `asyncio.to_thread`.
"""

import asyncio
import logging

from twilio.rest import Client
from twilio.twiml.voice_response import Connect, VoiceResponse

from app.config import Settings
from app.logging_config import log
from app.security import stream_token

logger = logging.getLogger(__name__)


class Telephony:
    def __init__(self, settings: Settings):
        self.s = settings
        self._client: Client | None = None

    @property
    def client(self) -> Client:
        if self._client is None:
            self._client = Client(self.s.twilio_account_sid, self.s.twilio_auth_token)
        return self._client

    def _base(self) -> str:
        return self.s.public_base_url.rstrip("/")

    async def place_call(self, call_id: str, to_number: str) -> str:
        base = self._base()

        def _create() -> str:
            call = self.client.calls.create(
                to=to_number,
                from_=self.s.twilio_from_number,
                url=f"{base}/twilio/voice?call_id={call_id}",
                method="POST",
                status_callback=f"{base}/twilio/status?call_id={call_id}",
                status_callback_method="POST",
                status_callback_event=["initiated", "ringing", "answered", "completed"],
                time_limit=self.s.max_call_duration_s,
                machine_detection="Enable",  # lets us skip voicemail if desired
            )
            return call.sid

        sid = await asyncio.to_thread(_create)
        log(logger, logging.INFO, "call_placed", call_id=call_id, call_sid=sid)
        return sid

    async def hangup(self, call_sid: str) -> None:
        def _end() -> None:
            self.client.calls(call_sid).update(status="completed")

        try:
            await asyncio.to_thread(_end)
        except Exception as exc:
            log(logger, logging.WARNING, "hangup_failed", call_sid=call_sid, error=str(exc))

    def stream_twiml(self, call_id: str) -> str:
        """TwiML that connects the answered call to our bidirectional media WebSocket."""
        ws_base = self._base().replace("https://", "wss://").replace("http://", "ws://")
        response = VoiceResponse()
        connect = Connect()
        stream = connect.stream(url=f"{ws_base}/twilio/media-stream")
        stream.parameter(name="call_id", value=call_id)
        stream.parameter(name="token", value=stream_token(call_id, self.s))
        response.append(connect)
        return str(response)

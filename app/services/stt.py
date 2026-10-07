"""Sarvam realtime streaming STT (saaras:v4) over WebSocket.

Twilio delivers 8 kHz mu-law audio and Sarvam accepts `encoding=mulaw,
sample_rate=8000` directly, so caller audio is forwarded with zero
transcoding. Sarvam's server-side VAD (`endpointing=vad`) decides when the
caller has finished a turn and emits `transcript.final` with the detected
language (because `language_code=auto`).
"""

import asyncio
import base64
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass

from sarvamai import AsyncSarvamAI, RealtimeAudioInput, RealtimeEnd

from app.config import Settings
from app.logging_config import log

logger = logging.getLogger(__name__)

# Overridable for tests (points the SDK at a local fake server)
SARVAM_ENVIRONMENT = None


@dataclass
class STTEvent:
    kind: str  # "speech_start" | "speech_end" | "partial" | "final" | "error"
    text: str = ""
    language: str | None = None
    language_confidence: float | None = None
    fatal: bool = False


EventHandler = Callable[[STTEvent], Awaitable[None]]


class SarvamRealtimeSTT:
    """One instance per phone call."""

    def __init__(self, settings: Settings, on_event: EventHandler, call_id: str):
        self.s = settings
        self.on_event = on_event
        self.call_id = call_id
        kwargs = {"environment": SARVAM_ENVIRONMENT} if SARVAM_ENVIRONMENT else {}
        self._client = AsyncSarvamAI(api_subscription_key=settings.sarvam_api_key, **kwargs)
        self._stack: AsyncExitStack | None = None
        self._ws = None
        self._reader: asyncio.Task | None = None
        self._closed = False

    async def start(self) -> None:
        self._stack = AsyncExitStack()
        params = dict(
            model=self.s.stt_model,
            language_code=self.s.stt_language_code,
            mode=self.s.stt_mode,
            stream_type=self.s.stt_stream_type,
            endpointing="vad",
            encoding="mulaw",  # Twilio native format, no conversion needed
            sample_rate="8000",
            threshold=str(self.s.stt_vad_threshold),
            silence_duration_ms=str(self.s.stt_silence_duration_ms),
            min_speech_duration_ms=str(self.s.stt_min_speech_duration_ms),
        )
        if self.s.stt_keyterms and self.s.stt_model == "saaras:v4":
            params["keyterms"] = json.dumps(self.s.stt_keyterms[:50])
        self._ws = await self._stack.enter_async_context(
            self._client.speech_to_text_realtime_streaming.connect(**params)
        )
        self._reader = asyncio.create_task(self._read_loop(), name=f"stt-reader-{self.call_id}")
        log(logger, logging.INFO, "stt_connected", call_id=self.call_id)

    async def send_audio(self, mulaw: bytes) -> None:
        if self._ws is None or self._closed:
            return
        try:
            await self._ws.send_realtime_audio_input(RealtimeAudioInput(audio=base64.b64encode(mulaw).decode("ascii")))
        except Exception as exc:
            if not self._closed:
                log(logger, logging.WARNING, "stt_send_failed", call_id=self.call_id, error=str(exc))
                self._closed = True
                await self.on_event(STTEvent(kind="error", text=str(exc), fatal=True))

    async def _read_loop(self) -> None:
        try:
            async for event in self._ws:
                name = getattr(event, "event", None)
                if name == "vad.speech_start":
                    await self.on_event(STTEvent(kind="speech_start"))
                elif name == "vad.speech_end":
                    await self.on_event(STTEvent(kind="speech_end"))
                elif name == "transcript.partial":
                    await self.on_event(STTEvent(kind="partial", text=event.text, language=event.language))
                elif name == "transcript.final":
                    await self.on_event(
                        STTEvent(
                            kind="final",
                            text=(event.text or "").strip(),
                            language=event.language,
                            language_confidence=event.language_confidence,
                        )
                    )
                elif name == "error":
                    log(
                        logger,
                        logging.WARNING,
                        "stt_error",
                        call_id=self.call_id,
                        code=event.code,
                        message=event.message,
                    )
                    await self.on_event(STTEvent(kind="error", text=event.message, fatal=event.is_fatal))
                    if event.is_fatal:
                        return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._closed:
                log(logger, logging.WARNING, "stt_read_failed", call_id=self.call_id, error=str(exc))
                await self.on_event(STTEvent(kind="error", text=str(exc), fatal=True))

    async def close(self) -> None:
        if self._closed and self._stack is None:
            return
        self._closed = True
        try:
            if self._ws is not None:
                await asyncio.wait_for(self._ws.send_realtime_end(RealtimeEnd()), timeout=1)
        except Exception:
            pass
        if self._reader:
            self._reader.cancel()
        if self._stack:
            try:
                await self._stack.aclose()
            except Exception:
                pass
            self._stack = None

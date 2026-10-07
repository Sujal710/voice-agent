"""Sarvam Bulbul v3 streaming TTS over WebSocket.

Latency tricks used here:
* Audio is requested as raw `linear16` at 8 kHz (telephony rate), so the
  only post-processing is a cheap numpy mu-law encode for Twilio.
* Each call keeps one *pre-warmed* TTS socket open (TCP + TLS + WS handshake
  already done), so the first sentence of a reply starts synthesising
  immediately. After a socket is used, the next one is warmed in background.
* On barge-in the in-flight socket is simply closed (Sarvam's recommended way
  to cancel) and the pre-warmed one takes over.
"""

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator

import websockets
from websockets.asyncio.client import ClientConnection, connect

from app.config import Settings
from app.logging_config import log
from app.services.audio import strip_wav_header

logger = logging.getLogger(__name__)

SARVAM_TTS_WS = "wss://api.sarvam.ai/text-to-speech/ws"
TTS_SAMPLE_RATE = 8000


class SarvamTTSConnection:
    """A single TTS WebSocket session (used for one agent turn)."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.ws: ClientConnection | None = None
        self._flushes = 0
        self._finals = 0
        self._input_done = asyncio.Event()
        self._configured_lang: str | None = None
        self._keepalive: asyncio.Task | None = None

    async def open(self) -> "SarvamTTSConnection":
        url = f"{SARVAM_TTS_WS}?model={self.s.tts_model}&send_completion_event=true"
        self.ws = await connect(
            url,
            additional_headers={"Api-Subscription-Key": self.s.sarvam_api_key},
            open_timeout=5,
            compression=None,  # audio is incompressible; skip deflate CPU cost
            max_size=None,
        )
        self._keepalive = asyncio.create_task(self._ping_loop())
        return self

    async def _ping_loop(self) -> None:
        # Sarvam closes idle sockets after ~60 s; keep the pre-warmed one alive.
        try:
            while True:
                await asyncio.sleep(25)
                await self._send({"type": "ping"})
        except (asyncio.CancelledError, Exception):
            return

    async def _send(self, msg: dict) -> None:
        assert self.ws is not None
        await self.ws.send(json.dumps(msg, ensure_ascii=False))

    async def configure(self, language_code: str) -> None:
        if self._configured_lang == language_code:
            return
        await self._send(
            {
                "type": "config",
                "data": {
                    "language_code": language_code,
                    "speaker": self.s.tts_speaker,
                    "pace": self.s.tts_pace,
                    "temperature": self.s.tts_temperature,
                    "speech_sample_rate": TTS_SAMPLE_RATE,
                    "output_audio_codec": "linear16",
                    "min_buffer_size": self.s.tts_min_buffer_size,
                    "max_chunk_length": self.s.tts_max_chunk_length,
                },
            }
        )
        self._configured_lang = language_code

    async def speak(self, text: str) -> None:
        """Send one sentence and flush so synthesis starts right away."""
        text = text.strip()
        if not text:
            return
        await self._send({"type": "text", "data": {"text": text}})
        await self._send({"type": "flush"})
        self._flushes += 1

    def end_input(self) -> None:
        """No more text will be sent for this turn."""
        self._input_done.set()

    async def audio(self, idle_timeout: float = 4.0) -> AsyncIterator[bytes]:
        """Yield raw PCM16 (8 kHz mono) chunks until the turn is fully synthesised."""
        assert self.ws is not None
        carry = b""
        while True:
            # Done when caller finished sending text and every flush is complete
            if self._input_done.is_set() and self._finals >= self._flushes:
                break
            try:
                raw = await asyncio.wait_for(self.ws.recv(), timeout=idle_timeout)
            except TimeoutError:
                if self._input_done.is_set():
                    break  # server didn't send completion events; assume done
                continue
            except websockets.ConnectionClosed:
                break
            msg = json.loads(raw)
            mtype = msg.get("type")
            if mtype == "audio":
                pcm = carry + strip_wav_header(base64.b64decode(msg["data"]["audio"]))
                if len(pcm) % 2:  # keep 16-bit sample alignment across chunks
                    pcm, carry = pcm[:-1], pcm[-1:]
                else:
                    carry = b""
                if pcm:
                    yield pcm
            elif mtype == "event" and msg.get("data", {}).get("event_type") == "final":
                self._finals += 1
            elif mtype == "error":
                log(logger, logging.WARNING, "tts_error", error=msg.get("data"))
                break

    async def close(self) -> None:
        if self._keepalive:
            self._keepalive.cancel()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None


class TTSPool:
    """Keeps one warm TTS connection ready for the next agent turn."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._next: asyncio.Task | None = None

    def warm(self) -> None:
        if self._next is None or (self._next.done() and self._next.exception() is not None):
            self._next = asyncio.create_task(SarvamTTSConnection(self.s).open())

    async def acquire(self, language_code: str) -> SarvamTTSConnection:
        self.warm()
        task, self._next = self._next, None
        try:
            conn = await task
        except Exception as exc:
            log(logger, logging.WARNING, "tts_prewarm_failed_retrying", error=str(exc))
            conn = await SarvamTTSConnection(self.s).open()
        if conn.ws is None or conn.ws.state.name != "OPEN":
            conn = await SarvamTTSConnection(self.s).open()
        await conn.configure(language_code)
        self.warm()  # get the next one ready in the background
        return conn

    async def close(self) -> None:
        if self._next is not None:
            if self._next.done() and not self._next.cancelled() and self._next.exception() is None:
                await self._next.result().close()
            else:
                self._next.cancel()
            self._next = None

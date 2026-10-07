"""Per-call orchestrator: Twilio <-> Sarvam STT <-> Groq <-> Sarvam TTS.

Turn pipeline (everything is streamed and overlapped):

  caller audio (mu-law 8k) ──► Sarvam STT (VAD + auto language)
        transcript.final (e.g. gu-IN)
            ──► Groq translate gu→en
            ──► Groq sales agent (English, token stream)
            ──► sentence chunker
            ──► Groq translate en→gu   (per sentence, in parallel, order kept)
            ──► Sarvam Bulbul v3 TTS   (per sentence, streamed)
            ──► PCM16 → mu-law → 20 ms frames ──► Twilio ──► caller

The caller hears sentence 1 while sentences 2..n are still being generated,
translated and synthesised. If the caller starts talking (barge-in), the
running turn is cancelled, Twilio's buffered audio is cleared, and the TTS
socket is dropped.
"""

import asyncio
import base64
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field

from fastapi import WebSocket

from app import repositories as repo
from app.config import Settings
from app.logging_config import log
from app.pipeline.memory import ConversationMemory
from app.pipeline.report import create_report_for_call
from app.pipeline.text import END_CALL_TOKEN, SentenceChunker, clean_for_speech, extract_followups
from app.security import verify_stream_token
from app.services.audio import FrameBuffer, pcm16_to_mulaw
from app.services.languages import is_english, normalize, to_tts_language
from app.services.llm import GroqLLM, build_sales_system_prompt
from app.services.stt import SarvamRealtimeSTT, STTEvent
from app.services.telephony import Telephony
from app.services.tts import SarvamTTSConnection, TTSPool

logger = logging.getLogger(__name__)

GREETING_EN = (
    "Hello {name}, this is {agent} calling from {company}. I'm calling about {product}. Do you have a minute to talk?"
)
NUDGE_EN = "Hello, are you still there?"
GOODBYE_EN = "It seems this isn't a good time. Thank you, have a great day!"

# Background tasks must be referenced or the GC may cancel them mid-flight
_background: set[asyncio.Task] = set()


def run_in_background(coro) -> None:
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


@dataclass
class TurnMetrics:
    turn: int
    language: str
    stt_final_at: float
    translate_in_ms: float | None = None
    llm_first_token_ms: float | None = None
    first_audio_ms: float | None = None  # STT final -> first audio frame sent to Twilio
    interrupted: bool = False

    def as_dict(self) -> dict:
        return {
            k: (round(v, 1) if isinstance(v, float) else v) for k, v in self.__dict__.items() if k != "stt_final_at"
        }


@dataclass
class CallSession:
    ws: WebSocket
    settings: Settings
    llm: GroqLLM
    telephony: Telephony

    call_id: str = ""
    call_sid: str = ""
    stream_sid: str = ""
    customer: dict = field(default_factory=dict)
    product: dict = field(default_factory=dict)
    language: str = "en-IN"
    system_prompt: str = ""
    memory: ConversationMemory | None = None  # LangChain-style summary-buffer chat history
    metrics: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stt: SarvamRealtimeSTT | None = None
        self.tts_pool = TTSPool(self.settings)
        self._send_lock = asyncio.Lock()
        self._turn_task: asyncio.Task | None = None
        self._turn_tts: SarvamTTSConnection | None = None
        self._turn_no = 0
        self._pending_mark: str | None = None
        self._agent_speaking = False
        self._hangup_after_mark: str | None = None
        self._last_activity = time.monotonic()
        self._nudges = 0
        self._watchdog: asyncio.Task | None = None
        self._closed = False

    # ------------------------------------------------------------------ Twilio I/O
    async def _send_json(self, payload: dict) -> None:
        if self._closed:
            return
        async with self._send_lock:
            with contextlib.suppress(Exception):
                await self.ws.send_text(json.dumps(payload))

    async def _send_audio(self, mulaw: bytes) -> None:
        await self._send_json(
            {"event": "media", "streamSid": self.stream_sid, "media": {"payload": base64.b64encode(mulaw).decode()}}
        )

    async def _send_mark(self, name: str) -> None:
        await self._send_json({"event": "mark", "streamSid": self.stream_sid, "mark": {"name": name}})

    async def _clear_twilio_audio(self) -> None:
        await self._send_json({"event": "clear", "streamSid": self.stream_sid})

    # ------------------------------------------------------------------ lifecycle
    async def on_start(self, start: dict) -> bool:
        """Handle Twilio's `start` message. Returns False if the stream must be rejected."""
        params = start.get("customParameters", {}) or {}
        self.call_id = params.get("call_id", "")
        if not verify_stream_token(self.call_id, params.get("token", ""), self.settings):
            log(logger, logging.WARNING, "stream_token_invalid", call_id=self.call_id)
            return False

        self.stream_sid = start["streamSid"]
        self.call_sid = start.get("callSid", "")
        call = await repo.get_call(self.call_id)
        if not call:
            return False
        self.customer = await repo.get_customer(call["customer_id"]) or {}
        self.product = await repo.get_product(call["product_sku"]) or {}
        self.language = normalize(self.customer.get("preferred_language"), self.settings.default_language)
        self.system_prompt = build_sales_system_prompt(
            self.customer, self.product, self.settings.company_name, self.settings.agent_name
        )
        self.memory = ConversationMemory(
            session_id=self.call_id,
            customer_id=call["customer_id"],
            summarize=self.llm.summarize_memory,
            persist=repo.append_message,
            max_recent_tokens=self.settings.memory_max_recent_tokens,
            past_calls=await repo.recent_call_reports(
                call["customer_id"], self.call_id, self.settings.memory_past_calls
            ),
        )

        # Open STT and warm TTS concurrently: both are network handshakes
        self.stt = SarvamRealtimeSTT(self.settings, self._on_stt_event, self.call_id)
        self.tts_pool.warm()
        await self.stt.start()
        await repo.update_call(
            self.call_id, {"status": "in-progress", "twilio_call_sid": self.call_sid, "language": self.language}
        )

        greeting = GREETING_EN.format(
            name=self.customer.get("name", "there"),
            agent=self.settings.agent_name,
            company=self.settings.company_name,
            product=self.product.get("name", "our product"),
        )
        self._start_turn(self._speak_fixed(greeting))
        self._watchdog = asyncio.create_task(self._silence_watchdog())
        log(logger, logging.INFO, "call_started", call_id=self.call_id, language=self.language)
        return True

    async def on_media(self, payload_b64: str) -> None:
        if self.stt:
            await self.stt.send_audio(base64.b64decode(payload_b64))

    async def on_mark(self, name: str) -> None:
        if name == self._pending_mark:
            self._agent_speaking = False
            self._pending_mark = None
            self._last_activity = time.monotonic()
        if self._hangup_after_mark and name == self._hangup_after_mark:
            run_in_background(self.telephony.hangup(self.call_sid))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._watchdog:
            self._watchdog.cancel()
        await self._cancel_turn(clear_twilio=False)
        if self.stt:
            await self.stt.close()
        await self.tts_pool.close()
        if self.call_id:
            run_in_background(self._finalize())
        log(logger, logging.INFO, "call_closed", call_id=self.call_id, turns=self._turn_no)

    async def _finalize(self) -> None:
        """After hang-up: save metrics + memory, then generate the verified call report."""
        if self.memory:
            await self.memory.drain()  # make sure every message is in MongoDB first
        await repo.update_call(
            self.call_id,
            {
                "metrics": {"turns": self.metrics},
                "language": self.language,
                "memory_summary": self.memory.summary if self.memory else None,
                "ended_at": repo.now(),
            },
        )
        report = await create_report_for_call(self.llm, self.call_id)
        log(logger, logging.INFO, "report_ready", call_id=self.call_id, outcome=(report or {}).get("outcome"))
        if self.customer.get("id"):
            await repo.set_customer_language(self.customer["id"], self.language)
            if report and report.get("outcome") == "do_not_call":
                await repo.mark_do_not_call(self.customer["id"])

    # ------------------------------------------------------------------ STT events
    async def _on_stt_event(self, ev: STTEvent) -> None:
        if ev.kind in ("speech_start", "partial"):
            self._last_activity = time.monotonic()
            self._nudges = 0
            # Barge-in: caller talks over the agent -> stop speaking immediately.
            # Partials (real words) are used rather than raw VAD so a cough
            # or background noise doesn't cut the agent off.
            if ev.kind == "partial" and self._agent_speaking and len(ev.text.strip()) >= 3:
                log(logger, logging.INFO, "barge_in", call_id=self.call_id, partial=ev.text)
                await self._cancel_turn(clear_twilio=True)
        elif ev.kind == "final" and ev.text:
            self._last_activity = time.monotonic()
            # Switch language only when the detector is confident
            if ev.language and (
                ev.language_confidence is None or ev.language_confidence >= self.settings.stt_min_language_confidence
            ):
                self.language = normalize(ev.language)
            self._start_turn(self._agent_turn(ev.text, self.language, time.perf_counter()))
        elif ev.kind == "error" and ev.fatal:
            log(logger, logging.ERROR, "stt_fatal", call_id=self.call_id, error=ev.text)
            run_in_background(self.telephony.hangup(self.call_sid))

    # ------------------------------------------------------------------ turns
    def _start_turn(self, coro) -> None:
        """Cancel whatever the agent is doing and start a new turn.

        The new turn waits for the old one to finish unwinding so the chat
        history stays in order (old reply is recorded before the new question).
        """
        previous = self._turn_task
        if previous and not previous.done():
            previous.cancel()

        async def runner():
            try:
                if previous is not None:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await previous
                await self._cancel_turn(clear_twilio=True)
            except asyncio.CancelledError:
                coro.close()  # never started; avoid "coroutine was never awaited"
                raise
            await coro

        self._turn_task = asyncio.create_task(runner())

    async def _cancel_turn(self, clear_twilio: bool) -> None:
        current = asyncio.current_task()
        task = self._turn_task
        if task and not task.done() and task is not current:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        if self._turn_tts:
            await self._turn_tts.close()
            self._turn_tts = None
        if clear_twilio and self._agent_speaking:
            await self._clear_twilio_audio()
        self._agent_speaking = False
        self._pending_mark = None

    async def _speak_fixed(self, text_en: str, end_call: bool = False) -> None:
        """Speak a scripted English line (greeting, nudge) in the caller's language."""
        lang = self.language
        text = text_en if is_english(lang) else await self.llm.translate(text_en, "en-IN", lang)
        # Recorded in memory too, so the LLM knows what the customer already heard
        self.memory.add("agent", text_en, text, lang, repo.now(), kind="scripted")

        async def sentences():
            yield text

        await self._synthesize(sentences(), lang, metrics=None, end_call=end_call)

    async def _agent_turn(self, customer_text: str, lang: str, t0: float) -> None:
        self._turn_no += 1
        m = TurnMetrics(turn=self._turn_no, language=lang, stt_final_at=t0)

        # 1) customer's words -> English for the LLM
        text_en = customer_text
        if not is_english(lang):
            text_en = await self.llm.translate(customer_text, lang, "en-IN")
            m.translate_in_ms = (time.perf_counter() - t0) * 1000
        self.memory.add("customer", text_en, customer_text, lang, repo.now())
        log(logger, logging.INFO, "customer_said", call_id=self.call_id, lang=lang, text=customer_text, text_en=text_en)

        # 2) English reply stream -> sentences -> translated sentences (order preserved)
        reply_parts: list[str] = []
        spoken_parts: list[str] = []  # what the customer actually heard (their language)
        end_call = False

        async def translated_sentences():
            nonlocal end_call
            chunker = SentenceChunker()
            pending: asyncio.Queue[asyncio.Task | None] = asyncio.Queue()

            async def producer():
                nonlocal end_call
                first = True
                try:
                    system = self.memory.build_system_prompt(self.system_prompt)
                    async for token in self.llm.stream_reply(system, self.memory.llm_messages()):
                        if first:
                            m.llm_first_token_ms = (time.perf_counter() - t0) * 1000
                            first = False
                        reply_parts.append(token)
                        for sentence in chunker.push(token):
                            await pending.put(self._translate_task(sentence, lang))
                    tail = chunker.flush()
                    if tail:
                        await pending.put(self._translate_task(tail, lang))
                finally:
                    await pending.put(None)

            prod = asyncio.create_task(producer())
            try:
                while (task := await pending.get()) is not None:
                    spoken = await task
                    if spoken:
                        spoken_parts.append(spoken)
                        yield spoken
                await prod  # surface LLM errors
            finally:
                prod.cancel()
                while not pending.empty():
                    t = pending.get_nowait()
                    if t:
                        t.cancel()

        try:
            await self._synthesize(
                translated_sentences(), lang, metrics=m, end_call_flag=lambda: END_CALL_TOKEN in "".join(reply_parts)
            )
        except asyncio.CancelledError:
            m.interrupted = True
            raise
        finally:
            raw_reply = "".join(reply_parts)
            reply_en = clean_for_speech(raw_reply)
            followups = extract_followups(raw_reply)
            if followups:
                log(logger, logging.INFO, "followup_flagged", call_id=self.call_id, followups=followups)
            if reply_en:
                original = " ".join(spoken_parts) if spoken_parts else reply_en
                self.memory.add(
                    "agent", reply_en, original, lang, repo.now(), interrupted=m.interrupted, followups=followups
                )
            self.metrics.append(m.as_dict())
            log(logger, logging.INFO, "turn_metrics", call_id=self.call_id, **m.as_dict())

    def _translate_task(self, sentence_en: str, lang: str) -> asyncio.Task:
        """Start translating a sentence immediately (runs in parallel with earlier sentences)."""
        clean = clean_for_speech(sentence_en)

        async def run() -> str:
            if not clean:
                return ""
            return clean if is_english(lang) else await self.llm.translate(clean, "en-IN", lang)

        return asyncio.create_task(run())

    async def _synthesize(
        self, sentences, lang: str, metrics: TurnMetrics | None, end_call: bool = False, end_call_flag=None
    ) -> None:
        """Feed sentences into TTS and stream the audio to Twilio concurrently."""
        tts = await self.tts_pool.acquire(to_tts_language(lang, self.settings.default_language))
        self._turn_tts = tts
        frames = FrameBuffer()
        mark = f"turn-{self.call_id[-6:]}-{time.monotonic_ns()}"

        async def feed():
            try:
                async for sentence in sentences:
                    await tts.speak(sentence)
            finally:
                tts.end_input()

        feeder = asyncio.create_task(feed())
        try:
            first = True
            async for pcm in tts.audio():
                mulaw = pcm16_to_mulaw(pcm)
                for frame in frames.push(mulaw):
                    if first:
                        self._agent_speaking = True
                        if metrics:
                            metrics.first_audio_ms = (time.perf_counter() - metrics.stt_final_at) * 1000
                        first = False
                    await self._send_audio(frame)
            tail = frames.flush()
            if tail:
                await self._send_audio(tail)
            await feeder  # propagate LLM / translation errors
        finally:
            feeder.cancel()
            await tts.close()
            if self._turn_tts is tts:
                self._turn_tts = None

        # Twilio echoes the mark back once all audio before it has been played
        self._pending_mark = mark
        should_end = end_call or (end_call_flag() if end_call_flag else False)
        if should_end:
            self._hangup_after_mark = mark
        await self._send_mark(mark)

    # ------------------------------------------------------------------ silence
    async def _silence_watchdog(self) -> None:
        """Nudge a silent caller, then end the call politely."""
        try:
            while not self._closed:
                await asyncio.sleep(1)
                busy = self._agent_speaking or (self._turn_task and not self._turn_task.done())
                if busy:
                    self._last_activity = time.monotonic()
                    continue
                if time.monotonic() - self._last_activity > 8:
                    self._last_activity = time.monotonic()
                    self._nudges += 1
                    if self._nudges <= 2:
                        self._start_turn(self._speak_fixed(NUDGE_EN))
                    else:
                        self._start_turn(self._speak_fixed(GOODBYE_EN, end_call=True))
                        return
        except asyncio.CancelledError:
            return

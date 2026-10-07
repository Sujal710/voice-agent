"""Runs our STT wrapper through the *real* sarvamai SDK against a local
WebSocket server that mimics Sarvam's realtime STT protocol."""

import asyncio
import base64
import json
from urllib.parse import parse_qs, urlparse

from sarvamai.environment import SarvamAIEnvironment
from websockets.asyncio.server import serve

from app.config import Settings
from app.services import stt as stt_mod


async def test_stt_wrapper_with_real_sdk(monkeypatch):
    seen = {"audio": b""}

    async def handler(ws):
        seen["key"] = ws.request.headers.get("Api-Subscription-Key")
        seen["query"] = parse_qs(urlparse(ws.request.path).query)
        await ws.send(json.dumps({"event": "session.begin", "request_id": "r1"}))
        async for raw in ws:
            msg = json.loads(raw)
            if msg["event"] == "audio_input":
                seen["audio"] += base64.b64decode(msg["audio"])
                if len(seen["audio"]) >= 320:
                    await ws.send(json.dumps({"event": "vad.speech_start", "utterance_idx": 0}))
                    await ws.send(
                        json.dumps(
                            {"event": "transcript.partial", "utterance_idx": 0, "text": "કિં", "language": "gu-IN"}
                        )
                    )
                    await ws.send(
                        json.dumps(
                            {
                                "event": "transcript.final",
                                "utterance_idx": 0,
                                "text": " કિંમત કેટલી છે? ",
                                "language": "gu-IN",
                                "language_confidence": 0.97,
                            }
                        )
                    )
            elif msg["event"] == "end":
                await ws.close()

    events = []
    done = asyncio.Event()

    async def on_event(ev):
        events.append(ev)
        if ev.kind == "final":
            done.set()

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        env = SarvamAIEnvironment(base=f"http://127.0.0.1:{port}", creative="", production=f"ws://127.0.0.1:{port}")
        monkeypatch.setattr(stt_mod, "SARVAM_ENVIRONMENT", env)
        settings = Settings(sarvam_api_key="sk-stt", stt_keyterms=["SolarMax", "EMI"])
        stt = stt_mod.SarvamRealtimeSTT(settings, on_event, "call-1")
        await stt.start()
        await stt.send_audio(b"\xff" * 160)
        await stt.send_audio(b"\x7f" * 160)
        await asyncio.wait_for(done.wait(), 3)
        await stt.close()

    q = seen["query"]
    assert seen["key"] == "sk-stt"
    assert q["encoding"] == ["mulaw"] and q["sample_rate"] == ["8000"]
    assert q["language_code"] == ["auto"] and q["model"] == ["saaras:v4"] and q["endpointing"] == ["vad"]
    assert json.loads(q["keyterms"][0]) == ["SolarMax", "EMI"]
    assert seen["audio"] == b"\xff" * 160 + b"\x7f" * 160
    assert [e.kind for e in events] == ["speech_start", "partial", "final"]
    final = events[-1]
    assert final.text == "કિંમત કેટલી છે?" and final.language == "gu-IN" and final.language_confidence == 0.97

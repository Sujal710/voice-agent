"""API routes (with patched repositories) and the TTS client against a local
WebSocket server that speaks Sarvam's TTS protocol."""

import asyncio
import base64
import json
from datetime import UTC

import pytest
from fastapi.testclient import TestClient
from websockets.asyncio.server import serve

from app import repositories as repo
from app.config import Settings
from app.main import app
from app.services import tts as tts_mod
from app.services.telephony import Telephony

API_KEY = "test-api-key-123"


@pytest.fixture
def client(monkeypatch):
    settings = Settings(api_key=API_KEY, twilio_validate_signature=False, public_base_url="https://example.ngrok.app")
    from app.config import get_settings

    app.dependency_overrides[get_settings] = lambda: settings
    app.state.settings = settings
    app.state.telephony = Telephony(settings)
    app.state.dial_semaphore = asyncio.Semaphore(2)

    class NoLLM:
        async def generate_report(self, lines, facts):
            return {}

    app.state.llm = NoLLM()

    placed = []

    async def fake_place_call(call_id, to):
        placed.append((call_id, to))
        return "CA123"

    monkeypatch.setattr(app.state.telephony, "place_call", fake_place_call)

    async def get_customer(cid):
        if cid == "dnc":
            return {"id": "dnc", "phone": "+911111111111", "do_not_call": True}
        return {"id": cid, "phone": "+919876543210", "do_not_call": False} if cid != "missing" else None

    async def get_product(sku):
        return {"sku": sku}

    async def create_call(cid, sku, **extra):
        from datetime import datetime

        return {
            "id": "call1",
            "customer_id": cid,
            "product_sku": sku,
            "status": "queued",
            "transcript": [],
            "created_at": datetime.now(UTC),
        }

    updates = []

    async def update_call(cid, fields):
        updates.append((cid, fields))

    monkeypatch.setattr(repo, "get_customer", get_customer)
    monkeypatch.setattr(repo, "get_product", get_product)
    monkeypatch.setattr(repo, "create_call", create_call)
    monkeypatch.setattr(repo, "update_call", update_call)
    c = TestClient(app)  # no context manager -> lifespan (real Mongo) is skipped
    c.placed, c.updates = placed, updates
    yield c
    app.dependency_overrides.clear()


def test_api_requires_key(client):
    assert client.post("/api/v1/calls", json={"customer_id": "u1", "product_sku": "P1"}).status_code == 401


def test_start_call(client):
    r = client.post("/api/v1/calls", json={"customer_id": "u1", "product_sku": "P1"}, headers={"X-API-Key": API_KEY})
    assert r.status_code == 201, r.text
    assert r.json()["twilio_call_sid"] == "CA123"
    assert client.placed == [("call1", "+919876543210")]


def test_do_not_call_is_respected(client):
    r = client.post("/api/v1/calls", json={"customer_id": "dnc", "product_sku": "P1"}, headers={"X-API-Key": API_KEY})
    assert r.status_code == 409
    assert client.placed == []


def test_campaign_reports_skips(client):
    r = client.post(
        "/api/v1/campaigns",
        json={"product_sku": "P1", "customer_ids": ["u1", "missing", "dnc"]},
        headers={"X-API-Key": API_KEY},
    )
    statuses = [x["status"] for x in r.json()["results"]]
    assert statuses == ["initiated", "skipped", "skipped"]


def test_voice_webhook_returns_stream_twiml(client):
    r = client.post("/twilio/voice?call_id=call1", data={"CallSid": "CA123"})
    assert r.status_code == 200
    assert '<Stream url="wss://example.ngrok.app/twilio/media-stream">' in r.text
    assert '<Parameter name="call_id" value="call1"' in r.text


def test_voicemail_is_hung_up(client):
    r = client.post("/twilio/voice?call_id=call1", data={"AnsweredBy": "machine_start"})
    assert "<Hangup" in r.text


def test_status_webhook(client):
    r = client.post("/twilio/status?call_id=call1", data={"CallStatus": "completed", "CallDuration": "42"})
    assert r.status_code == 204
    assert client.updates[-1][1]["duration_s"] == 42


def test_signature_enforced_when_enabled(client):
    app.dependency_overrides.clear()
    from app.config import get_settings

    app.dependency_overrides[get_settings] = lambda: Settings(
        api_key=API_KEY, twilio_auth_token="tok", twilio_validate_signature=True
    )
    r = client.post("/twilio/status?call_id=call1", data={"CallStatus": "completed"})
    assert r.status_code == 403


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


# ------------------------------------------------------------ TTS protocol
async def test_tts_connection_against_fake_sarvam_server(monkeypatch):
    received = []

    async def handler(ws):
        received.append(("headers", ws.request.headers.get("Api-Subscription-Key"), ws.request.path))
        async for raw in ws:
            msg = json.loads(raw)
            received.append(msg)
            if msg["type"] == "flush":
                # first chunk has a WAV header + odd byte to test realignment
                pcm = b"\x01\x00" * 100
                header = (
                    b"RIFF"
                    + (36 + len(pcm)).to_bytes(4, "little")
                    + b"WAVEfmt "
                    + (16).to_bytes(4, "little")
                    + b"\x01\x00\x01\x00@\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00"
                    + b"data"
                    + len(pcm).to_bytes(4, "little")
                )
                for part in (header + pcm[:51], pcm[51:]):
                    await ws.send(
                        json.dumps(
                            {
                                "type": "audio",
                                "data": {"content_type": "audio/wav", "audio": base64.b64encode(part).decode()},
                            }
                        )
                    )
                await ws.send(json.dumps({"type": "event", "data": {"event_type": "final"}}))

    async with serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setattr(tts_mod, "SARVAM_TTS_WS", f"ws://127.0.0.1:{port}/text-to-speech/ws")
        settings = Settings(sarvam_api_key="sk-test")
        pool = tts_mod.TTSPool(settings)
        conn = await pool.acquire("gu-IN")

        async def feed():
            await conn.speak("નમસ્તે.")
            await conn.speak("કેમ છો?")
            conn.end_input()

        asyncio.create_task(feed())
        audio = b"".join([chunk async for chunk in conn.audio(idle_timeout=2)])
        await conn.close()
        await pool.close()

    handshakes = [r for r in received if isinstance(r, tuple)]
    messages = [r for r in received if isinstance(r, dict)]
    # one socket used + one pre-warmed for the next turn
    assert len(handshakes) == 2
    assert handshakes[0] == ("headers", "sk-test", "/text-to-speech/ws?model=bulbul:v3&send_completion_event=true")
    config = messages[0]
    assert config["type"] == "config"
    assert config["data"]["language_code"] == "gu-IN"
    assert config["data"]["speech_sample_rate"] == 8000
    assert config["data"]["output_audio_codec"] == "linear16"
    assert [m["type"] for m in messages[1:]] == ["text", "flush", "text", "flush"]
    # two flushes x 100 samples, header stripped, 16-bit alignment kept
    assert audio == b"\x01\x00" * 200


def test_media_stream_websocket_routes_events(client, monkeypatch):
    from app.api import routes_twilio

    log = []

    class RecSession:
        def __init__(self, **kw):
            self.call_id = ""

        async def on_start(self, start):
            log.append(("start", start["streamSid"]))
            return True

        async def on_media(self, payload):
            log.append(("media", payload))

        async def on_mark(self, name):
            log.append(("mark", name))

        async def close(self):
            log.append(("close",))

    monkeypatch.setattr(routes_twilio, "CallSession", RecSession)
    app.state.llm = object()
    with client.websocket_connect("/twilio/media-stream") as ws:
        ws.send_json({"event": "connected"})
        ws.send_json({"event": "start", "start": {"streamSid": "MZ1", "customParameters": {}}})
        ws.send_json({"event": "media", "media": {"track": "inbound", "payload": "AAA="}})
        ws.send_json({"event": "media", "media": {"track": "outbound", "payload": "BBB="}})
        ws.send_json({"event": "mark", "mark": {"name": "m1"}})
        ws.send_json({"event": "stop"})
    assert log == [("start", "MZ1"), ("media", "AAA="), ("mark", "m1"), ("close",)]

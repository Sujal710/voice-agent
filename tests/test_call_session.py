"""End-to-end simulation of a phone call through CallSession, with fake
Sarvam STT/TTS, a fake Groq LLM, an in-memory DB and a fake Twilio socket."""

import asyncio
import json

import pytest

from app.config import Settings
from app.pipeline import call_session as cs
from app.security import stream_token


# ---------------------------------------------------------------- fakes
class FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, data: str):
        self.sent.append(json.loads(data))

    def events(self, name):
        return [m for m in self.sent if m["event"] == name]


class FakeLLM:
    def __init__(
        self, reply="Great question. Our system saves up to 90 percent on bills. Shall I share the price?", delay=0.0
    ):
        self.reply = reply
        self.delay = delay
        self.translations: list[tuple[str, str, str]] = []
        self.prompts: list[list[dict]] = []

    async def translate(self, text, src, dst):
        self.translations.append((text, src, dst))
        return f"<{dst}>{text}"

    async def stream_reply(self, system_prompt, history):
        self.prompts.append(list(history))
        for word in self.reply.split(" "):
            await asyncio.sleep(self.delay)
            yield word + " "

    async def summarize_memory(self, previous_summary, messages):
        return (previous_summary + " | " if previous_summary else "") + " / ".join(m.content[:10] for m in messages)

    async def generate_report(self, lines, product_facts):
        self.report_input = (lines, product_facts)
        return {
            "outcome": "interested",
            "outcome_evidence": [2],
            "sentiment": "positive",
            "summary": "Customer asked about price.",
            "customer_needs": [],
            "objections": [],
            "questions_asked": [{"text": "price?", "answered": True, "evidence": [2]}],
            "commitments": [],
            "callback_time": None,
            "compliance_flags": [],
        }


class FakeTTSConn:
    instances: list["FakeTTSConn"] = []

    def __init__(self, lang, slow=False):
        self.lang = lang
        self.spoken: list[str] = []
        self.closed = False
        self.slow = slow
        self._q: asyncio.Queue = asyncio.Queue()
        FakeTTSConn.instances.append(self)

    async def speak(self, text):
        self.spoken.append(text)
        await self._q.put(b"\x00\x10" * 400)  # 400 samples = 50 ms of PCM16

    def end_input(self):
        self._q.put_nowait(None)

    async def audio(self):
        while (chunk := await self._q.get()) is not None:
            if self.slow:
                await asyncio.sleep(0.05)
            yield chunk

    async def close(self):
        self.closed = True


class FakePool:
    slow = False

    def __init__(self, settings):
        pass

    def warm(self):
        pass

    async def acquire(self, lang):
        return FakeTTSConn(lang, slow=FakePool.slow)

    async def close(self):
        pass


class FakeSTT:
    instance = None

    def __init__(self, settings, on_event, call_id):
        self.on_event = on_event
        self.audio: list[bytes] = []
        FakeSTT.instance = self

    async def start(self):
        pass

    async def send_audio(self, b):
        self.audio.append(b)

    async def close(self):
        pass


class FakeTelephony:
    def __init__(self):
        self.hung_up: list[str] = []

    async def hangup(self, sid):
        self.hung_up.append(sid)


@pytest.fixture
def env(monkeypatch):
    db = {
        "calls": {"c1": {"id": "c1", "customer_id": "u1", "product_sku": "P1", "status": "in-progress"}},
        "messages": [],
        "updates": [],
        "reports": [],
    }

    async def get_call(cid, with_transcript=False):
        return db["calls"].get(cid)

    async def get_customer(uid):
        return {"id": "u1", "name": "Sujal", "preferred_language": "gu-IN", "city": "Ahmedabad"}

    async def get_product(sku):
        return {"sku": "P1", "name": "SolarMax", "price": "1 lakh", "features": [], "faqs": []}

    async def append_message(doc):
        db["messages"].append(doc)

    async def get_messages(cid):
        return sorted((m for m in db["messages"] if m["session_id"] == cid), key=lambda m: m["seq"])

    async def recent_call_reports(customer_id, exclude, limit):
        return db.get("past_calls", [])

    async def save_report(cid, report):
        db["reports"].append(report)

    async def update_call(cid, fields):
        db["updates"].append(fields)

    async def noop(*a, **k):
        return None

    r = cs.repo
    monkeypatch.setattr(r, "get_call", get_call)
    monkeypatch.setattr(r, "get_customer", get_customer)
    monkeypatch.setattr(r, "get_product", get_product)
    monkeypatch.setattr(r, "append_message", append_message)
    monkeypatch.setattr(r, "get_messages", get_messages)
    monkeypatch.setattr(r, "recent_call_reports", recent_call_reports)
    monkeypatch.setattr(r, "save_report", save_report)
    monkeypatch.setattr(r, "update_call", update_call)
    monkeypatch.setattr(r, "set_customer_language", noop)
    monkeypatch.setattr(r, "mark_do_not_call", noop)
    monkeypatch.setattr(cs, "SarvamRealtimeSTT", FakeSTT)
    monkeypatch.setattr(cs, "TTSPool", FakePool)
    FakeTTSConn.instances = []
    FakePool.slow = False
    return db


def make_session(llm=None):
    settings = Settings(twilio_auth_token="secret-token", api_key="test-api-key")
    ws, tel = FakeWS(), FakeTelephony()
    session = cs.CallSession(ws=ws, settings=settings, llm=llm or FakeLLM(), telephony=tel)
    start = {
        "streamSid": "MZ1",
        "callSid": "CA1",
        "customParameters": {"call_id": "c1", "token": stream_token("c1", settings)},
    }
    return session, ws, tel, start


async def wait_turn(session):
    for _ in range(200):
        if session._turn_task and session._turn_task.done():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("turn did not finish")


async def test_rejects_bad_stream_token(env):
    session, ws, _, start = make_session()
    start["customParameters"]["token"] = "forged"
    assert await session.on_start(start) is False


async def test_gujarati_turn_translates_both_ways_and_speaks_gujarati(env):
    llm = FakeLLM()
    session, ws, tel, start = make_session(llm)
    assert await session.on_start(start)
    await wait_turn(session)

    # Greeting: scripted English line translated to the customer's preferred language
    greeting_tts = FakeTTSConn.instances[0]
    assert greeting_tts.lang == "gu-IN"
    assert greeting_tts.spoken[0].startswith("<gu-IN>Hello Sujal")
    assert ws.events("media") and ws.events("mark")

    # Caller audio is forwarded to STT untouched (mu-law passthrough)
    await session.on_media("AAEC")
    assert FakeSTT.instance.audio == [b"\x00\x01\x02"]

    # Twilio confirms greeting playback finished
    await session.on_mark(ws.events("mark")[-1]["mark"]["name"])
    assert session._agent_speaking is False

    # Customer speaks Gujarati
    await FakeSTT.instance.on_event(
        cs.STTEvent(kind="final", text="કિંમત કેટલી છે?", language="gu-IN", language_confidence=0.95)
    )
    await wait_turn(session)

    # 1) Gujarati -> English before the LLM
    assert ("કિંમત કેટલી છે?", "gu-IN", "en-IN") in llm.translations
    assert llm.prompts[-1][-1] == {"role": "user", "content": "<en-IN>કિંમત કેટલી છે?"}
    # Greeting is in memory, so the LLM knows what the customer already heard
    assert llm.prompts[-1][0]["role"] == "assistant" and llm.prompts[-1][0]["content"].startswith("Hello Sujal")
    # 2) English reply -> Gujarati, sentence by sentence, in order
    turn_tts = FakeTTSConn.instances[-1]
    assert turn_tts.lang == "gu-IN"
    assert turn_tts.spoken == [
        "<gu-IN>Great question.",
        "<gu-IN>Our system saves up to 90 percent on bills.",
        "<gu-IN>Shall I share the price?",
    ]
    assert turn_tts.closed
    # 3) Audio frames are 20 ms mu-law (160 bytes base64-encoded) and a mark follows
    import base64

    sizes = {len(base64.b64decode(m["media"]["payload"])) for m in ws.events("media")}
    assert max(sizes) == 160
    assert ws.sent[-1]["event"] == "mark"
    # Metrics recorded
    m = session.metrics[-1]
    assert m["language"] == "gu-IN" and m["first_audio_ms"] is not None and m["llm_first_token_ms"] is not None
    # Messages stored LangChain-style: one doc per message, original + English
    await session.memory.drain()
    customer_msg = next(m for m in env["messages"] if m["role"] == "customer")
    assert customer_msg["original"] == "કિંમત કેટલી છે?" and customer_msg["content"].startswith("<en-IN>")
    assert customer_msg["session_id"] == "c1" and customer_msg["seq"] == 2
    agent_msg = env["messages"][-1]
    assert agent_msg["role"] == "agent" and agent_msg["original"].startswith("<gu-IN>Great question.")
    assert agent_msg["content"].startswith("Great question.")
    await session.close()


async def test_english_caller_skips_translation(env):
    llm = FakeLLM()
    session, ws, tel, start = make_session(llm)
    await session.on_start(start)
    await wait_turn(session)
    llm.translations.clear()

    await FakeSTT.instance.on_event(
        cs.STTEvent(kind="final", text="What is the price?", language="en-IN", language_confidence=0.99)
    )
    await wait_turn(session)
    assert llm.translations == []
    assert FakeTTSConn.instances[-1].lang == "en-IN"
    assert FakeTTSConn.instances[-1].spoken[0] == "Great question."
    await session.close()


async def test_low_confidence_language_keeps_previous_language(env):
    session, ws, tel, start = make_session()
    await session.on_start(start)
    await wait_turn(session)
    await FakeSTT.instance.on_event(cs.STTEvent(kind="final", text="ok", language="mr-IN", language_confidence=0.2))
    await wait_turn(session)
    assert session.language == "gu-IN"
    await session.close()


async def test_barge_in_clears_twilio_audio_and_cancels_turn(env):
    FakePool.slow = True
    llm = FakeLLM(reply=" ".join([f"This is a long sentence number {i}." for i in range(30)]), delay=0.01)
    session, ws, tel, start = make_session(llm)
    await session.on_start(start)
    await wait_turn(session)
    await session.on_mark(ws.events("mark")[-1]["mark"]["name"])

    await FakeSTT.instance.on_event(cs.STTEvent(kind="final", text="hello", language="en-IN", language_confidence=0.9))
    for _ in range(100):  # wait until the agent is audibly speaking
        if session._agent_speaking:
            break
        await asyncio.sleep(0.01)
    assert session._agent_speaking

    # Noise-level partial must NOT interrupt
    await FakeSTT.instance.on_event(cs.STTEvent(kind="partial", text="h"))
    assert session._agent_speaking

    await FakeSTT.instance.on_event(cs.STTEvent(kind="partial", text="wait wait"))
    assert ws.events("clear"), "Twilio buffer should be cleared on barge-in"
    assert session._agent_speaking is False
    assert session._turn_task.done()
    assert FakeTTSConn.instances[-1].closed
    assert session.metrics[-1]["interrupted"] is True
    await session.close()


async def test_end_call_token_hangs_up_after_playback(env):
    llm = FakeLLM(reply="Thank you for your time, have a great day! [END_CALL]")
    session, ws, tel, start = make_session(llm)
    await session.on_start(start)
    await wait_turn(session)
    await session.on_mark(ws.events("mark")[-1]["mark"]["name"])

    await FakeSTT.instance.on_event(
        cs.STTEvent(kind="final", text="not interested", language="en-IN", language_confidence=0.9)
    )
    await wait_turn(session)
    spoken = " ".join(FakeTTSConn.instances[-1].spoken)
    assert "END_CALL" not in spoken
    assert tel.hung_up == []  # not before the goodbye has been heard

    await session.on_mark(ws.events("mark")[-1]["mark"]["name"])
    await asyncio.sleep(0.01)
    assert tel.hung_up == ["CA1"]
    await session.close()


async def test_hangup_generates_verified_report(env):
    llm = FakeLLM()
    session, ws, tel, start = make_session(llm)
    await session.on_start(start)
    await wait_turn(session)
    await FakeSTT.instance.on_event(cs.STTEvent(kind="final", text="price?", language="en-IN", language_confidence=0.9))
    await wait_turn(session)
    await session.close()
    for _ in range(100):
        if env["reports"]:
            break
        await asyncio.sleep(0.01)
    report = env["reports"][-1]
    assert report["outcome"] == "interested"
    assert report["questions_asked"] == [{"text": "price?", "evidence": [2], "answered": True}]
    # LLM saw a numbered transcript and the product facts
    lines, facts = llm.report_input
    assert lines[0].startswith("[T1] AGENT: Hello Sujal") and lines[1] == "[T2] CUSTOMER: price?"
    assert "SolarMax" in facts
    assert any("metrics" in u and "memory_summary" in u for u in env["updates"])


async def test_new_utterance_mid_reply_keeps_history_in_order(env):
    FakePool.slow = True
    llm = FakeLLM(reply=" ".join(f"Point number {i}." for i in range(20)), delay=0.01)
    session, ws, tel, start = make_session(llm)
    await session.on_start(start)
    await wait_turn(session)
    await session.on_mark(ws.events("mark")[-1]["mark"]["name"])

    await FakeSTT.instance.on_event(cs.STTEvent(kind="final", text="first", language="en-IN", language_confidence=0.9))
    await asyncio.sleep(0.08)  # agent is mid-reply
    await FakeSTT.instance.on_event(cs.STTEvent(kind="final", text="second", language="en-IN", language_confidence=0.9))
    FakePool.slow = False
    llm.reply, llm.delay = "Sure.", 0
    await wait_turn(session)

    msgs = session.memory.messages
    roles = [(m.role, m.content[:6]) for m in msgs]
    assert roles[1] == ("customer", "first")
    assert roles[2][0] == "agent" and msgs[2].interrupted
    assert roles[3] == ("customer", "second")
    assert roles[4] == ("agent", "Sure.")
    assert [m.seq for m in msgs] == [1, 2, 3, 4, 5]
    await session.close()

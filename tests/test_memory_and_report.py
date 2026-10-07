"""Conversation memory (summary buffer), report grounding/rendering and report download routes."""

import asyncio
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app import repositories as repo
from app.config import Settings
from app.main import app
from app.pipeline.memory import ConversationMemory
from app.pipeline.report import build_call_report, ground_report, render_html, render_markdown
from app.security import report_token

NOW = datetime(2026, 10, 5, 10, 30, tzinfo=UTC)


# ------------------------------------------------------------------ memory
def make_memory(summarize=None, max_tokens=60, past_calls=None):
    stored = []

    async def persist(doc):
        stored.append(doc)

    async def default_summarize(prev, msgs):
        return (prev + " | " if prev else "") + "; ".join(m.content for m in msgs)

    mem = ConversationMemory(
        session_id="call-1",
        customer_id="cust-1",
        summarize=summarize or default_summarize,
        persist=persist,
        max_recent_tokens=max_tokens,
        past_calls=past_calls or [],
    )
    return mem, stored


async def test_messages_are_stored_one_document_each():
    mem, stored = make_memory(max_tokens=10_000)
    mem.add("agent", "Hello", "નમસ્તે", "gu-IN", NOW)
    mem.add("customer", "What is the price?", "કિંમત કેટલી છે?", "gu-IN", NOW)
    await mem.drain()
    assert [d["seq"] for d in stored] == [1, 2]
    assert stored[1] == {
        "session_id": "call-1",
        "customer_id": "cust-1",
        "seq": 2,
        "role": "customer",
        "content": "What is the price?",
        "original": "કિંમત કેટલી છે?",
        "language": "gu-IN",
        "interrupted": False,
        "kind": "dialogue",
        "followups": [],
        "created_at": NOW,
    }


async def test_old_messages_fold_into_summary_and_recent_stay_verbatim():
    mem, _ = make_memory(max_tokens=60)
    for i in range(10):
        role = "customer" if i % 2 else "agent"
        mem.add(role, f"message number {i} with some extra words here", f"orig {i}", "en-IN", NOW)
    await mem.drain()
    assert mem.summarized_upto > 0
    assert "message number 0" in mem.summary
    recent = mem.llm_messages()
    # last 4 messages are always kept word for word
    assert recent[-1]["content"] == "message number 9 with some extra words here"
    assert len(mem.messages[mem.summarized_upto :]) >= 4
    # nothing lost: every message is either summarised or still verbatim
    assert mem.summarized_upto + len(mem.messages[mem.summarized_upto :]) == 10
    system = mem.build_system_prompt("BASE")
    assert system.startswith("BASE") and "EARLIER IN THIS CALL" in system and "message number 0" in system


async def test_failed_summary_keeps_messages_verbatim():
    async def broken(prev, msgs):
        raise RuntimeError("groq down")

    mem, _ = make_memory(summarize=broken, max_tokens=20)
    for i in range(6):
        mem.add("customer" if i % 2 else "agent", f"important fact {i} " * 3, "", "en-IN", NOW)
    await mem.drain()
    assert mem.summary == "" and mem.summarized_upto == 0
    assert len(mem.llm_messages()) == 6  # nothing dropped


async def test_consecutive_same_role_messages_are_merged_and_interruptions_marked():
    mem, _ = make_memory(max_tokens=10_000)
    mem.add("agent", "Hello", "Hello", "en-IN", NOW, kind="scripted")
    mem.add("agent", "Are you there?", "Are you there?", "en-IN", NOW, kind="scripted")
    mem.add("customer", "Yes", "Yes", "en-IN", NOW)
    mem.add("agent", "Great, our plan", "Great, our plan", "en-IN", NOW, interrupted=True)
    msgs = mem.llm_messages()
    assert [m["role"] for m in msgs] == ["assistant", "user", "assistant"]
    assert msgs[0]["content"] == "Hello\nAre you there?"
    assert msgs[2]["content"].endswith("[interrupted by customer]")


def test_past_calls_are_labelled_as_possibly_outdated():
    mem, _ = make_memory(
        past_calls=[
            {
                "date": "01 Oct 2026",
                "outcome": "callback",
                "summary": "Asked for EMI details.",
                "next_steps": ["Call Monday"],
            }
        ]
    )
    prompt = mem.build_system_prompt("BASE")
    assert "may be outdated" in prompt and "Asked for EMI details." in prompt and "Call Monday" in prompt


# ------------------------------------------------------------------ grounding
MESSAGES = [
    {"seq": 1, "role": "agent", "content": "Hello, calling about SolarMax.", "original": "નમસ્તે, SolarMax વિશે."},
    {"seq": 2, "role": "customer", "content": "What is the price?", "original": "કિંમત કેટલી છે?"},
    {"seq": 3, "role": "agent", "content": "It is 1,65,000 rupees with free installation.", "original": "..."},
    {"seq": 4, "role": "customer", "content": "Too expensive. Call me Monday at 5 pm.", "original": "..."},
]


def test_grounding_keeps_cited_items_and_drops_invented_ones():
    raw = {
        "outcome": "callback",
        "outcome_evidence": [4],
        "sentiment": "neutral",
        "summary": "Customer asked price and wants a callback.",
        "customer_needs": [{"text": "Wants lower bills", "evidence": []}],  # no evidence -> dropped
        "objections": [
            {"text": "Price too high", "evidence": ["T4"]},
            {"text": "Worried about roof", "evidence": [9]},  # turn doesn't exist -> dropped
            {"text": "Agent's own words as objection", "evidence": [3]},  # wrong speaker -> dropped
        ],
        "questions_asked": [{"text": "Price?", "answered": True, "evidence": [2, 3]}],
        "commitments": [{"text": "Callback Monday 5 pm", "evidence": [4]}],
        "callback_time": "Monday 5 pm",
        "compliance_flags": [],
    }
    r = ground_report(raw, MESSAGES)
    assert r["outcome"] == "callback" and r["outcome_evidence"] == [4]
    assert r["objections"] == [{"text": "Price too high", "evidence": [4]}]
    assert r["customer_needs"] == []
    assert r["questions_asked"] == [{"text": "Price?", "evidence": [2], "answered": True}]
    assert r["commitments"] == [{"text": "Callback Monday 5 pm", "evidence": [4]}]
    assert len(r["verification"]["removed_items"]) == 3
    assert r["verification"]["needs_human_review"] is False


def test_unsupported_sale_is_downgraded_and_flagged():
    r = ground_report({"outcome": "sale", "outcome_evidence": [], "summary": "Sold!"}, MESSAGES)
    assert r["outcome"] == "unknown"
    assert r["verification"]["needs_human_review"] is True


def test_compliance_flags_require_agent_turns_and_trigger_review():
    raw = {
        "outcome": "interested",
        "compliance_flags": [
            {"text": "Agent promised free installation (not in product facts)", "evidence": [3]},
            {"text": "Bogus flag on customer turn", "evidence": [2]},
        ],
    }
    r = ground_report(raw, MESSAGES)
    assert [f["evidence"] for f in r["compliance_flags"]] == [[3]]
    assert r["verification"]["needs_human_review"] is True


def test_no_messages_means_no_conversation():
    assert ground_report({"outcome": "sale"}, [])["outcome"] == "no_conversation"


async def test_build_report_survives_llm_failure():
    class Down:
        async def generate_report(self, lines, facts):
            raise RuntimeError("timeout")

    r = await build_call_report(Down(), {"id": "c1"}, {"name": "Sujal"}, {"name": "SolarMax"}, MESSAGES)
    assert r["outcome"] == "unknown"
    assert "timeout" in r["verification"]["error"] and r["verification"]["needs_human_review"]


def test_rendering_markdown_and_html():
    r = ground_report(
        {
            "outcome": "interested",
            "outcome_evidence": [2],
            "summary": "Asked <price>.",
            "objections": [{"text": "Too costly", "evidence": [4]}],
        },
        MESSAGES,
    )
    r["meta"] = {
        "call_id": "c1",
        "customer": {"name": "Sujal", "phone": "+919876543210"},
        "product": {"sku": "P1", "name": "SolarMax"},
        "language": "gu-IN",
        "started_at": NOW,
        "generated_at": NOW,
    }
    md = render_markdown(r, MESSAGES)
    assert "# Call report: Sujal" in md and "Too costly [T4]" in md and "કિંમત કેટલી છે?" in md
    assert "| T2 | customer | કિંમત કેટલી છે? | What is the price? |" in md
    page = render_html(r, MESSAGES)
    assert '<meta charset="utf-8">' in page and "કિંમત કેટલી છે?" in page
    assert "Asked &lt;price&gt;." in page  # escaped, no HTML injection from transcripts


# ------------------------------------------------------------------ routes
API_KEY = "test-api-key-123"


@pytest.fixture
def client(monkeypatch):
    from app.config import get_settings

    settings = Settings(api_key=API_KEY, twilio_auth_token="tok", public_base_url="https://example.ngrok.app")
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.settings = settings

    calls = {
        "done": {"id": "done", "customer_id": "u1", "product_sku": "P1", "status": "completed", "created_at": NOW},
        "live": {"id": "live", "customer_id": "u1", "product_sku": "P1", "status": "in-progress", "created_at": NOW},
        "missed": {"id": "missed", "customer_id": "u1", "product_sku": "P1", "status": "no-answer", "created_at": NOW},
    }
    saved = {}

    async def get_call(cid, with_transcript=False):
        call = dict(calls[cid]) if cid in calls else None
        if call and cid in saved:
            call.update(report=saved[cid], report_ready=True, outcome=saved[cid]["outcome"])
        if call and with_transcript:
            call["transcript"] = [
                {**m, "language": "gu-IN", "created_at": NOW, "interrupted": False, "kind": "dialogue"}
                for m in (MESSAGES if cid == "done" else [])
            ]
        return call

    async def get_messages(cid):
        return MESSAGES if cid == "done" else []

    async def get_customer(cid):
        return {"id": "u1", "name": "Sujal", "phone": "+919876543210"}

    async def get_product(sku):
        return {"sku": "P1", "name": "SolarMax", "price": "1,65,000"}

    async def save_report(cid, report):
        saved[cid] = report

    class LLM:
        async def generate_report(self, lines, facts):
            return {"outcome": "interested", "outcome_evidence": [2], "summary": "Asked price."}

    app.state.llm = LLM()
    for name, fn in [
        ("get_call", get_call),
        ("get_messages", get_messages),
        ("get_customer", get_customer),
        ("get_product", get_product),
        ("save_report", save_report),
    ]:
        monkeypatch.setattr(repo, name, fn)
    c = TestClient(app)
    c.saved = saved
    yield c
    app.dependency_overrides.clear()


def test_download_markdown_report(client):
    r = client.get("/api/v1/calls/done/report?format=md", headers={"X-API-Key": API_KEY})
    assert r.status_code == 200
    assert r.headers["content-disposition"].startswith('attachment; filename="call-report-Sujal-05-Oct-2026.md"')
    assert "# Call report: Sujal" in r.text and "કિંમત કેટલી છે?" in r.text
    assert client.saved["done"]["outcome"] == "interested"  # stored in MongoDB too


def test_download_html_and_json(client):
    h = client.get("/api/v1/calls/done/report?format=html", headers={"X-API-Key": API_KEY})
    assert h.headers["content-type"].startswith("text/html") and "<table>" in h.text
    j = client.get("/api/v1/calls/done/report?format=json", headers={"X-API-Key": API_KEY}).json()
    assert j["report"]["outcome"] == "interested" and len(j["transcript"]) == 4


def test_report_not_available_while_call_is_live(client):
    r = client.get("/api/v1/calls/live/report", headers={"X-API-Key": API_KEY})
    assert r.status_code == 409


def test_missed_call_gets_a_no_conversation_report(client):
    r = client.get("/api/v1/calls/missed/report?format=json", headers={"X-API-Key": API_KEY})
    assert r.json()["report"]["outcome"] == "no_conversation"


def test_report_requires_api_key(client):
    assert client.get("/api/v1/calls/done/report").status_code == 401


def test_signed_link_and_report_url(client):
    client.post("/api/v1/calls/done/report", headers={"X-API-Key": API_KEY})
    call = client.get("/api/v1/calls/done", headers={"X-API-Key": API_KEY}).json()
    token = report_token("done", app.state.settings)
    assert call["report_url"] == f"https://example.ngrok.app/reports/done?token={token}"
    assert call["transcript"][1]["original"] == "કિંમત કેટલી છે?"

    ok = client.get(f"/reports/done?token={token}")
    assert ok.status_code == 200 and ok.headers["content-disposition"].startswith("inline")
    dl = client.get(f"/reports/done?token={token}&download=true&format=md")
    assert dl.headers["content-disposition"].startswith("attachment")
    assert client.get("/reports/done?token=forged").status_code == 403
    assert client.get(f"/reports/live?token={token}").status_code == 403  # token is per call


async def test_memory_drain_waits_for_writes():
    mem, stored = make_memory(max_tokens=10_000)

    async def slow_persist(doc):
        await asyncio.sleep(0.05)
        stored.append(doc)

    mem.persist = slow_persist
    mem.add("agent", "hi", "hi", "en-IN", NOW)
    await mem.drain()
    assert len(stored) == 1

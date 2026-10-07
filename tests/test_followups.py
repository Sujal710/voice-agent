"""Requests the agent can't answer from the product facts (e.g. a 20% discount):
never spoken, flagged live, always in the report, CSV and page; full conversation
stored inside the report."""

import asyncio

from app.pipeline import call_session as cs
from app.pipeline.report import build_call_report, render_markdown, transcript_lines
from app.pipeline.text import SentenceChunker, clean_for_speech, extract_followups
from tests.test_call_session import FakeLLM, FakeSTT, FakeTTSConn, make_session, wait_turn

DISCOUNT_REPLY = (
    "That's a fair question. I'll check with my team about a discount and get back to you. "
    "[FOLLOWUP: customer asked for a 20% discount. Wants it before Diwali.]"
)


def stream(text: str, step: int) -> list[str]:
    chunker = SentenceChunker()
    out = []
    for i in range(0, len(text), step):
        out += chunker.push(text[i : i + step])
    tail = chunker.flush()
    return out + ([tail] if tail else [])


def test_followup_tag_is_never_spoken_whatever_the_token_size():
    for step in (1, 2, 5, 13, 1000):
        spoken = " ".join(clean_for_speech(s) for s in stream(DISCOUNT_REPLY + " [END_CALL]", step)).strip()
        assert spoken == "That's a fair question. I'll check with my team about a discount and get back to you."
        assert "FOLLOWUP" not in spoken and "20%" not in spoken and "Diwali" not in spoken


def test_extract_followups():
    assert extract_followups(DISCOUNT_REPLY) == ["customer asked for a 20% discount. Wants it before Diwali."]
    assert extract_followups("[Follow-up: EMI for 36 months] ok [FOLLOWUP: EMI for 36 months]") == ["EMI for 36 months"]
    assert extract_followups("No tags here.") == []


async def test_live_call_flags_followup_and_report_always_lists_it(env):
    class ReportMissesIt(FakeLLM):
        async def generate_report(self, lines, product_facts):
            self.report_input = (lines, product_facts)
            # The report LLM "forgets" the discount request entirely
            return {"outcome": "interested", "outcome_evidence": [2], "summary": "Talked about price."}

    llm = ReportMissesIt(reply=DISCOUNT_REPLY)
    session, ws, tel, start = make_session(llm)
    await session.on_start(start)
    await wait_turn(session)
    await FakeSTT.instance.on_event(
        cs.STTEvent(kind="final", text="क्या 20% डिस्काउंट मिल सकता है?", language="hi-IN", language_confidence=0.95)
    )
    await wait_turn(session)

    # Spoken (in Hindi) without the tag
    spoken = " ".join(FakeTTSConn.instances[-1].spoken)
    assert "FOLLOWUP" not in spoken and "Diwali" not in spoken

    # Stored on the agent's message
    await session.memory.drain()
    agent_msg = env["messages"][-1]
    assert agent_msg["followups"] == ["customer asked for a 20% discount. Wants it before Diwali."]
    assert "FOLLOWUP" not in agent_msg["content"] and "FOLLOWUP" not in agent_msg["original"]

    await session.close()
    for _ in range(100):
        if env["reports"]:
            break
        await asyncio.sleep(0.01)
    report = env["reports"][-1]

    # The report LLM was told about the flag...
    lines, _ = llm.report_input
    assert "(agent flagged for follow-up: customer asked for a 20% discount" in lines[-1]
    # ...and even though it left it out, the live flag is merged in, linked to the customer's question (T2)
    assert report["unanswered_requests"] == [
        {
            "text": "customer asked for a 20% discount. Wants it before Diwali.",
            "evidence": [2],
            "source": "flagged live by agent",
        }
    ]
    assert report["follow_up_required"] is True

    # Full conversation stored inside the report, original language + English
    transcript = report["transcript"]
    assert [t["role"] for t in transcript] == ["agent", "customer", "agent"]
    assert transcript[1]["original"] == "क्या 20% डिस्काउंट मिल सकता है?"
    assert transcript[1]["content"].startswith("<en-IN>")


async def test_report_llm_item_is_not_duplicated_by_live_flag():
    messages = [
        {"seq": 1, "role": "agent", "content": "Hello", "original": "Hello"},
        {"seq": 2, "role": "customer", "content": "Can I get 20% off?", "original": "Can I get 20% off?"},
        {"seq": 3, "role": "agent", "content": "I'll check.", "original": "I'll check.", "followups": ["20% discount"]},
    ]

    class LLM:
        async def generate_report(self, lines, facts):
            return {
                "outcome": "interested",
                "outcome_evidence": [2],
                "unanswered_requests": [
                    {"text": "Asked for a 20% discount", "evidence": [2]},
                    {"text": "Invented request", "evidence": [3]},  # agent turn -> removed
                ],
            }

    r = await build_call_report(LLM(), {"id": "c1"}, {"name": "Rahul"}, {"name": "SolarMax"}, messages)
    assert r["unanswered_requests"] == [{"text": "Asked for a 20% discount", "evidence": [2]}]
    assert any(x["field"] == "unanswered_requests" for x in r["verification"]["removed_items"])
    assert len(r["transcript"]) == 3

    r["meta"] = {"customer": {"name": "Rahul"}, "product": {"name": "SolarMax"}}
    md = render_markdown(r, r["transcript"])
    assert "Needs follow-up:** yes, 1 request(s)" in md
    assert "Asked for a 20% discount [T2]" in md


def test_transcript_lines_mark_flags_for_the_report_llm():
    lines = transcript_lines(
        [
            {"seq": 1, "role": "customer", "content": "Any discount?"},
            {"seq": 2, "role": "agent", "content": "I'll check.", "followups": ["discount"]},
        ]
    )
    assert lines == ["[T1] CUSTOMER: Any discount?", "[T2] AGENT: I'll check. (agent flagged for follow-up: discount)"]

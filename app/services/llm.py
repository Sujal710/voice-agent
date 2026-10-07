"""Groq LLM services: translation, sales-agent reasoning, call summarisation.

One AsyncGroq client is shared by the whole process so HTTP connections are
reused (connection setup is a big part of per-request latency).
"""

import json
import logging
from collections.abc import AsyncIterator

from groq import AsyncGroq

from app.config import Settings
from app.logging_config import log
from app.pipeline.text import END_CALL_TOKEN
from app.services.languages import is_english, language_name

logger = logging.getLogger(__name__)


TRANSLATE_SYSTEM = (
    "You are a translation engine for a live phone call. Translate the user's text "
    "from {src} to {dst}. Rules: output ONLY the translation, no quotes, no notes. "
    "Keep product names, brand names, numbers, prices and phone numbers unchanged. "
    "Use natural, polite, spoken {dst} as a native speaker would say it on a call. "
    "{script_rule}"
)


def format_product_facts(product: dict) -> str:
    """The single source of truth about the product, shared by the agent prompt and the report checker."""
    features = "\n".join(f"- {f}" for f in product.get("features", [])) or "- (none listed)"
    faqs = "\n".join(f"Q: {q['q']}\nA: {q['a']}" for q in product.get("faqs", [])) or "(none)"
    return (
        f"- Name: {product.get('name')}\n"
        f"- Price: {product.get('price')}\n"
        f"- Offer: {product.get('offer') or 'none'}\n"
        f"- Pitch: {product.get('description', '')}\n"
        f"Features:\n{features}\nFAQs:\n{faqs}"
    )


def build_sales_system_prompt(customer: dict, product: dict, company_name: str, agent_name: str = "Riya") -> str:
    return f"""You are {agent_name}, a friendly, professional sales representative from {company_name},
on an OUTBOUND phone call with a customer. Your replies are converted to speech.

CUSTOMER
- Name: {customer.get("name", "the customer")}
- City: {customer.get("city") or "unknown"}
- Notes: {customer.get("notes") or "none"}

PRODUCT YOU ARE SELLING
{format_product_facts(product)}

HOW TO TALK
- Speak like a real person on a phone call: short, warm, 1-3 sentences per reply (under 45 words).
- Never use lists, markdown, emojis or special symbols. Write numbers the way you'd say them.
- Ask one question at a time. Understand the customer's needs before pitching.
- Handle objections politely with a relevant benefit. Never pressure.
- If the customer is busy, offer a callback time. If they say not interested twice,
  or ask not to be called, thank them and end the call politely.
- To end the call, finish your goodbye sentence and then append {END_CALL_TOKEN}.
- Always reply in English. Translation to the customer's language is handled for you.

GROUNDING RULES (very important, never break these)
- The PRODUCT section above is your ONLY source of product facts. Never invent or change
  prices, discounts, offers, warranties, timelines, features or policies.
- If the customer asks for something the PRODUCT section does not cover (for example a discount,
  price or offer that isn't listed, a feature, a delivery date, or a policy), do NOT agree, refuse
  or guess. Say honestly that you will check with the team and get back to them, then add a hidden
  tag at the very end of that reply: [FOLLOWUP: <what the customer asked, in a few words>]
  Example: "That's a fair question. I'll check with my team about that and get back to you.
  [FOLLOWUP: customer asked for a 20% discount]". The tag is never spoken; it alerts the sales team.
- Only refer to things the customer actually said in this conversation. Never claim the
  customer agreed to something unless they clearly said so.
- Treat notes about previous calls as possibly outdated; confirm them with the customer.
"""


MEMORY_SUMMARY_SYSTEM = """You maintain the memory of an ongoing sales phone call.
Update the running summary with the NEW MESSAGES. Rules:
- Record ONLY facts explicitly stated in the messages. Never infer, assume or add anything.
- Keep the customer's own words for numbers, dates, prices, names and decisions.
- Keep everything still relevant from the previous summary; drop nothing that was agreed.
- Use short lines under these headings (omit empty ones):
  Customer said: / Customer needs: / Objections: / Agent told the customer: / Agreed or decided: / Open questions:
- Maximum 150 words. Output only the summary text."""


REPORT_SYSTEM = """You write a post-call report for a sales manager from a phone-call transcript.
Each transcript line is numbered like [T3]. Use ONLY information in the transcript.

Return ONLY a JSON object with exactly these keys:
{
  "outcome": one of ["sale", "interested", "callback", "not_interested", "do_not_call", "no_conversation", "unknown"],
  "outcome_evidence": [turn numbers that prove the outcome],
  "sentiment": one of ["positive", "neutral", "negative"],
  "summary": "3-5 sentence factual summary of what was discussed",
  "customer_needs": [{"text": "...", "evidence": [turn numbers]}],
  "objections": [{"text": "...", "evidence": [turn numbers]}],
  "questions_asked": [{"text": "question the customer asked", "answered": true/false, "evidence": [turn numbers]}],
  "commitments": [{"text": "what was agreed / next step", "evidence": [turn numbers]}],
  "unanswered_requests": [{"text": "what the customer asked for that the PRODUCT FACTS don't cover, e.g. a 20% discount", "evidence": [customer turn numbers]}],
  "callback_time": "exact time/date the customer gave, or null",
  "compliance_flags": [{"text": "agent statement NOT supported by the PRODUCT FACTS", "evidence": [turn numbers]}]
}

STRICT RULES
- Every item must cite the turn numbers where it was actually said. If you cannot cite it, leave it out.
- Do not guess the customer's intent. If the outcome is unclear, use "unknown".
- "sale" only if the customer explicitly agreed to buy. "callback" only if a callback was explicitly agreed.
- unanswered_requests: every customer question or request the agent could not answer from the PRODUCT FACTS
  (discounts or prices not listed, missing features, dates, policies) and had to defer. Lines marked
  "(agent flagged for follow-up: ...)" are always such requests. These need action from the sales team.
- compliance_flags: compare each AGENT statement about price, offers, features, warranty or timelines
  with the PRODUCT FACTS; list only statements that contradict or go beyond them. Empty list if none.
- Write in English."""


class GroqLLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self.client = AsyncGroq(api_key=settings.groq_api_key, max_retries=1, timeout=15.0)

    async def translate(self, text: str, src: str, dst: str) -> str:
        """Translate between languages. Returns input unchanged if both are the same."""
        text = text.strip()
        if not text or language_name(src) == language_name(dst):
            return text
        script_rule = (
            f"Write the output in the native {language_name(dst)} script."
            if not is_english(dst)
            else "Write the output in English (Latin script)."
        )
        system = TRANSLATE_SYSTEM.format(src=language_name(src), dst=language_name(dst), script_rule=script_rule)
        try:
            resp = await self.client.chat.completions.create(
                model=self.s.groq_translate_model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
                temperature=0.0,
                max_tokens=400,
            )
            return (resp.choices[0].message.content or text).strip().strip('"')
        except Exception as exc:  # never kill a live call because translation failed
            log(logger, logging.WARNING, "translation_failed", error=str(exc), src=src, dst=dst)
            return text

    async def stream_reply(self, system_prompt: str, messages: list[dict]) -> AsyncIterator[str]:
        """Stream the sales agent's English reply token by token.

        `messages` comes from ConversationMemory (recent turns verbatim; older
        turns are already folded into the system prompt as a fact summary).
        """
        stream = await self.client.chat.completions.create(
            model=self.s.groq_agent_model,
            messages=[{"role": "system", "content": system_prompt}, *messages],
            temperature=self.s.llm_temperature,
            max_tokens=self.s.llm_max_tokens,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def summarize_memory(self, previous_summary: str, messages: list) -> str:
        """Fold older messages into the running fact summary (ConversationMemory hook)."""
        lines = "\n".join(f"{'AGENT' if m.role == 'agent' else 'CUSTOMER'}: {m.content}" for m in messages)
        user = f"PREVIOUS SUMMARY:\n{previous_summary or '(none)'}\n\nNEW MESSAGES:\n{lines}"
        resp = await self.client.chat.completions.create(
            model=self.s.groq_translate_model,  # small fast model is enough for this
            messages=[{"role": "system", "content": MEMORY_SUMMARY_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=300,
        )
        return (resp.choices[0].message.content or "").strip()

    async def generate_report(self, transcript_lines: list[str], product_facts: str) -> dict:
        """Ask the LLM for an evidence-cited JSON report. Validation happens in app.pipeline.report."""
        user = "PRODUCT FACTS:\n" + product_facts + "\n\nTRANSCRIPT:\n" + "\n".join(transcript_lines)
        resp = await self.client.chat.completions.create(
            model=self.s.groq_agent_model,
            messages=[{"role": "system", "content": REPORT_SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        return json.loads(resp.choices[0].message.content or "{}")

    async def close(self) -> None:
        await self.client.close()

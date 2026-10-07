"""Conversation memory, modelled on LangChain's chat-history design.

Storage (like LangChain's `MongoDBChatMessageHistory`)
    One MongoDB document per message in the `messages` collection, indexed by
    `session_id` (= call id). Each message keeps the English text the LLM
    sees *and* the original-language text the customer actually heard/said.

Context window (like LangChain's `ConversationSummaryBufferMemory`)
    The most recent messages are sent to the LLM word for word. When they
    exceed `max_recent_tokens`, the oldest ones are folded into a running
    *fact-only* summary by the LLM. This keeps prompts short (fast and cheap)
    without forgetting what was agreed earlier in a long call.

Anti-hallucination rules
    * The summary is updated in the background and only replaces messages
      after it succeeds, so nothing is ever lost while it runs.
    * The summariser may record only facts explicitly stated in the messages.
    * Long-term memory from previous calls is labelled as possibly outdated,
      and the agent is told to confirm it with the customer.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime

from app.logging_config import log

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token for English) — no tokenizer download needed."""
    return len(text) // 4 + 4


@dataclass
class Message:
    seq: int
    role: str  # "agent" | "customer"
    content: str  # English (what the LLM reasons over)
    original: str  # what was actually spoken on the call
    language: str
    at: datetime
    interrupted: bool = False
    kind: str = "dialogue"  # "dialogue" | "scripted" (greeting, nudges)
    followups: list[str] = field(default_factory=list)  # requests the agent couldn't answer from facts

    @property
    def llm_role(self) -> str:
        return "assistant" if self.role == "agent" else "user"

    def to_doc(self, session_id: str, customer_id: str | None) -> dict:
        return {
            "session_id": session_id,
            "customer_id": customer_id,
            "seq": self.seq,
            "role": self.role,
            "content": self.content,
            "original": self.original,
            "language": self.language,
            "interrupted": self.interrupted,
            "kind": self.kind,
            "followups": self.followups,
            "created_at": self.at,
        }


Persist = Callable[[dict], Awaitable[None]]
Summarize = Callable[[str, list[Message]], Awaitable[str]]


@dataclass
class ConversationMemory:
    session_id: str
    customer_id: str | None
    summarize: Summarize  # LLM call: (previous_summary, messages_to_fold) -> new summary
    persist: Persist  # writes one message document
    max_recent_tokens: int = 1200
    past_calls: list[dict] = field(default_factory=list)

    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    summarized_upto: int = 0  # messages[:summarized_upto] live only in `summary`

    def __post_init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------- writing
    def add(
        self,
        role: str,
        content: str,
        original: str,
        language: str,
        at: datetime,
        interrupted: bool = False,
        kind: str = "dialogue",
        followups: list[str] | None = None,
    ) -> Message:
        msg = Message(
            seq=len(self.messages) + 1,
            role=role,
            content=content.strip(),
            original=original.strip(),
            language=language,
            at=at,
            interrupted=interrupted,
            kind=kind,
            followups=list(followups or []),
        )
        self.messages.append(msg)
        self._spawn(self.persist(msg.to_doc(self.session_id, self.customer_id)))
        if self._recent_tokens() > self.max_recent_tokens:
            self._spawn(self._fold_old_messages())
        return msg

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        """Wait (max 10 s) for pending DB writes / summaries (used before the call report)."""
        if self._tasks:
            await asyncio.wait(list(self._tasks), timeout=10)

    # ------------------------------------------------------------- summarising
    def _recent(self) -> list[Message]:
        return self.messages[self.summarized_upto :]

    def _recent_tokens(self) -> int:
        return sum(estimate_tokens(m.content) for m in self._recent())

    async def _fold_old_messages(self) -> None:
        async with self._lock:  # one summary update at a time
            recent = self._recent()
            if sum(estimate_tokens(m.content) for m in recent) <= self.max_recent_tokens:
                return
            # Fold the oldest messages until the verbatim buffer is at half budget,
            # always keeping at least the last 4 messages word for word.
            budget = self.max_recent_tokens // 2
            keep_tokens, cut = 0, len(recent)
            for i in range(len(recent) - 1, -1, -1):
                keep_tokens += estimate_tokens(recent[i].content)
                if keep_tokens > budget and len(recent) - i > 4:
                    cut = i + 1
                    break
            else:
                cut = max(0, len(recent) - 4)
            to_fold = recent[:cut]
            if not to_fold:
                return
            try:
                new_summary = await self.summarize(self.summary, to_fold)
            except Exception as exc:  # keep messages verbatim; try again on next add
                log(logger, logging.WARNING, "memory_summary_failed", session_id=self.session_id, error=str(exc))
                return
            if new_summary.strip():
                self.summary = new_summary.strip()
                self.summarized_upto += len(to_fold)
                log(
                    logger,
                    logging.INFO,
                    "memory_summarized",
                    session_id=self.session_id,
                    folded=len(to_fold),
                    recent=len(self._recent()),
                )

    # ------------------------------------------------------------- reading
    def build_system_prompt(self, base_prompt: str) -> str:
        parts = [base_prompt]
        if self.past_calls:
            lines = []
            for c in self.past_calls:
                date = c.get("date", "")
                lines.append(f"- {date}: outcome={c.get('outcome', 'unknown')}. {c.get('summary', '')}".strip())
                for step in c.get("next_steps", [])[:3]:
                    lines.append(f"  next step recorded: {step}")
            parts.append(
                "PREVIOUS CALLS WITH THIS CUSTOMER (from our records; may be outdated, "
                "confirm with the customer before relying on any of it):\n" + "\n".join(lines)
            )
        if self.summary:
            parts.append(
                "EARLIER IN THIS CALL (verified fact summary of older messages; "
                "the most recent messages follow word for word):\n" + self.summary
            )
        return "\n\n".join(parts)

    def llm_messages(self) -> list[dict]:
        """Recent messages in chat format. Merges consecutive same-role messages
        (e.g. greeting + nudge) because some models require alternating roles."""
        out: list[dict] = []
        for m in self._recent():
            text = m.content + (" [interrupted by customer]" if m.interrupted else "")
            if out and out[-1]["role"] == m.llm_role:
                out[-1]["content"] += "\n" + text
            else:
                out.append({"role": m.llm_role, "content": text})
        return out

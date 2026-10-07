"""Text utilities for streaming LLM output into TTS.

The agent can add hidden control tags to its English reply. They are never
spoken and never translated:

* ``[END_CALL]``                 – hang up after this reply has been heard
* ``[FOLLOWUP: <what>]``         – the customer asked something the product facts
                                   don't cover (e.g. a 20% discount); the team must
                                   follow up. Recorded on the message and in the report.
"""

import re

END_CALL_TOKEN = "[END_CALL]"

# Sentence boundaries: . ! ? plus Hindi/Marathi/Bengali danda (। ॥)
_SENTENCE_END = re.compile(r"([.!?।॥]+)(\s+|$)")
_FOLLOWUP = re.compile(r"\[\s*FOLLOW[\s_-]?UP\s*:\s*([^\]]+?)\s*\]", re.I)
_ANY_TAG = re.compile(r"\[[^\]]*\]")
_MARKDOWN = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"\*(.+?)\*"), r"\1"),
    (re.compile(r"`(.+?)`"), r"\1"),
    (re.compile(r"^\s*[-*+•]\s+", re.M), ""),
    (re.compile(r"^\s*\d+\.\s+", re.M), ""),
    (re.compile(r"[#_|~]"), ""),
]


def extract_followups(text: str) -> list[str]:
    """['customer asked for a 20% discount', ...] from [FOLLOWUP: ...] tags."""
    seen: list[str] = []
    for item in _FOLLOWUP.findall(text or ""):
        item = re.sub(r"\s+", " ", item).strip()
        if item and item not in seen:
            seen.append(item)
    return seen


def clean_for_speech(text: str) -> str:
    """Remove control tags and markdown so only speakable words remain."""
    text = _ANY_TAG.sub("", text or "")
    for pattern, repl in _MARKDOWN:
        text = pattern.sub(repl, text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceChunker:
    """Turns a stream of LLM tokens into speakable sentences.

    The first chunk is released early (at a comma, after `first_min_chars`)
    so the caller hears audio as soon as possible; later chunks wait for full
    sentences, which sound more natural and translate better.

    Text inside an unfinished ``[ ... ]`` tag is held back until the tag closes,
    so a tag like ``[FOLLOWUP: asked for 20% off.]`` is never split in half
    (which could make half of it audible).
    """

    MAX_TAG_LEN = 300

    def __init__(self, first_min_chars: int = 25, max_chars: int = 220):
        self._buf = ""
        self._emitted_any = False
        self.first_min_chars = first_min_chars
        self.max_chars = max_chars

    def _open_tag_start(self) -> int:
        """Index of an unclosed '[' (or -1)."""
        start = self._buf.rfind("[")
        if start == -1 or "]" in self._buf[start:]:
            return -1
        if len(self._buf) - start > self.MAX_TAG_LEN:  # not a real tag, don't stall
            return -1
        return start

    def push(self, token: str) -> list[str]:
        self._buf += token
        out: list[str] = []
        while True:
            open_at = self._open_tag_start()
            region = self._buf if open_at == -1 else self._buf[:open_at]
            # Hide the inside of closed tags so their punctuation never ends a sentence
            region = _ANY_TAG.sub(lambda m: "x" * len(m.group()), region)
            match = _SENTENCE_END.search(region)
            if match:
                end = match.end()
                sentence, self._buf = self._buf[:end], self._buf[end:]
                if sentence.strip():
                    out.append(sentence.strip())
                    self._emitted_any = True
                continue
            # Early release for the very first chunk at a comma
            if not self._emitted_any and len(region) >= self.first_min_chars:
                comma = max(region.rfind(","), region.rfind("،"))
                if comma >= self.first_min_chars // 2:
                    out.append(self._buf[: comma + 1].strip())
                    self._buf = self._buf[comma + 1 :]
                    self._emitted_any = True
                    continue
            # Hard cap so a run-on sentence doesn't stall TTS
            if len(region) >= self.max_chars:
                cut = region.rfind(" ", 0, self.max_chars)
                cut = cut if cut > 0 else self.max_chars
                out.append(self._buf[:cut].strip())
                self._buf = self._buf[cut:]
                self._emitted_any = True
                continue
            break
        return out

    def flush(self) -> str | None:
        rest, self._buf = self._buf.strip(), ""
        return rest or None

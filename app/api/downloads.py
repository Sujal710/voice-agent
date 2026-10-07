"""Content-Disposition helper that works with non-English file names (e.g. Gujarati names)."""

import re
import unicodedata
from urllib.parse import quote


def content_disposition(filename: str, inline: bool = False) -> str:
    """RFC 6266/5987 header: ASCII fallback + UTF-8 name, so browsers keep the real name."""
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("_") or "download"
    kind = "inline" if inline else "attachment"
    return f"{kind}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


def slug(text: str | None) -> str:
    """File-name safe version of a person's name.

    Keeps letters, combining vowel signs and digits of every script (so "નિરવ પટેલ"
    stays intact; a plain \\w regex would drop the vowel signs) and removes symbols
    that are unsafe in file names.
    """
    kept = "".join(ch if unicodedata.category(ch)[0] in "LMN" else (" " if ch in " _-" else "") for ch in (text or ""))
    return re.sub(r"\s+", "-", kept.strip())[:40] or "customer"

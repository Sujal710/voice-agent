"""Language code helpers shared by STT, translation and TTS.

Sarvam STT (saaras) can detect ~23 Indic languages, but Bulbul TTS can speak
a smaller set. Anything TTS can't speak falls back to a close language.
"""

# BCP-47 code -> English name (used in translation prompts)
LANGUAGE_NAMES: dict[str, str] = {
    "en-IN": "English",
    "hi-IN": "Hindi",
    "gu-IN": "Gujarati",
    "mr-IN": "Marathi",
    "bn-IN": "Bengali",
    "ta-IN": "Tamil",
    "te-IN": "Telugu",
    "kn-IN": "Kannada",
    "ml-IN": "Malayalam",
    "pa-IN": "Punjabi",
    "od-IN": "Odia",
    "or-IN": "Odia",
    "as-IN": "Assamese",
    "ur-IN": "Urdu",
    "ne-IN": "Nepali",
    "kok-IN": "Konkani",
    "mai-IN": "Maithili",
    "sa-IN": "Sanskrit",
}

# Languages Bulbul TTS can speak
TTS_LANGUAGES = {"bn-IN", "en-IN", "gu-IN", "hi-IN", "kn-IN", "ml-IN", "mr-IN", "od-IN", "pa-IN", "ta-IN", "te-IN"}

# STT code -> TTS code, for languages TTS doesn't support directly
_TTS_FALLBACK = {
    "or-IN": "od-IN",  # STT uses or-IN for Odia, TTS uses od-IN
    "ur-IN": "hi-IN",
    "ne-IN": "hi-IN",
    "mai-IN": "hi-IN",
    "sa-IN": "hi-IN",
    "doi-IN": "hi-IN",
    "kok-IN": "mr-IN",
    "as-IN": "bn-IN",
}


def normalize(code: str | None, default: str = "en-IN") -> str:
    """Normalize codes like 'gu', 'GU-in', 'gu-IN' to 'gu-IN'."""
    if not code:
        return default
    code = code.strip()
    if "-" not in code:
        code = f"{code.lower()}-IN"
    lang, region = code.split("-", 1)
    return f"{lang.lower()}-{region.upper()}"


def to_tts_language(code: str | None, default: str = "en-IN") -> str:
    code = normalize(code, default)
    if code in TTS_LANGUAGES:
        return code
    return _TTS_FALLBACK.get(code, default)


def language_name(code: str | None) -> str:
    return LANGUAGE_NAMES.get(normalize(code), "English")


def is_english(code: str | None) -> bool:
    return normalize(code).startswith("en-")

from app.pipeline.text import END_CALL_TOKEN, SentenceChunker, clean_for_speech
from app.services.languages import is_english, language_name, normalize, to_tts_language


def stream(chunker, text, step=3):
    out = []
    for i in range(0, len(text), step):
        out += chunker.push(text[i : i + step])
    tail = chunker.flush()
    return out + ([tail] if tail else [])


def test_chunker_splits_sentences_and_keeps_all_text():
    text = "Hello Sujal, this is Riya from Acme. Do you have a minute? Great!"
    parts = stream(SentenceChunker(first_min_chars=200), text)
    assert parts == ["Hello Sujal, this is Riya from Acme.", "Do you have a minute?", "Great!"]


def test_chunker_releases_first_chunk_early_at_comma():
    parts = stream(
        SentenceChunker(first_min_chars=20),
        "Absolutely, I completely understand your concern about the price. It is fair.",
    )
    assert parts[0].endswith(",") or len(parts[0]) < 40
    assert "".join(parts).replace(" ", "") == "Absolutely,Icompletelyunderstandyourconcernabouttheprice.Itisfair."


def test_chunker_handles_danda():
    parts = stream(SentenceChunker(first_min_chars=200), "नमस्ते। आप कैसे हैं?")
    assert parts == ["नमस्ते।", "आप कैसे हैं?"]


def test_clean_for_speech_removes_markdown_and_end_token():
    assert clean_for_speech(f"**Great** choice! - item {END_CALL_TOKEN}") == "Great choice! - item"
    assert clean_for_speech(f"Bye! {END_CALL_TOKEN}") == "Bye!"


def test_languages():
    assert normalize("gu") == "gu-IN"
    assert normalize("GU-in") == "gu-IN"
    assert to_tts_language("gu-IN") == "gu-IN"
    assert to_tts_language("or-IN") == "od-IN"
    assert to_tts_language("ur-IN") == "hi-IN"
    assert to_tts_language("xx-IN") == "en-IN"
    assert language_name("mr-IN") == "Marathi"
    assert is_english("en-IN") and not is_english("hi-IN")

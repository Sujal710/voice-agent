"""Application settings, loaded from environment variables / .env."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "voice-sales-agent"
    company_name: str = "Acme Solutions"
    agent_name: str = "Riya"
    environment: str = "development"
    log_level: str = "INFO"
    # Public base URL Twilio can reach, e.g. https://abc.ngrok-free.app (no trailing slash)
    public_base_url: str = "http://localhost:8000"
    # Key required in the `X-API-Key` header for scripts / curl (never put it in a web page)
    api_key: str = Field(default="change-me", min_length=8)

    # --- Dashboard login (Call Desk page) ---
    dashboard_username: str = "admin"
    # scrypt hash created with: python -m scripts.hash_password   (empty = login disabled)
    dashboard_password_hash: str = ""
    # Signs session cookies. Generate: python -c "import secrets;print(secrets.token_urlsafe(48))"
    session_secret: str = ""
    session_hours: float = 12
    # Keep True in production (HTTPS). Browsers also accept Secure cookies on http://localhost.
    cookie_secure: bool = True

    # --- MongoDB ---
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "voice_sales_agent"

    # --- Twilio ---
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    # Validate X-Twilio-Signature on webhooks (disable only for local testing)
    twilio_validate_signature: bool = True
    max_call_duration_s: int = 600

    # --- Sarvam STT (realtime WebSocket) ---
    sarvam_api_key: str = ""
    stt_model: str = "saaras:v4"
    stt_language_code: str = "auto"
    stt_mode: str = "transcribe"
    stt_stream_type: str = "fast"
    stt_vad_threshold: float = 0.3
    stt_silence_duration_ms: int = 500
    stt_min_speech_duration_ms: int = 250
    stt_min_language_confidence: float = 0.6
    stt_keyterms: list[str] = Field(default_factory=list)

    # --- Sarvam TTS (Bulbul v3 WebSocket) ---
    tts_model: str = "bulbul:v3"
    tts_speaker: str = "shubh"
    tts_pace: float = 1.05
    tts_temperature: float = 0.6
    tts_min_buffer_size: int = 30
    tts_max_chunk_length: int = 150

    # --- Groq LLM ---
    groq_api_key: str = ""
    # Reasoning / sales conversation model
    groq_agent_model: str = "llama-3.3-70b-versatile"
    # Translation model: small and fast
    groq_translate_model: str = "llama-3.1-8b-instant"
    llm_max_tokens: int = 220
    llm_temperature: float = 0.3
    memory_max_recent_tokens: int = 1200
    memory_past_calls: int = 3

    # --- Language ---
    default_language: str = "en-IN"

    # --- Campaigns / daily CSV batches ---
    max_concurrent_outbound_calls: int = 10  # live calls at once (all replicas)
    dialer_enabled: bool = True
    dialer_poll_s: float = 3.0
    calling_timezone: str = "Asia/Kolkata"
    calling_hours_start: str = "09:00"  # TRAI allows telemarketing 9 am - 9 pm
    calling_hours_end: str = "21:00"
    default_country_code: str = "91"


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Auth for the management API and verification of Twilio requests."""

import hashlib
import hmac
import secrets

from fastapi import Depends, Header, HTTPException, Request, status
from twilio.request_validator import RequestValidator

from app.auth import COOKIE_NAME, CSRF_HEADER, CSRF_VALUE, read_session_token
from app.config import Settings, get_settings


def session_secret(settings: Settings) -> str:
    """Secret for signing dashboard session cookies (falls back to a hash of the API key)."""
    if settings.session_secret:
        return settings.session_secret
    return hashlib.sha256(f"session:{settings.api_key}".encode()).hexdigest()


def current_user(request: Request, settings: Settings) -> str | None:
    return read_session_token(request.cookies.get(COOKIE_NAME), session_secret(settings))


async def require_auth(
    request: Request,
    x_api_key: str | None = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> str:
    """Allow either the X-API-Key header (scripts, curl) or a logged-in dashboard session."""
    if x_api_key is not None:
        if secrets.compare_digest(x_api_key, settings.api_key):
            return "api-key"
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid X-API-Key")
    user = current_user(request, settings)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Please sign in")
    # Cross-site request forgery guard for cookie sessions: other sites can't set this header
    if request.method not in ("GET", "HEAD", "OPTIONS") and request.headers.get(CSRF_HEADER) != CSRF_VALUE:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Missing X-Requested-With header")
    return user


# Backwards-compatible name used by the routers
require_api_key = require_auth


async def verify_twilio_request(request: Request, settings: Settings = Depends(get_settings)) -> dict:
    """Validate X-Twilio-Signature and return the form params.

    The URL is rebuilt from PUBLIC_BASE_URL because behind ngrok / a load
    balancer `request.url` is the internal http:// URL, not what Twilio signed.
    """
    form = dict(await request.form())
    if not settings.twilio_validate_signature:
        return form
    url = settings.public_base_url.rstrip("/") + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    signature = request.headers.get("X-Twilio-Signature", "")
    if not RequestValidator(settings.twilio_auth_token).validate(url, form, signature):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid Twilio signature")
    return form


def stream_token(call_id: str, settings: Settings) -> str:
    """HMAC token passed to the media stream so only our calls can open it."""
    key = (settings.twilio_auth_token or settings.api_key).encode()
    return hmac.new(key, call_id.encode(), hashlib.sha256).hexdigest()


def verify_stream_token(call_id: str, token: str, settings: Settings) -> bool:
    return bool(call_id and token) and hmac.compare_digest(stream_token(call_id, settings), token)


def report_token(call_id: str, settings: Settings) -> str:
    """Signed token for the public report download link (separate purpose from the stream token)."""
    key = (settings.twilio_auth_token or settings.api_key).encode()
    return hmac.new(key, f"report:{call_id}".encode(), hashlib.sha256).hexdigest()


def verify_report_token(call_id: str, token: str, settings: Settings) -> bool:
    return bool(call_id and token) and hmac.compare_digest(report_token(call_id, settings), token)

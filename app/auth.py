"""Dashboard login: hashed password + signed, HttpOnly session cookie.

Why not put the API key in the web page?  Anything the browser's JavaScript can
read (an input box, localStorage) can be leaked by a screenshot, a browser
extension or an XSS bug. Instead:

* The password is stored only as a salted **scrypt** hash in `.env`
  (`python -m scripts.hash_password` creates it).
* After login the server sets a session cookie that is **HttpOnly** (JavaScript
  can't read it), **Secure** (HTTPS only), **SameSite=Strict** (not sent by other
  sites) and expires after `SESSION_HOURS`.
* The cookie is an HMAC-signed token, so it can't be forged or edited, and no
  server-side session store is needed (works across replicas).
* Cookie-authenticated requests that change data must also send the
  `X-Requested-With: CallDesk` header, which other websites cannot add
  (protection against cross-site request forgery).
* Repeated failed logins from one address are rate-limited.

The `X-API-Key` header keeps working for scripts and curl.
"""

import base64
import hashlib
import hmac
import json
import secrets
import time
from collections import defaultdict, deque

COOKIE_NAME = "calldesk_session"
CSRF_HEADER = "X-Requested-With"
CSRF_VALUE = "CallDesk"

_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1


# ---------------------------------------------------------------- passwords
def hash_password(password: str) -> str:
    """Return 'scrypt:N:r:p:salt_hex:hash_hex' (':' separators survive .env files and Docker)."""
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P, dklen=32)
    return f"scrypt:{_SCRYPT_N}:{_SCRYPT_R}:{_SCRYPT_P}:{salt.hex()}:{digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_hex, hash_hex = stored.split(":")
        if algo != "scrypt":
            return False
        digest = hashlib.scrypt(
            password.encode(), salt=bytes.fromhex(salt_hex), n=int(n), r=int(r), p=int(p), dklen=len(hash_hex) // 2
        )
        return hmac.compare_digest(digest.hex(), hash_hex)
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------- session tokens
def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload: str, secret: str) -> str:
    return _b64(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())


def make_session_token(username: str, secret: str, hours: float, now: float | None = None) -> str:
    exp = int((now or time.time()) + hours * 3600)
    payload = _b64(json.dumps({"u": username, "exp": exp}, separators=(",", ":")).encode())
    return f"{payload}.{_sign(payload, secret)}"


def read_session_token(token: str | None, secret: str, now: float | None = None) -> str | None:
    """Return the username if the token is genuine and not expired, else None."""
    if not token or "." not in token or not secret:
        return None
    payload, signature = token.rsplit(".", 1)
    if not hmac.compare_digest(signature, _sign(payload, secret)):
        return None
    try:
        data = json.loads(_unb64(payload))
    except (ValueError, json.JSONDecodeError):
        return None
    if data.get("exp", 0) < (now or time.time()):
        return None
    return data.get("u")


# ---------------------------------------------------------------- login rate limit
class LoginRateLimiter:
    """At most `attempts` failed logins per `window_s` per client address (per process)."""

    def __init__(self, attempts: int = 5, window_s: int = 300):
        self.attempts = attempts
        self.window_s = window_s
        self._failures: dict[str, deque] = defaultdict(deque)

    def _prune(self, key: str, now: float) -> deque:
        q = self._failures[key]
        while q and now - q[0] > self.window_s:
            q.popleft()
        return q

    def blocked(self, key: str, now: float | None = None) -> bool:
        return len(self._prune(key, now or time.time())) >= self.attempts

    def failed(self, key: str, now: float | None = None) -> None:
        now = now or time.time()
        self._prune(key, now).append(now)

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)


login_limiter = LoginRateLimiter()

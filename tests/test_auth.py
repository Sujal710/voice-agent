"""Dashboard login: hashed passwords, signed HttpOnly session cookies, CSRF guard, rate limit."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from app.auth import (
    COOKIE_NAME,
    LoginRateLimiter,
    hash_password,
    login_limiter,
    make_session_token,
    read_session_token,
    verify_password,
)
from app.config import Settings, get_settings
from app.main import app
from tests.fake_repo import InMemoryRepo

PASSWORD = "correct horse battery"
API_KEY = "test-api-key-123"
CSRF = {"X-Requested-With": "CallDesk"}


def test_password_hashing():
    stored = hash_password(PASSWORD)
    assert stored.startswith("scrypt:") and PASSWORD not in stored
    assert "$" not in stored  # safe inside .env files and docker compose
    assert verify_password(PASSWORD, stored)
    assert not verify_password("wrong password", stored)
    assert not verify_password(PASSWORD, "garbage")
    assert hash_password(PASSWORD) != stored  # random salt


def test_session_tokens_cannot_be_forged_or_reused_after_expiry():
    token = make_session_token("admin", "secret-1", hours=1, now=1000)
    assert read_session_token(token, "secret-1", now=1001) == "admin"
    assert read_session_token(token, "other-secret", now=1001) is None  # wrong key
    payload, sig = token.rsplit(".", 1)
    forged = make_session_token("root", "attacker", hours=1, now=1000).rsplit(".", 1)[0] + "." + sig
    assert read_session_token(forged, "secret-1", now=1001) is None  # edited payload
    assert read_session_token(token, "secret-1", now=1000 + 3601) is None  # expired
    assert read_session_token(None, "secret-1") is None and read_session_token("nonsense", "secret-1") is None


def test_rate_limiter():
    lim = LoginRateLimiter(attempts=3, window_s=60)
    for _ in range(3):
        assert not lim.blocked("1.2.3.4", now=100)
        lim.failed("1.2.3.4", now=100)
    assert lim.blocked("1.2.3.4", now=110)
    assert not lim.blocked("5.6.7.8", now=110)  # other clients unaffected
    assert not lim.blocked("1.2.3.4", now=161)  # window passed


@pytest.fixture
def client(monkeypatch):
    settings = Settings(
        api_key=API_KEY,
        dashboard_username="nidhi",
        dashboard_password_hash=hash_password(PASSWORD),
        session_secret="s" * 48,
        cookie_secure=False,  # TestClient talks plain http
    )
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.settings = settings

    class NoDialer:
        def wake(self):
            pass

    app.state.dialer = NoDialer()
    fake = InMemoryRepo().install(monkeypatch)
    asyncio.run(fake.upsert_product({"sku": "SOLAR", "name": "SolarMax"}))
    login_limiter._failures.clear()
    c = TestClient(app)
    c.settings = settings
    yield c
    app.dependency_overrides.clear()
    login_limiter._failures.clear()


def login(client, password=PASSWORD, username="nidhi"):
    return client.post("/auth/login", json={"username": username, "password": password}, headers=CSRF)


def test_page_needs_no_key_and_stores_none():
    html = TestClient(app).get("/").text
    assert 'id="apikey"' not in html and "X-API-Key" not in html
    assert "localStorage.setItem" not in html
    assert 'type="password"' in html and 'autocomplete="current-password"' in html


def test_dashboard_requires_sign_in(client):
    assert client.get("/api/v1/batches").status_code == 401
    assert client.get("/auth/me").status_code == 401


def test_login_sets_a_secure_httponly_cookie_and_unlocks_the_api(client):
    r = login(client)
    assert r.status_code == 200 and r.json() == {"user": "nidhi"}
    cookie = r.headers["set-cookie"]
    assert cookie.startswith(f"{COOKIE_NAME}=")
    assert "HttpOnly" in cookie and "SameSite=strict" in cookie and "Max-Age=43200" in cookie
    assert PASSWORD not in cookie and API_KEY not in cookie

    assert client.get("/auth/me").json()["user"] == "nidhi"
    assert client.get("/api/v1/batches").status_code == 200


def test_secure_flag_is_set_by_default():
    assert Settings().cookie_secure is True


def test_wrong_password_and_rate_limit(client):
    assert login(client, "nope").status_code == 401
    assert login(client, PASSWORD, username="someone").status_code == 401
    for _ in range(3):
        login(client, "nope")
    r = login(client)  # even the right password is refused while blocked
    assert r.status_code == 429


def test_cookie_requests_that_change_data_need_the_csrf_header(client):
    login(client)
    files = {"file": ("t.csv", b"name,phone\nRavi,9825012345\n", "text/csv")}
    no_header = client.post("/api/v1/batches", files=files, data={"product_sku": "SOLAR"})
    assert no_header.status_code == 403
    ok = client.post("/api/v1/batches", files=files, data={"product_sku": "SOLAR"}, headers=CSRF)
    assert ok.status_code == 201


def test_logout_ends_the_session(client):
    login(client)
    assert client.post("/auth/logout", headers=CSRF).status_code == 200
    assert client.get("/api/v1/batches").status_code == 401


def test_tampered_cookie_is_rejected(client):
    login(client)
    token = client.cookies.get(COOKIE_NAME)
    client.cookies.set(COOKIE_NAME, token[:-3] + "abc")
    assert client.get("/api/v1/batches").status_code == 401


def test_api_key_still_works_for_scripts(client):
    assert client.get("/api/v1/batches", headers={"X-API-Key": API_KEY}).status_code == 200
    assert client.get("/api/v1/batches", headers={"X-API-Key": "wrong"}).status_code == 401


def test_login_disabled_until_configured(monkeypatch):
    s = Settings(api_key=API_KEY, dashboard_password_hash="")
    app.dependency_overrides[get_settings] = lambda: s
    try:
        c = TestClient(app)
        assert c.get("/auth/status").json() == {"login_configured": False}
        r = c.post("/auth/login", json={"username": "admin", "password": "anything"}, headers=CSRF)
        assert r.status_code == 503 and "hash_password" in r.json()["detail"]
    finally:
        app.dependency_overrides.clear()

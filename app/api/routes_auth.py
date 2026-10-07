"""Dashboard sign-in / sign-out."""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from app.auth import COOKIE_NAME, login_limiter, make_session_token, verify_password
from app.config import Settings, get_settings
from app.security import current_user, session_secret

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginIn(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=200)


def _client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post("/login")
async def login(body: LoginIn, request: Request, response: Response, settings: Settings = Depends(get_settings)):
    if not settings.dashboard_password_hash:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Sign-in isn't set up yet. Add DASHBOARD_PASSWORD_HASH to .env (python -m scripts.hash_password).",
        )
    key = _client_key(request)
    if login_limiter.blocked(key):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Too many attempts. Try again in a few minutes.")
    # scrypt is deliberately slow (CPU-bound): run it off the event loop so live calls aren't affected
    password_ok = await asyncio.to_thread(verify_password, body.password, settings.dashboard_password_hash)
    user_ok = body.username == settings.dashboard_username
    if not (user_ok and password_ok):
        login_limiter.failed(key)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong username or password.")
    login_limiter.reset(key)
    token = make_session_token(body.username, session_secret(settings), settings.session_hours)
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=int(settings.session_hours * 3600),
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path="/",
    )
    return {"user": body.username}


@router.post("/logout")
async def logout(response: Response, settings: Settings = Depends(get_settings)):
    response.delete_cookie(COOKIE_NAME, path="/", secure=settings.cookie_secure, httponly=True, samesite="strict")
    return {"ok": True}


@router.get("/me")
async def me(request: Request, settings: Settings = Depends(get_settings)):
    user = current_user(request, settings)
    if not user:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Please sign in")
    return {"user": user, "login_configured": bool(settings.dashboard_password_hash)}


@router.get("/status")
async def auth_status(settings: Settings = Depends(get_settings)):
    """Lets the sign-in page explain when login hasn't been configured."""
    return {"login_configured": bool(settings.dashboard_password_hash)}

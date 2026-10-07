"""Pydantic request/response schemas."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

E164 = r"^\+[1-9]\d{7,14}$"


class FAQ(BaseModel):
    q: str
    a: str


class ProductIn(BaseModel):
    sku: str = Field(min_length=1, max_length=64)
    name: str
    price: str
    description: str = ""
    offer: str | None = None
    features: list[str] = []
    faqs: list[FAQ] = []


class ProductOut(ProductIn):
    id: str


class CustomerIn(BaseModel):
    name: str
    phone: str = Field(pattern=E164, description="E.164 format, e.g. +919876543210")
    preferred_language: str = Field(default="en-IN", description="BCP-47, e.g. gu-IN, hi-IN, en-IN")
    city: str | None = None
    notes: str | None = None
    do_not_call: bool = False


class CustomerOut(CustomerIn):
    id: str


class StartCallRequest(BaseModel):
    customer_id: str
    product_sku: str


class StartCampaignRequest(BaseModel):
    product_sku: str
    customer_ids: list[str] = Field(min_length=1, max_length=1000)


CallStatus = Literal[
    "queued", "initiated", "ringing", "in-progress", "completed", "busy", "no-answer", "failed", "canceled"
]


class TranscriptTurn(BaseModel):
    seq: int
    role: Literal["customer", "agent"]
    content: str = Field(description="English text (what the LLM reasons over)")
    original: str = Field(description="What was actually spoken on the call, in the caller's language")
    language: str | None = None
    interrupted: bool = False
    kind: str = "dialogue"
    created_at: datetime


class CallOut(BaseModel):
    id: str
    customer_id: str
    product_sku: str
    status: str
    twilio_call_sid: str | None = None
    language: str | None = None
    duration_s: int | None = None
    outcome: str | None = None
    report_ready: bool = False
    report_url: str | None = Field(
        default=None, description="Signed link to download the HTML report (no API key needed)"
    )
    transcript: list[TranscriptTurn] = []
    memory_summary: str | None = None
    metrics: dict | None = None
    created_at: datetime
    ended_at: datetime | None = None

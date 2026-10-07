"""Daily CSV uploads: parsing, dialer queue, batch API, per-name reports."""

import asyncio
import csv
import io
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from app.api.downloads import content_disposition, slug
from app.config import Settings, get_settings
from app.main import app
from app.services.csv_import import CSVImportError, normalize_phone, parse_contacts
from app.services.dialer import Dialer, within_calling_hours
from app.services.telephony import Telephony
from tests.fake_repo import InMemoryRepo

API_KEY = "test-api-key-123"
H = {"X-API-Key": API_KEY}

CSV = """Name,Mobile No.
Sujal Thakkar,98765 43210
Priya Patel,+91 99887 76655
Bad Number,12345
Dup,098765-43210
નિરવ પટેલ,9724012345
""".encode()


# ------------------------------------------------------------------ CSV parsing
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("98765 43210", "+919876543210"),
        ("098765-43210", "+919876543210"),
        ("919876543210", "+919876543210"),
        ("+91 98765 43210", "+919876543210"),
        ("(+91) 98765-43210", "+919876543210"),
        ("+1 415 555 0100", "+14155550100"),
        ("12345", None),
        ("9.19876E+11", None),  # Excel mangled it
        ("5876543210", None),  # Indian mobiles start with 6-9
        ("", None),
    ],
)
def test_normalize_phone(raw, expected):
    assert normalize_phone(raw) == expected


def test_parse_contacts_reports_every_rejected_line():
    valid, rejected = parse_contacts(CSV)
    assert [(r["line"], r["name"], r["phone"]) for r in valid] == [
        (2, "Sujal Thakkar", "+919876543210"),
        (3, "Priya Patel", "+919988776655"),
        (6, "નિરવ પટેલ", "+919724012345"),
    ]
    assert [(r["line"], r["reason"]) for r in rejected] == [
        (4, "not a valid phone number"),
        (5, "duplicate of line 2"),
    ]


def test_parse_contacts_handles_excel_formats():
    # UTF-8 BOM + semicolons + different header names
    valid, _ = parse_contacts("﻿Customer Name;Phone Number\nRavi;9825012345\n".encode())
    assert valid == [{"line": 2, "name": "Ravi", "phone": "+919825012345"}]
    # no header row at all
    valid, _ = parse_contacts(b"Ravi,9825012345\nMeera,9724012345\n")
    assert [v["name"] for v in valid] == ["Ravi", "Meera"]
    # name missing -> placeholder, still called
    valid, _ = parse_contacts(b"name,phone\n,9825012345\n")
    assert valid[0]["name"] == "Customer"


@pytest.mark.parametrize(
    "data,message",
    [
        (b"", "empty"),
        (b"name,email\nRavi,r@x.com\n", "phone number column"),
        (b"foo,bar\nbaz,qux\n", "Couldn't find the columns"),
    ],
)
def test_parse_contacts_explains_bad_files(data, message):
    with pytest.raises(CSVImportError, match=message):
        parse_contacts(data)


def test_download_names_support_indian_scripts():
    assert slug("Sujal Thakkar") == "Sujal-Thakkar"
    assert slug("નિરવ પટેલ") == "નિરવ-પટેલ"
    assert slug("A/B: test?") == "AB-test"
    header = content_disposition("call-report-નિરવ.md")
    header.encode("latin-1")  # must be a valid HTTP header
    assert "filename*=UTF-8''call-report-" in header


# ------------------------------------------------------------------ dialer
def settings(**kw):
    base = dict(api_key=API_KEY, calling_hours_start="00:00", calling_hours_end="23:59", dialer_enabled=False)
    return Settings(**{**base, **kw})


def test_calling_hours():
    s = settings(calling_hours_start="09:00", calling_hours_end="21:00")
    ist = ZoneInfo("Asia/Kolkata")
    assert within_calling_hours(s, datetime(2026, 10, 5, 9, 0, tzinfo=ist))
    assert within_calling_hours(s, datetime(2026, 10, 5, 20, 59, tzinfo=ist))
    assert not within_calling_hours(s, datetime(2026, 10, 5, 21, 0, tzinfo=ist))
    assert not within_calling_hours(s, datetime(2026, 10, 5, 7, 30, tzinfo=ist))


class FakeTelephony:
    def __init__(self, fail_for=()):
        self.dialed: list[tuple[str, str]] = []
        self.fail_for = set(fail_for)

    async def place_call(self, call_id, phone):
        if phone in self.fail_for:
            raise RuntimeError("Twilio: invalid number")
        self.dialed.append((call_id, phone))
        return f"CA{len(self.dialed)}"


async def queue_calls(fake: InMemoryRepo, phones: list[str]) -> list[dict]:
    calls = []
    for i, phone in enumerate(phones, start=1):
        cust = await fake.upsert_customer_from_csv(f"Person {i}", phone, "en-IN")
        calls.append(
            await fake.create_call(
                cust["id"], "P1", customer_name=f"Person {i}", phone=phone, batch_id="b1", batch_seq=i
            )
        )
    return calls


async def test_dialer_respects_concurrency_limit_and_order(monkeypatch):
    fake = InMemoryRepo().install(monkeypatch)
    await queue_calls(fake, [f"+91981234567{i}" for i in range(5)])
    tel = FakeTelephony()
    dialer = Dialer(settings(max_concurrent_outbound_calls=2), tel)

    assert await dialer.tick() == 2
    assert [p for _, p in tel.dialed] == ["+919812345670", "+919812345671"]
    assert await dialer.tick() == 0  # both lines busy

    first = next(c for c in fake.calls.values() if c["phone"] == "+919812345670")
    first["status"] = "completed"  # one call ends
    assert await dialer.tick() == 1
    assert tel.dialed[-1][1] == "+919812345672"


async def test_dialer_waits_outside_calling_hours(monkeypatch):
    fake = InMemoryRepo().install(monkeypatch)
    await queue_calls(fake, ["+919812345670"])
    s = settings(calling_hours_start="00:00", calling_hours_end="00:01")
    tel = FakeTelephony()
    if not within_calling_hours(s):
        assert await Dialer(s, tel).tick() == 0
        assert tel.dialed == []
        assert next(iter(fake.calls.values()))["status"] == "queued"


async def test_dialer_skips_do_not_call_and_records_failures(monkeypatch):
    fake = InMemoryRepo().install(monkeypatch)
    calls = await queue_calls(fake, ["+919812345670", "+919812345671"])
    fake.customers[calls[0]["customer_id"]]["do_not_call"] = True
    tel = FakeTelephony(fail_for={"+919812345671"})
    await Dialer(settings(), tel).tick()
    assert tel.dialed == []
    assert fake.calls[calls[0]["id"]]["status"] == "canceled"
    assert fake.calls[calls[1]["id"]]["status"] == "failed"
    assert "invalid number" in fake.calls[calls[1]["id"]]["error"]


async def test_two_dialers_never_dial_the_same_call(monkeypatch):
    fake = InMemoryRepo().install(monkeypatch)
    await queue_calls(fake, [f"+91981234567{i}" for i in range(6)])
    tel = FakeTelephony()
    s = settings(max_concurrent_outbound_calls=6)
    await asyncio.gather(Dialer(s, tel).tick(), Dialer(s, tel).tick())
    phones = [p for _, p in tel.dialed]
    assert len(phones) == len(set(phones)) == 6


# ------------------------------------------------------------------ batch API
@pytest.fixture
def client(monkeypatch):
    s = settings(public_base_url="https://desk.example.com")
    app.dependency_overrides[get_settings] = lambda: s
    app.state.settings = s
    app.state.telephony = Telephony(s)

    class WakeOnly:
        woken = 0

        def wake(self):
            WakeOnly.woken += 1

    app.state.dialer = WakeOnly()

    class LLM:
        async def generate_report(self, lines, facts):
            return {"outcome": "interested", "outcome_evidence": [2], "summary": "Asked about EMI."}

    app.state.llm = LLM()
    fake = InMemoryRepo().install(monkeypatch)
    asyncio.run(fake.upsert_product({"sku": "SOLAR", "name": "SolarMax 3 kW"}))
    c = TestClient(app)
    c.fake, c.dialer = fake, WakeOnly
    yield c
    app.dependency_overrides.clear()


def upload(client, data=CSV, sku="SOLAR"):
    return client.post(
        "/api/v1/batches", headers=H, files={"file": ("today.csv", data, "text/csv")}, data={"product_sku": sku}
    )


def test_preview_does_not_save_anything(client):
    r = client.post("/api/v1/batches/preview", headers=H, files={"file": ("today.csv", CSV, "text/csv")})
    assert r.status_code == 200
    assert r.json()["valid_count"] == 3 and r.json()["rejected_count"] == 2
    assert client.fake.calls == {} and client.fake.batches == {}


def test_upload_queues_calls_with_names_and_wakes_dialer(client):
    r = upload(client)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["queued"] == 3 and len(body["rejected"]) == 2
    assert client.dialer.woken == 1
    calls = sorted(client.fake.calls.values(), key=lambda c: c["batch_seq"])
    assert [(c["customer_name"], c["phone"], c["status"]) for c in calls] == [
        ("Sujal Thakkar", "+919876543210", "queued"),
        ("Priya Patel", "+919988776655", "queued"),
        ("નિરવ પટેલ", "+919724012345", "queued"),
    ]
    assert all(c["batch_id"] == body["batch_id"] for c in calls)


def test_reupload_next_day_reuses_customers_and_skips_do_not_call(client):
    upload(client)
    sujal = next(c for c in client.fake.customers.values() if c["phone"] == "+919876543210")
    sujal["do_not_call"] = True
    sujal["preferred_language"] = "gu-IN"  # learned on yesterday's call
    r = upload(client).json()
    assert r["queued"] == 2
    assert [s["name"] for s in r["skipped_do_not_call"]] == ["Sujal Thakkar"]
    assert len(client.fake.customers) == 3  # no duplicates
    assert sujal["preferred_language"] == "gu-IN"  # kept


def test_upload_errors_are_clear(client):
    assert upload(client, sku="NOPE").status_code == 404
    r = upload(client, data=b"name,email\nRavi,r@x.com\n")
    assert r.status_code == 422 and "phone number column" in r.json()["detail"]
    assert upload(client, data=b"name,phone\nRavi,123\n").status_code == 422
    assert (
        client.post("/api/v1/batches", files={"file": ("a.csv", CSV)}, data={"product_sku": "SOLAR"}).status_code == 401
    )


def finish_call(fake: InMemoryRepo, call: dict, text: str):
    call["status"] = "completed"
    fake.messages += [
        {"session_id": call["id"], "seq": 1, "role": "agent", "content": "Hello", "original": "નમસ્તે"},
        {"session_id": call["id"], "seq": 2, "role": "customer", "content": text, "original": text},
    ]


def test_batch_detail_zip_and_summary_are_named_per_person(client):
    batch_id = upload(client).json()["batch_id"]
    calls = sorted(client.fake.calls.values(), key=lambda c: c["batch_seq"])
    finish_call(client.fake, calls[0], "What is the EMI?")
    calls[1]["status"] = "no-answer"
    # calls[2] still queued

    detail = client.get(f"/api/v1/batches/{batch_id}", headers=H).json()
    assert detail["product_name"] == "SolarMax 3 kW"
    assert detail["status_counts"] == {"completed": 1, "no-answer": 1, "queued": 1}

    z = client.get(f"/api/v1/batches/{batch_id}/reports.zip", headers=H)
    assert z.status_code == 200 and z.headers["content-type"] == "application/zip"
    files = zipfile.ZipFile(io.BytesIO(z.content))
    names = sorted(files.namelist())
    assert names == ["001-Sujal-Thakkar-3210.md", "002-Priya-Patel-6655.md", "summary.csv"]
    assert "# Call report: Sujal Thakkar" in files.read("001-Sujal-Thakkar-3210.md").decode()
    assert "no conversation" in files.read("002-Priya-Patel-6655.md").decode()

    # reports were saved in the database (with the name), not only zipped
    assert calls[0]["report"]["meta"]["customer"]["name"] == "Sujal Thakkar"
    assert calls[0]["report_ready"] and calls[1]["report_ready"] and not calls[2].get("report_ready")

    rows = list(csv.DictReader(io.StringIO(files.read("summary.csv").decode("utf-8-sig"))))
    assert [(r["name"], r["status"], r["outcome"]) for r in rows] == [
        ("Sujal Thakkar", "completed", "interested"),
        ("Priya Patel", "no-answer", "no_conversation"),
        ("નિરવ પટેલ", "queued", ""),
    ]
    assert rows[0]["report_link"].startswith("https://desk.example.com/reports/")

    detail = client.get(f"/api/v1/batches/{batch_id}", headers=H).json()
    assert detail["calls"][0]["report_url"] and detail["calls"][2]["report_url"] is None

    one = client.get(f"/api/v1/calls/{calls[0]['id']}/report", headers=H)
    assert "call-report-Sujal-Thakkar-" in one.headers["content-disposition"]


def test_cancel_stops_only_waiting_calls(client):
    batch_id = upload(client).json()["batch_id"]
    calls = sorted(client.fake.calls.values(), key=lambda c: c["batch_seq"])
    calls[0]["status"] = "in-progress"
    assert client.post(f"/api/v1/batches/{batch_id}/cancel", headers=H).json() == {"canceled": 2}
    assert [c["status"] for c in calls] == ["in-progress", "canceled", "canceled"]


def test_upload_page_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Today's list" in r.text and "/api/v1/batches" in r.text

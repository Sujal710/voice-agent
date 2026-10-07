"""Daily CSV uploads ("batches"): upload name+phone list, track calls, download reports by name."""

import csv
import io
import zipfile
from datetime import datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import Response

from app import repositories as repo
from app.api.downloads import content_disposition, slug
from app.pipeline.report import create_report_for_call, render
from app.security import report_token, require_api_key
from app.services.csv_import import MAX_BYTES, CSVImportError, parse_contacts

router = APIRouter(prefix="/api/v1/batches", dependencies=[Depends(require_api_key)], tags=["batches"])

TERMINAL = {"completed", "busy", "no-answer", "failed", "canceled"}


async def _read_csv(file: UploadFile, request: Request) -> tuple[list[dict], list[dict]]:
    data = await file.read(MAX_BYTES + 1)
    try:
        return parse_contacts(data, request.app.state.settings.default_country_code)
    except CSVImportError as exc:
        raise HTTPException(422, str(exc)) from exc


def _report_url(request: Request, call_id: str) -> str:
    s = request.app.state.settings
    return f"{s.public_base_url.rstrip('/')}/reports/{call_id}?token={report_token(call_id, s)}"


@router.get("/dialer-status")
async def dialer_status(request: Request):
    """Calling window and live capacity, shown on the upload page."""
    from app.services.dialer import within_calling_hours

    s = request.app.state.settings
    return {
        "calling_hours": f"{s.calling_hours_start}–{s.calling_hours_end}",
        "timezone": s.calling_timezone,
        "open_now": within_calling_hours(s),
        "max_concurrent": s.max_concurrent_outbound_calls,
        "active_calls": await repo.count_active_calls(),
    }


@router.post("/preview")
async def preview_csv(request: Request, file: UploadFile = File(...)):
    """Check a CSV without saving or calling anyone."""
    valid, rejected = await _read_csv(file, request)
    return {"valid_count": len(valid), "rejected_count": len(rejected), "valid": valid[:200], "rejected": rejected}


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_batch(
    request: Request,
    file: UploadFile = File(..., description="CSV with columns: name, phone"),
    product_sku: str = Form(...),
    label: str | None = Form(None),
):
    """Upload today's CSV. Every valid number is queued and the dialer starts calling."""
    product = await repo.get_product(product_sku)
    if not product:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Product {product_sku} not found")
    valid, rejected = await _read_csv(file, request)
    if not valid:
        raise HTTPException(422, {"message": "No valid rows", "rejected": rejected})

    settings = request.app.state.settings
    today = datetime.now(ZoneInfo(settings.calling_timezone)).strftime("%d %b %Y")
    batch = await repo.create_batch(
        {
            "label": label or f"{today} – {(file.filename or 'upload').removesuffix('.csv')}",
            "filename": file.filename,
            "product_sku": product_sku,
            "total_rows": len(valid) + len(rejected),
            "queued": 0,
            "skipped_dnc": 0,
            "rejected": rejected,
        }
    )
    queued = skipped = 0
    skipped_rows = []
    for seq, row in enumerate(valid, start=1):
        customer = await repo.upsert_customer_from_csv(row["name"], row["phone"], settings.default_language)
        if customer.get("do_not_call"):
            skipped += 1
            skipped_rows.append({**row, "reason": "on the do-not-call list"})
            continue
        await repo.create_call(
            customer["id"],
            product_sku,
            customer_name=row["name"],
            phone=row["phone"],
            batch_id=batch["id"],
            batch_seq=seq,
        )
        queued += 1
    await repo.update_batch(batch["id"], {"queued": queued, "skipped_dnc": skipped})
    request.app.state.dialer.wake()
    return {
        "batch_id": batch["id"],
        "label": batch["label"],
        "queued": queued,
        "skipped_do_not_call": skipped_rows,
        "rejected": rejected,
    }


@router.get("")
async def list_batches():
    batches = await repo.list_batches()
    counts = await repo.batch_status_counts([b["id"] for b in batches])
    return [{**b, "status_counts": counts.get(b["id"], {})} for b in batches]


@router.get("/{batch_id}")
async def get_batch(batch_id: str, request: Request):
    batch = await repo.get_batch(batch_id)
    if not batch:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Batch not found")
    calls = await repo.list_batch_calls(batch_id)
    rows = []
    for c in calls:
        report = c.get("report") or {}
        rows.append(
            {
                "call_id": c["id"],
                "name": c.get("customer_name"),
                "phone": c.get("phone"),
                "status": c.get("status"),
                "outcome": c.get("outcome"),
                "summary": report.get("summary"),
                "callback_time": report.get("callback_time"),
                "follow_ups": [r["text"] for r in report.get("unanswered_requests") or []],
                "language": c.get("language"),
                "duration_s": c.get("duration_s"),
                "error": c.get("error"),
                "report_url": _report_url(request, c["id"]) if c.get("report_ready") else None,
            }
        )
    counts = (await repo.batch_status_counts([batch_id]))[batch_id]
    product = await repo.get_product(batch["product_sku"]) or {}
    return {**batch, "product_name": product.get("name"), "status_counts": counts, "calls": rows}


@router.post("/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    """Stop calling the remaining (not yet dialed) numbers in this batch."""
    if not await repo.get_batch(batch_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Batch not found")
    return {"canceled": await repo.cancel_queued_calls(batch_id)}


def _summary_csv(calls: list[dict], request: Request) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "name",
            "phone",
            "status",
            "outcome",
            "callback_time",
            "needs_follow_up",
            "summary",
            "language",
            "duration_s",
            "report_link",
        ]  # fmt: skip
    )
    for c in calls:
        report = c.get("report") or {}
        writer.writerow(
            [
                c.get("customer_name"),
                c.get("phone"),
                c.get("status"),
                c.get("outcome") or "",
                report.get("callback_time") or "",
                "; ".join(r["text"] for r in report.get("unanswered_requests") or []),
                report.get("summary") or "",
                c.get("language") or "",
                c.get("duration_s") or "",
                _report_url(request, c["id"]) if c.get("report_ready") else "",
            ]
        )
    return "﻿" + buf.getvalue()  # BOM so Excel shows Gujarati/Hindi names correctly


@router.get("/{batch_id}/summary.csv")
async def batch_summary_csv(batch_id: str, request: Request):
    """One row per person: name, phone, outcome, callback time, summary, report link."""
    batch = await repo.get_batch(batch_id)
    if not batch:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Batch not found")
    calls = await repo.list_batch_calls(batch_id)
    return Response(
        _summary_csv(calls, request),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": content_disposition(f"{slug(batch['label'])}-summary.csv")},
    )


@router.get("/{batch_id}/reports.zip")
async def batch_reports_zip(batch_id: str, request: Request, format: str = "md"):
    """Every finished call's report, one file per person, named after them, plus summary.csv."""
    if format not in {"md", "html", "json"}:
        raise HTTPException(422, "format must be md, html or json")
    batch = await repo.get_batch(batch_id)
    if not batch:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Batch not found")
    calls = await repo.list_batch_calls(batch_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for c in calls:
            if c.get("status") not in TERMINAL:
                continue  # still queued or on the phone
            full = await repo.get_call(c["id"])
            report = (full or {}).get("report") or await create_report_for_call(request.app.state.llm, c["id"])
            if not report:
                continue
            messages = await repo.get_messages(c["id"])
            phone_tail = (c.get("phone") or "")[-4:]
            name = f"{c.get('batch_seq', 0):03d}-{slug(c.get('customer_name'))}-{phone_tail}.{format}"
            zf.writestr(name, render(report, messages, format))
        calls = await repo.list_batch_calls(batch_id)  # refreshed (new reports)
        zf.writestr("summary.csv", _summary_csv(calls, request))
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": content_disposition(f"{slug(batch['label'])}-reports.zip")},
    )

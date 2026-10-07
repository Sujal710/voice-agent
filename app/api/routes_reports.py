"""Post-call report download.

* `GET /api/v1/calls/{id}/report?format=md|html|json`  (X-API-Key)
* `POST /api/v1/calls/{id}/report`                     regenerate (X-API-Key)
* `GET /reports/{id}?token=...&format=html`            signed link, opens in a browser
"""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import Response

from app import repositories as repo
from app.api.downloads import content_disposition, slug
from app.config import Settings, get_settings
from app.pipeline.report import RENDERERS, create_report_for_call, render
from app.security import require_api_key, verify_report_token

ReportFormat = Literal["md", "html", "json"]
TERMINAL = {"completed", "busy", "no-answer", "failed", "canceled"}

api_router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)], tags=["reports"])
public_router = APIRouter(tags=["reports"])


async def _report_response(request: Request, call_id: str, fmt: str, inline: bool) -> Response:
    call = await repo.get_call(call_id)
    if not call:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Call not found")
    report = call.get("report")
    if not report:
        if call.get("status") not in TERMINAL:
            raise HTTPException(
                status.HTTP_409_CONFLICT, "The report is created when the call ends. Try again shortly."
            )
        # Call ended without a live conversation (no answer, busy...) or the
        # report task didn't run: build it now.
        report = await create_report_for_call(request.app.state.llm, call_id)
    # Status callbacks can arrive after the report was built: show the latest call facts
    report.setdefault("meta", {}).update(
        {k: call.get(k) for k in ("status", "duration_s", "ended_at", "language") if call.get(k) is not None}
    )
    messages = await repo.get_messages(call_id)
    media_type, ext = RENDERERS[fmt]
    # File named after the person, e.g. call-report-Sujal-Thakkar-05-Oct-2026.md
    name = slug(call.get("customer_name") or (report.get("meta", {}).get("customer") or {}).get("name") or "")
    date = call["created_at"].strftime("%d-%b-%Y") if call.get("created_at") else call_id
    return Response(
        render(report, messages, fmt),
        media_type=media_type,
        headers={"Content-Disposition": content_disposition(f"call-report-{name}-{date}.{ext}", inline=inline)},
    )


@api_router.get("/calls/{call_id}/report")
async def download_report(request: Request, call_id: str, format: ReportFormat = "md"):
    """Download the post-call report as Markdown, HTML or JSON."""
    return await _report_response(request, call_id, format, inline=False)


@api_router.post("/calls/{call_id}/report", status_code=status.HTTP_201_CREATED)
async def regenerate_report(request: Request, call_id: str):
    """Rebuild the report (e.g. after a Groq outage)."""
    report = await create_report_for_call(request.app.state.llm, call_id)
    if report is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Call not found")
    return report


@public_router.get("/reports/{call_id}")
async def signed_report(
    request: Request,
    call_id: str,
    token: str,
    format: ReportFormat = "html",
    download: bool = False,
    settings: Settings = Depends(get_settings),
):
    """Open the report in a browser from the signed `report_url` (no API key needed)."""
    if not verify_report_token(call_id, token, settings):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Invalid report link")
    return await _report_response(request, call_id, format, inline=not download)

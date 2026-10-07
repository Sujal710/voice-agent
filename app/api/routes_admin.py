"""Management REST API: products, customers, calls and campaigns (X-API-Key protected)."""

import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from app import repositories as repo
from app.logging_config import log
from app.models import (
    CallOut,
    CustomerIn,
    CustomerOut,
    ProductIn,
    ProductOut,
    StartCallRequest,
    StartCampaignRequest,
)
from app.security import report_token, require_api_key

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", dependencies=[Depends(require_api_key)])


# ---------------- products ----------------
@router.put("/products", response_model=ProductOut)
async def upsert_product(body: ProductIn):
    return await repo.upsert_product(body.model_dump())


@router.get("/products", response_model=list[ProductOut])
async def list_products():
    return await repo.list_products()


# ---------------- customers ----------------
@router.put("/customers", response_model=CustomerOut)
async def upsert_customer(body: CustomerIn):
    return await repo.upsert_customer(body.model_dump())


@router.get("/customers", response_model=list[CustomerOut])
async def list_customers(limit: int = Query(100, le=500), skip: int = Query(0, ge=0)):
    return await repo.list_customers(limit, skip)


# ---------------- calls ----------------
async def _start_call(request: Request, customer_id: str, product_sku: str) -> dict:
    customer = await repo.get_customer(customer_id)
    if not customer:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Customer {customer_id} not found")
    if customer.get("do_not_call"):
        raise HTTPException(status.HTTP_409_CONFLICT, "Customer is on the do-not-call list")
    if not await repo.get_product(product_sku):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Product {product_sku} not found")

    call = await repo.create_call(
        customer_id, product_sku, customer_name=customer.get("name"), phone=customer.get("phone")
    )
    telephony = request.app.state.telephony
    async with request.app.state.dial_semaphore:  # cap concurrent Twilio API dials
        try:
            sid = await telephony.place_call(call["id"], customer["phone"])
        except Exception as exc:
            await repo.update_call(call["id"], {"status": "failed", "error": str(exc)})
            log(logger, logging.ERROR, "dial_failed", call_id=call["id"], error=str(exc))
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"Twilio error: {exc}") from exc
    await repo.update_call(call["id"], {"status": "initiated", "twilio_call_sid": sid})
    return {**call, "status": "initiated", "twilio_call_sid": sid}


@router.post("/calls", response_model=CallOut, status_code=status.HTTP_201_CREATED)
async def start_call(body: StartCallRequest, request: Request):
    """Place one outbound sales call."""
    return await _start_call(request, body.customer_id, body.product_sku)


@router.post("/campaigns", status_code=status.HTTP_202_ACCEPTED)
async def start_campaign(body: StartCampaignRequest, request: Request):
    """Dial many customers concurrently (bounded by MAX_CONCURRENT_OUTBOUND_CALLS)."""

    async def one(cid: str) -> dict:
        try:
            call = await _start_call(request, cid, body.product_sku)
            return {"customer_id": cid, "call_id": call["id"], "status": "initiated"}
        except HTTPException as exc:
            return {"customer_id": cid, "status": "skipped", "reason": exc.detail}

    results = await asyncio.gather(*(one(cid) for cid in body.customer_ids))
    return {"product_sku": body.product_sku, "results": results}


@router.get("/calls", response_model=list[CallOut])
async def list_calls(limit: int = Query(50, le=200), status_filter: str | None = Query(None, alias="status")):
    return await repo.list_calls(limit, status_filter)


@router.get("/calls/{call_id}", response_model=CallOut)
async def get_call(call_id: str, request: Request):
    """Call details with the full transcript (original language + English) and report link."""
    call = await repo.get_call(call_id, with_transcript=True)
    if not call:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Call not found")
    if call.get("report_ready"):
        settings = request.app.state.settings
        call["report_url"] = (
            f"{settings.public_base_url.rstrip('/')}/reports/{call_id}?token={report_token(call_id, settings)}"
        )
    return call

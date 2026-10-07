"""Data-access functions. Keeping Mongo queries here keeps routes and the
call pipeline free of database details."""

from datetime import UTC, datetime

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import ReturnDocument

from app.db import get_db


def now() -> datetime:
    return datetime.now(UTC)


def oid(value: str) -> ObjectId | None:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        return None


def serialize(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    doc = dict(doc)
    doc["id"] = str(doc.pop("_id"))
    return doc


# ---------- products ----------
async def upsert_product(data: dict) -> dict:
    db = get_db()
    await db.products.update_one({"sku": data["sku"]}, {"$set": {**data, "updated_at": now()}}, upsert=True)
    return serialize(await db.products.find_one({"sku": data["sku"]}))


async def get_product(sku: str) -> dict | None:
    return serialize(await get_db().products.find_one({"sku": sku}))


async def list_products() -> list[dict]:
    return [serialize(d) async for d in get_db().products.find().limit(500)]


# ---------- customers ----------
async def upsert_customer(data: dict) -> dict:
    db = get_db()
    await db.customers.update_one({"phone": data["phone"]}, {"$set": {**data, "updated_at": now()}}, upsert=True)
    return serialize(await db.customers.find_one({"phone": data["phone"]}))


async def get_customer(customer_id: str) -> dict | None:
    _id = oid(customer_id)
    return serialize(await get_db().customers.find_one({"_id": _id})) if _id else None


async def list_customers(limit: int = 100, skip: int = 0) -> list[dict]:
    cursor = get_db().customers.find().sort("_id", -1).skip(skip).limit(limit)
    return [serialize(d) async for d in cursor]


async def set_customer_language(customer_id: str, language: str) -> None:
    _id = oid(customer_id)
    if _id:
        await get_db().customers.update_one({"_id": _id}, {"$set": {"preferred_language": language}})


async def mark_do_not_call(customer_id: str) -> None:
    _id = oid(customer_id)
    if _id:
        await get_db().customers.update_one({"_id": _id}, {"$set": {"do_not_call": True}})


# ---------- calls ----------
async def create_call(customer_id: str, product_sku: str, **extra) -> dict:
    """extra: customer_name, phone, batch_id (name/phone are copied so reports keep the
    name exactly as it appeared in that day's CSV)."""
    doc = {
        **extra,
        "customer_id": customer_id,
        "product_sku": product_sku,
        "status": "queued",
        "twilio_call_sid": None,
        "created_at": now(),
    }
    res = await get_db().calls.insert_one(doc)
    doc["_id"] = res.inserted_id
    return serialize(doc)


async def get_call(call_id: str, with_transcript: bool = False) -> dict | None:
    _id = oid(call_id)
    call = serialize(await get_db().calls.find_one({"_id": _id})) if _id else None
    if call and with_transcript:
        call["transcript"] = await get_messages(call_id)
    return call


async def list_calls(limit: int = 50, status: str | None = None) -> list[dict]:
    query = {"status": status} if status else {}
    cursor = get_db().calls.find(query, {"report": 0}).sort("created_at", -1).limit(limit)
    return [serialize(d) async for d in cursor]


async def update_call(call_id: str, fields: dict) -> None:
    _id = oid(call_id)
    if _id:
        await get_db().calls.update_one({"_id": _id}, {"$set": fields})


async def save_report(call_id: str, report: dict) -> None:
    await update_call(call_id, {"report": report, "outcome": report.get("outcome"), "report_ready": True})


async def recent_call_reports(customer_id: str, exclude_call_id: str, limit: int = 3) -> list[dict]:
    """Long-term memory: compact notes from this customer's previous calls."""
    cursor = (
        get_db()
        .calls.find(
            {"customer_id": customer_id, "report_ready": True, "_id": {"$ne": oid(exclude_call_id)}},
            {"report.summary": 1, "report.outcome": 1, "report.commitments": 1, "created_at": 1},
        )
        .sort("created_at", -1)
        .limit(limit)
    )
    out = []
    async for d in cursor:
        r = d.get("report", {})
        out.append(
            {
                "date": d["created_at"].strftime("%d %b %Y"),
                "outcome": r.get("outcome", "unknown"),
                "summary": r.get("summary", ""),
                "next_steps": [c["text"] for c in r.get("commitments", [])],
            }
        )
    return out


# ---------- messages (LangChain-style chat history: one document per message) ----------
async def append_message(doc: dict) -> None:
    await get_db().messages.insert_one(dict(doc))


async def get_messages(session_id: str) -> list[dict]:
    cursor = get_db().messages.find({"session_id": session_id}, {"_id": 0}).sort("seq", 1)
    return [d async for d in cursor]


async def update_call_by_sid(call_sid: str, fields: dict) -> None:
    await get_db().calls.update_one({"twilio_call_sid": call_sid}, {"$set": fields})


# ---------- batches (daily CSV uploads) ----------
ACTIVE_STATUSES = ["dialing", "initiated", "ringing", "in-progress"]


async def upsert_customer_from_csv(name: str, phone: str, default_language: str) -> dict:
    """Create the customer, or refresh the name of an existing one. Keeps their learned
    preferred language and do-not-call flag."""
    db = get_db()
    await db.customers.update_one(
        {"phone": phone},
        {
            "$set": {"name": name, "updated_at": now()},
            "$setOnInsert": {"preferred_language": default_language, "do_not_call": False, "created_at": now()},
        },
        upsert=True,
    )
    return serialize(await db.customers.find_one({"phone": phone}))


async def create_batch(doc: dict) -> dict:
    doc = {**doc, "created_at": now()}
    res = await get_db().batches.insert_one(doc)
    doc["_id"] = res.inserted_id
    return serialize(doc)


async def update_batch(batch_id: str, fields: dict) -> None:
    _id = oid(batch_id)
    if _id:
        await get_db().batches.update_one({"_id": _id}, {"$set": fields})


async def get_batch(batch_id: str) -> dict | None:
    _id = oid(batch_id)
    return serialize(await get_db().batches.find_one({"_id": _id})) if _id else None


async def list_batches(limit: int = 30) -> list[dict]:
    cursor = get_db().batches.find({}, {"rejected": 0}).sort("created_at", -1).limit(limit)
    return [serialize(d) async for d in cursor]


async def batch_status_counts(batch_ids: list[str]) -> dict[str, dict]:
    """{batch_id: {"queued": 3, "completed": 5, ...}, ...} with one aggregation."""
    pipeline = [
        {"$match": {"batch_id": {"$in": batch_ids}}},
        {"$group": {"_id": {"b": "$batch_id", "s": "$status"}, "n": {"$sum": 1}}},
    ]
    out: dict[str, dict] = {b: {} for b in batch_ids}
    async for row in await get_db().calls.aggregate(pipeline):
        out.setdefault(row["_id"]["b"], {})[row["_id"]["s"]] = row["n"]
    return out


async def list_batch_calls(batch_id: str) -> list[dict]:
    cursor = get_db().calls.find(
        {"batch_id": batch_id},
        {"report.summary": 1, "report.callback_time": 1, "report.outcome": 1, "report.unanswered_requests": 1,
         "customer_name": 1, "phone": 1,
         "status": 1, "outcome": 1, "report_ready": 1, "duration_s": 1, "language": 1, "error": 1,
         "customer_id": 1, "product_sku": 1, "created_at": 1, "ended_at": 1, "batch_seq": 1},
    ).sort("batch_seq", 1)  # fmt: skip
    return [serialize(d) async for d in cursor]


async def cancel_queued_calls(batch_id: str) -> int:
    res = await get_db().calls.update_many(
        {"batch_id": batch_id, "status": "queued"}, {"$set": {"status": "canceled", "ended_at": now()}}
    )
    return res.modified_count


# ---------- dialer queue ----------
async def count_active_calls() -> int:
    return await get_db().calls.count_documents({"status": {"$in": ACTIVE_STATUSES}})


async def claim_next_queued_call(worker_id: str) -> dict | None:
    """Atomically move the oldest queued batch call to 'dialing'.

    find_one_and_update is atomic in MongoDB, so with several app replicas
    running a dialer each, a call can never be dialed twice."""
    doc = await get_db().calls.find_one_and_update(
        {"status": "queued", "batch_id": {"$exists": True}},
        {"$set": {"status": "dialing", "dialer": worker_id, "dialing_at": now()}},
        sort=[("created_at", 1), ("batch_seq", 1)],
        return_document=ReturnDocument.AFTER,
    )
    return serialize(doc)


async def requeue_stale_dialing(older_than_s: int = 120) -> int:
    """Recover calls stuck in 'dialing' (e.g. the server restarted mid-dial)."""
    from datetime import timedelta

    res = await get_db().calls.update_many(
        {"status": "dialing", "dialing_at": {"$lt": now() - timedelta(seconds=older_than_s)}},
        {"$set": {"status": "queued"}},
    )
    return res.modified_count

"""In-memory stand-in for app.repositories, used by tests (no MongoDB needed)."""

import itertools
from datetime import UTC, datetime, timedelta

from app import repositories as repo

ACTIVE = repo.ACTIVE_STATUSES


class InMemoryRepo:
    def __init__(self):
        self._ids = itertools.count(1)
        self.products: dict[str, dict] = {}
        self.customers: dict[str, dict] = {}
        self.calls: dict[str, dict] = {}
        self.batches: dict[str, dict] = {}
        self.messages: list[dict] = []
        self.clock = datetime(2026, 10, 5, 4, 0, tzinfo=UTC)

    def _id(self) -> str:
        return f"{next(self._ids):024x}"

    def now(self):
        self.clock += timedelta(seconds=1)
        return self.clock

    def install(self, monkeypatch=None):
        """Replace the real repository functions with these in-memory ones."""
        names = [
            "get_product", "list_products", "upsert_product", "get_customer", "upsert_customer_from_csv",
            "create_call", "get_call", "update_call", "list_calls", "save_report", "get_messages",
            "append_message", "recent_call_reports", "create_batch", "update_batch", "get_batch", "list_batches",
            "batch_status_counts", "list_batch_calls", "cancel_queued_calls", "count_active_calls",
            "claim_next_queued_call", "requeue_stale_dialing", "set_customer_language", "mark_do_not_call",
        ]  # fmt: skip
        for name in names:
            if monkeypatch:
                monkeypatch.setattr(repo, name, getattr(self, name))
            else:
                setattr(repo, name, getattr(self, name))
        return self

    # products / customers
    async def upsert_product(self, data):
        self.products[data["sku"]] = {"id": data["sku"], **data}
        return self.products[data["sku"]]

    async def get_product(self, sku):
        return self.products.get(sku)

    async def list_products(self):
        return list(self.products.values())

    async def get_customer(self, cid):
        return self.customers.get(cid)

    async def upsert_customer_from_csv(self, name, phone, default_language):
        for c in self.customers.values():
            if c["phone"] == phone:
                c["name"] = name
                return c
        cid = self._id()
        self.customers[cid] = {
            "id": cid,
            "name": name,
            "phone": phone,
            "preferred_language": default_language,
            "do_not_call": False,
        }
        return self.customers[cid]

    async def set_customer_language(self, cid, lang):
        self.customers.get(cid, {})["preferred_language"] = lang

    async def mark_do_not_call(self, cid):
        self.customers.get(cid, {})["do_not_call"] = True

    # calls
    async def create_call(self, customer_id, product_sku, **extra):
        cid = self._id()
        self.calls[cid] = {
            **extra,
            "id": cid,
            "customer_id": customer_id,
            "product_sku": product_sku,
            "status": "queued",
            "twilio_call_sid": None,
            "created_at": self.now(),
        }
        return dict(self.calls[cid])

    async def get_call(self, cid, with_transcript=False):
        call = self.calls.get(cid)
        if call is None:
            return None
        call = dict(call)
        if with_transcript:
            call["transcript"] = await self.get_messages(cid)
        return call

    async def update_call(self, cid, fields):
        if cid in self.calls:
            self.calls[cid].update(fields)

    async def list_calls(self, limit=50, status=None):
        return [c for c in self.calls.values() if not status or c["status"] == status][:limit]

    async def save_report(self, cid, report):
        await self.update_call(cid, {"report": report, "outcome": report.get("outcome"), "report_ready": True})

    async def get_messages(self, cid):
        return sorted((m for m in self.messages if m["session_id"] == cid), key=lambda m: m["seq"])

    async def append_message(self, doc):
        self.messages.append(dict(doc))

    async def recent_call_reports(self, customer_id, exclude, limit=3):
        return []

    # batches
    async def create_batch(self, doc):
        bid = self._id()
        self.batches[bid] = {**doc, "id": bid, "created_at": self.now()}
        return dict(self.batches[bid])

    async def update_batch(self, bid, fields):
        self.batches[bid].update(fields)

    async def get_batch(self, bid):
        return dict(self.batches[bid]) if bid in self.batches else None

    async def list_batches(self, limit=30):
        rows = sorted(self.batches.values(), key=lambda b: b["created_at"], reverse=True)[:limit]
        return [{k: v for k, v in b.items() if k != "rejected"} for b in rows]

    async def batch_status_counts(self, batch_ids):
        out = {b: {} for b in batch_ids}
        for c in self.calls.values():
            if c.get("batch_id") in out:
                out[c["batch_id"]][c["status"]] = out[c["batch_id"]].get(c["status"], 0) + 1
        return out

    async def list_batch_calls(self, bid):
        rows = [dict(c) for c in self.calls.values() if c.get("batch_id") == bid]
        return sorted(rows, key=lambda c: c.get("batch_seq", 0))

    async def cancel_queued_calls(self, bid):
        n = 0
        for c in self.calls.values():
            if c.get("batch_id") == bid and c["status"] == "queued":
                c["status"] = "canceled"
                n += 1
        return n

    # dialer queue
    async def count_active_calls(self):
        return sum(1 for c in self.calls.values() if c["status"] in ACTIVE)

    async def claim_next_queued_call(self, worker_id):
        queued = [c for c in self.calls.values() if c["status"] == "queued" and "batch_id" in c]
        if not queued:
            return None
        call = min(queued, key=lambda c: (c["created_at"], c.get("batch_seq", 0)))
        call.update(status="dialing", dialer=worker_id)
        return dict(call)

    async def requeue_stale_dialing(self, older_than_s=120):
        return 0

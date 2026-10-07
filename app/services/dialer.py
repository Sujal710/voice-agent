"""Background dialer: works through queued calls from CSV batches.

* Keeps at most MAX_CONCURRENT_OUTBOUND_CALLS calls live at once (across all replicas,
  because the count comes from MongoDB).
* Only dials inside the allowed calling window (default 09:00–21:00 IST, in line with
  TRAI telemarketing hours). Outside it, calls simply wait in the queue.
* Claims each call atomically in MongoDB, so running several app replicas is safe.
* Recovers calls stuck in 'dialing' after a crash.
"""

import asyncio
import logging
import socket
import uuid
from datetime import datetime, time
from zoneinfo import ZoneInfo

from app import repositories as repo
from app.config import Settings
from app.logging_config import log
from app.services.telephony import Telephony

logger = logging.getLogger(__name__)


def within_calling_hours(settings: Settings, now: datetime | None = None) -> bool:
    tz = ZoneInfo(settings.calling_timezone)
    local = (now or datetime.now(tz)).astimezone(tz).time()
    start = time.fromisoformat(settings.calling_hours_start)
    end = time.fromisoformat(settings.calling_hours_end)
    return start <= local < end


class Dialer:
    def __init__(self, settings: Settings, telephony: Telephony):
        self.s = settings
        self.telephony = telephony
        self.worker_id = f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()

    def start(self) -> None:
        if self.s.dialer_enabled and self._task is None:
            self._task = asyncio.create_task(self._loop(), name="dialer")
            log(logger, logging.INFO, "dialer_started", worker=self.worker_id)

    def wake(self) -> None:
        """Called after a CSV upload so dialing starts immediately instead of on the next tick."""
        self._wake.set()

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        ticks = 0
        while True:
            try:
                if ticks % 30 == 0:
                    await repo.requeue_stale_dialing()
                ticks += 1
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a DB hiccup must not kill the dialer
                log(logger, logging.ERROR, "dialer_error", error=str(exc))
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=self.s.dialer_poll_s)
            except TimeoutError:
                pass

    async def tick(self) -> int:
        """Dial as many queued calls as free capacity allows. Returns how many were dialed."""
        if not within_calling_hours(self.s):
            return 0
        free = self.s.max_concurrent_outbound_calls - await repo.count_active_calls()
        dialed = 0
        for _ in range(max(0, free)):
            call = await repo.claim_next_queued_call(self.worker_id)
            if call is None:
                break
            await self._dial(call)
            dialed += 1
        return dialed

    async def _dial(self, call: dict) -> None:
        customer = await repo.get_customer(call["customer_id"]) or {}
        if customer.get("do_not_call"):
            await repo.update_call(call["id"], {"status": "canceled", "error": "do-not-call", "ended_at": repo.now()})
            return
        phone = call.get("phone") or customer.get("phone")
        try:
            sid = await self.telephony.place_call(call["id"], phone)
            await repo.update_call(call["id"], {"status": "initiated", "twilio_call_sid": sid})
        except Exception as exc:
            await repo.update_call(call["id"], {"status": "failed", "error": str(exc)[:300], "ended_at": repo.now()})
            log(logger, logging.WARNING, "dial_failed", call_id=call["id"], error=str(exc))

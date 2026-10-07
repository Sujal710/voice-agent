"""Structured (JSON) logging so logs can be shipped to any log platform."""

import json
import logging
import sys
from datetime import UTC, datetime


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # Quiet noisy libraries
    for name in ("websockets", "httpx", "httpcore", "pymongo", "twilio.http_client"):
        logging.getLogger(name).setLevel(logging.WARNING)


def log(logger: logging.Logger, level: int, msg: str, **fields) -> None:
    """Log with structured key/value fields: log(logger, logging.INFO, "x", call_id=...)"""
    logger.log(level, msg, extra={"extra_fields": fields})

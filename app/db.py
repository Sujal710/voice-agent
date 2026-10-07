"""MongoDB access using PyMongo's native async client (Motor's successor)."""

import logging

from pymongo import ASCENDING, DESCENDING, AsyncMongoClient
from pymongo.asynchronous.database import AsyncDatabase

from app.config import Settings

logger = logging.getLogger(__name__)


class Mongo:
    client: AsyncMongoClient | None = None
    db: AsyncDatabase | None = None

    @classmethod
    async def connect(cls, settings: Settings) -> None:
        cls.client = AsyncMongoClient(
            settings.mongodb_uri,
            maxPoolSize=100,
            serverSelectionTimeoutMS=5000,
            tz_aware=True,
        )
        cls.db = cls.client[settings.mongodb_db]
        await cls.client.admin.command("ping")
        await cls.ensure_indexes()

    @classmethod
    async def ensure_indexes(cls) -> None:
        db = cls.db
        await db.customers.create_index([("phone", ASCENDING)], unique=True)
        await db.products.create_index([("sku", ASCENDING)], unique=True)
        await db.calls.create_index([("twilio_call_sid", ASCENDING)], sparse=True)
        await db.calls.create_index([("customer_id", ASCENDING), ("created_at", DESCENDING)])
        await db.calls.create_index([("status", ASCENDING)])
        await db.calls.create_index([("batch_id", ASCENDING), ("batch_seq", ASCENDING)], sparse=True)
        await db.calls.create_index([("status", ASCENDING), ("created_at", ASCENDING)])  # dialer queue
        await db.batches.create_index([("created_at", DESCENDING)])
        # Chat history, one document per message (same idea as LangChain's MongoDBChatMessageHistory)
        await db.messages.create_index([("session_id", ASCENDING), ("seq", ASCENDING)], unique=True)
        await db.messages.create_index([("customer_id", ASCENDING), ("created_at", DESCENDING)])

    @classmethod
    async def close(cls) -> None:
        if cls.client is not None:
            await cls.client.close()
            cls.client = None
            cls.db = None


def get_db() -> AsyncDatabase:
    if Mongo.db is None:
        raise RuntimeError("MongoDB is not connected")
    return Mongo.db

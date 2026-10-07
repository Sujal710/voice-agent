"""Seed MongoDB with a sample product and customers.

Usage:  python -m scripts.seed_db --phone +919876543210 --language gu-IN
"""

import argparse
import asyncio

from app import repositories as repo
from app.config import get_settings
from app.db import Mongo

SAMPLE_PRODUCT = {
    "sku": "SOLARMAX-3KW",
    "name": "SolarMax 3 kilowatt rooftop solar system",
    "price": "1,65,000 rupees, or EMI from 4,999 rupees per month",
    "offer": "Free installation and a 30,000 rupee government subsidy assistance this month",
    "description": "A rooftop solar system that cuts home electricity bills by up to 90 percent.",
    "features": [
        "Saves around 3,000 to 4,000 rupees per month on electricity",
        "25 year panel warranty and 5 year free maintenance",
        "Mobile app to track power generation",
        "Installation completed in 3 days",
    ],
    "faqs": [
        {
            "q": "Does it work on cloudy days?",
            "a": "Yes, it generates less power but still works; extra power is stored to the grid via net metering.",
        },
        {"q": "How much roof space is needed?", "a": "About 300 square feet of shadow-free roof."},
        {
            "q": "Is there a subsidy?",
            "a": "Yes, we help you apply for the PM Surya Ghar subsidy of up to 30,000 rupees for this size.",
        },
    ],
}


async def main(phone: str, language: str, name: str) -> None:
    settings = get_settings()
    await Mongo.connect(settings)
    product = await repo.upsert_product(SAMPLE_PRODUCT)
    customer = await repo.upsert_customer(
        {
            "name": name,
            "phone": phone,
            "preferred_language": language,
            "city": "Ahmedabad",
            "notes": "Owns an independent house; high summer electricity bills.",
            "do_not_call": False,
        }
    )
    print(f"Product SKU : {product['sku']}")
    print(f"Customer ID : {customer['id']}  ({customer['name']}, {customer['phone']}, {language})")
    await Mongo.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phone", required=True, help="E.164 number to call, e.g. +919876543210")
    parser.add_argument("--language", default="en-IN", help="Greeting language, e.g. gu-IN, hi-IN, en-IN")
    parser.add_argument("--name", default="Sujal")
    args = parser.parse_args()
    asyncio.run(main(args.phone, args.language, args.name))

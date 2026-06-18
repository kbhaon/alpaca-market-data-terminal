from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

@dataclass(frozen=True)
class AlpacaSettings:
    api_key: str
    secret_key: str
    data_feed: str = "iex"


def get_settings() -> AlpacaSettings:
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY", "").strip()
    secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
    data_feed = os.getenv("ALPACA_DATA_FEED", "iex").strip().lower()

    if not api_key or not secret_key:
        raise ValueError(
            "Missing Alpaca credentials. Create a .env file with valid "
            "ALPACA_API_KEY and ALPACA_SECRET_KEY."
        )

    return AlpacaSettings(
        api_key=api_key,
        secret_key=secret_key,
        data_feed=data_feed,
    )

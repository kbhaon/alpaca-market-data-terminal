from __future__ import annotations

from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.trading.client import TradingClient

from src.config import AlpacaSettings, get_settings


def resolve_data_feed(feed_name: str) -> DataFeed:
    normalized = feed_name.strip().upper()
    if normalized == "SIP":
        return DataFeed.SIP
    return DataFeed.IEX


def get_historical_client(settings: AlpacaSettings | None = None) -> StockHistoricalDataClient:
    settings = settings or get_settings()
    return StockHistoricalDataClient(settings.api_key, settings.secret_key)


def get_stream_client(settings: AlpacaSettings | None = None) -> StockDataStream:
    settings = settings or get_settings()
    return StockDataStream(
        settings.api_key,
        settings.secret_key,
        feed=resolve_data_feed(settings.data_feed),
    )


def get_paper_trading_client(settings: AlpacaSettings | None = None) -> TradingClient:
    settings = settings or get_settings()
    return TradingClient(
        settings.api_key,
        settings.secret_key,
        paper=True,
    )

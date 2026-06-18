from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Queue
from threading import Thread
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

from src.config import get_settings
from src.data_connector import get_stream_client, resolve_data_feed


@dataclass(frozen=True)
class MarketSnapshot:
    symbol: str
    bid_price: float | None
    ask_price: float | None
    last_trade_price: float | None
    updated_at: datetime | None

    @property
    def bid_display(self) -> str:
        return _format_price(self.bid_price)

    @property
    def ask_display(self) -> str:
        return _format_price(self.ask_price)

    @property
    def last_trade_display(self) -> str:
        return _format_price(self.last_trade_price)

    @property
    def updated_at_display(self) -> str:
        if self.updated_at is None:
            return "n/a"
        updated_at = self.updated_at
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return updated_at.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S E.T.")


def _format_price(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def get_latest_quote_trade(
    client: StockHistoricalDataClient,
    symbol: str,
) -> MarketSnapshot:
    settings = get_settings()
    feed = resolve_data_feed(settings.data_feed)

    quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol, feed=feed)
    trade_request = StockLatestTradeRequest(symbol_or_symbols=symbol, feed=feed)

    latest_quote = client.get_stock_latest_quote(quote_request).get(symbol)
    latest_trade = client.get_stock_latest_trade(trade_request).get(symbol)

    quote_time = getattr(latest_quote, "timestamp", None)
    trade_time = getattr(latest_trade, "timestamp", None)

    return MarketSnapshot(
        symbol=symbol,
        bid_price=getattr(latest_quote, "bid_price", None),
        ask_price=getattr(latest_quote, "ask_price", None),
        last_trade_price=getattr(latest_trade, "price", None),
        updated_at=trade_time or quote_time,
    )


class QuoteStreamer:
    """A class to stream real-time quotes for a given symbol."""

    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.events: Queue[dict[str, Any]] = Queue()
        self.stream = get_stream_client()
        self.thread: Thread | None = None

    async def _handle_quote(self, quote: Any) -> None:
        self.events.put(
            {
                "type": "quote",
                "symbol": self.symbol,
                "bid_price": getattr(quote, "bid_price", None),
                "ask_price": getattr(quote, "ask_price", None),
                "timestamp": getattr(quote, "timestamp", None),
            }
        )

    async def _handle_trade(self, trade: Any) -> None:
        self.events.put(
            {
                "type": "trade",
                "symbol": self.symbol,
                "price": getattr(trade, "price", None),
                "timestamp": getattr(trade, "timestamp", None),
            }
        )

    def start(self) -> None:
        self.stream.subscribe_quotes(self._handle_quote, self.symbol)
        self.stream.subscribe_trades(self._handle_trade, self.symbol)
        self.thread = Thread(target=self.stream.run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stream.stop()

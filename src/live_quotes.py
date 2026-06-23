from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from queue import Empty, Queue
from threading import Thread
from typing import Any
from zoneinfo import ZoneInfo

from alpaca.data.requests import StockLatestQuoteRequest, StockLatestTradeRequest

from src.config import get_settings
from src.data_connector import get_historical_client, get_stream_client, resolve_data_feed


LIVE_QUOTE_MANAGER_STATE_KEY = "live_quote_manager"


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


def get_latest_quote_snapshot(symbol: str) -> MarketSnapshot:
    """Fetch the latest known quote/trade snapshot before stream events arrive."""
    symbol = symbol.strip().upper()
    settings = get_settings()
    feed = resolve_data_feed(settings.data_feed)
    client = get_historical_client(settings)

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


class LiveQuoteStreamManager:
    """Manage one active live quote stream and its latest display snapshot."""

    def __init__(self) -> None:
        self.streamer: QuoteStreamer | None = None
        self.symbol: str | None = None
        self.snapshot: MarketSnapshot | None = None
        self.error: str | None = None
        self.snapshot_error: str | None = None

    def stop(self) -> None:
        """Stop the active stream and reset its accumulated quote state."""
        if self.streamer is not None:
            try:
                self.streamer.stop()
            except Exception:
                pass

        self.streamer = None
        self.symbol = None
        self.snapshot = None

    def get_snapshot(self, symbol: str) -> MarketSnapshot | None:
        """Ensure a stream is running for symbol, then return the latest snapshot."""
        symbol = symbol.strip().upper()
        if not symbol:
            self.stop()
            self.error = None
            return None

        if self.streamer is None or self.symbol != symbol:
            self._restart(symbol)

        if self.streamer is None:
            return None

        return self._drain_events()

    def _restart(self, symbol: str) -> None:
        self.stop()
        self.error = None
        self.snapshot_error = None

        try:
            streamer = QuoteStreamer(symbol)
            streamer.start()
        except Exception as exc:
            self.error = str(exc)
            return

        self.streamer = streamer
        self.symbol = symbol
        self.snapshot = self._load_initial_snapshot(symbol)

    def _load_initial_snapshot(self, symbol: str) -> MarketSnapshot:
        try:
            return get_latest_quote_snapshot(symbol)
        except Exception as exc:
            self.snapshot_error = str(exc)
            return self._empty_snapshot(symbol)

    def _drain_events(self) -> MarketSnapshot:
        if self.streamer is None or self.symbol is None:
            raise RuntimeError("Live quote stream is not running.")

        snapshot = self.snapshot or self._empty_snapshot(self.symbol)

        while True:
            try:
                event = self.streamer.events.get_nowait()
            except Empty:
                break

            event_type = event.get("type")
            timestamp = event.get("timestamp") or snapshot.updated_at

            if event_type == "quote":
                snapshot = MarketSnapshot(
                    symbol=self.symbol,
                    bid_price=event.get("bid_price"),
                    ask_price=event.get("ask_price"),
                    last_trade_price=snapshot.last_trade_price,
                    updated_at=timestamp,
                )
            elif event_type == "trade":
                snapshot = MarketSnapshot(
                    symbol=self.symbol,
                    bid_price=snapshot.bid_price,
                    ask_price=snapshot.ask_price,
                    last_trade_price=event.get("price"),
                    updated_at=timestamp,
                )

        self.snapshot = snapshot
        return snapshot

    @staticmethod
    def _empty_snapshot(symbol: str) -> MarketSnapshot:
        return MarketSnapshot(
            symbol=symbol,
            bid_price=None,
            ask_price=None,
            last_trade_price=None,
            updated_at=None,
        )


def get_live_quote_manager(session_state: Any) -> LiveQuoteStreamManager:
    """Return the live quote manager stored in Streamlit session state."""
    manager = session_state.get(LIVE_QUOTE_MANAGER_STATE_KEY)
    if isinstance(manager, LiveQuoteStreamManager):
        return manager

    if hasattr(manager, "stop"):
        try:
            manager.stop()
        except Exception:
            pass

    manager = LiveQuoteStreamManager()
    session_state[LIVE_QUOTE_MANAGER_STATE_KEY] = manager
    return manager

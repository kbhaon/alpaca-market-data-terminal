from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from src.backtester import STRATEGY_SPECS, StrategySpec, build_ml_strategy_spec
from src.config import AlpacaSettings, get_settings
from src.historical import fetch_daily_ohlcv


# No switch for live env. only paper trading
PAPER_ONLY = True

DEFAULT_ORDER_NOTIONAL = 10_000.0
DEFAULT_HISTORY_YEARS = 5

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "paper_trading.log"
HISTORY_FILE = LOG_DIR / "paper_trading_history.csv"

HISTORY_COLUMNS = [
    "run_at",
    "symbol",
    "strategy",
    "bar_timestamp",
    "close",
    "signal",
    "action",
    "order_id",
    "order_status",
    "order_qty",
    "market_open",
    "note",
]

ML_STRATEGY_NAME = "ML Signal"

_ML_CONTRACT_HINT = (
    "src/features.py and src/models.py are placeholders and do not implement "
    "add_ml_features / generate_ml_signals yet."
)

# One tick may run from the UI while another runs from a loop thread; guard the
# shared history CSV.
_HISTORY_LOCK = threading.Lock()


@dataclass(frozen=True)
class AccountSnapshot:
    """Paper account state pulled straight from Alpaca."""

    cash: float
    equity: float
    buying_power: float
    status: str


@dataclass(frozen=True)
class PositionSnapshot:
    """One open paper position pulled straight from Alpaca."""

    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float


@dataclass(frozen=True)
class ExecutionStrategy:
    """A backtester StrategySpec plus the optional feature step it needs live.

    Rule-based strategies add their indicators inside the signal function, so
    feature_prep stays None. The ML strategy needs add_ml_features applied to
    the raw bars first.
    """

    spec: StrategySpec
    feature_prep: object | None = None  # Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class ExecutionReport:
    """Outcome of one run_execution_tick() call, for display and history."""

    symbol: str
    strategy: str
    run_at: pd.Timestamp
    bar_timestamp: pd.Timestamp | None
    close_price: float | None
    signal: str  # "LONG", "FLAT", or "ERROR"
    action: str  # "BUY", "SELL", "HOLD", "NONE", "SKIP", or "ERROR"
    order_id: str | None = None
    order_status: str | None = None
    order_qty: float | None = None
    market_open: bool | None = None
    note: str = ""
    log_lines: list[str] = field(default_factory=list)


def get_paper_trading_logger() -> logging.Logger:
    """Return a logger that writes signal/order events to console and logs/paper_trading.log."""
    logger = logging.getLogger("paper_trading")

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def get_trading_client(settings: AlpacaSettings | None = None) -> TradingClient:
    """Build an Alpaca trading client locked to the paper environment."""
    settings = settings or get_settings()
    return TradingClient(settings.api_key, settings.secret_key, paper=PAPER_ONLY)


def get_account_snapshot(client: TradingClient | None = None) -> AccountSnapshot:
    """Fetch the current paper account balances from Alpaca."""
    client = client or get_trading_client()
    account = client.get_account()
    return AccountSnapshot(
        cash=float(account.cash),
        equity=float(account.equity),
        buying_power=float(account.buying_power),
        status=str(account.status),
    )


def get_open_position(
    symbol: str,
    client: TradingClient | None = None,
) -> PositionSnapshot | None:
    """Return the open paper position for symbol, or None when flat.

    Alpaca answers with a 404 APIError when there is no position.
    """
    client = client or get_trading_client()
    try:
        position = client.get_open_position(symbol)
    except APIError:
        return None

    return PositionSnapshot(
        symbol=symbol,
        qty=float(position.qty),
        avg_entry_price=float(position.avg_entry_price),
        market_value=float(position.market_value),
        unrealized_pl=float(position.unrealized_pl),
        unrealized_plpc=float(position.unrealized_plpc),
    )


def get_recent_orders(
    symbol: str | None = None,
    limit: int = 25,
    client: TradingClient | None = None,
) -> pd.DataFrame:
    """Fetch recent paper orders from Alpaca as a flat DataFrame."""
    client = client or get_trading_client()
    request = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        symbols=[symbol] if symbol else None,
        limit=limit,
    )
    orders = client.get_orders(request)

    rows = [
        {
            "submitted_at": order.submitted_at,
            "symbol": order.symbol,
            "side": getattr(order.side, "value", order.side),
            "qty": order.qty,
            "notional": order.notional,
            "type": getattr(order.order_type, "value", order.order_type),
            "status": getattr(order.status, "value", order.status),
            "filled_qty": order.filled_qty,
            "filled_avg_price": order.filled_avg_price,
            "id": str(order.id),
        }
        for order in orders
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "submitted_at",
            "symbol",
            "side",
            "qty",
            "notional",
            "type",
            "status",
            "filled_qty",
            "filled_avg_price",
            "id",
        ],
    )


def _has_open_order(client: TradingClient, symbol: str) -> bool:
    """True when an order for symbol is still open (e.g. queued after hours)."""
    request = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    return len(client.get_orders(request)) > 0


def _get_open_position_qty(client: TradingClient, symbol: str) -> float:
    try:
        position = client.get_open_position(symbol)
    except APIError:
        return 0.0
    return float(position.qty)


def available_strategy_names() -> list[str]:
    """Names accepted by resolve_execution_strategy, ML spec included."""
    return [*STRATEGY_SPECS, ML_STRATEGY_NAME]


def resolve_execution_strategy(strategy_name: str) -> ExecutionStrategy:
    """Map a strategy name to its spec and any live feature-prep step.

    The ML strategy imports lazily so this module keeps working while
    src/features.py and src/models.py are placeholders.
    """
    if strategy_name in STRATEGY_SPECS:
        return ExecutionStrategy(spec=STRATEGY_SPECS[strategy_name])

    if strategy_name == ML_STRATEGY_NAME:
        try:
            from src.features import add_ml_features
        except ImportError as exc:
            raise ImportError(_ML_CONTRACT_HINT) from exc
        return ExecutionStrategy(
            spec=build_ml_strategy_spec(),
            feature_prep=add_ml_features,
        )

    raise ValueError(
        f"Unknown strategy '{strategy_name}'. "
        f"Choose from: {', '.join(available_strategy_names())}."
    )


def _compute_signals(strategy: ExecutionStrategy, bars: pd.DataFrame) -> pd.DataFrame:
    df = strategy.feature_prep(bars) if strategy.feature_prep is not None else bars
    return strategy.spec.signal_function(df, price_col="close")


def append_history_row(report: ExecutionReport, path: Path = HISTORY_FILE) -> None:
    """Append one execution report to the persistent history CSV."""
    row = pd.DataFrame(
        [
            {
                "run_at": report.run_at,
                "symbol": report.symbol,
                "strategy": report.strategy,
                "bar_timestamp": report.bar_timestamp,
                "close": report.close_price,
                "signal": report.signal,
                "action": report.action,
                "order_id": report.order_id,
                "order_status": report.order_status,
                "order_qty": report.order_qty,
                "market_open": report.market_open,
                "note": report.note,
            }
        ],
        columns=HISTORY_COLUMNS,
    )

    with _HISTORY_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        row.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_history(
    path: Path = HISTORY_FILE,
    symbol: str | None = None,
    strategy: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load the persisted signal/trade history; empty frame if none exists yet."""
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    with _HISTORY_LOCK:
        history = pd.read_csv(path)

    if symbol is not None:
        history = history[history["symbol"] == symbol]
    if strategy is not None:
        history = history[history["strategy"] == strategy]
    if limit is not None:
        history = history.tail(limit)

    return history.reset_index(drop=True)


def get_last_acted_bar(
    symbol: str,
    strategy: str,
    path: Path = HISTORY_FILE,
) -> pd.Timestamp | None:
    """Latest bar timestamp we already traded on, so restarts don't re-order."""
    history = load_history(path=path, symbol=symbol, strategy=strategy)
    acted = history[history["action"].isin(["BUY", "SELL"])]
    if acted.empty:
        return None
    return pd.to_datetime(acted["bar_timestamp"]).max()


def run_execution_tick(
    symbol: str,
    strategy: ExecutionStrategy | str,
    notional: float = DEFAULT_ORDER_NOTIONAL,
    years: int = DEFAULT_HISTORY_YEARS,
    trading_client: TradingClient | None = None,
    last_acted_bar: pd.Timestamp | None = None,
    record: bool = True,
) -> ExecutionReport:
    """
    Evaluate the strategy once and reconcile with the current paper position.

    Fetch daily bars, generate signals, read the latest position column, then:
        LONG signal + no position  -> submit market BUY (paper only)
        LONG signal + position     -> HOLD, no order
        FLAT signal + position     -> close the position (market SELL)
        FLAT signal + no position  -> no order

    Orders are skipped when the latest bar was already acted on
    (last_acted_bar) or an order for the symbol is still open, so re-running
    on the same daily bar never stacks orders. Long-only and unleveraged: the
    buy notional is capped at available cash. When the market is closed the
    order is still submitted; paper DAY orders queue for the next open.
    """
    logger = get_paper_trading_logger()
    log_lines: list[str] = []

    def log(message: str) -> None:
        logger.info(message)
        log_lines.append(message)

    if isinstance(strategy, str):
        strategy = resolve_execution_strategy(strategy)
    strategy_name = strategy.spec.name

    client = trading_client or get_trading_client()
    run_at = pd.Timestamp.now(tz="UTC")
    notes: list[str] = []

    log(f"=== Paper trading tick: {strategy_name} on {symbol} (paper={PAPER_ONLY}) ===")

    bars = fetch_daily_ohlcv(symbol, years=years)
    if bars.empty:
        raise ValueError(f"No daily bars returned for {symbol}.")

    signals = _compute_signals(strategy, bars)

    latest = signals.iloc[-1]
    bar_timestamp = pd.Timestamp(latest["timestamp"])
    close_price = float(latest["close"])
    is_long = int(latest[strategy.spec.position_col]) == 1
    signal_label = "LONG" if is_long else "FLAT"

    if "ml_probability" in signals.columns:
        notes.append(f"P(up)={float(latest['ml_probability']):.4f}")

    log(
        f"Latest bar {bar_timestamp.date()} close={close_price:.2f} "
        f"-> signal {signal_label}"
    )

    clock = client.get_clock()
    market_open = bool(clock.is_open)
    log(f"Market open: {market_open} (next open {clock.next_open}, next close {clock.next_close})")

    held_qty = _get_open_position_qty(client, symbol)
    log(f"Current paper position in {symbol}: {held_qty:g} shares")

    if is_long and held_qty == 0:
        intended = "BUY"
    elif is_long:
        intended = "HOLD"
    elif held_qty > 0:
        intended = "SELL"
    else:
        intended = "NONE"

    action = intended
    order = None
    order_qty: float | None = None

    if intended in ("BUY", "SELL"):
        if last_acted_bar is not None and bar_timestamp == pd.Timestamp(last_acted_bar):
            action = "SKIP"
            notes.append(f"already acted on bar {bar_timestamp.date()}")
            log(f"{intended} signal but bar {bar_timestamp.date()} was already acted on. No order.")
        elif _has_open_order(client, symbol):
            action = "SKIP"
            notes.append("open order pending")
            log(f"{intended} signal but an order for {symbol} is still open. No order.")

    if action == "BUY":
        account = client.get_account()
        available_cash = float(account.cash)
        effective_notional = min(notional, available_cash)
        qty = math.floor(effective_notional / close_price)

        if qty < 1:
            action = "NONE"
            notes.append(
                f"insufficient cash (${available_cash:,.2f}) for one share at ${close_price:.2f}"
            )
            log(
                f"LONG signal but cannot afford one share "
                f"(cash ${available_cash:,.2f}, close ${close_price:.2f}). No order."
            )
        else:
            order_qty = float(qty)
            order = client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            log(
                f"Submitted paper BUY {qty} {symbol} @ market "
                f"(~${qty * close_price:,.2f}) | order id {order.id} | status {order.status}"
            )

    elif action == "SELL":
        order_qty = held_qty
        order = client.close_position(symbol)
        log(
            f"FLAT signal: closing {held_qty:g} {symbol} @ market | "
            f"order id {order.id} | status {order.status}"
        )

    elif action == "HOLD":
        log(f"LONG signal and already holding {held_qty:g} shares. No order.")

    elif action == "NONE" and intended == "NONE":
        log("FLAT signal and no open position. No order.")

    if order is not None and not market_open:
        notes.append("market closed; order queued for next open")

    log(f"=== Tick complete: signal={signal_label}, action={action} ===")

    report = ExecutionReport(
        symbol=symbol,
        strategy=strategy_name,
        run_at=run_at,
        bar_timestamp=bar_timestamp,
        close_price=close_price,
        signal=signal_label,
        action=action,
        order_id=str(order.id) if order is not None else None,
        order_status=str(order.status.value if hasattr(order.status, "value") else order.status)
        if order is not None
        else None,
        order_qty=order_qty,
        market_open=market_open,
        note="; ".join(notes),
        log_lines=log_lines,
    )

    if record:
        append_history_row(report)

    return report


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a strategy once and submit paper-only orders: "
            "buy if the latest signal is LONG, close the position if FLAT."
        )
    )
    parser.add_argument("--symbol", default="AAPL", help="Ticker to trade (default: AAPL)")
    parser.add_argument(
        "--strategy",
        default="Trend Following",
        choices=available_strategy_names(),
        help="Strategy that drives the trade (default: Trend Following)",
    )
    parser.add_argument(
        "--notional",
        type=float,
        default=DEFAULT_ORDER_NOTIONAL,
        help="Max order notional in USD, capped at paper cash (default: 10000)",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Do not append this tick to logs/paper_trading_history.csv",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    last_acted_bar = get_last_acted_bar(symbol, args.strategy)

    try:
        report = run_execution_tick(
            symbol,
            args.strategy,
            notional=args.notional,
            last_acted_bar=last_acted_bar,
            record=not args.no_record,
        )
    except ImportError as exc:
        raise SystemExit(f"Cannot run '{args.strategy}': {exc}") from exc
    print(
        f"{report.symbol} [{report.strategy}]: signal={report.signal} "
        f"action={report.action} order_status={report.order_status or 'no order'}"
        + (f" | {report.note}" if report.note else "")
    )


if __name__ == "__main__":
    _main()

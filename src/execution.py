from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.config import AlpacaSettings
from src.data_connector import get_paper_trading_client


DEFAULT_ORDER_NOTIONAL = 10_000.0

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "paper_trading.log"


@dataclass(frozen=True)
class OrderPlan:
    symbol: str
    action: str
    side: str | None
    quantity: float | None
    latest_position: int
    current_position: float
    close_price: float
    probability: float
    bar_timestamp: pd.Timestamp | None
    requested_notional: float
    reason: str


@dataclass(frozen=True)
class ExecutionReport:
    """Outcome of one paper-trading decision, for display and submission logs."""

    symbol: str
    bar_timestamp: pd.Timestamp | None
    close_price: float
    probability: float
    signal: str
    action: str
    current_position: float
    order_id: str | None = None
    order_status: str | None = None
    order_qty: float | None = None
    market_open: bool | None = None
    dry_run: bool = False
    message: str = ""
    log_lines: list[str] = field(default_factory=list)


def get_paper_trading_logger() -> logging.Logger:
    """Return a logger that writes signal and order events to logs/paper_trading.log."""
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


def get_current_position(
    symbol: str,
    trading_client: TradingClient | None = None,
) -> float:
    """Return current paper shares for symbol, or 0 when there is no open position."""
    client = trading_client or get_paper_trading_client()

    try:
        position = client.get_open_position(symbol)
    except APIError:
        return 0.0

    return float(position.qty)


def get_latest_signal(signal_df: pd.DataFrame) -> dict[str, Any]:
    """Extract the latest usable ML signal row from a model-generated signal DataFrame."""
    required_columns = {"close", "ml_probability", "ml_position"}
    missing = sorted(required_columns - set(signal_df.columns))
    if missing:
        raise ValueError(f"Signal DataFrame is missing required columns: {missing}")

    ready = signal_df.dropna(subset=["close", "ml_probability", "ml_position"])
    if ready.empty:
        raise ValueError("Signal DataFrame does not contain a usable latest signal row.")

    latest = ready.iloc[-1]
    timestamp = (
        pd.Timestamp(latest["timestamp"])
        if "timestamp" in ready.columns and pd.notna(latest["timestamp"])
        else None
    )
    latest_position = int(latest["ml_position"])
    probability = float(latest["ml_probability"])
    close_price = float(latest["close"])
    trade_signal = (
        int(latest["ml_trade_signal"])
        if "ml_trade_signal" in ready.columns and pd.notna(latest["ml_trade_signal"])
        else None
    )

    return {
        "timestamp": timestamp,
        "close_price": close_price,
        "probability": probability,
        "position": latest_position,
        "trade_signal": trade_signal,
        "label": "LONG" if latest_position == 1 else "FLAT",
    }


def build_order_plan(
    symbol: str,
    latest_signal: dict[str, Any],
    current_position: float,
    notional: float = DEFAULT_ORDER_NOTIONAL,
    available_cash: float | None = None,
) -> OrderPlan:
    """Convert latest long/flat signal and current paper position into an order plan."""
    if notional <= 0:
        raise ValueError("notional must be positive.")

    close_price = float(latest_signal["close_price"])
    if close_price <= 0:
        raise ValueError("latest close price must be positive.")

    latest_position = int(latest_signal["position"])
    probability = float(latest_signal["probability"])
    timestamp = latest_signal.get("timestamp")

    if latest_position == 1 and current_position <= 0:
        effective_notional = min(notional, available_cash) if available_cash is not None else notional
        quantity = math.floor(effective_notional / close_price)

        if quantity < 1:
            return OrderPlan(
                symbol=symbol,
                action="NONE",
                side=None,
                quantity=None,
                latest_position=latest_position,
                current_position=current_position,
                close_price=close_price,
                probability=probability,
                bar_timestamp=timestamp,
                requested_notional=notional,
                reason="LONG signal, but available cash is insufficient to buy one share.",
            )

        return OrderPlan(
            symbol=symbol,
            action="BUY",
            side="buy",
            quantity=float(quantity),
            latest_position=latest_position,
            current_position=current_position,
            close_price=close_price,
            probability=probability,
            bar_timestamp=timestamp,
            requested_notional=notional,
            reason="LONG signal and no current paper position.",
        )

    if latest_position == 1 and current_position > 0:
        return OrderPlan(
            symbol=symbol,
            action="HOLD",
            side=None,
            quantity=None,
            latest_position=latest_position,
            current_position=current_position,
            close_price=close_price,
            probability=probability,
            bar_timestamp=timestamp,
            requested_notional=notional,
            reason="LONG signal and paper account already holds shares.",
        )

    if latest_position == 0 and current_position > 0:
        return OrderPlan(
            symbol=symbol,
            action="SELL",
            side="sell",
            quantity=float(current_position),
            latest_position=latest_position,
            current_position=current_position,
            close_price=close_price,
            probability=probability,
            bar_timestamp=timestamp,
            requested_notional=notional,
            reason="FLAT signal and paper account has an open position.",
        )

    return OrderPlan(
        symbol=symbol,
        action="NONE",
        side=None,
        quantity=None,
        latest_position=latest_position,
        current_position=current_position,
        close_price=close_price,
        probability=probability,
        bar_timestamp=timestamp,
        requested_notional=notional,
        reason="FLAT signal and paper account is already flat.",
    )


def submit_paper_order(
    order_plan: OrderPlan,
    trading_client: TradingClient | None = None,
) -> Any:
    """Submit the market order described by an order plan to Alpaca paper trading."""
    client = trading_client or get_paper_trading_client()

    if order_plan.action == "BUY":
        if order_plan.quantity is None:
            raise ValueError("BUY order plan is missing quantity.")

        return client.submit_order(
            MarketOrderRequest(
                symbol=order_plan.symbol,
                qty=int(order_plan.quantity),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )

    if order_plan.action == "SELL":
        return client.close_position(order_plan.symbol)

    return None


def execute_latest_signal(
    symbol: str,
    signal_df: pd.DataFrame,
    notional: float = DEFAULT_ORDER_NOTIONAL,
    settings: AlpacaSettings | None = None,
    trading_client: TradingClient | None = None,
    dry_run: bool = False,
) -> ExecutionReport:
    """
    Execute the latest model signal in Alpaca paper trading.

    This function intentionally does not fetch market data, compute features,
    apply PCA, train a model, or generate ML signals. It receives the model
    output and handles only paper-account inspection, order planning, order
    submission, and logging.
    """
    logger = get_paper_trading_logger()
    log_lines: list[str] = []

    def log(message: str) -> None:
        logger.info(message)
        log_lines.append(message)

    client = trading_client or get_paper_trading_client(settings)
    latest_signal = get_latest_signal(signal_df)
    signal_label = str(latest_signal["label"])

    log(f"=== Paper execution run for {symbol} ===")
    log(
        f"Latest signal: {signal_label} | "
        f"P(next-day up)={float(latest_signal['probability']):.4f} | "
        f"close={float(latest_signal['close_price']):.2f}"
    )

    clock = client.get_clock()
    log(f"Market open: {clock.is_open} (next open {clock.next_open}, next close {clock.next_close})")

    current_position = get_current_position(symbol, trading_client=client)
    log(f"Current paper position in {symbol}: {current_position:g} shares")

    available_cash = None
    if int(latest_signal["position"]) == 1 and current_position <= 0:
        account = client.get_account()
        available_cash = float(account.cash)

    order_plan = build_order_plan(
        symbol=symbol,
        latest_signal=latest_signal,
        current_position=current_position,
        notional=notional,
        available_cash=available_cash,
    )
    log(f"Order plan: {order_plan.action} | {order_plan.reason}")

    order = None
    if not dry_run and order_plan.action in {"BUY", "SELL"}:
        order = submit_paper_order(order_plan, trading_client=client)
        log(
            f"Submitted paper {order_plan.action} for {symbol} | "
            f"order id {order.id} | status {order.status}"
        )
    elif dry_run:
        log("Dry run enabled. No paper order submitted.")

    log(f"=== Run complete: signal={signal_label}, action={order_plan.action} ===")

    return ExecutionReport(
        symbol=symbol,
        bar_timestamp=order_plan.bar_timestamp,
        close_price=order_plan.close_price,
        probability=order_plan.probability,
        signal=signal_label,
        action=order_plan.action,
        current_position=current_position,
        order_id=str(order.id) if order is not None else None,
        order_status=str(order.status.value if hasattr(order.status, "value") else order.status)
        if order is not None
        else None,
        order_qty=order_plan.quantity if order is not None else None,
        market_open=bool(clock.is_open),
        dry_run=dry_run,
        message=order_plan.reason,
        log_lines=log_lines,
    )

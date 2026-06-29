from __future__ import annotations

import pandas as pd
from src.indicators import add_exponential_moving_average, add_required_indicators


# Strategy 1
def generate_macd_sma_trend_signals(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Strategy 1: MACD + SMA200 Trend Filter.

    Buy when:
        MACD > MACD Signal
        Close > SMA200

    Sell when:
        MACD < MACD Signal
        Close < SMA200

    This is a long-only strategy:
        1 = long
        0 = cash
    """
    result = df.copy()

    required_columns = {
        price_col,
        "sma_200",
        "macd",
        "macd_signal",
    }

    if not required_columns.issubset(result.columns):
        result = add_required_indicators(result, price_col=price_col)

    entry_condition = (
        (result["macd"] > result["macd_signal"])
        & (result[price_col] > result["sma_200"])
    )

    exit_condition = (
        (result["macd"] < result["macd_signal"])
        | (result[price_col] < result["sma_200"])
    )

    current_position = 0
    positions = []
    trade_signals = []

    for should_enter, should_exit in zip(
        entry_condition.fillna(False),
        exit_condition.fillna(False),
    ):
        trade_signal = 0

        if current_position == 0 and should_enter:
            current_position = 1
            trade_signal = 1

        elif current_position == 1 and should_exit:
            current_position = 0
            trade_signal = -1

        positions.append(current_position)
        trade_signals.append(trade_signal)

    result["trend_entry_condition"] = entry_condition.fillna(False)
    result["trend_exit_condition"] = exit_condition.fillna(False)
    result["trend_position"] = positions
    result["trend_trade_signal"] = trade_signals
    result["trend_buy_signal"] = result["trend_trade_signal"] == 1
    result["trend_sell_signal"] = result["trend_trade_signal"] == -1

    return result


# Strategy 2
def generate_rsi_bollinger_mean_reversion_signals(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Strategy 2: RSI + Bollinger Band mean reversion.

    Buy when:
        RSI14 < 30
        Close < lower Bollinger Band

    Sell when:
        RSI14 > 70
        Close > upper Bollinger Band

    This is a long-only strategy:
        1 = long
        0 = cash
    """
    result = df.copy()

    required_columns = {
        price_col,
        "rsi_14",
        "bb_lower_20",
        "bb_upper_20",
    }

    if not required_columns.issubset(result.columns):
        result = add_required_indicators(result, price_col=price_col)

    entry_condition = (
        (result["rsi_14"] < 30)
        & (result[price_col] < result["bb_lower_20"])
    )

    exit_condition = (
        (result["rsi_14"] > 70)
        & (result[price_col] > result["bb_upper_20"])
    )

    current_position = 0
    positions = []
    trade_signals = []

    for should_enter, should_exit in zip(
        entry_condition.fillna(False),
        exit_condition.fillna(False),
    ):
        trade_signal = 0

        if current_position == 0 and should_enter:
            current_position = 1
            trade_signal = 1

        elif current_position == 1 and should_exit:
            current_position = 0
            trade_signal = -1

        positions.append(current_position)
        trade_signals.append(trade_signal)

    result["mean_reversion_entry_condition"] = entry_condition.fillna(False)
    result["mean_reversion_exit_condition"] = exit_condition.fillna(False)
    result["mean_reversion_position"] = positions
    result["mean_reversion_trade_signal"] = trade_signals
    result["mean_reversion_buy_signal"] = result["mean_reversion_trade_signal"] == 1
    result["mean_reversion_sell_signal"] = result["mean_reversion_trade_signal"] == -1

    return result

# Custom Strategy
def generate_custom_multifactor_signals(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Strategy 3: Multi-factor trend, momentum, and volatility strategy.

    Indicators used:
        Trend: SMA50, SMA200, EMA20
        Momentum: MACD, RSI14
        Volatility: Bollinger Bands

    Buy when:
        Close > SMA200
        EMA20 > SMA50
        MACD > MACD Signal
        50 < RSI14 < 70
        Close > upper Bollinger Band

    Sell when:
        Close < EMA20
        MACD < MACD Signal
        RSI14 < 45
        Close < middle Bollinger Band

    This is a long-only strategy:
        1 = long
        0 = cash
    """
    result = df.copy()

    required_columns = {
        price_col,
        "sma_50",
        "sma_200",
        "macd",
        "macd_signal",
        "rsi_14",
        "bb_upper_20",
        "bb_middle_20",
    }

    if not required_columns.issubset(result.columns):
        result = add_required_indicators(result, price_col=price_col)

    if "ema_20" not in result.columns:
        result = add_exponential_moving_average(
            result,
            span=20,
            price_col=price_col,
        )

    trend_filter = (
        (result[price_col] > result["sma_200"])
        & (result["ema_20"] > result["sma_50"])
    )

    momentum_filter = (
        (result["macd"] > result["macd_signal"])
        & (result["rsi_14"] > 50)
        & (result["rsi_14"] < 70)
    )

    volatility_breakout = result[price_col] > result["bb_upper_20"]

    entry_condition = trend_filter & momentum_filter & volatility_breakout

    exit_condition = (
        (result[price_col] < result["ema_20"])
        | (result["macd"] < result["macd_signal"])
        | (result["rsi_14"] < 45)
        | (result[price_col] < result["bb_middle_20"])
    )

    current_position = 0
    positions = []
    trade_signals = []

    for should_enter, should_exit in zip(
        entry_condition.fillna(False),
        exit_condition.fillna(False),
    ):
        trade_signal = 0

        if current_position == 0 and should_enter:
            current_position = 1
            trade_signal = 1

        elif current_position == 1 and should_exit:
            current_position = 0
            trade_signal = -1

        positions.append(current_position)
        trade_signals.append(trade_signal)

    result["custom_trend_filter"] = trend_filter.fillna(False)
    result["custom_momentum_filter"] = momentum_filter.fillna(False)
    result["custom_volatility_breakout"] = volatility_breakout.fillna(False)
    result["custom_entry_condition"] = entry_condition.fillna(False)
    result["custom_exit_condition"] = exit_condition.fillna(False)
    result["custom_position"] = positions
    result["custom_trade_signal"] = trade_signals
    result["custom_buy_signal"] = result["custom_trade_signal"] == 1
    result["custom_sell_signal"] = result["custom_trade_signal"] == -1

    return result

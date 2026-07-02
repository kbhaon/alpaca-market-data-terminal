from __future__ import annotations

import pandas as pd


# TODO: fill in after implementing add_ml_features. Every column name listed
# here is treated as a model input by src/models.py.
FEATURE_COLUMNS: list[str] = []


def add_ml_features(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
    """
    Return a copy of df with all ML feature columns added.

    Input df is the raw output of historical.fetch_daily_ohlcv():
    `timestamp` (UTC) + OHLCV columns, sorted by timestamp. Keep those
    original columns intact — the backtester and plots need them. Warmup
    rows may contain NaN features.

    TODO:
        - add at least 6 technical indicators across categories
          (trend / momentum / volatility / volume); src/indicators.py already
          has SMA/EMA/MACD/RSI/Bollinger/Stochastic helpers, so mainly the
          volatility/volume ones (e.g. ATR, OBV) need writing
        - add log returns
        - add rolling mean and rolling std
        - list every added column in FEATURE_COLUMNS
    """
    raise NotImplementedError("add_ml_features is not implemented yet.")

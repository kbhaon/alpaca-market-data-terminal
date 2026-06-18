from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.config import get_settings
from src.data_connector import resolve_data_feed


def get_historical_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    days: int = 30,
    timeframe_value: int = 5,
    timeframe_unit: TimeFrameUnit = TimeFrameUnit.Minute,
) -> pd.DataFrame:
    end = datetime.now(UTC)
    start = end - timedelta(days=days)
    settings = get_settings()

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(timeframe_value, timeframe_unit),
        start=start,
        end=end,
        feed=resolve_data_feed(settings.data_feed),
    )

    bars = client.get_stock_bars(request)
    df = bars.df
    if df.empty:
        return df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    return df.reset_index()

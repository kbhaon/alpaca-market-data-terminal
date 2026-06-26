from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.config import get_settings
from src.data_connector import get_historical_client, resolve_data_feed

REQUIRED_DAILY_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
OPTIONAL_DAILY_COLUMNS = ["trade_count", "vwap"]
DEFAULT_TICKERS = ["AAPL", "MSFT", "SPY", "QQQ", "NVDA"]


def normalize_symbol(symbol: str) -> str:
    """Normalize user-entered ticker text before sending it to Alpaca."""
    normalized_symbol = str(symbol).strip().upper()
    if not normalized_symbol:
        raise ValueError("Ticker symbol cannot be blank.")
    return normalized_symbol


def _to_utc_datetime(value: str | date | datetime | None) -> datetime | None:
    """Convert a date-like value into a timezone-aware UTC datetime."""
    if value is None:
        return None

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)

    return timestamp.to_pydatetime()


def get_default_date_range(years: int = 5, extra_days: int = 14) -> tuple[datetime, datetime]:
    """Return a buffered date range for at least five years of daily bars."""
    if years < 1:
        raise ValueError("years must be at least 1.")

    end = datetime.now(UTC)
    start = end - timedelta(days=int(years * 365.25) + extra_days)
    return start, end


def _empty_daily_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_DAILY_COLUMNS + OPTIONAL_DAILY_COLUMNS)


def _require_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _flatten_alpaca_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Flatten Alpaca's usual symbol/timestamp MultiIndex into normal columns."""
    if df is None or df.empty:
        return _empty_daily_frame()

    result = df.copy()
    if isinstance(result.index, pd.MultiIndex):
        symbol_level = "symbol" if "symbol" in result.index.names else 0
        try:
            result = result.xs(symbol, level=symbol_level, drop_level=True)
        except KeyError:
            return _empty_daily_frame()

    result = result.reset_index()
    result.columns = [str(column).lower() for column in result.columns]

    if "timestamp" not in result.columns and "index" in result.columns:
        result = result.rename(columns={"index": "timestamp"})

    result["symbol"] = symbol
    return result


def clean_daily_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Alpaca daily bars into assignment-friendly OHLCV columns."""
    if df is None or df.empty:
        return _empty_daily_frame()

    result = df.copy()
    result.columns = [str(column).lower() for column in result.columns]

    _require_columns(result, REQUIRED_DAILY_COLUMNS)

    result["symbol"] = result["symbol"].astype(str).str.upper().str.strip()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")

    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in OPTIONAL_DAILY_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=REQUIRED_DAILY_COLUMNS)
    result = result.sort_values("timestamp")
    result = result.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    result = result.reset_index(drop=True)

    ordered_columns = REQUIRED_DAILY_COLUMNS + [
        column for column in OPTIONAL_DAILY_COLUMNS if column in result.columns
    ]
    return result[ordered_columns]


def validate_daily_history(
    df: pd.DataFrame,
    years: int = 5,
    min_trading_days_per_year: int = 200,
) -> None:
    """Raise if the returned dataset is clearly too short for the assignment."""
    if df is None or df.empty:
        raise ValueError("No historical data was returned by Alpaca.")

    minimum_rows = years * min_trading_days_per_year
    if len(df) >= minimum_rows:
        return

    symbol = df["symbol"].iloc[0] if "symbol" in df.columns and not df.empty else "the selected ticker"
    raise ValueError(
        f"Only {len(df):,} daily bars were returned for {symbol}. "
        f"The assignment needs about {minimum_rows:,}+ daily bars for {years} years. "
        "Try a large, liquid ticker such as AAPL, MSFT, SPY, QQQ, or NVDA, "
        "or check your Alpaca data feed setting."
    )


def fetch_daily_ohlcv_checked(
    symbol: str,
    years: int = 5,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    client: StockHistoricalDataClient | None = None,
    feed_name: str | None = None,
    validate_history: bool = True,
) -> pd.DataFrame:
    """
    Reference implementation with input normalization, date handling, and validation.

    This lives outside src for now so src.historical can stay focused on fetching
    bars. A future app2.py can decide whether to reuse these UI/workflow helpers.
    """
    normalized_symbol = normalize_symbol(symbol)
    default_start, default_end = get_default_date_range(years=years)
    request_start = _to_utc_datetime(start) or default_start
    request_end = _to_utc_datetime(end) or default_end

    if request_start >= request_end:
        raise ValueError("start must be earlier than end.")

    settings = get_settings()
    historical_client = client or get_historical_client(settings)
    data_feed = resolve_data_feed(feed_name or settings.data_feed)

    request = StockBarsRequest(
        symbol_or_symbols=normalized_symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=request_start,
        end=request_end,
        feed=data_feed,
        limit=10_000,
    )

    bars = historical_client.get_stock_bars(request)
    raw_df = _flatten_alpaca_bars(bars.df, normalized_symbol)
    clean_df = clean_daily_ohlcv(raw_df)

    if validate_history:
        validate_daily_history(clean_df, years=years)

    return clean_df


def fetch_multiple_daily_ohlcv(
    symbols: Iterable[str],
    years: int = 5,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    feed_name: str | None = None,
    validate_history: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch checked daily OHLCV data for several tickers."""
    settings = get_settings()
    client = get_historical_client(settings)
    results: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        normalized_symbol = normalize_symbol(symbol)
        results[normalized_symbol] = fetch_daily_ohlcv_checked(
            symbol=normalized_symbol,
            years=years,
            start=start,
            end=end,
            client=client,
            feed_name=feed_name or settings.data_feed,
            validate_history=validate_history,
        )

    return results


def save_daily_ohlcv_to_csv(
    df: pd.DataFrame,
    output_dir: str | Path = "data",
) -> Path:
    """Save one ticker's cleaned daily OHLCV data to CSV."""
    if df is None or df.empty:
        raise ValueError("Cannot save an empty DataFrame.")

    _require_columns(df, ["symbol", "timestamp"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    symbol = str(df["symbol"].iloc[0]).upper()
    start_date = pd.to_datetime(df["timestamp"].min()).date().isoformat()
    end_date = pd.to_datetime(df["timestamp"].max()).date().isoformat()
    file_path = output_path / f"{symbol}_daily_ohlcv_{start_date}_to_{end_date}.csv"

    df.to_csv(file_path, index=False)
    return file_path


def load_assignment_data(
    symbol: str,
    years: int = 5,
    save_csv: bool = False,
    output_dir: str | Path = "data",
) -> pd.DataFrame:
    """Fetch checked daily OHLCV data and optionally save a CSV copy."""
    df = fetch_daily_ohlcv_checked(symbol=symbol, years=years)

    if save_csv:
        save_daily_ohlcv_to_csv(df, output_dir=output_dir)

    return df

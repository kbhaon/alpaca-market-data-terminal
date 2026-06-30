from __future__ import annotations

import pandas as pd
import streamlit as st
from alpaca.data.timeframe import TimeFrameUnit

from src.backtester import STRATEGY_SPECS, build_buy_hold_result, run_backtest
from src.company import get_company_name
from src.company_search import CompanyMatch, get_company_choices
from src.data_connector import get_historical_client
from src.historical import get_historical_bars
from src.indicators import add_exponential_moving_average, add_required_indicators
from src.metrics import build_metrics_table, infer_periods_per_year
from src.plots import plot_drawdowns, plot_portfolio_values, plot_signal_chart


st.set_page_config(page_title="Mini Trading Strategy Backtester", layout="wide")


RANGE_PRESETS = {
    "1D": pd.DateOffset(days=1),
    "5D": pd.DateOffset(days=5),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "5Y": pd.DateOffset(years=5),
}

INDICATOR_OPTIONS = [
    "SMA 50",
    "SMA 200",
    "EMA 12",
    "EMA 26",
    "EMA 20",
    "MACD",
    "RSI 14",
    "Bollinger Bands",
    "Momentum 10",
    "Stochastic Oscillator",
]


def resolve_date_range(
    selected_range: str,
    custom_days: int | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    resolved_end = (
        pd.Timestamp(end)
        if end is not None
        else pd.Timestamp.now(tz="UTC").floor("min")
    )

    if resolved_end.tzinfo is None:
        resolved_end = resolved_end.tz_localize("UTC")
    else:
        resolved_end = resolved_end.tz_convert("UTC")

    if selected_range == "Custom":
        offset = pd.DateOffset(days=int(custom_days or 365))
    else:
        offset = RANGE_PRESETS[selected_range]

    return resolved_end - offset, resolved_end


def resolve_tick_spec(
    selected_tick: str,
    custom_tick: int | None = None,
) -> tuple[int, TimeFrameUnit, int]:
    if selected_tick == "Custom":
        custom_tick_minutes = int(custom_tick or 1)

        if custom_tick_minutes <= 59:
            return custom_tick_minutes, TimeFrameUnit.Minute, 1

        if custom_tick_minutes % 60 == 0:
            return custom_tick_minutes // 60, TimeFrameUnit.Hour, 1

        raise ValueError(
            "Custom tick must be 1-59 minutes or a whole-hour minute value "
            "(60, 120, 180, ...)."
        )

    if selected_tick.endswith("m"):
        return int(selected_tick[:-1]), TimeFrameUnit.Minute, 1

    if selected_tick in {"1D", "5D"}:
        aggregate = 5 if selected_tick == "5D" else 1
        return 1, TimeFrameUnit.Day, aggregate

    if selected_tick in {"1M", "3M"}:
        return int(selected_tick[:-1]), TimeFrameUnit.Month, 1

    if selected_tick == "1h":
        return 1, TimeFrameUnit.Hour, 1

    return 1, TimeFrameUnit.Minute, 1


def aggregate_bars_by_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if days <= 1 or df.empty:
        return df

    resampled = (
        df.set_index("timestamp")
        .resample(f"{days}D", label="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
    )

    return resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()


def render_invalid_symbol_message(target=st) -> None:
    target.markdown(
        """
        <div style="
            height: 360px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
        ">
            <div>
                <div style="font-size: 1.2rem; font-weight: 700;">
                    This symbol does not exist
                </div>
                <div style="margin-top: 0.5rem; color: #6b7280;">
                    Pick another equity or enter another ticker.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    if "timestamp" not in result.columns and "index" in result.columns:
        result = result.rename(columns={"index": "timestamp"})

    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")

    for column in ["open", "high", "low", "close", "volume"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=["timestamp", "close"])
    result = result.sort_values("timestamp")
    result = result.drop_duplicates(subset=["timestamp"], keep="last")
    return result.reset_index(drop=True)


def add_selected_indicators(df: pd.DataFrame, selected_indicators: list[str]) -> pd.DataFrame:
    result = add_required_indicators(df, price_col="close")

    if "EMA 20" in selected_indicators and "ema_20" not in result.columns:
        result = add_exponential_moving_average(result, span=20, price_col="close")

    return result


def safe_company_name(
    symbol: str,
    selected_match: CompanyMatch | None,
    is_valid_symbol: bool,
) -> str:
    if not is_valid_symbol:
        return symbol or "Invalid symbol"

    if selected_match is not None and selected_match.symbol == symbol:
        return selected_match.name

    try:
        return get_company_name(symbol)
    except Exception:
        return symbol


st.title("Mini Trading Strategy Backtester")

if "bt_ticker_input" not in st.session_state:
    st.session_state.bt_ticker_input = "HOOD"

if "bt_selected_symbol" not in st.session_state:
    st.session_state.bt_selected_symbol = st.session_state.bt_ticker_input

equity_choices: list[CompanyMatch] = get_company_choices()
equity_by_label = {match.display: match for match in equity_choices}
equity_by_symbol = {match.symbol: match for match in equity_choices}

equity_placeholder = "Select or search an equity"
equity_options = [equity_placeholder, *equity_by_label.keys()]


def sync_from_equity() -> None:
    match = equity_by_label.get(st.session_state.bt_equity_selection)
    if match is None:
        return

    st.session_state.bt_selected_symbol = match.symbol
    st.session_state.bt_ticker_input = match.symbol


def sync_from_ticker() -> None:
    symbol = st.session_state.bt_ticker_input.strip().upper()
    if symbol:
        st.session_state.bt_selected_symbol = symbol


current_symbol = st.session_state.bt_selected_symbol.strip().upper()
current_match = equity_by_symbol.get(current_symbol)

st.session_state.bt_ticker_input = current_symbol
if current_match is not None:
    st.session_state.bt_equity_selection = current_match.display
else:
    st.session_state.bt_equity_selection = equity_placeholder

st.sidebar.selectbox(
    "Stocks & ETFs",
    options=equity_options,
    key="bt_equity_selection",
    on_change=sync_from_equity,
)

symbol_input = st.sidebar.text_input(
    "Ticker",
    key="bt_ticker_input",
    on_change=sync_from_ticker,
)

symbol_input = symbol_input.strip().upper()
if symbol_input:
    st.session_state.bt_selected_symbol = symbol_input

symbol = symbol_input
selected_match = equity_by_symbol.get(symbol)
is_valid_symbol = bool(symbol) and (not equity_by_symbol or symbol in equity_by_symbol)

time_range = st.sidebar.radio(
    "Time range",
    options=[*RANGE_PRESETS.keys(), "Custom"],
    index=list(RANGE_PRESETS.keys()).index("5Y"),
    horizontal=True,
)

if time_range == "Custom":
    custom_days = st.sidebar.slider(
        "Custom range (calendar days)",
        min_value=1,
        max_value=1827,
        value=365,
    )
else:
    custom_days = None

tick_choice = st.sidebar.radio(
    "Tick size",
    options=["1m", "5m", "15m", "30m", "1h", "1D", "5D", "1M", "3M", "Custom"],
    index=5,
    horizontal=True,
)

if tick_choice == "Custom":
    custom_tick = st.sidebar.slider(
        "Custom tick size (minutes)",
        min_value=1,
        max_value=240,
        value=60,
    )
else:
    custom_tick = None

selected_indicators = st.sidebar.multiselect(
    "Indicators",
    options=INDICATOR_OPTIONS,
    default=["SMA 50", "SMA 200", "MACD", "RSI 14", "Bollinger Bands"],
)

selected_strategy_names = st.sidebar.multiselect(
    "Strategies",
    options=list(STRATEGY_SPECS.keys()),
    default=list(STRATEGY_SPECS.keys()),
)

range_start, range_end = resolve_date_range(time_range, custom_days)

try:
    timeframe_value, timeframe_unit, aggregate_factor = resolve_tick_spec(
        tick_choice,
        custom_tick,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

company_name = safe_company_name(symbol, selected_match, is_valid_symbol)
st.subheader(f"{company_name} ({symbol})")

if not selected_strategy_names:
    st.warning("Select at least one strategy.")
    st.stop()

if not is_valid_symbol:
    render_invalid_symbol_message(st)
    st.stop()

try:
    client = get_historical_client()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

requested_key = (
    f"{symbol}|{range_start.isoformat()}|{range_end.isoformat()}|{timeframe_value}|"
    f"{timeframe_unit.value}|{aggregate_factor}"
)

has_data = (
    "bt_historical_df" in st.session_state
    and st.session_state.get("bt_historical_key") == requested_key
)

if not has_data:
    with st.spinner("Loading historical bars..."):
        request_value = timeframe_value
        request_unit = timeframe_unit

        if timeframe_unit == TimeFrameUnit.Day and aggregate_factor > 1:
            request_value = 1

        bars = get_historical_bars(
            client=client,
            symbol=symbol,
            timeframe_value=request_value,
            timeframe_unit=request_unit,
            start=range_start.to_pydatetime(),
            end=range_end.to_pydatetime(),
        )

        bars = clean_price_data(bars)

        if timeframe_unit == TimeFrameUnit.Day and aggregate_factor > 1:
            bars = aggregate_bars_by_days(bars, aggregate_factor)
            bars = clean_price_data(bars)

        st.session_state.bt_historical_df = bars
        st.session_state.bt_historical_key = requested_key

price_df = st.session_state.bt_historical_df

if price_df.empty:
    st.warning("No historical bars returned for this symbol.")
    st.stop()

analysis_df = add_selected_indicators(price_df, selected_indicators)
periods_per_year = infer_periods_per_year(
    timeframe_value,
    timeframe_unit,
    aggregate_factor,
)

buy_hold_result = build_buy_hold_result(analysis_df)
strategy_results = []

for strategy_name in selected_strategy_names:
    spec = STRATEGY_SPECS[strategy_name]
    signals = spec.signal_function(analysis_df.copy(), price_col="close")
    strategy_results.append(run_backtest(signals, spec))

all_results = [buy_hold_result, *strategy_results]

st.markdown(
    '<div style="font-size: 24px; font-weight: 700; margin: 1.25rem 0 0.5rem;">'
    "Buy/Sell Signals"
    "</div>",
    unsafe_allow_html=True,
)
signal_columns = st.columns(len(strategy_results))

for column, result in zip(signal_columns, strategy_results):
    with column:
        signal_fig = plot_signal_chart(result, selected_indicators, timeframe_unit)
        st.plotly_chart(signal_fig, width="stretch")

portfolio_fig = plot_portfolio_values(all_results, timeframe_unit)
st.plotly_chart(portfolio_fig, width="stretch")

metrics_table = build_metrics_table(all_results, periods_per_year)
st.dataframe(metrics_table, width="stretch")

drawdown_fig = plot_drawdowns(all_results, timeframe_unit)
st.plotly_chart(drawdown_fig, width="stretch")

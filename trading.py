from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from alpaca.data.timeframe import TimeFrameUnit

from src.backtester import build_buy_hold_result, build_ml_strategy_spec, run_backtest
from src.company import get_company_name
from src.company_search import CompanyMatch, get_company_choices
from src.data_connector import get_historical_client
from src.execution import LOG_FILE, execute_latest_signal
from src.features import build_feature_pca_pipeline, transform_latest_features
from src.historical import fetch_daily_ohlcv, get_historical_bars
from src.live_quotes import get_live_quote_manager
from src.metrics import build_metrics_table
from src.models import PROBABILITY_THRESHOLD, run_ml_signal_pipeline, score_pca_features
from src.plots import (
    plot_drawdowns,
    plot_pca_explained_variance,
    plot_portfolio_values,
    plot_signal_chart,
)


st.set_page_config(page_title="Alpaca Market Data Terminal", layout="wide")


LIVE_QUOTE_REFRESH_SECONDS = 1.0
EASTERN_TZ = "America/New_York"
ML_HISTORY_YEARS = 5
ML_PERIODS_PER_YEAR = 252
ML_SIGNAL_INDICATORS = ["MACD", "RSI 14", "Bollinger Bands"]
ML_MODEL_CACHE_STATE_KEY = "ml_model_cache"
ML_EXECUTION_REPORTS_STATE_KEY = "ml_last_execution_reports"
ML_LATEST_SIGNALS_STATE_KEY = "ml_latest_signal_frames"


RANGE_PRESETS = {
    "1D": pd.DateOffset(days=1),
    "5D": pd.DateOffset(days=5),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "5Y": pd.DateOffset(years=5),
}


def resolve_date_range(
    selected_range: str,
    custom_days: int | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Map a range button to an explicit calendar start/end range."""
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
        offset = pd.DateOffset(days=int(custom_days or 30))
    else:
        offset = RANGE_PRESETS[selected_range]

    return resolved_end - offset, resolved_end


def resolve_tick_spec(
    selected_tick: str,
    custom_tick: int | None = None,
) -> tuple[int, TimeFrameUnit, int]:
    """Map a tick selector value to request timeframe and optional aggregate factor."""
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
    """Aggregate daily bars into multi-day OHLCV bars."""
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


def prepare_historical_display_df(
    df: pd.DataFrame,
    timeframe_unit: TimeFrameUnit,
) -> pd.DataFrame:
    """Return a display copy with chart timestamps shown in Eastern time."""
    if df.empty or "timestamp" not in df.columns:
        return df

    display_df = df.copy()
    timestamps = pd.to_datetime(display_df["timestamp"], utc=True)

    if timeframe_unit in {TimeFrameUnit.Minute, TimeFrameUnit.Hour}:
        display_df["timestamp"] = (
            timestamps.dt.tz_convert(EASTERN_TZ).dt.tz_localize(None)
        )
    else:
        display_df["timestamp"] = timestamps.dt.date

    return display_df


def render_invalid_symbol_message(target=st) -> None:
    """Show an empty-chart style message for invalid ticker input."""
    target.markdown(
        """
        <div style="
            height: 560px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
        ">
            <div>
                <div style="font-size: 1.2rem; font-weight: 700;">
                    This symbol doesn't exist
                </div>
                <div style="margin-top: 0.5rem; color: #6b7280;">
                    Try picking another one for your analysis, and you'll see the data here.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every=LIVE_QUOTE_REFRESH_SECONDS)
def render_live_quote(symbol: str, is_valid_symbol: bool) -> None:
    """
    Refresh only the live quote area.

    Because this is a fragment, only this function reruns on the timer while
    the Alpaca websocket stream keeps receiving quote and trade events.
    """
    manager = get_live_quote_manager(st.session_state)

    if not is_valid_symbol:
        manager.stop()
        st.info("No live quote for invalid symbol.")
        return

    snapshot = manager.get_snapshot(symbol)
    if snapshot is None:
        st.error(f"Could not start live quote stream: {manager.error}")
        return

    st.metric("Bid", snapshot.bid_display)
    st.metric("Ask", snapshot.ask_display)
    st.metric("Last", snapshot.last_trade_display)
    if snapshot.updated_at is None:
        st.caption("Waiting for first streamed update.")
    else:
        st.caption(f"Updated at: {snapshot.updated_at_display}")


def _latest_ml_signal_row(signal_df: pd.DataFrame) -> pd.Series | None:
    if signal_df.empty or "ml_probability" not in signal_df.columns:
        return None

    ready = signal_df.dropna(subset=["ml_probability"])
    if ready.empty:
        return None

    return ready.iloc[-1]


def _get_ml_model_cache() -> dict:
    return st.session_state.setdefault(ML_MODEL_CACHE_STATE_KEY, {})


def _get_ml_execution_reports() -> dict:
    return st.session_state.setdefault(ML_EXECUTION_REPORTS_STATE_KEY, {})


def _get_ml_latest_signal_frames() -> dict:
    return st.session_state.setdefault(ML_LATEST_SIGNALS_STATE_KEY, {})


def _read_paper_trading_log(max_lines: int = 80) -> str:
    if not LOG_FILE.exists():
        return "No paper-trading log has been written yet."

    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    recent_lines = lines[-max_lines:]
    return "\n".join(recent_lines) if recent_lines else "No paper-trading log entries yet."


def _build_ml_results(
    symbol: str,
    years: int,
    probability_threshold: float,
    test_size: float,
) -> dict:
    price_df = fetch_daily_ohlcv(symbol, years=years)
    if price_df.empty:
        raise ValueError(f"No daily OHLCV bars returned for {symbol}.")

    pca_result = build_feature_pca_pipeline(
        price_df,
        price_col="close",
        test_size=test_size,
    )
    ml_signal_result = run_ml_signal_pipeline(
        pca_result,
        probability_threshold=probability_threshold,
        trade_on_test_only=True,
    )
    signal_df = ml_signal_result.signal_df

    test_signal_df = signal_df[signal_df["ml_sample_type"].eq("test")].copy()
    if len(test_signal_df) < 2:
        test_signal_df = signal_df.dropna(subset=["ml_probability"]).copy()

    if len(test_signal_df) < 2:
        raise ValueError("Not enough ML signal rows to run the backtest.")

    ml_spec = build_ml_strategy_spec()
    ml_result = run_backtest(test_signal_df, ml_spec)
    buy_hold_result = build_buy_hold_result(test_signal_df)
    results = [buy_hold_result, ml_result]

    return {
        "price_df": price_df,
        "pca_result": pca_result,
        "ml_signal_result": ml_signal_result,
        "signal_df": signal_df,
        "backtest_df": test_signal_df,
        "ml_result": ml_result,
        "buy_hold_result": buy_hold_result,
        "results": results,
        "metrics_table": build_metrics_table(
            results,
            periods_per_year=ML_PERIODS_PER_YEAR,
        ),
        "trained_at": pd.Timestamp.now(tz="UTC"),
    }


def _train_ml_panel_state(
    symbol: str,
    years: int,
    probability_threshold: float,
    test_size: float,
) -> dict:
    return {
        "symbol": symbol,
        "years": years,
        "probability_threshold": probability_threshold,
        "test_size": test_size,
        **_build_ml_results(
            symbol=symbol,
            years=years,
            probability_threshold=probability_threshold,
            test_size=test_size,
        ),
    }


def _build_fresh_latest_signal(
    symbol: str,
    panel_state: dict,
) -> pd.DataFrame:
    fresh_price_df = fetch_daily_ohlcv(symbol, years=int(panel_state["years"]))
    if fresh_price_df.empty:
        raise ValueError(f"No latest daily OHLCV bars returned for {symbol}.")

    latest_pca_df = transform_latest_features(
        fresh_price_df,
        panel_state["pca_result"],
        price_col="close",
    )
    ml_signal_result = panel_state["ml_signal_result"]
    return score_pca_features(
        pca_frame=latest_pca_df,
        component_columns=ml_signal_result.component_columns,
        model=ml_signal_result.model,
        probability_threshold=ml_signal_result.probability_threshold,
    )


def _execution_report_dict(report) -> dict:
    report_dict = asdict(report)
    report_dict["bar_timestamp"] = str(report_dict["bar_timestamp"])
    return report_dict


def render_ml_trading_panel(symbol: str, is_valid_symbol: bool) -> None:
    st.subheader("ML Trading Signal")

    if not is_valid_symbol:
        st.info("Choose a valid ticker before training the ML signal.")
        return

    model_cache = _get_ml_model_cache()
    panel_state = model_cache.get(symbol)

    cached_threshold = (
        float(panel_state.get("probability_threshold", PROBABILITY_THRESHOLD))
        if panel_state is not None
        else float(PROBABILITY_THRESHOLD)
    )
    cached_test_size = (
        float(panel_state.get("test_size", 0.20))
        if panel_state is not None
        else 0.20
    )

    control_cols = st.columns([1, 1, 1])
    with control_cols[0]:
        probability_threshold = st.slider(
            "Long probability threshold",
            min_value=0.50,
            max_value=0.90,
            value=cached_threshold,
            step=0.01,
            key=f"ml_threshold_{symbol}",
        )
    with control_cols[1]:
        test_size = st.slider(
            "Backtest holdout",
            min_value=0.10,
            max_value=0.50,
            value=cached_test_size,
            step=0.05,
            key=f"ml_holdout_{symbol}",
        )
    with control_cols[2]:
        order_notional = st.number_input(
            "Paper order notional",
            min_value=100.0,
            max_value=1_000_000.0,
            value=100_000.0,
            step=10_000.0,
            key=f"ml_order_notional_{symbol}",
        )

    years = ML_HISTORY_YEARS
    train_button_label = (
        "Train Model / Run Backtest"
        if panel_state is None
        else "Retrain Model / Run Backtest"
    )

    if panel_state is None:
        st.info("Train once for this equity. Later paper orders reuse the cached model and refresh only the latest signal.")
    elif (
        abs(float(probability_threshold) - cached_threshold) > 1e-9
        or abs(float(test_size) - cached_test_size) > 1e-9
    ):
        st.info("The changed model controls will apply after retraining.")

    if st.button(train_button_label, key=f"ml_retrain_{symbol}"):
        with st.spinner("Fetching 5 years of daily bars, fitting PCA, and training ML model..."):
            try:
                model_cache[symbol] = _train_ml_panel_state(
                    symbol=symbol,
                    years=years,
                    probability_threshold=probability_threshold,
                    test_size=test_size,
                )
                panel_state = model_cache[symbol]
                _get_ml_latest_signal_frames().pop(symbol, None)
                _get_ml_execution_reports().pop(symbol, None)
                st.success("Model trained for this equity.")
            except Exception as exc:
                st.error(f"Could not train ML signal: {exc}")
                return

    if panel_state is None:
        return

    trained_at = panel_state.get("trained_at")
    trained_at_display = str(trained_at) if trained_at is not None else "current session"
    st.caption(
        "Using cached model for "
        f"{symbol} trained at {trained_at_display}. "
        "Paper orders refresh latest data without retraining."
    )

    signal_df = panel_state["signal_df"]
    latest = _latest_ml_signal_row(signal_df)
    if latest is None:
        st.warning("The ML pipeline did not produce a latest signal row.")
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Cached Signal", str(latest["ml_signal"]))
    summary_cols[1].metric("P(next day up)", f"{float(latest['ml_probability']):.2%}")
    summary_cols[2].metric("Training Rows", int(signal_df["ml_sample_type"].eq("train").sum()))
    summary_cols[3].metric("Backtest Rows", int(signal_df["ml_sample_type"].eq("test").sum()))

    tab_metrics, tab_charts, tab_trades, tab_paper = st.tabs(
        ["Metrics", "Charts", "Trades", "Paper Order"]
    )

    with tab_metrics:
        st.dataframe(panel_state["metrics_table"], width="stretch")
        st.dataframe(
            signal_df[
                [
                    "timestamp",
                    "close",
                    "ml_sample_type",
                    "ml_probability",
                    "ml_signal",
                    "ml_position",
                    "ml_trade_signal",
                ]
            ].tail(20),
            width="stretch",
        )

    with tab_charts:
        variance = signal_df.attrs.get("ml_pca_explained_variance_ratio", [])
        if variance:
            st.plotly_chart(
                plot_pca_explained_variance(variance, threshold=0.80),
                width="stretch",
            )
        st.plotly_chart(
            plot_signal_chart(
                panel_state["ml_result"],
                ML_SIGNAL_INDICATORS,
                TimeFrameUnit.Day,
            ),
            width="stretch",
        )
        st.plotly_chart(
            plot_portfolio_values(panel_state["results"], TimeFrameUnit.Day),
            width="stretch",
        )
        st.plotly_chart(
            plot_drawdowns(panel_state["results"], TimeFrameUnit.Day),
            width="stretch",
        )

    with tab_trades:
        trades = panel_state["ml_result"].trades
        if trades.empty:
            st.info("No closed ML trades in the current backtest window.")
        else:
            st.dataframe(trades, width="stretch")

    with tab_paper:
        st.caption("Orders submitted here use Alpaca paper trading credentials only.")
        if st.button("Refresh Signal and Submit Paper Order", key=f"ml_submit_{symbol}"):
            with st.spinner("Fetching latest bars, scoring the cached model, and submitting a paper order..."):
                try:
                    latest_signal_df = _build_fresh_latest_signal(symbol, panel_state)
                    _get_ml_latest_signal_frames()[symbol] = latest_signal_df
                    report = execute_latest_signal(
                        symbol,
                        signal_df=latest_signal_df,
                        notional=float(order_notional),
                    )
                    _get_ml_execution_reports()[symbol] = report
                except Exception as exc:
                    st.error(f"Could not submit paper-trading action: {exc}")

        latest_signal_df = _get_ml_latest_signal_frames().get(symbol)
        if latest_signal_df is not None:
            st.dataframe(
                latest_signal_df[
                    [
                        "timestamp",
                        "close",
                        "ml_probability",
                        "ml_signal",
                        "ml_position",
                        "ml_trade_signal",
                    ]
                ],
                width="stretch",
            )

        report = _get_ml_execution_reports().get(symbol)
        if report is not None:
            st.json(_execution_report_dict(report))
            if report.log_lines:
                st.markdown("Latest execution log")
                st.code("\n".join(report.log_lines), language="text")

        st.markdown("Recent paper-trading log")
        st.code(_read_paper_trading_log(), language="text")

st.title("Mini Market Data Terminal v1.0")


if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = "HOOD"

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = st.session_state.ticker_input


equity_choices: list[CompanyMatch] = get_company_choices()
equity_by_label = {match.display: match for match in equity_choices}
equity_by_symbol = {match.symbol: match for match in equity_choices}

equity_placeholder = "Select or search an equity"
equity_options = [equity_placeholder, *equity_by_label.keys()]


def sync_from_equity() -> None:
    match = equity_by_label.get(st.session_state.equity_selection)

    if match is None:
        return

    st.session_state.selected_symbol = match.symbol
    st.session_state.ticker_input = match.symbol


def sync_from_ticker() -> None:
    symbol = st.session_state.ticker_input.strip().upper()

    if symbol:
        st.session_state.selected_symbol = symbol


current_symbol = st.session_state.selected_symbol.strip().upper()
current_match = equity_by_symbol.get(current_symbol)

st.session_state.ticker_input = current_symbol

if current_match is not None:
    st.session_state.equity_selection = current_match.display
else:
    st.session_state.equity_selection = equity_placeholder


st.sidebar.selectbox(
    "Stocks & ETFs",
    options=equity_options,
    key="equity_selection",
    on_change=sync_from_equity,
)


symbol_input = st.sidebar.text_input(
    "Ticker",
    key="ticker_input",
    on_change=sync_from_ticker,
)

symbol_input = symbol_input.strip().upper()

if symbol_input:
    st.session_state.selected_symbol = symbol_input

symbol = symbol_input
selected_match = equity_by_symbol.get(symbol)
is_valid_symbol = bool(symbol) and (not equity_by_symbol or symbol in equity_by_symbol)


time_range = st.sidebar.radio(
    "Time range",
    options=[*RANGE_PRESETS.keys(), "Custom"],
    index=0,
    horizontal=True,
)

if time_range == "Custom":
    custom_days = st.sidebar.slider(
        "Custom range (calendar days)",
        min_value=1,
        max_value=1827,
        value=30,
    )
else:
    custom_days = None


tick_choice = st.sidebar.radio(
    "Tick size",
    options=["1m", "5m", "15m", "30m", "1h", "1D", "5D", "1M", "3M", "Custom"],
    index=1,
    horizontal=True,
)

if tick_choice == "Custom":
    custom_tick = st.sidebar.slider(
        "Custom tick size (minutes)",
        min_value=1,
        max_value=240,
        value=5,
    )
else:
    custom_tick = None


range_start, range_end = resolve_date_range(time_range, custom_days)

try:
    timeframe_value, timeframe_unit, aggregate_factor = resolve_tick_spec(
        tick_choice,
        custom_tick,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()


try:
    client = get_historical_client()
except ValueError as exc:
    st.error(str(exc))
    st.stop()


left, right = st.columns([2, 1])


with left:
    if not is_valid_symbol:
        company_name = symbol or "Invalid symbol"
    elif selected_match is not None and selected_match.symbol == symbol:
        company_name = selected_match.name
    else:
        company_name = get_company_name(symbol)

    st.subheader(f"{company_name} ({symbol})")

    chart_area = st.empty()
    table_area = st.empty()

    if not is_valid_symbol:
        render_invalid_symbol_message(chart_area)
        table_area.markdown("")
    else:
        requested_key = (
            f"{symbol}|{range_start.isoformat()}|{range_end.isoformat()}|"
            f"{timeframe_value}|"
            f"{timeframe_unit.value}|{aggregate_factor}"
        )

        has_data = (
            "historical_df" in st.session_state
            and st.session_state.get("historical_key") == requested_key
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

                if timeframe_unit == TimeFrameUnit.Day and aggregate_factor > 1:
                    bars = aggregate_bars_by_days(bars, aggregate_factor)

                st.session_state.historical_df = bars
                st.session_state.historical_key = requested_key

        df = st.session_state.historical_df

        if df.empty:
            chart_area.warning("No historical bars returned for this symbol.")
            table_area.markdown("")
        else:
            display_df = prepare_historical_display_df(df, timeframe_unit)

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                row_heights=[0.72, 0.28],
                vertical_spacing=0.05,
            )

            fig.add_trace(
                go.Candlestick(
                    x=display_df["timestamp"],
                    open=display_df["open"],
                    high=display_df["high"],
                    low=display_df["low"],
                    close=display_df["close"],
                    name="Price",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Bar(
                    x=display_df["timestamp"],
                    y=display_df["volume"],
                    name="Volume",
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                height=640,
                xaxis_rangeslider_visible=False,
            )

            fig.update_xaxes(
                title_text="Time (E.T.)",
                row=2,
                col=1,
            )

            # Fixed deprecation warning:
            # use_container_width=True -> width="stretch"
            chart_area.plotly_chart(fig, width="stretch")

            # Fixed deprecation warning:
            # use_container_width=True -> width="stretch"
            table_area.dataframe(display_df.tail(50), width="stretch")


with right:
    st.subheader("Live Quote")

    # Only this quote area refreshes automatically.
    # The chart/table/sidebar will not refresh every second anymore.
    render_live_quote(symbol, is_valid_symbol)


st.divider()
with st.expander("ML Trading Signal", expanded=False):
    render_ml_trading_panel(symbol, is_valid_symbol)

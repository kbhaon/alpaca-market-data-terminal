from __future__ import annotations
from typing import Any
import math
import pandas as pd
from alpaca.data.timeframe import TimeFrameUnit


INITIAL_CAPITAL = 100000

def infer_periods_per_year(
    timeframe_value: int,
    timeframe_unit: TimeFrameUnit,
    aggregate_factor: int,
) -> float:
    if timeframe_unit == TimeFrameUnit.Minute:
        return max(1.0, 252 * 390 / max(timeframe_value, 1))

    if timeframe_unit == TimeFrameUnit.Hour:
        return max(1.0, 252 * 6.5 / max(timeframe_value, 1))

    if timeframe_unit == TimeFrameUnit.Month:
        return max(1.0, 12 / max(timeframe_value, 1))

    return max(1.0, 252 / max(timeframe_value * aggregate_factor, 1))


def calculate_drawdown(values: pd.Series) -> pd.Series:
    running_peak = values.cummax()
    return values / running_peak - 1

# All the required metrics: 
# 1. Total Return
# 2. CAGR
# 3. Volatility
# 4. Sharpe Ratio
# 5. Sortino Ratio
# 6. Max Drawdown
# 7. Win Rate

def calculate_performance_metrics(
    result: Any,
    periods_per_year: float,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict[str, float]:
    history = result.history
    returns = pd.to_numeric(history["strategy_return"], errors="coerce").dropna()
    ending_value = float(history["portfolio_value"].iloc[-1])
    total_return = ending_value / initial_capital - 1 # total return

    years = max(len(history) / periods_per_year, 1 / periods_per_year)
    cagr = (ending_value / initial_capital) ** (1 / years) - 1 # CAGR
    volatility = returns.std(ddof=0) * math.sqrt(periods_per_year) # Volatility

    sharpe_ratio = math.nan
    if returns.std(ddof=0) != 0:
        sharpe_ratio = returns.mean() / returns.std(ddof=0) * math.sqrt(periods_per_year) # Sharpe Ratio

    downside_returns = returns[returns < 0]
    sortino_ratio = math.nan
    if not downside_returns.empty and downside_returns.std(ddof=0) != 0:
        sortino_ratio = (
            returns.mean()
            / downside_returns.std(ddof=0)
            * math.sqrt(periods_per_year)
        ) # Sortino Ratio

    max_drawdown = float(history["drawdown"].min()) # Max Drawdown

    closed_trades = result.trades.dropna(subset=["return"])
    win_rate = math.nan
    if not closed_trades.empty:
        win_rate = float((closed_trades["return"] > 0).mean()) # Win Rate

    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Max Drawdown": max_drawdown,
        "Win Rate": win_rate,
    }


def format_metric_value(metric: str, value: float) -> str:
    if pd.isna(value):
        return "n/a"
    if metric in {"Total Return", "CAGR", "Volatility", "Max Drawdown", "Win Rate"}:
        return f"{value:.2%}"
    return f"{value:.2f}"


def build_metrics_table(
    results: list[Any],
    periods_per_year: float,
) -> pd.DataFrame:
    rows = []

    for result in results:
        metrics = calculate_performance_metrics(result, periods_per_year)
        rows.append(
            {
                "Strategy": result.name,
                **{
                    metric: format_metric_value(metric, value)
                    for metric, value in metrics.items()
                },
            }
        )

    return pd.DataFrame(rows).set_index("Strategy")

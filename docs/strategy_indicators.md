# Strategy and Indicator Documentation

This document summarizes the strategies and indicators used by the strategy
backtester.

## Strategy Overview

All implemented strategies are long-only:

- `1` means the strategy is invested.
- `0` means the strategy is in cash.

The backtester also includes a buy-and-hold benchmark for comparison.

## Strategy 1: Trend Following

The trend-following strategy uses MACD and a 200-period simple moving average.

Buy when:

- MACD is above the MACD signal line.
- Closing price is above SMA200.

Sell when:

- MACD falls below the MACD signal line.
- Closing price falls below SMA200.

Purpose:

This strategy tries to participate when price momentum is positive and the stock
is trading above its longer-term trend.

## Strategy 2: Mean Reversion

The mean-reversion strategy uses RSI and Bollinger Bands.

Buy when:

- RSI14 is below 30.
- Closing price is below the lower Bollinger Band.

Sell when:

- RSI14 is above 70.
- Closing price is above the upper Bollinger Band.

Purpose:

This strategy looks for short-term oversold conditions and exits after the price
recovers into an overbought area.

## Strategy 3: Custom Multi-Factor

The custom strategy combines trend, momentum, and volatility breakout signals.

Buy when all of these are true:

- Closing price is above SMA200.
- EMA20 is above SMA50.
- MACD is above the MACD signal line.
- RSI14 is between 50 and 70.
- Closing price is above the upper Bollinger Band.

Sell when any of these are true:

- Closing price falls below EMA20.
- MACD falls below the MACD signal line.
- RSI14 falls below 45.
- Closing price falls below the middle Bollinger Band.

Purpose:

This strategy is stricter on entry than exit. It waits for trend, momentum, and
breakout conditions to align before buying, then exits when price or momentum
starts to weaken.

## Indicators

The backtester can calculate and display these indicators:

- SMA50
- SMA200
- EMA12
- EMA20
- EMA26
- MACD
- MACD signal
- MACD histogram
- RSI14
- Bollinger Bands
- Momentum 10
- Stochastic oscillator

## Notes

These strategies are for exploratory analysis and classroom demonstration. They
do not include commissions, slippage, taxes, funding costs, or dividend
reinvestment.

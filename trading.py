from __future__ import annotations

import argparse

from src.execution import DEFAULT_ORDER_NOTIONAL, execute_latest_signal


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the ML signal pipeline once and submit paper-only orders: "
            "buy if the signal is LONG, close the position if FLAT."
        )
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default="AAPL",
        help="Ticker to trade (default: AAPL)",
    )
    parser.add_argument(
        "--notional",
        type=float,
        default=DEFAULT_ORDER_NOTIONAL,
        help="Max order notional in USD, capped at paper cash (default: 10000)",
    )
    args = parser.parse_args()

    report = execute_latest_signal(args.symbol.upper(), notional=args.notional)
    print(
        f"{report.symbol}: signal={report.signal} "
        f"P(up)={report.probability:.2%} action={report.action}"
    )


if __name__ == "__main__":
    main()

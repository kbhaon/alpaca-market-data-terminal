# alpaca-market-data-terminal

Mini market data terminal using Alpaca APIs for historical OHLCV charts and live bid/ask quote updates in a simple Python-based UI.

## Executive Summary

This project built a terminal that connects to Alpaca market data, retrieves historical OHLCV bars, displays a chart, and provides a simple UI for real-time quotes for US-listed stocks and ETFs.

## Demo Video

Demo video: https://youtu.be/Zx6PTew7rmc?si=mOBUHANlP98lEoAR

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate alpaca-terminal
```

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Then add your Alpaca paper-trading API key and secret to `.env`.

## Run

```bash
streamlit run app.py
```

## Repository Structure

```text
alpaca-market-data-terminal/
├── app.py                  # Streamlit app entrypoint and UI layout
├── src/
│   ├── __init__.py
│   ├── config.py           # Loads Alpaca credentials and app settings from environment variables
│   ├── data_connector.py   # Builds Alpaca market data clients and resolves data feed settings
│   ├── historical.py       # Fetches historical OHLCV bar data
│   ├── live_quotes.py      # Streams latest quotes/trades with Alpaca websocket
│   ├── company.py          # Resolves ticker symbols to company names
│   └── company_search.py   # Provides Stocks/ETFs dropdown and fuzzy company/ticker search logic
├── screenshots/
│    ├── UI_1.png           # Screenshot of the historical data chart and live quote panel
│    └── UI_2.png           # Screenshot of the historical data table
├── .env.example            # Template for required Alpaca API credentials
├── .gitignore              # Excludes local secrets, caches, and system files
├── environment.yml         # Conda environment specification
├── requirements.txt        # Python package requirements
├── LICENSE                 # Project license
├── SKILL.md                # Project-specific workflow notes
└── README.md               # Project overview, setup, and usage instructions
```

## Behavioral Notes

The live quote panel initiates with Alpaca's latest known quote/trade snapshot, then updates from websocket quote and trade events when new market data arrives. During after-hours periods, streamed updates may be sparse, but the panel should still show the latest available snapshot.

## Security Notes

DO NOT commit `.env` or real API credentials. Commit `.env.example` only.

# alpaca-market-data-terminal

Mini market data terminal using Alpaca APIs for historical OHLCV charts and live bid/ask quote updates in a simple Python-based UI.

## Project Goal

This project is for FINM 250 Homework #. It connects to Alpaca market data, retrieves historical OHLCV bars, displays a chart, and provides a simple UI for current bid, ask, and last trade data.

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate finm250-alpaca-terminal
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
app.py                  Streamlit UI entrypoint
src/config.py           Environment variable loading
src/data_connector.py   Alpaca client construction
src/historical.py       Historical OHLCV retrieval
src/live_quotes.py      Latest quote/trade helpers and stream starter
screenshots/            UI screenshots for submission
```

## Security Notes

Do not commit `.env` or real API credentials. Commit `.env.example` only.

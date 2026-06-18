from functools import lru_cache

from alpaca.trading.client import TradingClient

from .config import get_settings


@lru_cache(maxsize=128)
def get_company_name(symbol: str) -> str:
    """Return a human-readable company name for a symbol, with fallback to symbol."""

    if not symbol:
        return ""

    symbol = symbol.strip().upper()

    settings = get_settings()

    try:
        trading_client = TradingClient(
            settings.api_key,
            settings.secret_key,
            paper=True,
        )
        asset = trading_client.get_asset(symbol)
        name = getattr(asset, "name", "")
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:
        pass

    return symbol

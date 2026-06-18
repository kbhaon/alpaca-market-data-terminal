from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover
    from difflib import SequenceMatcher

    def _ratio(a: str, b: str) -> float:
        return SequenceMatcher(None, a.lower(), b.lower()).ratio() * 100

    class _FallbackFuzz:
        @staticmethod
        def WRatio(a: str, b: str) -> float:
            return _ratio(a, b)

    fuzz = _FallbackFuzz()

from alpaca.trading.client import TradingClient

from src.config import get_settings


@dataclass(frozen=True)
class CompanyMatch:
    symbol: str
    name: str
    score: float
    exchange: str | None = None

    @property
    def display(self) -> str:
        exchange = f" · {self.exchange}" if self.exchange else ""
        return f"{self.symbol} — {self.name}{exchange}"


def _score_symbol(query_upper: str, symbol: str) -> float:
    """Score ticker matches conservatively so short tickers do not dominate name search."""

    if query_upper == symbol:
        return 120.0

    if len(query_upper) < 2:
        return 0.0

    if symbol.startswith(query_upper):
        return 98.0

    # Support common brand-name inputs such as "Google" -> GOOG/GOOGL.
    if len(symbol) >= 3 and query_upper.startswith(symbol):
        return 92.0

    return 0.0


GENERIC_NAME_TOKENS = {
    "american",
    "class",
    "common",
    "company",
    "corp",
    "corporation",
    "depositary",
    "each",
    "inc",
    "incorporated",
    "limited",
    "ltd",
    "ordinary",
    "representing",
    "share",
    "shares",
    "stock",
    "the",
}


def _score_name(query_lower: str, query_words: list[str], name: str) -> float:
    """Score company-name matches with fuzzy typo tolerance."""

    name_lower = name.lower()
    name_tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", name_lower)
        if len(token) >= 3 and token not in GENERIC_NAME_TOKENS
    ]
    meaningful_name = " ".join(name_tokens)
    token_score = max((fuzz.ratio(query_lower, token) for token in name_tokens), default=0.0)
    phrase_score = fuzz.token_set_ratio(query_lower, meaningful_name) if meaningful_name else 0.0
    name_score = max(phrase_score, token_score)

    if query_lower in name_lower:
        name_score = max(name_score, 100.0)

    if query_words:
        token_hits = sum(word in name_lower for word in query_words)
        if token_hits == len(query_words):
            name_score = max(name_score, 95.0)
        elif token_hits > 0:
            name_score = max(name_score, 70.0)

    return float(name_score)


def _adjust_security_type_score(score: float, name: str) -> float:
    """Prefer operating-company common stocks over derivative ETFs for company-name searches."""

    name_lower = name.lower()
    is_common_equity = any(
        term in name_lower
        for term in (
            "capital stock",
            "common stock",
            "ordinary share",
            "ordinary shares",
        )
    )
    if is_common_equity:
        score += 5.0

    derivative_terms = (
        "etf",
        "fund",
        "trust",
        "2x",
        "3x",
        "inverse",
        "bear",
        "bull",
        "daily",
        "option income",
        "yield",
    )
    if not is_common_equity and any(term in name_lower for term in derivative_terms):
        score -= 8.0

    return max(score, 0.0)


@lru_cache(maxsize=1)
def _get_all_assets() -> list[dict[str, str]]:
    settings = get_settings()
    trading_client = TradingClient(settings.api_key, settings.secret_key, paper=True)

    methods_to_try = [
        ("get_all_assets", {}),
        ("get_all_assets", {"status": "active"}),
        ("list_assets", {}),
        ("list_assets", {"status": "active"}),
        ("list_assets", {"status": "active", "asset_class": "us_equity"}),
        ("get_assets", {}),
        ("get_assets", {"status": "active"}),
        ("get_assets", {"asset_class": "us_equity"}),
    ]

    assets = None
    last_error: str | None = None
    attempt_log = []
    for method_name, kwargs in methods_to_try:
        fn = getattr(trading_client, method_name, None)
        if fn is None:
            attempt_log.append(f"{method_name}(missing)")
            continue
        try:
            attempt_log.append(f"{method_name}({kwargs})")
            assets = fn(**kwargs)
            break
        except TypeError:
            # Some alpaca-py versions may require no arguments for this method
            try:
                assets = fn()
                break
            except Exception as exc:
                last_error = f"{method_name}() -> {exc}"
                attempt_log.append(f"{last_error}")
                pass
        except Exception as exc:
            last_error = f"{method_name}({kwargs}) -> {exc}"
            attempt_log.append(last_error)
            pass

    if assets is None:
        raise RuntimeError(
            f"Unable to fetch assets from TradingClient. Last error: {last_error}. Attempts: {attempt_log}"
        )

    normalized = []
    for asset in assets:
        symbol = str(getattr(asset, "symbol", "") or "").strip().upper()
        name = str(getattr(asset, "name", "") or "").strip()
        raw_exchange = getattr(asset, "exchange", None)
        exchange = str(getattr(raw_exchange, "value", raw_exchange) or "")
        raw_status = getattr(asset, "status", "")
        status = str(getattr(raw_status, "value", raw_status) or "").lower()
        if not symbol or not name:
            continue
        if status and status != "active":
            continue
        normalized.append(
            {
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
            }
        )

    return normalized


def search_companies(query: str, limit: int = 10, min_score: float = 45.0) -> list[CompanyMatch]:
    """Return ranked fuzzy matches for company name or ticker query."""

    return _search_companies(query, limit=limit, min_score=min_score)


def get_company_choices() -> list[CompanyMatch]:
    """Return active equities for the searchable equity dropdown."""

    try:
        assets = _get_all_assets()
    except Exception:
        return []

    choices = [
        CompanyMatch(
            symbol=row["symbol"],
            name=row["name"],
            score=0.0,
            exchange=row.get("exchange"),
        )
        for row in assets
    ]
    return sorted(choices, key=lambda match: (match.symbol, match.name))


def _search_companies(
    query: str,
    limit: int = 10,
    min_score: float = 45.0,
) -> list[CompanyMatch]:
    query = (query or "").strip()
    if len(query) < 2:
        return []

    query_lower = query.lower()
    query_upper = query.upper()
    query_words = [token for token in query_lower.split() if token]
    matches: list[CompanyMatch] = []

    try:
        assets = _get_all_assets()
    except Exception:
        return []

    for row in assets:
        symbol = row["symbol"]
        name = row["name"]

        symbol_score = _score_symbol(query_upper, symbol)
        name_score = _score_name(query_lower, query_words, name)
        score = _adjust_security_type_score(max(symbol_score, name_score), name)

        if score >= min_score:
            matches.append(
                CompanyMatch(
                    symbol=symbol,
                    name=name,
                    score=float(score),
                    exchange=row.get("exchange"),
                )
            )

    matches.sort(key=lambda m: (m.score, m.symbol), reverse=True)
    dedup = []
    seen = set()
    for m in matches:
        if m.symbol in seen:
            continue
        seen.add(m.symbol)
        dedup.append(m)
        if len(dedup) >= limit:
            break

    return dedup

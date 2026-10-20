"""Market data from Yahoo Finance: ticker search, price history and company info."""

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

# Exchange suffix of a Yahoo ticker -> (benchmark index ticker, display name).
# Tickers without a suffix (US listings, crypto) fall back to the S&P 500.
BENCHMARKS: dict[str, tuple[str, str]] = {
    "PA": ("^FCHI", "CAC 40"),
    "DE": ("^GDAXI", "DAX"),
    "L": ("^FTSE", "FTSE 100"),
    "AS": ("^AEX", "AEX"),
    "MI": ("FTSEMIB.MI", "FTSE MIB"),
    "MC": ("^IBEX", "IBEX 35"),
    "SW": ("^SSMI", "SMI"),
    "BR": ("^BFX", "BEL 20"),
    "T": ("^N225", "Nikkei 225"),
    "HK": ("^HSI", "Hang Seng"),
    "TO": ("^GSPTSE", "S&P/TSX"),
    "AX": ("^AXJO", "S&P/ASX 200"),
}
DEFAULT_BENCHMARK = ("^GSPC", "S&P 500")

SUPPORTED_QUOTE_TYPES = {"EQUITY", "ETF", "CRYPTOCURRENCY"}


class TickerNotFoundError(ValueError):
    """Raised when Yahoo Finance returns no data for a ticker."""


@dataclass(frozen=True)
class SearchResult:
    symbol: str
    name: str
    exchange: str

    @property
    def label(self) -> str:
        return f"{self.name} ({self.symbol}) · {self.exchange}"


@dataclass(frozen=True)
class CompanyInfo:
    name: str
    currency: str
    sector: str | None
    country: str | None
    exchange: str | None


def parse_search_quotes(quotes: Iterable[dict]) -> list[SearchResult]:
    """Keep the tradable instruments we support from raw Yahoo search quotes."""
    results = []
    for quote in quotes:
        symbol = quote.get("symbol")
        if not symbol or quote.get("quoteType") not in SUPPORTED_QUOTE_TYPES:
            continue
        name = quote.get("longname") or quote.get("shortname") or symbol
        exchange = quote.get("exchDisp") or quote.get("exchange") or ""
        results.append(SearchResult(symbol=symbol, name=name, exchange=exchange))
    return results


def search_tickers(query: str, max_results: int = 8) -> list[SearchResult]:
    """Search Yahoo Finance by company name or ticker."""
    query = query.strip()
    if not query:
        return []
    search = yf.Search(query, max_results=max_results, news_count=0)
    return parse_search_quotes(search.quotes)


def benchmark_for(symbol: str) -> tuple[str, str]:
    """Pick the benchmark index matching the ticker's home market."""
    _, dot, suffix = symbol.rpartition(".")
    if not dot:
        return DEFAULT_BENCHMARK
    return BENCHMARKS.get(suffix.upper(), DEFAULT_BENCHMARK)


def fetch_prices(symbol: str, period: str = "1y") -> pd.Series:
    """Daily adjusted close prices for the given period (e.g. "1y", "5y")."""
    history = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if history.empty or "Close" not in history:
        raise TickerNotFoundError(f"No price data for {symbol!r}")
    prices = history["Close"].dropna()
    if prices.empty:
        raise TickerNotFoundError(f"No price data for {symbol!r}")
    prices.name = symbol
    return prices


def fetch_company_info(symbol: str) -> CompanyInfo:
    """Basic descriptive info; falls back to the symbol when Yahoo has no name."""
    info = yf.Ticker(symbol).info or {}
    return CompanyInfo(
        name=info.get("longName") or info.get("shortName") or symbol,
        currency=info.get("currency") or "",
        sector=info.get("sector"),
        country=info.get("country"),
        exchange=info.get("fullExchangeName") or info.get("exchange"),
    )

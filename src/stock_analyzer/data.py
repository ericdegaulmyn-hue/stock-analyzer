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

# 13-week US Treasury bill yield, quoted by Yahoo in percent (4.08 = 4.08%).
RISK_FREE_SYMBOL = "^IRX"


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
    """Daily adjusted close prices for the given period (e.g. "1y", "5y").

    The index is plain calendar dates (no timezone) so series from different
    exchanges line up on the same day.
    """
    history = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if history.empty or "Close" not in history:
        raise TickerNotFoundError(f"No price data for {symbol!r}")
    prices = history["Close"].dropna()
    if prices.empty:
        raise TickerNotFoundError(f"No price data for {symbol!r}")
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index = prices.index.normalize()
    prices.name = symbol
    return prices


@dataclass(frozen=True)
class Fundamentals:
    summary: str
    ratios: dict[str, float | None]
    # fiscal year (e.g. "2025") -> line item (e.g. "Total Revenue") -> value
    income_statement: dict[str, dict[str, float]]


@dataclass(frozen=True)
class AnalystView:
    recommendation: str | None
    analyst_count: int | None
    current_price: float | None
    target_mean: float | None
    target_low: float | None
    target_high: float | None
    # rating ("strongBuy", "buy", ...) -> number of analysts, latest month
    rating_counts: dict[str, int]


# Yahoo `info` keys used by the fundamental analyst. Margins, growth and ROE are decimals.
RATIO_KEYS = (
    "marketCap",
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "profitMargins",
    "operatingMargins",
    "revenueGrowth",
    "earningsGrowth",
    "returnOnEquity",
    "debtToEquity",
    "freeCashflow",
)
INCOME_ROWS = ("Total Revenue", "Gross Profit", "Operating Income", "Net Income", "Diluted EPS")
RATING_COLUMNS = ("strongBuy", "buy", "hold", "sell", "strongSell")


def fetch_fundamentals(symbol: str, years: int = 4) -> Fundamentals:
    """Key ratios and the last annual income statements (empty for ETFs and crypto)."""
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    statement = ticker.income_stmt
    income: dict[str, dict[str, float]] = {}
    if statement is not None and not statement.empty:
        rows = [row for row in INCOME_ROWS if row in statement.index]
        for column in statement.columns[:years]:
            values = statement.loc[rows, column].dropna()
            if not values.empty:
                income[str(column.year)] = {row: float(v) for row, v in values.items()}
    return Fundamentals(
        summary=info.get("longBusinessSummary") or "",
        ratios={key: info.get(key) for key in RATIO_KEYS},
        income_statement=income,
    )


def fetch_analyst_view(symbol: str) -> AnalystView:
    """Wall Street consensus: rating, price targets and rating distribution."""
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    counts: dict[str, int] = {}
    recommendations = ticker.recommendations
    if recommendations is not None and not recommendations.empty:
        latest = recommendations.iloc[0]
        counts = {col: int(latest[col]) for col in RATING_COLUMNS if col in latest}
    return AnalystView(
        recommendation=info.get("recommendationKey"),
        analyst_count=info.get("numberOfAnalystOpinions"),
        current_price=info.get("currentPrice"),
        target_mean=info.get("targetMeanPrice"),
        target_low=info.get("targetLowPrice"),
        target_high=info.get("targetHighPrice"),
        rating_counts=counts,
    )


def fetch_risk_free_rate(period: str = "1y") -> float:
    """Average annual T-bill yield over the period, as a decimal (0.04 = 4%)."""
    return fetch_prices(RISK_FREE_SYMBOL, period).mean() / 100


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

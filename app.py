"""Streamlit entry point: `uv run streamlit run app.py`."""

import math

import streamlit as st

from stock_analyzer.charts import price_and_drawdown_figure
from stock_analyzer.committee import StockBrief
from stock_analyzer.committee_view import render_committee
from stock_analyzer.data import (
    TickerNotFoundError,
    benchmark_for,
    fetch_analyst_view,
    fetch_company_info,
    fetch_fundamentals,
    fetch_prices,
    fetch_risk_free_rate,
    search_tickers,
)
from stock_analyzer.metrics import compute_metrics

PERIODS = {"1Y": "1y", "3Y": "3y", "5Y": "5y"}
CACHE_TTL = 60 * 60  # seconds; limits calls to Yahoo Finance


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_search(query: str):
    return search_tickers(query)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_prices(symbol: str, period: str):
    return fetch_prices(symbol, period)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_info(symbol: str):
    return fetch_company_info(symbol)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_fundamentals(symbol: str):
    return fetch_fundamentals(symbol)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_analyst_view(symbol: str):
    return fetch_analyst_view(symbol)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def cached_risk_free_rate(period: str):
    return fetch_risk_free_rate(period)


def fmt_pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:+.2%}"


def fmt_num(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.2f}"


st.set_page_config(page_title="Stock Analyzer", page_icon="📈", layout="wide")
st.title("Stock Analyzer")
st.caption("Risk analytics for any listed stock, ETF or crypto. Not investment advice.")

query = st.text_input("Search a stock", placeholder="e.g. Apple, LVMH, MC.PA, BTC-USD")
if not query:
    st.stop()

with st.spinner("Searching..."):
    results = cached_search(query)
if not results:
    st.warning(f"No stock found for “{query}”. Try the company name or its ticker.")
    st.stop()

selected = st.selectbox("Select a result", results, format_func=lambda r: r.label)
period_label = st.radio("Period", list(PERIODS), horizontal=True)

period = PERIODS[period_label]
benchmark_symbol, benchmark_name = benchmark_for(selected.symbol)
try:
    with st.spinner("Downloading market data..."):
        prices = cached_prices(selected.symbol, period)
        benchmark_prices = cached_prices(benchmark_symbol, period)
        info = cached_info(selected.symbol)
except TickerNotFoundError:
    st.error(f"Yahoo Finance has no price data for {selected.symbol}.")
    st.stop()

try:
    risk_free_rate = cached_risk_free_rate(period)
    rate_note = f"Risk-free rate: {risk_free_rate:.2%} (avg. 13-week US T-bill)"
except TickerNotFoundError:
    risk_free_rate = 0.0
    rate_note = "Risk-free rate unavailable, 0% used"

try:
    metrics = compute_metrics(prices, benchmark_prices, risk_free_rate)
except ValueError:
    st.error("Not enough price history to compute metrics for this period.")
    st.stop()

st.subheader(info.name)
details = [
    selected.symbol,
    info.exchange,
    info.sector,
    info.country,
    f"Benchmark: {benchmark_name}",
]
st.caption(" · ".join(d for d in details if d))

last, first = prices.iloc[-1], prices.iloc[0]
st.metric(
    "Last close",
    f"{last:,.2f} {info.currency}",
    f"{(last / first - 1):+.2%} over {period_label}",
)

rows = [
    [
        ("Annualized return", fmt_pct(metrics.annualized_return)),
        ("Volatility (ann.)", fmt_pct(metrics.volatility).lstrip("+")),
        ("Max drawdown", fmt_pct(metrics.max_drawdown)),
    ],
    [
        ("Sharpe ratio", fmt_num(metrics.sharpe)),
        ("Sortino ratio", fmt_num(metrics.sortino)),
        ("VaR 95% (1 day)", fmt_pct(-metrics.var_95)),
    ],
    [
        (f"Beta vs {benchmark_name}", fmt_num(metrics.beta)),
        ("Alpha (ann.)", fmt_pct(metrics.alpha)),
        ("Correlation", fmt_num(metrics.correlation)),
    ],
]
for row in rows:
    for column, (label, value) in zip(st.columns(len(row)), row, strict=True):
        column.metric(label, value)
st.caption(f"{rate_note}. Computed on daily closes over {period_label}.")

st.plotly_chart(
    price_and_drawdown_figure(prices, benchmark_prices, info.name, benchmark_name),
    use_container_width=True,
)

brief = StockBrief(
    symbol=selected.symbol,
    name=info.name,
    sector=info.sector,
    currency=info.currency,
    benchmark_name=benchmark_name,
    period_label=period_label,
    risk_free_rate=risk_free_rate,
    metrics=metrics,
    fundamentals=cached_fundamentals(selected.symbol),
    analyst_view=cached_analyst_view(selected.symbol),
)
render_committee(brief, run_key=f"{selected.symbol}:{period}")

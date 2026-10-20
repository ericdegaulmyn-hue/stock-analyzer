"""Streamlit entry point: `uv run streamlit run app.py`."""

import streamlit as st

from stock_analyzer.data import (
    TickerNotFoundError,
    benchmark_for,
    fetch_company_info,
    fetch_prices,
    search_tickers,
)

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

try:
    with st.spinner("Downloading market data..."):
        prices = cached_prices(selected.symbol, PERIODS[period_label])
        info = cached_info(selected.symbol)
except TickerNotFoundError:
    st.error(f"Yahoo Finance has no price data for {selected.symbol}.")
    st.stop()

_, benchmark_name = benchmark_for(selected.symbol)
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
st.line_chart(prices)

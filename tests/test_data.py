import pandas as pd
import pytest

from stock_analyzer import data
from stock_analyzer.data import (
    TickerNotFoundError,
    benchmark_for,
    fetch_prices,
    parse_search_quotes,
)


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [
        ("AAPL", "^GSPC"),
        ("MC.PA", "^FCHI"),
        ("mc.pa", "^FCHI"),
        ("SAP.DE", "^GDAXI"),
        ("7203.T", "^N225"),
        ("BTC-USD", "^GSPC"),
        ("XYZ.UNKNOWN", "^GSPC"),
    ],
)
def test_benchmark_for(symbol, expected):
    assert benchmark_for(symbol)[0] == expected


def test_parse_search_quotes_keeps_supported_types_only():
    quotes = [
        {"symbol": "MC.PA", "longname": "LVMH", "exchDisp": "Paris", "quoteType": "EQUITY"},
        {"symbol": "^FCHI", "shortname": "CAC 40", "quoteType": "INDEX"},
        {"symbol": "SPY", "shortname": "SPDR S&P 500", "exchange": "PCX", "quoteType": "ETF"},
        {"longname": "No symbol", "quoteType": "EQUITY"},
    ]

    results = parse_search_quotes(quotes)

    assert [r.symbol for r in results] == ["MC.PA", "SPY"]
    assert results[0].label == "LVMH (MC.PA) · Paris"
    assert results[1].name == "SPDR S&P 500"
    assert results[1].exchange == "PCX"


class FakeTicker:
    def __init__(self, history):
        self._history = history

    def history(self, **_kwargs):
        history = self._history
        if not history.empty and not isinstance(history.index, pd.DatetimeIndex):
            # Yahoo always returns dated rows; give plain test frames a date index
            history = history.set_axis(pd.bdate_range("2026-01-01", periods=len(history)))
        return history


def test_fetch_prices_drops_missing_values(monkeypatch):
    history = pd.DataFrame({"Close": [100.0, 101.0, None]})
    monkeypatch.setattr(data.yf, "Ticker", lambda _symbol: FakeTicker(history))

    prices = fetch_prices("AAPL")

    assert prices.tolist() == [100.0, 101.0]
    assert prices.name == "AAPL"


def test_fetch_prices_strips_timezone_to_calendar_dates(monkeypatch):
    index = pd.DatetimeIndex(["2026-10-19 00:00", "2026-10-20 00:00"], tz="Europe/Paris")
    history = pd.DataFrame({"Close": [100.0, 101.0]}, index=index)
    monkeypatch.setattr(data.yf, "Ticker", lambda _symbol: FakeTicker(history))

    prices = fetch_prices("MC.PA")

    assert prices.index.tz is None
    assert prices.index.tolist() == [pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-20")]


def test_fetch_risk_free_rate_converts_percent_to_decimal(monkeypatch):
    history = pd.DataFrame({"Close": [4.0, 4.2]})
    monkeypatch.setattr(data.yf, "Ticker", lambda _symbol: FakeTicker(history))

    assert data.fetch_risk_free_rate() == pytest.approx(0.041)


class FakeInfoTicker:
    def __init__(self, info, income_stmt=None, recommendations=None):
        self.info = info
        self.income_stmt = income_stmt if income_stmt is not None else pd.DataFrame()
        self.recommendations = recommendations


def test_fetch_fundamentals_keeps_recent_years_and_known_rows(monkeypatch):
    statement = pd.DataFrame(
        {
            pd.Timestamp("2025-09-30"): [400.0, 100.0, 1.0],
            pd.Timestamp("2024-09-30"): [380.0, None, 2.0],
        },
        index=["Total Revenue", "Net Income", "Some Other Row"],
    )
    info = {"longBusinessSummary": "Phones.", "marketCap": 5e12}
    monkeypatch.setattr(data.yf, "Ticker", lambda _s: FakeInfoTicker(info, statement))

    result = data.fetch_fundamentals("AAPL")

    assert result.summary == "Phones."
    assert result.ratios["marketCap"] == 5e12
    assert result.ratios["trailingPE"] is None
    assert result.income_statement == {
        "2025": {"Total Revenue": 400.0, "Net Income": 100.0},
        "2024": {"Total Revenue": 380.0},
    }


def test_fetch_analyst_view_reads_latest_month(monkeypatch):
    recommendations = pd.DataFrame(
        {"period": ["0m", "-1m"], "strongBuy": [6, 5], "buy": [18, 19], "hold": [14, 13]}
    )
    info = {"recommendationKey": "buy", "numberOfAnalystOpinions": 39, "targetMeanPrice": 328.4}
    monkeypatch.setattr(
        data.yf, "Ticker", lambda _s: FakeInfoTicker(info, recommendations=recommendations)
    )

    view = data.fetch_analyst_view("AAPL")

    assert view.recommendation == "buy"
    assert view.analyst_count == 39
    assert view.target_mean == 328.4
    assert view.rating_counts == {"strongBuy": 6, "buy": 18, "hold": 14}


@pytest.mark.parametrize(
    "history",
    [pd.DataFrame(), pd.DataFrame({"Close": [None, None]})],
)
def test_fetch_prices_raises_when_no_data(monkeypatch, history):
    monkeypatch.setattr(data.yf, "Ticker", lambda _symbol: FakeTicker(history))

    with pytest.raises(TickerNotFoundError):
        fetch_prices("NOPE")

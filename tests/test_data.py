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
        return self._history


def test_fetch_prices_drops_missing_values(monkeypatch):
    history = pd.DataFrame({"Close": [100.0, 101.0, None]})
    monkeypatch.setattr(data.yf, "Ticker", lambda _symbol: FakeTicker(history))

    prices = fetch_prices("AAPL")

    assert prices.tolist() == [100.0, 101.0]
    assert prices.name == "AAPL"


@pytest.mark.parametrize(
    "history",
    [pd.DataFrame(), pd.DataFrame({"Close": [None, None]})],
)
def test_fetch_prices_raises_when_no_data(monkeypatch, history):
    monkeypatch.setattr(data.yf, "Ticker", lambda _symbol: FakeTicker(history))

    with pytest.raises(TickerNotFoundError):
        fetch_prices("NOPE")

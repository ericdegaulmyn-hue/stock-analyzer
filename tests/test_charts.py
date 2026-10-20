import pandas as pd
import pytest

from stock_analyzer.charts import price_and_drawdown_figure, rebase


def test_rebase_starts_at_100():
    prices = pd.Series([50.0, 55.0, 45.0])
    assert rebase(prices).tolist() == pytest.approx([100.0, 110.0, 90.0])


def test_figure_has_stock_benchmark_and_drawdown():
    dates = pd.bdate_range("2026-01-01", periods=3)
    stock = pd.Series([10.0, 11.0, 9.0], index=dates)
    bench = pd.Series([100.0, 101.0, 102.0], index=dates)

    fig = price_and_drawdown_figure(stock, bench, "Stock", "Index")

    assert [trace.name for trace in fig.data] == ["Stock", "Index", "Drawdown"]

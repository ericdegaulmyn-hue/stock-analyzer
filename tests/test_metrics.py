"""Expected values are computed by hand in the comments, not with the code under test."""

import math

import pandas as pd
import pytest

from stock_analyzer import metrics as m


def series(values, start="2025-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)), dtype=float)


def test_daily_returns():
    # 100 -> 110 is +10%, 110 -> 99 is -10%
    assert m.daily_returns(series([100, 110, 99])).tolist() == pytest.approx([0.10, -0.10])


def test_daily_rate_compounds_back_to_annual():
    assert (1 + m.daily_rate(0.05)) ** 252 == pytest.approx(1.05)


def test_annualized_return_of_a_doubling_year():
    # 252 equal daily returns that double the price -> +100% per year
    r = 2 ** (1 / 252) - 1
    assert m.annualized_return(series([r] * 252)) == pytest.approx(1.0)


def test_annualized_volatility():
    # returns [+1%, -1%]: sample std = sqrt(2 * 0.01^2 / 1) = 0.0141421
    expected = 0.0141421356 * math.sqrt(252)
    assert m.annualized_volatility(series([0.01, -0.01])) == pytest.approx(expected)


def test_sharpe_ratio_with_zero_rate():
    # returns [2%, 0%, 1%]: mean 1%, sample std 1% -> 1 * sqrt(252)
    assert m.sharpe_ratio(series([0.02, 0.0, 0.01])) == pytest.approx(math.sqrt(252))


def test_sortino_ratio_with_zero_rate():
    # returns [2%, -1%, 2%, -1%]: mean 0.5%
    # downside deviation = sqrt((0 + 0.0001 + 0 + 0.0001) / 4) = 0.00707107
    expected = 0.005 / 0.0070710678 * math.sqrt(252)
    assert m.sortino_ratio(series([0.02, -0.01, 0.02, -0.01])) == pytest.approx(expected)


def test_ratios_are_nan_without_risk():
    flat = series([0.01, 0.01, 0.01])
    assert math.isnan(m.sharpe_ratio(flat))
    assert math.isnan(m.sortino_ratio(flat))


def test_max_drawdown():
    # peak 120, trough 90 -> 90 / 120 - 1 = -25%; the later 100 is only -16.7% below 120
    prices = series([100, 120, 90, 110, 100, 130])
    assert m.max_drawdown(prices) == pytest.approx(-0.25)


def test_value_at_risk_95():
    # 101 returns from -50% to +50% in 1% steps: the 5th percentile is -45%
    returns = series([i / 100 for i in range(-50, 51)])
    assert m.value_at_risk(returns) == pytest.approx(0.45)


def test_beta_alpha_correlation_for_a_leveraged_stock():
    # stock always moves exactly twice the benchmark -> beta 2, correlation 1
    bench = series([0.01, -0.02, 0.03, 0.005])
    stock = bench * 2
    assert m.beta(stock, bench) == pytest.approx(2.0)
    assert stock.corr(bench) == pytest.approx(1.0)
    # with a zero rate: alpha = (mean_s - 2 * mean_b) * 252 = 0
    assert m.jensen_alpha(stock, bench) == pytest.approx(0.0)


def test_alpha_of_a_constant_outperformance():
    # stock = benchmark + 0.1% per day -> beta 1, alpha = 0.001 * 252 = 25.2%
    bench = series([0.01, -0.02, 0.03, 0.005])
    assert m.jensen_alpha(bench + 0.001, bench, 0.04) == pytest.approx(0.252)


def test_compute_metrics_aligns_dates():
    stock = series([100, 102, 101, 105, 104], start="2025-01-06")
    # benchmark has one extra day before and is missing 2025-01-08
    bench = series([50, 50, 51, 50.5, 52.5, 52], start="2025-01-03").drop(
        pd.Timestamp("2025-01-08")
    )
    result = m.compute_metrics(stock, bench, risk_free_rate=0.0)
    # 4 common dates (06, 07, 09, 10): stock 100, 102, 105, 104 -> drawdown 104 / 105 - 1
    assert result.max_drawdown == pytest.approx(104 / 105 - 1)
    assert all(math.isfinite(v) for v in vars(result).values())


def test_compute_metrics_needs_overlap():
    with pytest.raises(ValueError):
        m.compute_metrics(series([1, 2]), series([1, 2]), 0.0)

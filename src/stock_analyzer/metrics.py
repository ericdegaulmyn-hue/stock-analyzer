"""Risk and performance metrics computed from daily prices.

Conventions: 252 trading days per year, simple daily returns, sample standard
deviation, historical (non-parametric) VaR. Rates are decimals (0.04 = 4%).
"""

import math
from dataclasses import dataclass

import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class RiskMetrics:
    annualized_return: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    var_95: float
    beta: float
    alpha: float
    correlation: float


def daily_returns(prices: pd.Series) -> pd.Series:
    return prices.pct_change().dropna()


def daily_rate(annual_rate: float) -> float:
    """Convert an annual rate into the equivalent compounded daily rate."""
    return (1 + annual_rate) ** (1 / TRADING_DAYS) - 1


def annualized_return(returns: pd.Series) -> float:
    """Geometric (compounded) return scaled to one year."""
    growth = (1 + returns).prod()
    return growth ** (TRADING_DAYS / len(returns)) - 1


def annualized_volatility(returns: pd.Series) -> float:
    return returns.std() * math.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - daily_rate(risk_free_rate)
    std = excess.std()
    if std == 0:
        return math.nan
    return excess.mean() / std * math.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Like Sharpe, but only penalizes returns below the risk-free rate."""
    excess = returns - daily_rate(risk_free_rate)
    downside = math.sqrt((excess.clip(upper=0) ** 2).mean())
    if downside == 0:
        return math.nan
    return excess.mean() / downside * math.sqrt(TRADING_DAYS)


def drawdown(prices: pd.Series) -> pd.Series:
    """Percentage below the running peak at each date (0 at a new high)."""
    return prices / prices.cummax() - 1


def max_drawdown(prices: pd.Series) -> float:
    """Worst peak-to-trough fall, as a negative number."""
    return drawdown(prices).min()


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """One-day historical VaR, as a positive loss (0.03 = 3%)."""
    return -returns.quantile(1 - confidence)


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    return returns.cov(benchmark_returns) / benchmark_returns.var()


def jensen_alpha(
    returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0
) -> float:
    """Annualized excess return not explained by beta exposure (CAPM)."""
    rf = daily_rate(risk_free_rate)
    b = beta(returns, benchmark_returns)
    daily_alpha = (returns.mean() - rf) - b * (benchmark_returns.mean() - rf)
    return daily_alpha * TRADING_DAYS


def compute_metrics(
    prices: pd.Series, benchmark_prices: pd.Series, risk_free_rate: float
) -> RiskMetrics:
    """All metrics, computed on the dates where both stock and benchmark traded."""
    aligned = pd.concat([prices, benchmark_prices], axis=1, join="inner").dropna()
    if len(aligned) < 3:
        raise ValueError("Not enough overlapping prices to compute metrics")
    stock, bench = aligned.iloc[:, 0], aligned.iloc[:, 1]
    stock_returns, bench_returns = daily_returns(stock), daily_returns(bench)

    return RiskMetrics(
        annualized_return=annualized_return(stock_returns),
        volatility=annualized_volatility(stock_returns),
        sharpe=sharpe_ratio(stock_returns, risk_free_rate),
        sortino=sortino_ratio(stock_returns, risk_free_rate),
        max_drawdown=max_drawdown(stock),
        var_95=value_at_risk(stock_returns),
        beta=beta(stock_returns, bench_returns),
        alpha=jensen_alpha(stock_returns, bench_returns, risk_free_rate),
        correlation=stock_returns.corr(bench_returns),
    )

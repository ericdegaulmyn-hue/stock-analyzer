"""Plotly figures for the analysis page."""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock_analyzer.metrics import drawdown


def rebase(prices: pd.Series, base: float = 100) -> pd.Series:
    """Rescale a price series so it starts at `base`, to compare assets side by side."""
    return prices / prices.iloc[0] * base


def price_and_drawdown_figure(
    prices: pd.Series, benchmark_prices: pd.Series, name: str, benchmark_name: str
) -> go.Figure:
    """Top: stock vs benchmark rebased to 100. Bottom: stock drawdown."""
    aligned = pd.concat([prices, benchmark_prices], axis=1, join="inner").dropna()
    stock, bench = aligned.iloc[:, 0], aligned.iloc[:, 1]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.06,
        subplot_titles=("Performance (rebased to 100)", "Drawdown"),
    )
    fig.add_trace(
        go.Scatter(x=stock.index, y=rebase(stock), name=name, line={"width": 2}),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bench.index,
            y=rebase(bench),
            name=benchmark_name,
            line={"width": 1.5, "dash": "dot"},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=stock.index,
            y=drawdown(stock),
            name="Drawdown",
            fill="tozeroy",
            line={"width": 1},
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.update_layout(
        height=560,
        hovermode="x unified",
        margin={"l": 10, "r": 10, "t": 80, "b": 10},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.06, "x": 0},
    )
    return fig

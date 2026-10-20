# Stock Analyzer

Search any stock, see its price chart and risk metrics (volatility, Sharpe, Sortino, alpha, beta),
then ask an **AI Investment Committee** of four Claude agents to debate it and issue a
Buy / Hold / Sell rating.

> Not investment advice. Educational project.

## Run locally

```bash
uv sync
uv run streamlit run app.py
```

## Checks

```bash
uv run ruff format --check && uv run ruff check && uv run pytest
```

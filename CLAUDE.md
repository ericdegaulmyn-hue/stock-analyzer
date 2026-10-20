# Stock Analyzer

Portfolio demo site for job interviews: enter a ticker → price chart + risk metrics → optional "AI Investment Committee" (4 LLM agents debate and output a Buy/Hold/Sell report).

## Working agreements
- The owner makes product decisions; ask before deciding anything user-facing.
- Explain technical choices in plain language; finance concepts need no explanation.
- Keep process light and token usage low.

## Stack
- Python + Streamlit, Plotly charts, yfinance data (cached), official `anthropic` SDK → `claude-haiku-5-5`, server-side web search (max 2 uses, sentiment + fundamental agents only).
- Site UI in English. Every AI output shows "Not investment advice".
- Benchmark for alpha/beta chosen automatically from the ticker's market.
- Metrics (`metrics.py`): 252 trading days, risk-free rate = average 13-week US T-bill (`^IRX`) over the period, historical 1-day VaR 95%, Jensen's alpha. Computed on dates common to stock and benchmark. Periods 1Y (default) / 3Y / 5Y.
- Hosted on Streamlit Community Cloud. API keys only in `.streamlit/secrets.toml` (gitignored) / Streamlit secrets — never committed.

## Agents
Market Sentiment, Fundamental, Quant/Risk (interprets the site's own metrics), Rapporteur (summarizes). 2-round debate then synthesis, streamed live.
- `committee.py`: agents, prompts, Claude calls (no Streamlit). `committee_view.py`: UI and session-state cache of finished debates.
- Rapporteur returns a structured `Verdict` (BUY/HOLD/SELL + confidence) via `messages.parse`.
- Tests use a fake client — never call the real API in tests.

## Quality
- One branch + PR per feature; CI (GitHub Actions) runs format, lint, tests.
- Every financial formula has a unit test with hand-computed values.
- Review with `/code-review low`.

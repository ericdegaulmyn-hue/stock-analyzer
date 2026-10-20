# Stock Analyzer

Portfolio demo site for job interviews: enter a ticker → price chart + risk metrics → optional "AI Investment Committee" (4 LLM agents debate and output a Buy/Hold/Sell report).

## Working agreements
- The owner makes product decisions; ask before deciding anything user-facing.
- Explain technical choices in plain language; finance concepts need no explanation.
- Keep process light and token usage low.

## Stack
- Python + Streamlit, Plotly charts, yfinance data (cached), LiteLLM → Claude (Haiku 5.5 default), native web search.
- Site UI in English. Every AI output shows "Not investment advice".
- Benchmark for alpha/beta chosen automatically from the ticker's market.
- Hosted on Streamlit Community Cloud. API keys only in `.streamlit/secrets.toml` (gitignored) / Streamlit secrets — never committed.

## Agents
Market Sentiment, Fundamental, Quant/Risk (interprets the site's own metrics), Rapporteur (summarizes). 2-round debate then synthesis, streamed live.

## Quality
- One branch + PR per feature; CI (GitHub Actions) runs format, lint, tests.
- Every financial formula has a unit test with hand-computed values.
- Review with `/code-review low`.

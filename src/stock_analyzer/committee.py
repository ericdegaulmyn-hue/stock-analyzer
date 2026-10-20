"""AI Investment Committee: three Claude analysts debate a stock in two rounds,
then a rapporteur writes the final report with a Buy / Hold / Sell rating.

The UI drives the order of turns (see app.py); this module builds the prompts,
streams each statement and produces the structured verdict.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import anthropic
from pydantic import BaseModel

from stock_analyzer.data import AnalystView, Fundamentals
from stock_analyzer.metrics import RiskMetrics

MODEL = "claude-haiku-5-5"
ANALYST_MAX_TOKENS = 4000
REPORT_MAX_TOKENS = 8000
# Web search is billed per search, so each analyst that has it gets at most two.
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 2}
MAX_PAUSE_RESUMES = 2

DISCLAIMER = "Educational role-play by AI agents. Not investment advice."

COMMON_RULES = """You sit on an investment committee in an educational demo app.
Stay in your role and argue from your speciality only.
Keep each statement under 180 words, in plain markdown (no headings).
Use the committee briefing; when you use web search, cite what you found in a few words.
Say clearly when data is missing instead of guessing numbers.
Do not describe your searches or your process: speak directly to the committee."""


@dataclass(frozen=True)
class Agent:
    key: str
    name: str
    avatar: str
    role: str
    uses_web: bool

    @property
    def system(self) -> str:
        return f"You are the committee's {self.name}. {self.role}\n\n{COMMON_RULES}"


SENTIMENT = Agent(
    key="sentiment",
    name="Market Sentiment Analyst",
    avatar="📰",
    role=(
        "You read the market's mood: analyst ratings and price targets, recent news, "
        "upgrades and downgrades, and how investors are positioned."
    ),
    uses_web=True,
)
FUNDAMENTAL = Agent(
    key="fundamental",
    name="Fundamental Analyst",
    avatar="📊",
    role=(
        "You judge the business from its annual reports: revenue and earnings trends, "
        "margins, balance sheet strength, cash generation and valuation multiples."
    ),
    uses_web=True,
)
QUANT = Agent(
    key="quant",
    name="Quant / Risk Analyst",
    avatar="📐",
    role=(
        "You interpret the risk metrics computed by the app: return, volatility, "
        "Sharpe and Sortino ratios, drawdown, VaR, beta, alpha and correlation."
    ),
    uses_web=False,
)
ANALYSTS = (SENTIMENT, FUNDAMENTAL, QUANT)

RAPPORTEUR = Agent(
    key="rapporteur",
    name="Rapporteur",
    avatar="📝",
    role=(
        "You listened silently to the whole debate. You write the committee's final "
        "report for the client: a fair synthesis of every view, the main points of "
        "agreement and disagreement, and one rating the committee would most likely "
        "converge on, with an honest confidence level."
    ),
    uses_web=False,
)


@dataclass(frozen=True)
class StockBrief:
    """Everything the committee knows about the stock, gathered by the app."""

    symbol: str
    name: str
    sector: str | None
    currency: str
    benchmark_name: str
    period_label: str
    risk_free_rate: float
    metrics: RiskMetrics
    fundamentals: Fundamentals
    analyst_view: AnalystView


@dataclass(frozen=True)
class Statement:
    agent: Agent
    round: int
    text: str


class Verdict(BaseModel):
    rating: Literal["BUY", "HOLD", "SELL"]
    confidence: Literal["LOW", "MEDIUM", "HIGH"]
    summary: str
    bull_points: list[str]
    bear_points: list[str]
    disagreements: list[str]


# ---------- Prompt building ----------


def fmt(value: float | int | None, kind: str = "num") -> str:
    """Human-readable numbers for prompts: pct (0.12 -> 12.0%), money (1.2e9 -> 1.20B)."""
    if value is None:
        return "n/a"
    if kind == "pct":
        return f"{value:.1%}"
    if kind == "money":
        for divisor, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
            if abs(value) >= divisor:
                return f"{value / divisor:.2f}{suffix}"
    return f"{value:,.2f}"


RATIO_FORMATS = {
    "marketCap": ("Market cap", "money"),
    "trailingPE": ("P/E (trailing)", "num"),
    "forwardPE": ("P/E (forward)", "num"),
    "priceToBook": ("Price / book", "num"),
    "profitMargins": ("Net margin", "pct"),
    "operatingMargins": ("Operating margin", "pct"),
    "revenueGrowth": ("Revenue growth (yoy)", "pct"),
    "earningsGrowth": ("Earnings growth (yoy)", "pct"),
    "returnOnEquity": ("Return on equity", "pct"),
    "debtToEquity": ("Debt / equity (%)", "num"),
    "freeCashflow": ("Free cash flow", "money"),
}


def header(brief: StockBrief) -> str:
    sector = f", sector: {brief.sector}" if brief.sector else ""
    return f"Stock under review: {brief.name} ({brief.symbol}{sector}), prices in {brief.currency}."


def sentiment_data(brief: StockBrief) -> str:
    view = brief.analyst_view
    counts = ", ".join(f"{k}: {v}" for k, v in view.rating_counts.items()) or "n/a"
    return "\n".join(
        [
            f"- Consensus rating: {view.recommendation or 'n/a'} "
            f"({view.analyst_count or 'n/a'} analysts)",
            f"- Rating distribution this month: {counts}",
            f"- Current price: {fmt(view.current_price)}",
            f"- Price target: mean {fmt(view.target_mean)}, "
            f"low {fmt(view.target_low)}, high {fmt(view.target_high)}",
            "Use web search for the latest news and analyst moves.",
        ]
    )


def fundamental_data(brief: StockBrief) -> str:
    f = brief.fundamentals
    lines = [
        f"- {label}: {fmt(f.ratios.get(key), kind)}" for key, (label, kind) in RATIO_FORMATS.items()
    ]
    for year, items in f.income_statement.items():
        values = ", ".join(
            f"{item} {fmt(v, 'num' if item == 'Diluted EPS' else 'money')}"
            for item, v in items.items()
        )
        lines.append(f"- FY{year}: {values}")
    if not f.income_statement:
        lines.append("- No annual income statement available (ETF, crypto or missing data).")
    if f.summary:
        lines.append(f"- Business: {f.summary[:600]}")
    lines.append("Use web search for the latest annual report or earnings if useful.")
    return "\n".join(lines)


def quant_data(brief: StockBrief) -> str:
    m = brief.metrics
    return "\n".join(
        [
            f"Period: {brief.period_label}, benchmark: {brief.benchmark_name}, "
            f"risk-free rate: {fmt(brief.risk_free_rate, 'pct')}.",
            f"- Annualized return: {fmt(m.annualized_return, 'pct')}",
            f"- Annualized volatility: {fmt(m.volatility, 'pct')}",
            f"- Sharpe ratio: {fmt(m.sharpe)}",
            f"- Sortino ratio: {fmt(m.sortino)}",
            f"- Max drawdown: {fmt(m.max_drawdown, 'pct')}",
            f"- 1-day historical VaR 95%: {fmt(m.var_95, 'pct')}",
            f"- Beta: {fmt(m.beta)}",
            f"- Jensen's alpha (annualized): {fmt(m.alpha, 'pct')}",
            f"- Correlation with benchmark: {fmt(m.correlation)}",
            "Conventions: daily data, 252 trading days. The annualized return is "
            "geometric (compounded); Sharpe and Sortino use the arithmetic mean of daily "
            "excess returns scaled by sqrt(252), so they will not equal "
            "(annualized return - rate) / volatility. This is expected, not an error.",
        ]
    )


AGENT_DATA = {
    SENTIMENT.key: sentiment_data,
    FUNDAMENTAL.key: fundamental_data,
    QUANT.key: quant_data,
}


def format_transcript(statements: list[Statement]) -> str:
    return "\n\n".join(f"[Round {s.round}] {s.agent.name}:\n{s.text}" for s in statements)


def briefing(agent: Agent, brief: StockBrief) -> str:
    data = AGENT_DATA[agent.key](brief)
    return f"{header(brief)}\n\nCommittee briefing (prepared by the app):\n{data}"


def opening_prompt(agent: Agent, brief: StockBrief) -> str:
    return (
        f"{briefing(agent, brief)}\n\n"
        "Round 1: give your opening view on this stock and say whether it leans "
        "Buy, Hold or Sell from your angle."
    )


def rebuttal_prompt(agent: Agent, brief: StockBrief, statements: list[Statement]) -> str:
    return (
        f"{briefing(agent, brief)}\n\n"
        f"Debate so far:\n{format_transcript(statements)}\n\n"
        "Round 2: react to your colleagues. Challenge what you disagree with, "
        "acknowledge strong points, and state your final lean (Buy, Hold or Sell)."
    )


def report_prompt(brief: StockBrief, statements: list[Statement]) -> str:
    return (
        f"{header(brief)}\n\nFull debate:\n{format_transcript(statements)}\n\n"
        "Write the committee report: a summary of 3 to 5 sentences, the main bull "
        "and bear points (2 to 4 each, one short sentence each), the key "
        "disagreements, and the final rating with your confidence."
    )


# ---------- Claude calls ----------


def stream_statement(client: anthropic.Anthropic, agent: Agent, prompt: str) -> Iterator[str]:
    """Stream one agent's statement as text chunks, for st.write_stream."""
    messages = [{"role": "user", "content": prompt}]
    tools = [WEB_SEARCH_TOOL] if agent.uses_web else []
    for _ in range(MAX_PAUSE_RESUMES + 1):
        with client.messages.stream(
            model=MODEL,
            max_tokens=ANALYST_MAX_TOKENS,
            system=agent.system,
            messages=messages,
            tools=tools,
            output_config={"effort": "low"},
        ) as stream:
            yield from stream.text_stream
            message = stream.get_final_message()
        if message.stop_reason == "refusal":
            yield "\n\n_This analyst declined to comment._"
            return
        if message.stop_reason != "pause_turn":
            return
        # The server-side search loop paused: send the turn back so it resumes.
        messages = [*messages, {"role": "assistant", "content": message.content}]


def write_report(
    client: anthropic.Anthropic, brief: StockBrief, statements: list[Statement]
) -> Verdict | None:
    """The rapporteur's structured report; None if the model declined."""
    response = client.messages.parse(
        model=MODEL,
        max_tokens=REPORT_MAX_TOKENS,
        system=RAPPORTEUR.system,
        messages=[{"role": "user", "content": report_prompt(brief, statements)}],
        output_format=Verdict,
        output_config={"effort": "medium"},
    )
    return response.parsed_output

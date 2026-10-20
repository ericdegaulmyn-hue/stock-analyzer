"""Committee tests use a fake Claude client: no network, no API cost."""

from types import SimpleNamespace

import pytest

from stock_analyzer import committee as c
from stock_analyzer.data import AnalystView, Fundamentals
from stock_analyzer.metrics import RiskMetrics


def make_brief(income=None):
    return c.StockBrief(
        symbol="AAPL",
        name="Apple Inc.",
        sector="Technology",
        currency="USD",
        benchmark_name="S&P 500",
        period_label="1Y",
        risk_free_rate=0.04,
        metrics=RiskMetrics(
            annualized_return=0.30,
            volatility=0.24,
            sharpe=1.06,
            sortino=1.52,
            max_drawdown=-0.138,
            var_95=0.02,
            beta=0.62,
            alpha=0.19,
            correlation=0.33,
        ),
        fundamentals=Fundamentals(
            summary="Makes iPhones.",
            ratios={"marketCap": 4.95e12, "profitMargins": 0.276, "trailingPE": 38.9},
            income_statement={"2025": {"Total Revenue": 4.16e11}} if income is None else income,
        ),
        analyst_view=AnalystView(
            recommendation="buy",
            analyst_count=39,
            current_price=339.5,
            target_mean=328.35,
            target_low=215.0,
            target_high=405.0,
            rating_counts={"strongBuy": 6, "buy": 18},
        ),
    )


@pytest.mark.parametrize(
    ("value", "kind", "expected"),
    [
        (None, "num", "n/a"),
        (0.276, "pct", "27.6%"),
        (4.95e12, "money", "4.95T"),
        (4.16e11, "money", "416.00B"),
        (2.5e6, "money", "2.50M"),
        (38.934, "num", "38.93"),
    ],
)
def test_fmt(value, kind, expected):
    assert c.fmt(value, kind) == expected


def test_each_analyst_only_gets_its_own_data():
    brief = make_brief()
    sentiment = c.opening_prompt(c.SENTIMENT, brief)
    fundamental = c.opening_prompt(c.FUNDAMENTAL, brief)
    quant = c.opening_prompt(c.QUANT, brief)

    assert "Consensus rating: buy (39 analysts)" in sentiment
    assert "Market cap: 4.95T" in fundamental
    assert "FY2025: Total Revenue 416.00B" in fundamental
    assert "Sharpe ratio: 1.06" in quant
    assert "arithmetic mean of daily excess returns" in quant
    assert "Sharpe" not in sentiment
    assert "Market cap" not in quant


def test_fundamental_prompt_flags_missing_statements():
    prompt = c.opening_prompt(c.FUNDAMENTAL, make_brief(income={}))
    assert "No annual income statement available" in prompt


def test_rebuttal_and_report_include_the_debate():
    statements = [c.Statement(agent=c.QUANT, round=1, text="Beta is low.")]
    rebuttal = c.rebuttal_prompt(c.SENTIMENT, make_brief(), statements)
    report = c.report_prompt(make_brief(), statements)

    assert "[Round 1] Quant / Risk Analyst:\nBeta is low." in rebuttal
    assert "Round 2" in rebuttal
    assert "Beta is low." in report


def test_only_sentiment_and_fundamental_can_search_the_web():
    assert [a.key for a in c.ANALYSTS if a.uses_web] == ["sentiment", "fundamental"]


class FakeStream:
    def __init__(self, chunks, stop_reason):
        self.text_stream = iter(chunks)
        self._message = SimpleNamespace(stop_reason=stop_reason, content=["block"])

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def get_final_message(self):
        return self._message


class FakeMessages:
    def __init__(self, turns):
        self.turns = list(turns)  # [(chunks, stop_reason), ...] one per API call
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(*self.turns.pop(0))

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(parsed_output="the verdict")


def fake_client(*turns):
    return SimpleNamespace(messages=FakeMessages(turns))


def test_stream_statement_yields_text_and_passes_web_tool():
    client = fake_client((["Hello ", "committee."], "end_turn"))

    text = "".join(c.stream_statement(client, c.SENTIMENT, "prompt"))

    assert text == "Hello committee."
    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["tools"] == [c.WEB_SEARCH_TOOL]


def test_quant_has_no_tools():
    client = fake_client((["ok"], "end_turn"))
    "".join(c.stream_statement(client, c.QUANT, "prompt"))
    assert client.messages.calls[0]["tools"] == []


def test_stream_statement_resumes_after_pause_turn():
    client = fake_client((["Searching..."], "pause_turn"), ([" Done."], "end_turn"))

    text = "".join(c.stream_statement(client, c.SENTIMENT, "prompt"))

    assert text == "Searching... Done."
    resumed = client.messages.calls[1]["messages"]
    assert [m["role"] for m in resumed] == ["user", "assistant"]


def test_stream_statement_reports_a_refusal():
    client = fake_client(([], "refusal"))
    text = "".join(c.stream_statement(client, c.SENTIMENT, "prompt"))
    assert "declined" in text


def test_write_report_returns_the_parsed_verdict():
    client = fake_client()
    assert c.write_report(client, make_brief(), []) == "the verdict"
    assert client.messages.calls[0]["output_format"] is c.Verdict

"""Streamlit section that runs the AI Investment Committee and shows its answer.

Clicking the button only sets `committee_pending` and reruns the page: on that
rerun the search and period controls render disabled (see `committee_locked`),
so nothing can interrupt the debate. When it ends, the lock is released with
another rerun. Results are kept per stock, whatever the period shown.
"""

import anthropic
import streamlit as st

from stock_analyzer.committee import (
    DISCLAIMER,
    RAPPORTEUR,
    CommitteeResult,
    Statement,
    StockBrief,
    run_round,
    write_report,
)

ROUND_TITLES = {1: "Round 1 · Opening views", 2: "Round 2 · Debate"}
RATING_COLORS = {"BUY": "green", "HOLD": "orange", "SELL": "red"}
PENDING = "committee_pending"
RESULTS = "committee_results"
ERROR = "committee_error"


def as_markdown(text: str) -> str:
    """Escape `$` so prices like $339 are not rendered as LaTeX math by Streamlit."""
    return text.replace("$", r"\$")


def committee_locked() -> bool:
    """True while the committee is deliberating; the page disables its controls."""
    return st.session_state.get(PENDING) is not None


def api_key() -> str | None:
    """The key from .streamlit/secrets.toml locally, or Streamlit Cloud secrets."""
    try:
        return st.secrets.get("ANTHROPIC_API_KEY") or None
    except FileNotFoundError:
        return None


def render_statement(statement: Statement) -> None:
    with st.chat_message(statement.agent.name, avatar=statement.agent.avatar):
        st.markdown(f"**{statement.agent.name}**")
        st.markdown(as_markdown(statement.text))


def render_result(result: CommitteeResult) -> None:
    verdict = result.verdict
    if verdict is None:
        st.warning("The rapporteur could not produce an answer this time.")
    else:
        with st.chat_message(RAPPORTEUR.name, avatar=RAPPORTEUR.avatar):
            color = RATING_COLORS[verdict.rating]
            st.markdown(f"### :{color}[{verdict.rating}] · confidence {verdict.confidence.lower()}")
            st.markdown(as_markdown(verdict.summary))
            st.caption(f"Based on {result.period_label} data. {DISCLAIMER}")
    with st.expander("See the full debate"):
        for round_no, title in ROUND_TITLES.items():
            st.markdown(f"**{title}**")
            for statement in result.statements:
                if statement.round == round_no:
                    render_statement(statement)


def run_committee(client: anthropic.Anthropic, brief: StockBrief) -> CommitteeResult:
    with st.status("The committee is in session...", expanded=True) as status:
        st.write("Round 1 · the three analysts give their opening views")
        opening = run_round(client, brief, 1, [])
        st.write("Round 2 · the analysts debate")
        debate = run_round(client, brief, 2, opening)
        st.write("Rapporteur · writing the answer")
        verdict = write_report(client, brief, opening + debate)
        status.update(label="The committee has decided", state="complete", expanded=False)
    return CommitteeResult(
        statements=opening + debate, verdict=verdict, period_label=brief.period_label
    )


def deliberate(brief: StockBrief, key: str) -> None:
    """Run the debate, store the result or the error, then release the lock."""
    try:
        result = run_committee(anthropic.Anthropic(api_key=key), brief)
        st.session_state[RESULTS][brief.symbol] = result
    except anthropic.AuthenticationError:
        st.session_state[ERROR] = "The Anthropic API key was rejected. Check the app secrets."
    except anthropic.RateLimitError:
        st.session_state[ERROR] = "Too many requests to Claude right now. Try again in a minute."
    except anthropic.APIConnectionError:
        st.session_state[ERROR] = "Could not reach the Claude API. Check the connection."
    except anthropic.APIStatusError as error:
        st.session_state[ERROR] = f"Claude API error ({error.status_code}). Try again later."
    finally:
        st.session_state[PENDING] = None
    st.rerun()


def render_committee(brief: StockBrief) -> None:
    st.divider()
    st.subheader("AI Investment Committee")
    st.caption(
        "Three Claude analysts (sentiment, fundamentals, quant) debate in two rounds, "
        f"then a rapporteur gives the committee's answer. {DISCLAIMER}"
    )

    results = st.session_state.setdefault(RESULTS, {})
    if error := st.session_state.pop(ERROR, None):
        st.error(error)
    if brief.symbol in results:
        render_result(results[brief.symbol])
        return

    key = api_key()
    if key is None:
        st.info("Add `ANTHROPIC_API_KEY` to the app secrets to enable the committee.")
        return
    if st.session_state.get(PENDING) == brief.symbol:
        deliberate(brief, key)
    elif st.button("Convene the committee", type="primary", icon="🏛️"):
        st.session_state[PENDING] = brief.symbol
        st.rerun()

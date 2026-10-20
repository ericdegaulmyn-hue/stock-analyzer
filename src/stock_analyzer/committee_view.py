"""Streamlit section that runs the AI Investment Committee and shows the debate live."""

import anthropic
import streamlit as st

from stock_analyzer.committee import (
    ANALYSTS,
    DISCLAIMER,
    RAPPORTEUR,
    Statement,
    StockBrief,
    Verdict,
    opening_prompt,
    rebuttal_prompt,
    stream_statement,
    write_report,
)

ROUND_TITLES = {1: "Round 1 · Opening views", 2: "Round 2 · Debate"}
RATING_COLORS = {"BUY": "green", "HOLD": "orange", "SELL": "red"}


def api_key() -> str | None:
    """The key from .streamlit/secrets.toml locally, or Streamlit Cloud secrets."""
    try:
        return st.secrets.get("ANTHROPIC_API_KEY") or None
    except FileNotFoundError:
        return None


def render_statement(statement: Statement) -> None:
    with st.chat_message(statement.agent.name, avatar=statement.agent.avatar):
        st.markdown(f"**{statement.agent.name}**")
        st.markdown(statement.text)


def render_verdict(verdict: Verdict) -> None:
    color = RATING_COLORS[verdict.rating]
    with st.chat_message(RAPPORTEUR.name, avatar=RAPPORTEUR.avatar):
        st.markdown(f"**{RAPPORTEUR.name} · Committee report**")
        st.markdown(f"### :{color}[{verdict.rating}] · confidence {verdict.confidence.lower()}")
        st.markdown(verdict.summary)
        bull, bear = st.columns(2)
        bull.markdown("**Bull case**\n" + "".join(f"\n- {p}" for p in verdict.bull_points))
        bear.markdown("**Bear case**\n" + "".join(f"\n- {p}" for p in verdict.bear_points))
        if verdict.disagreements:
            st.markdown(
                "**Where the committee disagreed**"
                + "".join(f"\n- {p}" for p in verdict.disagreements)
            )
        st.caption(DISCLAIMER)


def run_debate(client: anthropic.Anthropic, brief: StockBrief) -> dict:
    """Stream both rounds live, then ask the rapporteur for the report."""
    statements: list[Statement] = []
    for round_no in (1, 2):
        st.markdown(f"#### {ROUND_TITLES[round_no]}")
        for agent in ANALYSTS:
            if round_no == 1:
                prompt = opening_prompt(agent, brief)
            else:
                prompt = rebuttal_prompt(agent, brief, statements)
            with st.chat_message(agent.name, avatar=agent.avatar):
                st.markdown(f"**{agent.name}**")
                text = st.write_stream(stream_statement(client, agent, prompt))
            statements.append(Statement(agent=agent, round=round_no, text=text))

    st.markdown("#### Final report")
    with st.spinner("The rapporteur is writing the report..."):
        verdict = write_report(client, brief, statements)
    if verdict is None:
        st.warning("The rapporteur could not produce a report this time.")
    else:
        render_verdict(verdict)
    return {"statements": statements, "verdict": verdict}


def render_committee(brief: StockBrief, run_key: str) -> None:
    """Button + debate. Finished debates are kept in the session to survive reruns."""
    st.divider()
    st.subheader("AI Investment Committee")
    st.caption(
        "Three Claude analysts (sentiment, fundamentals, quant) debate in two rounds, "
        f"then a rapporteur issues a rating. {DISCLAIMER}"
    )

    results = st.session_state.setdefault("committee_runs", {})
    if run_key in results:
        previous = results[run_key]
        for round_no in (1, 2):
            st.markdown(f"#### {ROUND_TITLES[round_no]}")
            for statement in previous["statements"]:
                if statement.round == round_no:
                    render_statement(statement)
        st.markdown("#### Final report")
        if previous["verdict"] is not None:
            render_verdict(previous["verdict"])
        return

    key = api_key()
    if key is None:
        st.info("Add `ANTHROPIC_API_KEY` to `.streamlit/secrets.toml` to enable the committee.")
        return
    if not st.button("Convene the committee", type="primary", icon="🏛️"):
        return

    client = anthropic.Anthropic(api_key=key)
    try:
        results[run_key] = run_debate(client, brief)
    except anthropic.AuthenticationError:
        st.error("The Anthropic API key was rejected. Check `.streamlit/secrets.toml`.")
    except anthropic.RateLimitError:
        st.error("Too many requests to Claude right now. Try again in a minute.")
    except anthropic.APIConnectionError:
        st.error("Could not reach the Claude API. Check your internet connection.")
    except anthropic.APIStatusError as error:
        st.error(f"Claude API error ({error.status_code}). Try again later.")

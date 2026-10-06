"""Streamlit UI. UI only: all logic lives in ``onboardiq.service``.

Run with ``onboardiq ui`` or ``streamlit run src/onboardiq/app.py``.
"""
from __future__ import annotations

import streamlit as st

from onboardiq.config import Settings
from onboardiq.feedback.store import FeedbackStore
from onboardiq.render import answer_markdown, escape
from onboardiq.service import Assistant
from onboardiq.types import LEVEL_LABELS, LEVELS, ROLE_LABELS, ROLES


@st.cache_resource(show_spinner="Loading the index...")
def get_assistant() -> Assistant:
    # cached for the whole server process: the index is loaded (or built) once, not per message
    return Assistant(Settings.from_env())


@st.cache_resource
def get_feedback_store(path: str) -> FeedbackStore:
    return FeedbackStore(path)


def main() -> None:
    st.set_page_config(page_title="onboardiq", page_icon=None, layout="centered")
    st.title("onboardiq")
    st.caption("Onboarding answers grounded in your team's documents, tuned to your role and seniority.")

    assistant = get_assistant()
    feedback = get_feedback_store(str(assistant.settings.feedback_db))

    with st.sidebar:
        role = st.selectbox("Role", ROLES, format_func=ROLE_LABELS.get, index=2)
        level = st.selectbox("Level", LEVELS, format_func=LEVEL_LABELS.get, index=1)
        st.divider()
        st.write(f"Index: {len(assistant.store.chunks)} chunks, embedder `{assistant.store.embedder_name}`")
        counts = feedback.summary()
        st.write(f"Feedback so far: {counts['helpful']} helpful, {counts['not_helpful']} not helpful")
        if st.button("Clear conversation"):
            st.session_state.turns = []

    turns = st.session_state.setdefault("turns", [])

    question = st.chat_input("Ask about tools, data, processes or your first weeks...")
    if question:
        history = [(t.question, t.text) for t in turns]
        with st.spinner("Searching the onboarding docs..."):
            turns.append(assistant.ask(question, role, level, history))

    for i, answer in enumerate(turns):
        with st.chat_message("user"):
            st.markdown(escape(answer.question))
        with st.chat_message("assistant"):
            st.markdown(answer_markdown(answer))
            if answer.citations:
                with st.expander("Retrieved passages"):
                    for c in answer.citations:
                        st.markdown(f"**[{c.marker}]** {escape(c.source)} - {escape(c.heading)}")
                        st.text(c.snippet)
            with st.form(key=f"feedback-{answer.answer_id}"):
                rating = st.radio("Was this helpful?", ("helpful", "not_helpful"), index=None, horizontal=True,
                                  format_func=lambda r: "Yes" if r == "helpful" else "No", key=f"r-{i}")
                comment = st.text_input("Comment (optional)", key=f"c-{i}")
                if st.form_submit_button("Submit feedback"):
                    if rating is None:
                        st.warning("Pick Yes or No first.")
                    else:
                        feedback.submit(answer, rating, comment)
                        st.success("Thanks, feedback saved.")


main()

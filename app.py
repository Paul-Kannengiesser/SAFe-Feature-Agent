"""Simple web UI.  Start with:  .venv/bin/streamlit run app.py"""

import streamlit as st

from feature_agent import export
from feature_agent.agent import Agent
from feature_agent.retrieval import Retriever
from feature_agent.schema import FIELD_LABELS, FIELD_ORDER
from feature_agent.sources import get_source

st.set_page_config(page_title="Feature Agent", layout="wide")


@st.cache_resource
def load_retriever() -> Retriever:
    return Retriever(get_source())


if "agent" not in st.session_state or st.sidebar.button("New feature"):
    st.session_state.agent = Agent(load_retriever())
agent: Agent = st.session_state.agent
if agent.turns:
    st.sidebar.caption(f"Run log: runs/{export.run_path(agent).name}")

chat, ticket = st.columns([3, 2])

with chat:
    st.subheader("Chat")
    for role, text in agent.messages:
        st.chat_message(role).write(text)
    if prompt := st.chat_input("Your answer"):
        st.chat_message("user").write(prompt)
        with st.spinner("Thinking ..."):
            agent.handle(prompt)
            export.save(agent)
        st.rerun()

with ticket:
    st.subheader("Feature")
    if agent.epic:
        st.caption(f"Epic: {agent.epic.key} {agent.epic.name}")
    for field in FIELD_ORDER:
        st.markdown(f"**{FIELD_LABELS[field]}**")
        text = agent.draft.field_text(field)
        if text:
            st.code(text, language=None, wrap_lines=True)
            if field == "name" and not agent.name_confirmed:
                st.caption("proposed by the agent, not yet confirmed")
        else:
            st.caption("missing")
    if agent.findings:
        st.markdown("**Open quality issues**")
        for i, f in enumerate(list(agent.findings)):
            st.caption(f"{FIELD_LABELS[f.field]} ({f.dimension}): {f.message}")
            if st.button("Keep as is", key=f"keep-{i}-{f.field}-{f.dimension}-{f.text[:40]}"):
                agent.accept(f)
                export.save(agent)
                st.rerun()
    if agent.accepted:
        st.markdown("**Kept as is (bridged by you)**")
        for f in agent.accepted:
            st.caption(f"{FIELD_LABELS[f.field]} ({f.dimension}): {f.message}")

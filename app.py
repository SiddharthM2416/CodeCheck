"""
CodeCheck -- Streamlit UI.

Two views:
  - Chat: free-form questions to the agent (test-gap detection, risk
    explanation, test drafting, general "where is X handled" Q&A)
  - Scan for Gaps: one-click test-gap scan, the headline feature, with
    optional path_prefix/language scoping

Run with:
    streamlit run app.py
"""

import asyncio
import os
import streamlit as st

from retrieval import list_indexed_repos
from agent_multi import run_agent
from index_repo import index_repo as run_indexing

st.set_page_config(page_title="CodeCheck", page_icon="\U0001f50e", layout="wide")


def run_agent_sync(query: str, trace: list) -> str:
    """Streamlit callbacks are synchronous -- run_agent is async, so wrap
    it in asyncio.run() here."""
    def on_event(event: dict):
        trace.append(event)

    return asyncio.run(run_agent(query, on_event=on_event))


def render_trace(trace: list):
    """Renders the same info the CLI prints to stderr, but as a readable
    expander in the UI -- which provider handled each turn, which tools
    got called with what arguments, and a preview of what came back."""
    if not trace:
        return
    with st.expander("Show agent reasoning trace", expanded=False):
        for event in trace:
            if event["type"] == "provider":
                st.markdown(f"**Provider:** `{event['provider']}`")
            elif event["type"] == "rate_limited":
                st.warning(f"Rate limited on `{event['provider']}`, switching provider...")
            elif event["type"] == "tool_call":
                st.markdown(f"**Tool call:** `{event['name']}({event['input']})`")
            elif event["type"] == "tool_result":
                st.code(event["preview"], language="json")


def get_repo_options() -> dict:
    """collection_name -> display label, e.g. 'requests (291 chunks)'."""
    try:
        repos = list_indexed_repos()
    except Exception as e:
        st.sidebar.error(f"Could not load indexed repos: {e}")
        return {}
    if not repos:
        return {}
    options = {}
    for r in repos:
        repo_name = r["repo_path"].replace("\\", "/").rstrip("/").split("/")[-1]
        label = f"{repo_name} ({r['chunk_count']} chunks)"
        options[r["collection_name"]] = label
    return options


# ---------------------------------------------------------------------------
# Sidebar: repo selection
# ---------------------------------------------------------------------------

st.sidebar.title("CodeCheck")

repo_options = get_repo_options()

with st.sidebar.expander("\u2795 Index a new repo", expanded=not repo_options):
    new_repo_path = st.text_input(
        "Local path to repo",
        key="new_repo_path",
        placeholder=r"e.g. E:\Projects2026\CodeCheck\repos\my-repo",
    )
    st.caption("Python, Java, or JavaScript.")

    if st.button("Index this repo"):
        if not new_repo_path:
            st.warning("Enter a path first.")
        elif not os.path.isdir(new_repo_path):
            st.error(f"Not a valid directory: {new_repo_path}")
        else:
            with st.spinner("Indexing... this can take a minute or two."):
                try:
                    run_indexing(new_repo_path)
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                else:
                    st.success("Indexed successfully!")
                    st.rerun()  # refresh so the new repo shows up in the dropdown below

if not repo_options:
    st.sidebar.warning(
        "No indexed repos yet -- use \u201cIndex a new repo\u201d above to add one."
    )
    st.stop()

selected_collection = st.sidebar.selectbox(
    "Repo",
    options=list(repo_options.keys()),
    format_func=lambda cid: repo_options[cid],
)

# ---------------------------------------------------------------------------
# Main area: two views
# ---------------------------------------------------------------------------

tab_scan, tab_chat = st.tabs(["\U0001f50d Scan for Gaps", "\U0001f4ac Chat"])

with tab_scan:
    st.subheader("Find untested, risky code")
    st.caption("One-click scan for test gaps, with why-it's-risky explanations.")

    col1, col2 = st.columns(2)
    with col1:
        path_prefix = st.text_input(
            "Limit to a subdirectory or file (optional)",
            placeholder="e.g. adapters.py or src/requests/",
        )
    with col2:
        language_filter = st.selectbox("Language", options=["Any", "python", "java"])

    if st.button("Scan for gaps", type="primary"):
        scope_note = f" in {path_prefix}" if path_prefix else ""
        lang_note = f" ({language_filter} only)" if language_filter != "Any" else ""
        query = (
            f"Using the repo with collection_name='{selected_collection}', "
            f"scan for untested functions/classes/methods{scope_note}{lang_note}. "
            f"For each one, cite file:line and explain specifically why it's "
            f"risky to leave untested, grounded in what that exact code does."
        )
        if language_filter != "Any":
            query += f" Only include {language_filter} code."

        trace = []
        with st.spinner("Scanning..."):
            try:
                answer = run_agent_sync(query, trace)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                answer = None

        if answer:
            st.markdown(answer)
            render_trace(trace)

with tab_chat:
    st.subheader("Ask anything about this codebase")
    st.caption("Test gaps, risk explanations, test drafting, or general Q&A.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            st.markdown(entry["content"])
            if entry["role"] == "assistant" and entry.get("trace"):
                render_trace(entry["trace"])

    user_input = st.chat_input("e.g. Draft a pytest stub for merge_setting")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # each turn is answered independently (no shared memory across
        # turns beyond what's visible on screen) -- a deliberate scope cut,
        # see PROGRESS.md. Inject the selected repo explicitly so the agent
        # doesn't have to guess it from the query wording.
        scoped_query = (
            f"(Using the repo with collection_name='{selected_collection}') "
            f"{user_input}"
        )

        trace = []
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = run_agent_sync(scoped_query, trace)
                except Exception as e:
                    answer = f"Something went wrong: {e}"
            st.markdown(answer)
            render_trace(trace)

        st.session_state.chat_history.append({
            "role": "assistant", "content": answer, "trace": trace,
        })
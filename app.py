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
from walker import get_latest_source_mtime

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
            elif event["type"] == "tool_generation_failed":
                st.warning(f"Tool-call generation failed on `{event['provider']}`, switching provider...")
            elif event["type"] == "tool_call":
                st.markdown(f"**Tool call:** `{event['name']}({event['input']})`")
            elif event["type"] == "tool_result":
                st.code(event["preview"], language="json")


def load_repos() -> list[dict]:
    try:
        return list_indexed_repos()
    except Exception as e:
        st.sidebar.error(f"Could not load indexed repos: {e}")
        return []


# ---------------------------------------------------------------------------
# Sidebar: repo selection
# ---------------------------------------------------------------------------

st.sidebar.title("CodeCheck")

repos = load_repos()
repo_by_collection = {r["collection_name"]: r for r in repos}
repo_options = {
    r["collection_name"]: f"{r['repo_path'].replace(chr(92), '/').rstrip('/').split('/')[-1]} ({r['chunk_count']} chunks)"
    for r in repos
}

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

# --- Staleness check: does the repo have file changes since it was last
# indexed? Code is NOT automatically re-indexed on change (that would mean
# re-embedding on every single query -- slow and wasteful) -- instead we
# detect staleness and offer a one-click re-index button here.
selected_repo = repo_by_collection[selected_collection]
repo_path = selected_repo["repo_path"]
indexed_at = selected_repo.get("indexed_at")

is_stale = False
if indexed_at is not None and os.path.isdir(repo_path):
    try:
        latest_mtime = get_latest_source_mtime(repo_path)
        if latest_mtime is not None and latest_mtime > indexed_at:
            is_stale = True
    except Exception:
        pass  # staleness check is best-effort; don't block the UI on it

if is_stale:
    st.sidebar.warning("This repo has changed since it was last indexed.")
elif indexed_at is None:
    st.sidebar.caption("(Indexed before staleness tracking was added -- can't check for changes.)")

if st.sidebar.button("\U0001f504 Re-index this repo", type="primary" if is_stale else "secondary"):
    with st.sidebar:
        with st.spinner("Re-indexing..."):
            try:
                run_indexing(repo_path)
            except Exception as e:
                st.error(f"Re-indexing failed: {e}")
            else:
                st.success("Re-indexed!")
                st.rerun()

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
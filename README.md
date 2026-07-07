# CodeCheck

**An agent that finds untested, risky code in a real codebase — and explains why it matters.**

CodeCheck scans a Python, Java, or JavaScript repository, identifies functions and classes with no corresponding tests, explains *why* leaving each one untested is risky (grounded in what that specific code actually does, not a generic template), and can draft a test stub for it. It can also answer general questions about a codebase, and — when asked — write new code that matches a project's existing conventions.

Everything is grounded in the real, retrieved source code via retrieval-augmented generation (RAG) — the agent never guesses from memory about what your code does.

---

## Why this exists

"Chat with your codebase" is a saturated category. This project is narrower and more specific on purpose: **it finds test-gaps and explains risk**, using AST-level chunking (not naive fixed-size text splitting) so retrieved code is always a complete, meaningful unit — a whole function, not half of one.

## Features

- 🔍 **One-click gap scanning** — finds untested functions/classes/methods, with file:line citations
- 🧠 **Risk explanations grounded in real code** — reasons about what a specific function does (mutates state, touches auth/I/O, etc.), not boilerplate
- ✍️ **Test drafting** — generates pytest/JUnit stubs as text (never auto-writes files)
- 💬 **General Q&A** — "where is X handled?", plus grounded code generation for new features
- 🌐 **Multi-language** — Python, Java, and JavaScript (including React/JSX components)
- 🆓 **Zero-cost by default** — runs on Groq + Gemini's free tiers with automatic failover between them; Claude supported as a drop-in upgrade if you have API credits
- 🖥️ **Streamlit UI** — chat view, one-click scan view, in-app repo indexing, and a transparent "reasoning trace" showing every tool call the agent makes

## Architecture

```
User (Streamlit or CLI)
        │
        ▼
   Agent (Groq/Gemini/Claude, via tool-use loop)
        │
   decides which tool(s) to call
        │
   ┌────┴─────────────┬──────────────────┐
   ▼                   ▼                  ▼
MCP Server tool:    MCP Server tool:   MCP Server tool:
search_code(query)  read_file(path)    find_untested(path)
        │                   │                  │
        ▼                   ▼                  ▼
   Chroma vector DB    Direct file        Chroma metadata
   (semantic search)    read from disk     filter (has_test)
        │
        ▼
Agent explains risk, cites file:line, drafts tests as text
```

**Indexing pipeline** (run once per repo, via `index_repo.py`):

```
repo folder → walker.py (find source/test files, skip build artifacts)
            → chunkers/*.py (tree-sitter AST parsing → function/class chunks)
            → test_linkage.py (name-matching heuristic → has_test flag)
            → sentence-transformers (embed each chunk)
            → ChromaDB (store vectors + metadata)
```

## Tech Stack

| Layer | Tech |
|---|---|
| Chunking | `tree-sitter` (uniform AST parsing across Python/Java/JS) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, local, free) |
| Vector store | ChromaDB |
| Tool protocol | MCP (Model Context Protocol) |
| LLM backends | Groq, Gemini (free tier, with fallback) — Claude optional |
| UI | Streamlit |

## Setup

```bash
git clone <this-repo>
cd codecheck
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Create a `.env` file (see `.env.example`):
```
GROQ_API_KEY=your-groq-key         # console.groq.com — free, no card needed
GEMINI_API_KEY=your-gemini-key     # aistudio.google.com/apikey — free, no card needed
# ANTHROPIC_API_KEY=your-key       # optional, for agent.py instead of agent_multi.py
```

## Usage

**1. Index a repo** (one-time per repo, re-run after significant code changes):
```bash
python index_repo.py path/to/your/repo
```

**2. Ask questions** via CLI:
```bash
python agent_multi.py "What's untested and risky in this repo?"
python agent_multi.py "Draft a pytest stub for calculate_refund"
```

**3. Or launch the UI:**
```bash
streamlit run app.py
```
Index new repos directly from the sidebar, scan for gaps with one click, or chat freely — no CLI needed after initial setup.


## Project Structure

```
chunkers/          # per-language AST chunk extractors (Python/Java/JS)
walker.py          # repo directory walker, source/test file classification
test_linkage.py    # has_test heuristic per language
index_repo.py       # standalone indexing script
retrieval.py         # search_code / find_untested / read_file
mcp_server.py         # MCP server exposing the above as tools
providers.py           # Groq/Gemini adapters + fallback logic
agent.py                # agent loop (Claude)
agent_multi.py            # agent loop (Groq/Gemini with fallback)
app.py                     # Streamlit UI
```
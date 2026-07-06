"""
Phase 4 (multi-provider variant): same agent loop as agent.py, but backed
by Groq and/or Gemini instead of Claude, with automatic fallback -- if the
current provider hits a rate limit (HTTP 429), the loop switches to the
next configured provider and retries the SAME request, without losing
conversation context (the canonical history in providers.py makes this
possible).

Usage:
    python agent_multi.py "your question here"

Requires GROQ_API_KEY and/or GEMINI_API_KEY in the environment (.env).
Provider order is Gemini first, then Groq, unless overridden (Gemini
generally reasons better on code-tracing tasks; Groq is the fast/high-volume
fallback -- see ProviderPool for details)."""

import asyncio
import os
import sys

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from providers import (
    mcp_tools_to_groq_format, mcp_tools_to_gemini_format,
    call_groq, call_gemini, is_rate_limit_error, is_tool_generation_error,
)

load_dotenv()

GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = """You are a codebase assistant with two capabilities:

1. PRIMARY: test-gap detection -- finding functions/classes with little or \
no test coverage, explaining why leaving them untested is risky, and \
drafting pytest or JUnit test stubs for them.

2. SECONDARY: general code assistance grounded in this specific codebase -- \
answering "where is X handled" questions, and when asked, WRITING NEW CODE \
(e.g. a new controller, function, or feature) that follows the existing \
codebase's actual conventions. You are allowed and expected to do this when \
asked -- do not refuse or redirect back to test-gap detection just because \
that's your primary purpose. If someone asks for a new controller/function, \
use search_code and read_file to look at how similar existing code in this \
repo is structured (naming, error handling, response format, imports) \
before writing something that matches those patterns, rather than writing \
generic code from scratch.

Rules you must always follow:
- Always cite file:line when referencing any piece of EXISTING code.
- When flagging untested code, you MUST ground your risk explanation in the \
  SPECIFIC operations that specific chunk of code performs -- not a generic \
  template applied to every function. Look at the actual code body (call \
  read_file if the chunk shown to you is truncated or you need more context) \
  and reference concrete details: what does it mutate, what external system \
  does it touch, what input does it trust, what would break downstream if \
  it's wrong.
- When asked to draft a test OR write new code, follow real conventions for \
  the target language/codebase and present it as TEXT ONLY. You cannot \
  write files -- the person will copy what you write into their own files.
- Decide which tool(s) a question actually needs.
- If unsure which repo/collection to use, call list_repos first.

Here is the difference between a GENERIC risk explanation (do not write \
these) and a SPECIFIC one (write these instead):

GENERIC (bad -- do not do this):
  "HTTPAdapter.__getstate__ is risky to leave untested because it is used \
  to store and retrieve the state of the adapter, which could lead to \
  issues if not implemented correctly."
  -- This is generic boilerplate. It doesn't say what would actually break,
  and the exact same sentence could be pasted under almost any method.

SPECIFIC (good -- do this):
  "HTTPAdapter.__getstate__ controls exactly which internal attributes \
  survive pickling (used when a Session is passed across process \
  boundaries, e.g. multiprocessing). If a new attribute is added to \
  HTTPAdapter later but this method isn't updated to include it, that \
  attribute will silently vanish after unpickling with no error raised -- \
  a bug that would only surface as confusing behavior far from its actual \
  cause. That silent-failure mode is exactly why it's worth a test that \
  pickles/unpickles an adapter and asserts the restored object's state \
  matches the original."
  -- This references what the code actually does (controls pickled state),
  the specific failure mode (silent attribute loss), and why that failure
  mode is worse than a loud error (hard to debug).

Two different untested functions should almost never get the same risk \
justification. If you notice you're about to write a similar sentence for \
multiple functions, stop and look at what's actually different about each \
one's code.

CRITICAL rule for drafting tests: before writing ANY assertion about what \
exception a function raises, what it returns, or what state it mutates, \
you must have actually SEEN that specific behavior in the code you \
retrieved. If a tool result is marked [TRUNCATED], do not guess what the \
missing part does -- call read_file to see the full function first. \
Getting an error message's exact text right is not the same as knowing \
when that error actually fires -- trace the real control flow (if/else \
branches, what happens after an exception path, what the function does \
with None inputs) before asserting anything about it. When in doubt, write \
a weaker but CORRECT assertion (e.g. "does not raise") rather than a \
specific but unverified one.

The same grounding discipline applies to writing NEW code: look at how \
existing similar code in the repo actually does things (imports used, \
error-handling style, naming conventions, response shape) before writing \
something new, rather than inventing a plausible-looking pattern that \
doesn't match the rest of the codebase.

If you know which part of the repo is relevant (e.g. "frontend" vs \
"backend"), use search_code's path_prefix parameter to scope the search \
directly, rather than trying many different keyword phrasings of the same \
query -- each tool call costs real API quota, and repeatedly re-searching \
with slightly different wording rarely finds something a well-scoped \
search wouldn't have. If your first 2-3 searches haven't found what you \
need, report that honestly rather than continuing to retry indefinitely."""


class ProviderPool:
    """Tries providers in order, falling back to the next on a rate-limit
    error. Remembers which provider last succeeded and starts there next
    time (rather than always starting from the top), so a temporarily
    rate-limited provider gets skipped for the rest of this run."""

    def __init__(self, groq_client, gemini_client, groq_tools, gemini_tools, on_event=None):
        self.groq_client = groq_client
        self.gemini_client = gemini_client
        self.groq_tools = groq_tools
        self.gemini_tools = gemini_tools
        self.on_event = on_event

        self.providers = []
        # Gemini tried first: for this specific task (tracing exact control
        # flow before asserting behavior), reasoning quality matters more
        # than raw throughput -- Gemini 2.5 Flash generally benchmarks
        # stronger than Llama 3.3 70B on code-reasoning tasks. Groq's real
        # advantage is speed/volume, which matters less here than getting
        # the assertion right. Falls back to Groq automatically if Gemini
        # is rate-limited.
        if gemini_client:
            self.providers.append("gemini")
        if groq_client:
            self.providers.append("groq")
        if not self.providers:
            raise RuntimeError("No provider configured -- set GROQ_API_KEY and/or GEMINI_API_KEY.")

        self._current = 0

    def _call(self, provider: str, system: str, history: list[dict]):
        if provider == "groq":
            return call_groq(self.groq_client, GROQ_MODEL, system, self.groq_tools, history)
        else:
            return call_gemini(self.gemini_client, GEMINI_MODEL, system, self.gemini_tools, history)

    def call_with_fallback(self, system: str, history: list[dict]):
        n = len(self.providers)
        last_error = None
        for attempts in range(n):
            provider = self.providers[self._current]
            try:
                result = self._call(provider, system, history)
                return result, provider
            except Exception as e:
                last_error = e
                if is_rate_limit_error(e):
                    if self.on_event:
                        self.on_event({"type": "rate_limited", "provider": provider})
                    else:
                        print(f"  [rate limited on {provider}, switching provider...]", file=sys.stderr)
                    self._current = (self._current + 1) % n
                    continue
                if is_tool_generation_error(e):
                    # different root cause than rate-limiting (the model
                    # malformed a tool call, e.g. Groq/Llama emitting
                    # `<function=search_code [...]}](</function>` instead
                    # of valid JSON args) -- but the right response is the
                    # same: switch provider, since a different model is
                    # very likely to format the call correctly.
                    if self.on_event:
                        self.on_event({"type": "tool_generation_failed", "provider": provider})
                    else:
                        print(f"  [tool-call generation failed on {provider}, switching provider...]", file=sys.stderr)
                    self._current = (self._current + 1) % n
                    continue
                raise  # genuinely unrelated errors surface immediately, no point retrying another provider

        raise RuntimeError(
            f"All {n} configured provider(s) failed (rate limits or tool-call errors). Last error: {last_error}"
        )


async def run_agent(user_query: str, on_event=None) -> str:
    """on_event(dict) is called for each trace event -- {"type": "provider", ...},
    {"type": "tool_call", ...}, {"type": "tool_result", ...} -- so a UI (e.g.
    Streamlit) can display a structured reasoning trace instead of parsing
    stderr. If on_event is None, falls back to printing to stderr (the
    original CLI behavior)."""

    def emit(event: dict):
        if on_event:
            on_event(event)
        else:
            if event["type"] == "provider":
                print(f"  [provider: {event['provider']}]", file=sys.stderr)
            elif event["type"] == "tool_call":
                print(f"  [tool call] {event['name']}({event['input']})", file=sys.stderr)
            elif event["type"] == "tool_result":
                print(f"  [tool result] {event['preview']}", file=sys.stderr)
            elif event["type"] == "rate_limited":
                print(f"  [rate limited on {event['provider']}, switching provider...]", file=sys.stderr)
            elif event["type"] == "tool_generation_failed":
                print(f"  [tool-call generation failed on {event['provider']}, switching provider...]", file=sys.stderr)

    groq_client = None
    gemini_client = None

    if os.environ.get("GROQ_API_KEY"):
        from groq import Groq
        groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    if os.environ.get("GEMINI_API_KEY"):
        from google import genai
        gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools = (await session.list_tools()).tools

                groq_tools = mcp_tools_to_groq_format(mcp_tools) if groq_client else None
                gemini_tools = mcp_tools_to_gemini_format(mcp_tools) if gemini_client else None

                pool = ProviderPool(groq_client, gemini_client, groq_tools, gemini_tools, on_event=emit)

                history = [{"role": "user", "text": user_query}]

                while True:
                    response, used_provider = pool.call_with_fallback(SYSTEM_PROMPT, history)
                    emit({"type": "provider", "provider": used_provider})

                    if not response.tool_calls:
                        return response.text or ""

                    history.append({
                        "role": "assistant",
                        "text": response.text,
                        "tool_calls": [{"id": tc.id, "name": tc.name, "input": tc.input} for tc in response.tool_calls],
                    })

                    for tc in response.tool_calls:
                        emit({"type": "tool_call", "name": tc.name, "input": tc.input})
                        result = await session.call_tool(tc.name, tc.input)
                        result_text = "".join(c.text for c in result.content if hasattr(c, "text"))
                        preview = result_text[:300] + ("..." if len(result_text) > 300 else "")
                        emit({"type": "tool_result", "name": tc.name, "preview": preview, "full": result_text})
                        history.append({
                            "role": "tool_result",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": result_text,
                        })
    except BaseException as e:
        # anyio/asyncio wrap exceptions raised inside nested async context
        # managers (like the two `async with` above) in an opaque
        # BaseExceptionGroup / "unhandled errors in a TaskGroup" wrapper --
        # this was a REAL bug found via testing: a RuntimeError with a
        # perfectly clear message ("All 2 configured provider(s) are
        # rate-limited...") was getting buried inside that wrapper, so the
        # person just saw "unhandled errors in a TaskGroup (1 sub-exception)"
        # with no useful information. Unwrap it here so the real message
        # actually reaches the caller (CLI print, or app.py's st.error).
        leaf = e
        while hasattr(leaf, "exceptions") and leaf.exceptions:
            leaf = leaf.exceptions[0]
        raise RuntimeError(f"Agent failed: {leaf}") from leaf


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python agent_multi.py "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    answer = asyncio.run(run_agent(query))
    print(answer)
"""
Phase 4: the agent loop.

Connects to mcp_server.py as an MCP client (spawns it as a subprocess over
stdio), converts its tool definitions into the format the Anthropic
Messages API expects, and runs a manual tool_use loop: send the
conversation to Claude, and whenever it wants to call a tool, execute that
tool against the MCP server, feed the result back, and continue until
Claude produces a final text answer.

Usage:
    python agent.py "Scan the requests repo and tell me the riskiest
                      untested functions"

Requires ANTHROPIC_API_KEY set in the environment.
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()  # reads .env in the current working directory, if present

MODEL = "claude-sonnet-4-6"

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
- Always cite file:line when referencing any piece of EXISTING code (e.g. \
  "sessions.py:134"). Never describe code without grounding it in a \
  specific file and line range you actually retrieved via a tool.
- When flagging untested code, you MUST ground your risk explanation in the \
  SPECIFIC operations that specific chunk of code performs -- not a generic \
  template applied to every function. Look at the actual code body (call \
  read_file if the chunk shown to you is truncated or you need more context) \
  and reference concrete details: what does it mutate, what external system \
  does it touch, what input does it trust, what would break downstream if \
  it's wrong.
- When asked to draft a test OR write new code, follow the target \
  language/codebase's real conventions (pytest for Python, JUnit for Java) \
  and present it as TEXT ONLY in your response. Never claim to have written \
  it to a file -- you have no file-writing capability, only read/search tools.
- Decide which tool(s) a question actually needs. Don't call find_untested \
  for a question that's really asking "where is X handled" (that's \
  search_code), and don't call search_code for "what's untested in \
  directory Y" (that's find_untested, with path_prefix).
- If you don't know which repo/collection to use, call list_repos first \
  rather than guessing a collection_name.
- Be honest about the test-linkage heuristic's limitations if relevant: \
  it's name-based matching against test file contents, not true call-graph \
  analysis, so it can have false positives (generic names) and false \
  negatives (indirect testing via another function).

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
doesn't match the rest of the codebase."""


def mcp_tool_to_anthropic_format(tool) -> dict:
    """Convert an MCP tool definition into the Anthropic Messages API's
    tool format. MCP tools already use JSON Schema for inputSchema, which
    is exactly what Anthropic's `input_schema` field expects -- so this is
    mostly just a field rename."""
    return {
        "name": tool.name,
        "description": tool.description or "",
        "input_schema": tool.inputSchema,
    }


async def run_agent(user_query: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set in the environment.")

    client = Anthropic(api_key=api_key)
    server_params = StdioServerParameters(command="python", args=["mcp_server.py"])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = (await session.list_tools()).tools
            anthropic_tools = [mcp_tool_to_anthropic_format(t) for t in mcp_tools]

            messages = [{"role": "user", "content": user_query}]

            while True:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=2048,
                    system=SYSTEM_PROMPT,
                    tools=anthropic_tools,
                    messages=messages,
                )

                if response.stop_reason != "tool_use":
                    # final answer -- collect any text blocks
                    return "".join(
                        block.text for block in response.content if block.type == "text"
                    )

                # Claude wants to call one or more tools -- append its turn,
                # execute each tool call against the MCP server, append results
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result = await session.call_tool(block.name, block.input)
                    result_text = "".join(
                        c.text for c in result.content if hasattr(c, "text")
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result_text,
                    })

                messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python agent.py "your question here"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    answer = asyncio.run(run_agent(query))
    print(answer)
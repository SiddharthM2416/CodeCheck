"""
Adapters for Groq and Gemini, so agent_multi.py can call either provider
through one canonical interface and switch between them mid-conversation
on a rate-limit error.

Design: conversation history is kept in a canonical, provider-agnostic
format (a list of dicts). Each provider adapter converts that canonical
history into its own required format fresh on every call -- this is what
makes switching providers mid-conversation possible without losing context,
since the canonical history never depends on either provider's SDK types.

Canonical history entry shapes:
    {"role": "user", "text": "..."}
    {"role": "assistant", "text": "...", "tool_calls": [{"id", "name", "input"}] or None}
    {"role": "tool_result", "tool_call_id": "...", "name": "...", "content": "..."}
"""

from dataclasses import dataclass


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class ProviderResponse:
    text: str | None            # final answer text, if this turn is done
    tool_calls: list[ToolCall]  # empty if no tool calls requested


def is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort, SDK-agnostic rate-limit detection.

    429 is the conventional rate-limit status code, but Groq's free tier
    returns 413 ("Request too large") with a body code of
    'rate_limit_exceeded' when a single request's token count exceeds the
    Tokens Per Minute budget -- this is still a rate-limit condition (the
    fix is either to wait or switch provider), just signaled differently
    than a classic RPM-style 429. Check both the status code AND the
    error body/message text, since relying on status code alone misses
    this real case."""
    status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status == 429:
        return True

    name = type(exc).__name__.lower()
    if "ratelimit" in name or "resourceexhausted" in name:
        return True

    # fall back to string-matching the error body/message -- covers Groq's
    # 413 + 'rate_limit_exceeded' case and similar provider quirks
    text = str(exc).lower()
    if "rate_limit_exceeded" in text or "tokens per minute" in text or "requests per minute" in text:
        return True

    return False


# ---------------------------------------------------------------------------
# Groq (OpenAI-compatible tool-calling format)
# ---------------------------------------------------------------------------

def mcp_tools_to_groq_format(mcp_tools) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            },
        }
        for t in mcp_tools
    ]


def canonical_to_groq_messages(system: str, history: list[dict]) -> list[dict]:
    messages = [{"role": "system", "content": system}]
    for entry in history:
        if entry["role"] == "user":
            messages.append({"role": "user", "content": entry["text"]})
        elif entry["role"] == "assistant":
            msg = {"role": "assistant", "content": entry.get("text") or None}
            if entry.get("tool_calls"):
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {"name": tc["name"], "arguments": _json_dumps(tc["input"])},
                    }
                    for tc in entry["tool_calls"]
                ]
            messages.append(msg)
        elif entry["role"] == "tool_result":
            messages.append({
                "role": "tool",
                "tool_call_id": entry["tool_call_id"],
                "content": entry["content"],
            })
    return messages


def call_groq(client, model: str, system: str, groq_tools: list[dict], history: list[dict]) -> ProviderResponse:
    messages = canonical_to_groq_messages(system, history)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=groq_tools,
        max_tokens=2048,
    )
    choice = response.choices[0]
    msg = choice.message

    if msg.tool_calls:
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, input=_json_loads(tc.function.arguments))
            for tc in msg.tool_calls
        ]
        return ProviderResponse(text=msg.content, tool_calls=calls)

    return ProviderResponse(text=msg.content, tool_calls=[])


# ---------------------------------------------------------------------------
# Gemini (google-genai function-calling format)
# ---------------------------------------------------------------------------

def mcp_tools_to_gemini_format(mcp_tools):
    from google.genai import types

    declarations = [
        types.FunctionDeclaration(
            name=t.name,
            description=t.description or "",
            parameters=_json_schema_to_gemini_schema(t.inputSchema),
        )
        for t in mcp_tools
    ]
    return [types.Tool(function_declarations=declarations)]


def _json_schema_to_gemini_schema(schema: dict) -> dict:
    """Gemini's function parameter schema is a constrained subset of JSON
    Schema (OpenAPI 3.0-style) -- strip fields it doesn't understand
    (e.g. $schema, additionalProperties) rather than passing them through
    and risking a rejected request."""
    if not isinstance(schema, dict):
        return schema
    DROP_KEYS = {"$schema", "additionalProperties", "title"}
    cleaned = {k: v for k, v in schema.items() if k not in DROP_KEYS}
    if "properties" in cleaned:
        cleaned["properties"] = {
            k: _json_schema_to_gemini_schema(v) for k, v in cleaned["properties"].items()
        }
    return cleaned


def canonical_to_gemini_contents(history: list[dict]):
    from google.genai import types

    contents = []
    for entry in history:
        if entry["role"] == "user":
            contents.append(types.Content(role="user", parts=[types.Part(text=entry["text"])]))
        elif entry["role"] == "assistant":
            parts = []
            if entry.get("text"):
                parts.append(types.Part(text=entry["text"]))
            for tc in entry.get("tool_calls") or []:
                parts.append(types.Part(function_call=types.FunctionCall(name=tc["name"], args=tc["input"])))
            contents.append(types.Content(role="model", parts=parts))
        elif entry["role"] == "tool_result":
            contents.append(types.Content(
                role="user",
                parts=[types.Part(function_response=types.FunctionResponse(
                    name=entry["name"], response={"result": entry["content"]},
                ))],
            ))
    return contents


def call_gemini(client, model: str, system: str, gemini_tools, history: list[dict]) -> ProviderResponse:
    from google.genai import types

    contents = canonical_to_gemini_contents(history)
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            tools=gemini_tools,
            max_output_tokens=2048,
        ),
    )

    candidate = response.candidates[0]
    text_parts = []
    tool_calls = []
    for i, part in enumerate(candidate.content.parts):
        if part.text:
            text_parts.append(part.text)
        if part.function_call:
            tool_calls.append(ToolCall(
                id=f"gemini-call-{i}",  # Gemini doesn't issue call IDs -- synthesize one
                name=part.function_call.name,
                input=dict(part.function_call.args),
            ))

    return ProviderResponse(text="".join(text_parts) or None, tool_calls=tool_calls)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

import json


def _json_dumps(obj) -> str:
    return json.dumps(obj)


def _json_loads(s: str) -> dict:
    return json.loads(s) if s else {}
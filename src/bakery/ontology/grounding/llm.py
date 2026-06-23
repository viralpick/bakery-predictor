"""Provider-neutral LLM interface (docs/.../grounding-eval-design.md §4).

Higher layers (questions/arms/scorer) use only the neutral types here; each
provider's quirks (tool format, structured-output format, params) live inside
its adapter. Today: OpenAI(gpt-5-mini). Future: an Anthropic adapter slots in
without touching anything above this file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict          # JSON Schema for the tool's arguments


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    content: str


@dataclass
class Message:
    role: str                              # system | user | assistant | tool
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None        # set on role="tool" replies


@dataclass(frozen=True)
class LLMResponse:
    text: str | None
    tool_calls: list[ToolCall]
    parsed: dict | None                    # structured-output result, if requested


@runtime_checkable
class LLMClient(Protocol):
    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSpec] | None = None,
        output_schema: dict | None = None,
    ) -> LLMResponse: ...


# ============================================================================
# OpenAI Adapter (S3 provider-specific code)
# ============================================================================


def _to_openai_tools(tools: list[ToolSpec]) -> list[dict]:
    return [{"type": "function",
             "function": {"name": t.name, "description": t.description,
                          "parameters": t.parameters, "strict": True}}
            for t in tools]


def _to_response_format(schema: dict) -> dict:
    return {"type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema, "strict": True}}


def _to_openai_messages(messages: list[Message]) -> list[dict]:
    out = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            out.append({"role": "assistant", "content": m.content,
                        "tool_calls": [{"id": c.id, "type": "function",
                                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                                       for c in m.tool_calls]})
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _parse_response(completion) -> LLMResponse:
    msg = completion.choices[0].message
    calls = [ToolCall(id=tc.id, name=tc.function.name, arguments=json.loads(tc.function.arguments))
             for tc in (msg.tool_calls or [])]
    parsed = getattr(msg, "parsed", None)
    # Fallback: if response_format is raw json_schema (not Pydantic), OpenAI puts JSON in .content
    if parsed is None and msg.content:
        try:
            parsed = json.loads(msg.content)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return LLMResponse(text=msg.content, tool_calls=calls, parsed=parsed)


class OpenAIClient:
    """OpenAI adapter (gpt-5-mini). The only provider-specific code in S3."""

    def __init__(self, model: str = "gpt-5-mini", api_key: str | None = None):
        self._model = model
        self._api_key = api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key or os.getenv("OPENAI_API_KEY"))
        return self._client

    def generate(self, messages, *, tools=None, output_schema=None) -> LLMResponse:
        client = self._ensure_client()
        kwargs: dict = {"model": self._model, "messages": _to_openai_messages(messages)}
        if tools:
            kwargs["tools"] = _to_openai_tools(tools)
        if output_schema:
            kwargs["response_format"] = _to_response_format(output_schema)
        completion = client.chat.completions.parse(**kwargs)
        return _parse_response(completion)


def make_llm_client(provider: str, model: str, **kw) -> LLMClient:
    if provider == "openai":
        return OpenAIClient(model=model, **kw)
    raise ValueError(f"unknown provider: {provider} (anthropic adapter: add when 발급)")

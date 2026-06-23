"""Provider-neutral LLM interface (docs/.../grounding-eval-design.md §4).

Higher layers (questions/arms/scorer) use only the neutral types here; each
provider's quirks (tool format, structured-output format, params) live inside
its adapter. Today: OpenAI(gpt-5-mini). Future: an Anthropic adapter slots in
without touching anything above this file.
"""

from __future__ import annotations

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

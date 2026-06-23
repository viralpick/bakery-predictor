"""Two eval arms over the same model — grounded (tools) vs rag-only (knowledge).

Fairness §8: identical model/params; the ONLY difference is whether the
OntologyFunction tools are exposed. The grounded arm runs a provider-neutral
tool loop (LLMClient.generate → dispatch → re-ask); rag-only gets the ontology
knowledge chunks as context and must answer without tools.
"""

from __future__ import annotations

from ..schema import BAKERY_ONTOLOGY
from ...data.loader import DailyDataset
from .llm import LLMClient, Message
from .questions import Question
from .tools import TOOL_SPECS, dispatch

MAX_TOOL_TURNS = 6

OUTPUT_SCHEMAS: dict[str, dict] = {
    "numeric": {"type": "object", "properties": {"answer_value": {"type": "number"}},
                "required": ["answer_value"], "additionalProperties": False},
    "ranking": {"type": "object", "properties": {"top_items": {"type": "array", "items": {"type": "string"}}},
                "required": ["top_items"], "additionalProperties": False},
    "decomposition": {"type": "object", "properties": {"order_qty": {"type": "number"}},
                      "required": ["order_qty"], "additionalProperties": False},
}

_GROUNDED_SYS = (
    "You answer bakery ordering questions using ONLY the provided tools. "
    "Call the relevant tool(s), then return the final answer in the required JSON schema. "
    "Never guess a number you can compute with a tool."
)


def _knowledge_text() -> str:
    return "\n".join(f"- {k.name}: {k.content}" for k in BAKERY_ONTOLOGY.knowledge)


_RAG_SYS = (
    "You answer bakery ordering questions using ONLY the domain knowledge below. "
    "You have no data access; give your best estimate in the required JSON schema.\n\n"
    + _knowledge_text()
)


def run_grounded(client: LLMClient, question: Question, dataset: DailyDataset) -> dict:
    schema = OUTPUT_SCHEMAS[question.grader_type]
    messages = [Message(role="system", content=_GROUNDED_SYS),
                Message(role="user", content=question.text)]
    for _ in range(MAX_TOOL_TURNS):
        resp = client.generate(messages, tools=TOOL_SPECS, output_schema=schema)
        if not resp.tool_calls:
            return resp.parsed or {}
        messages.append(Message(role="assistant", tool_calls=resp.tool_calls))
        for call in resp.tool_calls:
            result = dispatch(call, dataset)
            messages.append(Message(role="tool", content=result.content, tool_call_id=result.call_id))
    # tool budget exhausted — force a final answer with no tools
    return (client.generate(messages, output_schema=schema).parsed) or {}


def run_rag_only(client: LLMClient, question: Question, dataset: DailyDataset) -> dict:
    schema = OUTPUT_SCHEMAS[question.grader_type]
    messages = [Message(role="system", content=_RAG_SYS),
                Message(role="user", content=question.text)]
    return client.generate(messages, output_schema=schema).parsed or {}

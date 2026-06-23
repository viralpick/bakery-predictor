import dataclasses

import pytest

from bakery.ontology.grounding.llm import ToolSpec, ToolCall, Message, LLMResponse

def test_toolspec_is_frozen_dataclass():
    t = ToolSpec(name="f", description="d", parameters={"type": "object", "properties": {}})
    assert t.name == "f"
    assert t.parameters["type"] == "object"
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.name = "changed"

def test_message_defaults():
    m = Message(role="user", content="hi")
    assert m.tool_calls == []
    assert m.tool_call_id is None

def test_llmresponse_holds_tool_calls_and_parsed():
    r = LLMResponse(text=None, tool_calls=[ToolCall(id="1", name="f", arguments={"x": 1})], parsed=None)
    assert r.tool_calls[0].arguments == {"x": 1}

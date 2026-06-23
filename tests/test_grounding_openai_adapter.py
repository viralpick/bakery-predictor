import os
import pytest
from bakery.ontology.grounding.llm import ToolSpec, make_llm_client
from bakery.ontology.grounding import llm as L


def test_to_openai_tools_shape():
    spec = ToolSpec("f", "desc", {"type": "object", "properties": {"x": {"type": "number"}},
                                  "required": ["x"], "additionalProperties": False})
    out = L._to_openai_tools([spec])
    assert out[0]["type"] == "function"
    assert out[0]["function"]["name"] == "f"
    assert out[0]["function"]["parameters"]["properties"]["x"]["type"] == "number"


def test_to_response_format_wraps_json_schema():
    rf = L._to_response_format({"type": "object", "properties": {}, "additionalProperties": False})
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert "schema" in rf["json_schema"]


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_make_llm_client_openai():
    client = make_llm_client("openai", "gpt-5-mini")
    assert isinstance(client, L.OpenAIClient)


def test_make_llm_client_unknown_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        make_llm_client("grok", "x")


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="no OPENAI_API_KEY")
def test_live_structured_answer():
    client = make_llm_client("openai", "gpt-5-mini")
    from bakery.ontology.grounding.llm import Message
    schema = {"type": "object", "properties": {"answer_value": {"type": "number"}},
              "required": ["answer_value"], "additionalProperties": False}
    resp = client.generate([Message(role="user", content="Return answer_value 7.")], output_schema=schema)
    assert resp.parsed["answer_value"] == 7

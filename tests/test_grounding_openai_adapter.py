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


def test_make_llm_client_openai():
    client = make_llm_client("openai", "gpt-5-mini")
    assert isinstance(client, L.OpenAIClient)


def test_make_llm_client_unknown_raises():
    with pytest.raises(ValueError, match="unknown provider"):
        make_llm_client("grok", "x")


def test_parse_response_falls_back_to_content_json():
    """When msg.parsed is None (raw json_schema) and msg.content is JSON, parse it."""
    import types
    msg = types.SimpleNamespace(
        content='{"answer_value": 7}',
        parsed=None,
        tool_calls=None
    )
    choice = types.SimpleNamespace(message=msg)
    completion = types.SimpleNamespace(choices=[choice])

    resp = L._parse_response(completion)
    assert resp.parsed == {"answer_value": 7}
    assert resp.text == '{"answer_value": 7}'
    assert resp.tool_calls == []


def test_parse_response_handles_missing_content_and_tool_call():
    """When content is None but there's a tool_call, parsed stays None, tool_calls are extracted."""
    import types
    tc = types.SimpleNamespace(
        id="call_123",
        function=types.SimpleNamespace(
            name="my_tool",
            arguments='{"x": 1}'
        )
    )
    msg = types.SimpleNamespace(
        content=None,
        parsed=None,
        tool_calls=[tc]
    )
    choice = types.SimpleNamespace(message=msg)
    completion = types.SimpleNamespace(choices=[choice])

    resp = L._parse_response(completion)
    assert resp.parsed is None
    assert resp.text is None
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "my_tool"
    assert resp.tool_calls[0].arguments == {"x": 1}


@pytest.mark.skipif(
    not (os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")),
    reason="no OPENAI_API_KEY / AZURE_OPENAI_API_KEY",
)
def test_live_structured_answer():
    client = make_llm_client("auto", "gpt-5-mini")
    from bakery.ontology.grounding.llm import Message
    schema = {"type": "object", "properties": {"answer_value": {"type": "number"}},
              "required": ["answer_value"], "additionalProperties": False}
    resp = client.generate([Message(role="user", content="Return answer_value 7.")], output_schema=schema)
    assert resp.parsed["answer_value"] == 7


def test_make_llm_client_azure(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    client = make_llm_client("azure", "gpt-5-mini")
    assert isinstance(client, L.AzureOpenAIClient)
    assert isinstance(client, L.OpenAIClient)          # generate 로직 상속·재사용


def test_azure_endpoint_normalizes_trailing_openai():
    """SDK가 /openai를 자동으로 붙이므로 .env endpoint 끝의 /openai는 제거."""
    assert (L._normalize_azure_endpoint("https://x.azure-api.net/openai")
            == "https://x.azure-api.net")
    assert (L._normalize_azure_endpoint("https://x.azure-api.net/openai/")
            == "https://x.azure-api.net")
    assert (L._normalize_azure_endpoint("https://x.openai.azure.com")
            == "https://x.openai.azure.com")


def test_azure_deployment_env_fallback(monkeypatch):
    """model 미지정 시 AZURE_OPENAI_DEPLOYMENT env를 deployment로 사용."""
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deploy")
    client = make_llm_client("azure", "")
    assert client._model == "my-deploy"


def test_make_llm_client_auto_prefers_azure(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "k")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
    client = make_llm_client("auto", "gpt-5-mini")
    assert isinstance(client, L.AzureOpenAIClient)


def test_make_llm_client_auto_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    client = make_llm_client("auto", "gpt-5-mini")
    assert isinstance(client, L.OpenAIClient)
    assert not isinstance(client, L.AzureOpenAIClient)

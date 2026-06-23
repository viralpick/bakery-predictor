import pytest
from bakery.data.loader import load_dataset
from bakery.ontology.grounding.llm import LLMResponse, ToolCall, Message
from bakery.ontology.grounding.questions import Question
from bakery.ontology.grounding import arms


@pytest.fixture(scope="module")
def dataset():
    return load_dataset("synthetic")


class FakeLLM:
    """Scripted client: first call emits a tool_call, second returns parsed answer."""
    def __init__(self, tool_call, parsed):
        self._tool_call, self._parsed, self.calls = tool_call, parsed, 0

    def generate(self, messages, *, tools=None, output_schema=None):
        self.calls += 1
        if tools and self.calls == 1:
            return LLMResponse(text=None, tool_calls=[self._tool_call], parsed=None)
        return LLMResponse(text=None, tool_calls=[], parsed=self._parsed)


def test_run_grounded_runs_tool_then_answers(dataset):
    store = dataset.daily["store_id"].iloc[0]
    import pandas as pd
    dd = pd.to_datetime(dataset.daily.loc[dataset.daily["store_id"] == store, "date"])
    tc = ToolCall(id="c1", name="rank_stockout_risk",
                  arguments={"store_id": store, "period": [str(dd.min().date()), str(dd.max().date())], "k": 3})
    fake = FakeLLM(tc, {"top_items": ["X"]})
    q = Question(id="q", text="", grader_type="ranking", source_fn="rank_stockout_risk", fn_kwargs={"k": 3})
    answer = arms.run_grounded(fake, q, dataset)
    assert fake.calls == 2                       # tool turn + answer turn
    assert answer == {"top_items": ["X"]}


def test_run_rag_only_no_tools(dataset):
    fake = FakeLLM(None, {"answer_value": 1.0})
    q = Question(id="q", text="", grader_type="numeric", source_fn="waste_cost")
    answer = arms.run_rag_only(fake, q, dataset)
    assert fake.calls == 1                        # single turn, no tool loop
    assert answer == {"answer_value": 1.0}


def test_output_schemas_cover_grader_types():
    assert set(arms.OUTPUT_SCHEMAS) == {"numeric", "ranking", "decomposition"}

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import humanizer_graph as graph
import rag_engine
from rewriting import build_prompt, clean_output


@pytest.mark.parametrize(
    "text",
    [
        "گزینهٔ دوم برای PostgreSQL مناسب است.\nمنبع: https://example.com/paper [12]",
        "این شش کلمه در هر دو متن\nباید بدون حذف بند اول باقی بماند.",
        "## API\n**OpenAI** and PostgreSQL\nنتیجه در ۲۰۲۶ تأیید نشد.",
        "«این نقل‌قول و مبلغ ۱۲۰۰ افغانی را تغییر ندهید.»",
    ],
)
def test_cleanup_keeps_source_content(text):
    assert clean_output(text, text) == text


def test_cleanup_removes_only_response_wrapper():
    text = "اسناد را تا دوشنبه بفرستید."
    assert clean_output("بازنویسی:\n" + text, "متن اصلی") == text
    assert clean_output("```text\n" + text + "\n```", "متن اصلی") == text
    assert clean_output("خروجی: ۱۲", "خروجی: ۱۲") == "خروجی: ۱۲"


def test_json_and_streaming_use_identical_prompt_and_preserve_content(monkeypatch):
    prompts = []
    output = "گزینهٔ دوم همچنان مناسب است.\nمنبع: https://example.com [12]"

    class Model:
        def invoke(self, prompt):
            prompts.append(prompt)
            return output

        def stream(self, prompt):
            prompts.append(prompt)
            yield output[:20]
            yield output[20:]

    examples = [
        {"ai": "لطفاً اسناد را ارسال نمایید.", "human": "لطفاً اسناد را بفرستید."}
    ]
    monkeypatch.setattr(graph, "get_llm", lambda model: Model())
    monkeypatch.setattr(graph, "_get_vs", lambda: object())

    def retrieve(vs, text, *, k):
        assert k == 4
        return examples

    monkeypatch.setattr(graph, "retrieve_examples", retrieve)
    text = "این گزینه برای پروژهٔ {id} مناسب است."
    state = graph.node_retrieve({"input_text": text, "model": "llama3.1:8b"})
    result = graph.node_humanize(state)
    events = list(graph.stream_humanizer(text))
    assert result["humanized_text"] == output
    assert prompts == [build_prompt(text, examples)] * 2
    assert "{id}" in prompts[0]
    assert events[-1] == "data: __DONE__\n\n"
    assert "__TEXT__" in events[-3]


def test_retrieval_failure_uses_same_fallback_for_both_paths(monkeypatch):
    def unavailable():
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(graph, "_get_vs", unavailable)
    state = graph.node_retrieve({"input_text": "سلام"})
    assert state["retrieved_examples"] == []
    assert graph.find_examples("سلام") == []


def test_full_graph_runs_one_policy_without_style_state(monkeypatch):
    prompts = []

    class Model:
        def invoke(self, prompt):
            prompts.append(prompt)
            return "لطفاً اسناد را تا دوشنبه بفرستید."

    monkeypatch.setattr(graph, "get_llm", lambda model: Model())
    monkeypatch.setattr(graph, "find_examples", lambda text: [])
    monkeypatch.setattr(graph, "_graph", None)
    text = "خواهشمند است اسناد مربوطه را تا روز دوشنبه ارسال نمایید."
    result = graph.run_humanizer(text)
    assert "style" not in result
    assert result["humanized_text"] == "لطفاً اسناد را تا دوشنبه بفرستید."
    assert prompts == [build_prompt(text, [])]
    assert result["evaluation"]


def test_production_retrieval_has_no_style_filter_or_bonus(monkeypatch):
    calls = []

    class Store:
        def similarity_search_with_score(self, query, **kwargs):
            calls.append(kwargs)
            return [
                (
                    SimpleNamespace(
                        metadata={"ai": "دوم", "human": "دو", "style": "casual"}
                    ),
                    0.2,
                ),
                (
                    SimpleNamespace(
                        metadata={"ai": "اول", "human": "یک", "style": "formal"}
                    ),
                    0.1,
                ),
            ]

    monkeypatch.setattr(rag_engine, "get_reranker", lambda: None)
    results = rag_engine.retrieve_examples(Store(), "پرسش")
    assert calls == [{"k": 20}]
    assert [item["ai"] for item in results] == ["اول", "دوم"]
    assert all(item["raw_score"] == item["adjusted_score"] for item in results)


def test_wrapper_only_generation_is_a_failure(monkeypatch):
    class Model:
        def invoke(self, prompt):
            return "بازنویسی:"

    monkeypatch.setattr(graph, "get_llm", lambda model: Model())
    result = graph.node_humanize({"input_text": "متن اصلی"})
    assert result["humanized_text"] == ""
    assert result["error"]


def test_original_code_fences_are_not_treated_as_model_wrappers():
    text = "```sql\nSELECT 1;\n```"
    assert clean_output(text, text) == text

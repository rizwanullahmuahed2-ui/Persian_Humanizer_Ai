"""Retrieve examples and apply one natural-rewriting policy through LangGraph or SSE."""

from typing import TypedDict, Optional, List, Dict, Any, Iterator
from model_provider import DEFAULT_MODEL, GPT_MODEL, GPTClient, validate_model
from rewriting import build_prompt, clean_output
import re
import time
import difflib


# ── State ──────────────────────────────────────────────────────────────────────
class HumanizerState(TypedDict):
    input_text: str
    model: str
    retrieved_examples: List[Dict]
    humanized_text: str
    evaluation: Dict[str, Any]
    error: Optional[str]


# ── LLM ────────────────────────────────────────────────────────────────────────
def get_llm(model: str = DEFAULT_MODEL):
    validate_model(model)
    if model == GPT_MODEL:
        return GPTClient()
    from langchain_ollama import OllamaLLM

    return OllamaLLM(
        model=model,
        temperature=0.8,
        num_predict=512,
    )


def clean(raw: str, original: str) -> str:
    return clean_output(raw, original)


def retrieve_examples(*args, **kwargs):
    from rag_engine import retrieve_examples as retrieve

    return retrieve(*args, **kwargs)


def rule_based_eval(original: str, humanized: str) -> Dict[str, Any]:
    """
    Real metrics — no LLM judging.
    1. meaning_preserved  → character n-gram overlap (simhash-like)
    2. naturalness        → avg sentence length + punctuation ratio
    3. formality_reduced  → drop in formal markers
    4. length_ratio       → not too long, not too short
    5. dari_markers       → count Dari-specific words
    """

    # ── 1. Meaning preservation (sequence similarity on words) ──
    orig_words = original.split()
    hum_words = humanized.split()
    sm = difflib.SequenceMatcher(None, orig_words, hum_words)
    meaning_score = round(min(sm.ratio() * 1.4, 1.0) * 10, 1)  # scale to 0-10
    # We want some similarity (meaning kept) but not 1.0 (means no change)
    meaning_score = max(3.0, min(meaning_score, 10.0))

    # ── 2. Naturalness (shorter sentences = more natural) ──
    sentences = [s.strip() for s in re.split(r"[.،؟!]", humanized) if s.strip()]
    avg_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    # Sweet spot: 5-12 words per sentence
    if avg_len <= 12:
        naturalness = 9.0
    elif avg_len <= 18:
        naturalness = 7.5
    else:
        naturalness = 5.5

    # ── 3. Formality reduction ──
    formal_markers = [
        "می‌گردد",
        "می‌شود",
        "می‌باشد",
        "خواهد شد",
        "مقرر",
        "مقرر گردید",
        "استدعا",
        "محضر",
        "استحضار",
        "مبذول",
        "متشکریم",
        "حائز",
        "مستلزم",
        "اتخاذ",
        "مقتضی",
        "لازم‌الاجرا",
        "موصوف",
        "معظم",
    ]
    orig_formal = sum(1 for m in formal_markers if m in original)
    hum_formal = sum(1 for m in formal_markers if m in humanized)
    if orig_formal == 0:
        formality_score = 8.0
    else:
        reduction = (orig_formal - hum_formal) / orig_formal
        formality_score = round(min(reduction * 10 + 4, 10), 1)
        formality_score = max(3.0, formality_score)

    # ── 4. Length ratio ──
    ratio = len(humanized) / max(len(original), 1)
    if 0.3 <= ratio <= 0.95:
        length_score = 9.0
    elif ratio < 0.3:
        length_score = 5.0  # too short
    else:
        length_score = 6.0  # too long (repetition?)

    # ── 5. Dari dialect markers ──
    dari_words = [
        "میتم",
        "میتی",
        "میته",
        "میتیم",
        "میتین",
        "میتن",
        "استم",
        "استی",
        "استه",
        "استیم",
        "استین",
        "استن",
        "میایم",
        "میرم",
        "میگم",
        "میخوام",
        "داریم",
        "باید",
        "کنین",
        "بگین",
        "بیاین",
        "ببینین",
        "بدین",
    ]
    dari_count = sum(1 for w in dari_words if w in humanized)
    dari_score = min(5.0 + dari_count * 1.5, 10.0)

    overall = round(
        (meaning_score + naturalness + formality_score + length_score + dari_score) / 5,
        1,
    )

    # Feedback
    issues = []
    if meaning_score < 5:
        issues.append("معنا کمی تغییر کرده")
    if naturalness < 6:
        issues.append("جملات هنوز بلند است")
    if formality_score < 5:
        issues.append("کلمات رسمی هنوز زیاد است")
    if dari_count == 0:
        issues.append("نشانه‌های دری دیده نشد")
    if ratio > 1.0:
        issues.append("متن طولانی‌تر از اصل شد")
    feedback = " | ".join(issues) if issues else "بازنویسی خوب به دری"

    return {
        "meaning_preserved": meaning_score,
        "naturalness": naturalness,
        "formality_reduced": formality_score,
        "dari_dialect": round(dari_score, 1),
        "length_ratio": round(ratio, 2),
        "overall_score": overall,
        "feedback": feedback,
    }


# ── Nodes ──────────────────────────────────────────────────────────────────────
_vs = None


def _get_vs():
    global _vs
    if _vs is None:
        from rag_engine import get_or_build_vectorstore

        _vs = get_or_build_vectorstore()
    return _vs


def find_examples(text: str) -> list[dict]:
    try:
        examples = retrieve_examples(_get_vs(), text, k=4)
        print(f"[GRAPH] retrieve → {len(examples)} examples")
        return examples
    except Exception as exc:
        print(f"[GRAPH] retrieve error: {type(exc).__name__}")
        return []


def node_retrieve(state: HumanizerState) -> HumanizerState:
    return {
        **state,
        "retrieved_examples": find_examples(state["input_text"]),
        "error": None,
    }


def node_humanize(state: HumanizerState) -> HumanizerState:
    try:
        llm = get_llm(state.get("model", DEFAULT_MODEL))
        prompt = build_prompt(state["input_text"], state.get("retrieved_examples", []))

        print(f"[GRAPH] humanize → calling {state.get('model', DEFAULT_MODEL)}")
        t0 = time.time()
        raw = llm.invoke(prompt)
        print(f"[GRAPH] humanize ← {len(raw)} chars in {time.time() - t0:.1f}s")

        humanized = clean(raw, state["input_text"])
        if not humanized:
            raise RuntimeError(
                "The selected model returned empty output. Please try again."
            )

        return {**state, "humanized_text": humanized, "error": None}

    except Exception as e:
        print(f"[GRAPH] humanize error: {type(e).__name__}")
        return {**state, "humanized_text": "", "error": str(e)}


def node_evaluate(state: HumanizerState) -> HumanizerState:
    humanized = state.get("humanized_text", "")
    if not humanized:
        return {
            **state,
            "evaluation": {"overall_score": 0, "feedback": "خروجی خالی"},
            "error": None,
        }

    ev = rule_based_eval(state["input_text"], humanized)
    print(f"[GRAPH] evaluate → overall={ev['overall_score']}")
    return {**state, "evaluation": ev, "error": None}


def _should_evaluate(state: HumanizerState) -> str:
    return "evaluate" if state.get("humanized_text") else "end"


# ── Graph ──────────────────────────────────────────────────────────────────────
def build_graph():
    from langgraph.graph import StateGraph, END

    g = StateGraph(HumanizerState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("humanize", node_humanize)
    g.add_node("evaluate", node_evaluate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "humanize")
    g.add_conditional_edges(
        "humanize", _should_evaluate, {"evaluate": "evaluate", "end": END}
    )
    g.add_edge("evaluate", END)
    return g.compile()


_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_humanizer(input_text: str, model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    validate_model(model)
    result = get_graph().invoke(
        {
            "input_text": input_text,
            "model": model,
            "retrieved_examples": [],
            "humanized_text": "",
            "evaluation": {},
            "error": None,
        }
    )
    return result


# ── Streaming ──────────────────────────────────────────────────────────────────
def stream_humanizer(input_text: str, model: str = DEFAULT_MODEL) -> Iterator[str]:
    """
    Yields tokens one-by-one for SSE streaming.
    Retrieval runs first (blocking), then inference streams.
    """
    validate_model(model)
    import json

    # Step 1: retrieve (fast)
    yield "data: __RETRIEVING__\n\n"
    examples = find_examples(input_text)
    prompt = build_prompt(input_text, examples)

    # Step 2: stream inference
    yield "data: __GENERATING__\n\n"

    llm = get_llm(model)

    full_text = ""
    for chunk in llm.stream(prompt):
        full_text += chunk
        yield f"data: __TOKEN__{json.dumps(chunk, ensure_ascii=False)}\n\n"

    # Step 3: evaluation — از ارزیاب واحد سیستم استفاده می‌کنیم
    humanized = clean(full_text, input_text)
    if not humanized:
        raise RuntimeError(
            "The selected model returned empty output. Please try again."
        )
    yield f"data: __TEXT__{json.dumps(humanized, ensure_ascii=False)}\n\n"
    from Evaluator import evaluate_humanization

    ev = evaluate_humanization(input_text, humanized)
    import json

    yield f"data: __EVAL__{json.dumps(ev, ensure_ascii=False)}\n\n"
    yield "data: __DONE__\n\n"

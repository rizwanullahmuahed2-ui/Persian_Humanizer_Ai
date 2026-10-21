"""Content-based example retrieval for the single natural-rewriting workflow."""

from __future__ import annotations

import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from typing import List, Dict, Any, TYPE_CHECKING
from rewriting import format_examples_for_prompt

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import Chroma

# ── Paths ──────────────────────────────────────────────────────────────────────
# اگر USE_TRAIN_SET=1 باشد، فقط روی train_set ایندکس ساخته می‌شود (برای ارزیابی بدون leakage).
# در حالت عادی روی کل دیتاست ایندکس می‌سازد.
_USE_TRAIN_SET = os.environ.get("USE_TRAIN_SET", "0") == "1"

if _USE_TRAIN_SET:
    DATASET_PATH = os.path.join(os.path.dirname(__file__), "../data/train_set.csv")
    CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db_train")
    print(
        "[RAG] حالت ارزیابی: ایندکس فقط روی train_set ساخته می‌شود (test جدا نگه داشته شده)"
    )
else:
    DATASET_PATH = os.path.join(
        os.path.dirname(__file__), "../data/persian_humanizer_dataset.csv"
    )
    CHROMA_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db_v2")

# ── Models ─────────────────────────────────────────────────────────────────────
# Embedding: only on AI text (not paired, no chunking)
# سند (مورد ۴): MiniLM قدیمی است ولی بد نیست. با EMBEDDING_MODEL می‌توان
# E5 یا BGE را امتحان کرد — ولی اول با ab_test_retrieval.py اندازه بگیر.
EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Reranker: cross-encoder for precision scoring
# سند (مورد ۳): cross-encoder فعلی روی داده انگلیسی آموزش دیده.
# با RERANKER_MODEL می‌توان نسخه چندزبانه (مثلاً BAAI/bge-reranker-v2-m3) را امتحان کرد.
# پیش‌فرض همان نسخه فعلی است تا رفتار عوض نشود مگر اینکه عمداً تغییر دهی.
RERANKER_MODEL = os.environ.get(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

_embeddings = None
_reranker = None
_vectorstore = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_reranker():
    """Load cross-encoder reranker (lazy). Falls back to None if not installed."""
    global _reranker
    if _reranker is False:  # already tried and failed
        return None
    if _reranker is not None:
        return _reranker
    try:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_MODEL)
        print("[RAG] Reranker loaded ✓")
    except Exception as e:
        print(f"[RAG] Reranker unavailable ({e}) — using raw similarity scores")
        _reranker = False
    return _reranker if _reranker is not False else None


# ── Document loading — NO chunking, embed only AI text ─────────────────────────
def load_documents(csv_path: str) -> List[Document]:
    import pandas as pd
    from langchain_core.documents import Document

    df = pd.read_csv(csv_path)
    docs = []
    for _, row in df.iterrows():
        ai = str(row.get("ai", "")).strip()
        human = str(row.get("human", "")).strip()
        style = str(row.get("style", "casual")).strip()
        transformation = str(row.get("transformation", "")).strip()
        notes = str(row.get("notes", "")).strip()

        if not ai or not human:
            continue

        # ★ Page content = ONLY the AI text (what we search against)
        #   No pairing, no chunking — keeps vectors clean
        docs.append(
            Document(
                page_content=ai,
                metadata={
                    "ai": ai,
                    "human": human,
                    "style": style,
                    "transformation": transformation,
                    "notes": notes,
                },
            )
        )
    print(f"[RAG] Loaded {len(docs)} documents (no chunking)")
    return docs


def build_vectorstore(docs: List[Document], persist_dir: str) -> Chroma:
    from langchain_community.vectorstores import Chroma

    vs = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
    )
    print(f"[RAG] Vectorstore built → {persist_dir}")
    return vs


def load_vectorstore(persist_dir: str) -> Chroma:
    from langchain_community.vectorstores import Chroma

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings(),
    )


def get_or_build_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        print("[RAG] Loading existing vectorstore (v2)...")
        _vectorstore = load_vectorstore(CHROMA_DIR)
    else:
        print("[RAG] Building vectorstore v2 ...")
        docs = load_documents(DATASET_PATH)
        _vectorstore = build_vectorstore(docs, CHROMA_DIR)
    return _vectorstore


# ── Stage 1: Coarse retrieval with style filter ────────────────────────────────
def _coarse_retrieve(
    vs: Chroma,
    query: str,
    style: str | None,
    top_n: int = 20,
) -> List[Dict[str, Any]]:
    """Retrieve candidates by content; optional labels are for offline comparisons."""

    # Production uses semantic similarity only. An explicit label remains
    # available to the historical retrieval-comparison scripts.
    if style is None:
        results = vs.similarity_search_with_score(query, k=top_n)
    else:
        try:
            results = vs.similarity_search_with_score(
                query, k=top_n, filter={"style": style}
            )
            if len(results) < 4:
                raise ValueError("too few style matches")
        except Exception:
            results = vs.similarity_search_with_score(query, k=top_n)

    candidates = []
    seen = set()
    for doc, score in results:
        ai = doc.metadata.get("ai", "")
        if ai in seen or not ai:
            continue
        seen.add(ai)

        # Style bonus: boost score if style matches
        style_match = style is not None and doc.metadata.get("style", "") == style
        adjusted_score = float(score) * (0.85 if style_match else 1.0)  # lower = better

        candidates.append(
            {
                "ai": ai,
                "human": doc.metadata.get("human", ""),
                "style": doc.metadata.get("style", ""),
                "transformation": doc.metadata.get("transformation", ""),
                "notes": doc.metadata.get("notes", ""),
                "raw_score": float(score),
                "adjusted_score": adjusted_score,
                "style_match": style_match,
            }
        )

    # Sort by adjusted score (lower distance = better)
    candidates.sort(key=lambda x: x["adjusted_score"])
    return candidates


# ── Stage 2: Reranker ──────────────────────────────────────────────────────────
def _rerank(query: str, candidates: List[Dict], top_k: int = 4) -> List[Dict]:
    """Cross-encoder reranking for precision. Falls back to raw order."""
    reranker = get_reranker()
    if reranker is None or len(candidates) <= top_k:
        return candidates[:top_k]

    try:
        pairs = [(query, c["ai"]) for c in candidates]
        scores = reranker.predict(pairs)
        for i, c in enumerate(candidates):
            c["rerank_score"] = float(scores[i])
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        print(f"[RAG] Reranked {len(candidates)} → top {top_k}")
    except Exception as e:
        print(f"[RAG] Rerank failed ({e}), using coarse order")

    return candidates[:top_k]


# ── Public API ─────────────────────────────────────────────────────────────────
def retrieve_examples(
    vectorstore: Chroma,
    query: str,
    k: int = 4,
    style: str | None = None,
) -> List[Dict[str, Any]]:
    """Full pipeline: coarse top-20 → reranker → top-k."""
    candidates = _coarse_retrieve(vectorstore, query, style, top_n=20)
    final = _rerank(query, candidates, top_k=k)
    print(f"[RAG] Retrieved {len(final)} examples (style={style})")
    return final


# ── Index rebuild helper ───────────────────────────────────────────────────────
def rebuild_index() -> Dict[str, int]:
    import shutil

    global _vectorstore
    _vectorstore = None
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    docs = load_documents(DATASET_PATH)
    build_vectorstore(docs, CHROMA_DIR)
    _vectorstore = load_vectorstore(CHROMA_DIR)
    return {"documents": len(docs)}

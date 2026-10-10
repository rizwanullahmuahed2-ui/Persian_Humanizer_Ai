"""
موارد ۳ و ۴ سند — آزمایش Reranker و Embedding
سند درست می‌گوید: «اول اندازه بگیر، بدون آزمایش تغییر نده.»

این اسکریپت کیفیت retrieval را بدون نیاز به LLM می‌سنجد.
ایده: از خود دیتاست به‌عنوان ground-truth استفاده می‌کنیم.

روش سنجش (بدون قضاوت انسانی، کاملاً عینی):
  برای هر کوئری در test، می‌دانیم جفت درست (human مرجع) چیست.
  retrieval را اجرا می‌کنیم و می‌بینیم آیا نمونه‌های بازیابی‌شده
  از نظر سبک و transformation با کوئری همخوان‌اند یا نه.

سنجه‌ها:
  - style_match@k : چند درصد از نمونه‌های بازیابی‌شده همان style کوئری را دارند
  - transform_match@k : همان transformation
  - avg_similarity : میانگین شباهت معنایی بازیابی‌ها

این به تو می‌گوید کدام embedding / reranker retrieval بهتری دارد،
بدون اینکه حتی یک‌بار LLM را صدا بزنی (سریع و ارزان).

مدل‌های قابل مقایسه:
  Embeddings:
    paraphrase-multilingual-MiniLM-L12-v2  (فعلی)
    intfloat/multilingual-e5-base          (E5)
    BAAI/bge-m3                             (BGE)
  Rerankers:
    cross-encoder/ms-marco-MiniLM-L-6-v2   (فعلی، انگلیسی)
    BAAI/bge-reranker-v2-m3                 (چندزبانه)
"""

import os
import sys
import pandas as pd
from typing import List, Dict

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "../backend"))

TEST_SET = os.path.join(HERE, "test_set.csv")
RESULTS_OUT = os.path.join(HERE, "retrieval_ab_results.csv")

EMBEDDING_CANDIDATES = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "e5":     "intfloat/multilingual-e5-base",
    "bge":    "BAAI/bge-m3",
}

LIMIT = int(os.environ.get("AB_LIMIT", "50"))
TOP_K = 4


def build_temp_index(embedding_model: str, docs: List, persist_dir: str):
    """ساخت ایندکس موقت با یک embedding مشخص."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    import shutil

    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    emb = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vs = Chroma.from_documents(documents=docs, embedding=emb, persist_directory=persist_dir)
    return vs, emb


def evaluate_embedding(name: str, model: str, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict:
    """یک embedding را روی test می‌سنجد."""
    from langchain_core.documents import Document

    print(f"\n=== آزمایش embedding: {name} ({model}) ===")
    try:
        docs = [
            Document(page_content=str(r["ai"]).strip(),
                     metadata={"style": str(r.get("style", "")), "transformation": str(r.get("transformation", ""))})
            for _, r in train_df.iterrows()
            if str(r.get("ai", "")).strip()
        ]
        persist = os.path.join(HERE, f"_tmp_index_{name}")
        vs, emb = build_temp_index(model, docs, persist)
    except Exception as e:
        print(f"  ❌ بارگذاری {name} شکست خورد: {e}")
        return {"embedding": name, "error": str(e)}

    style_hits = 0
    transform_hits = 0
    total_retrieved = 0

    for _, row in test_df.iterrows():
        query = str(row["ai"]).strip()
        q_style = str(row.get("style", ""))
        q_transform = str(row.get("transformation", ""))
        if not query:
            continue
        try:
            results = vs.similarity_search(query, k=TOP_K)
        except Exception:
            continue
        for doc in results:
            total_retrieved += 1
            if doc.metadata.get("style") == q_style:
                style_hits += 1
            if doc.metadata.get("transformation") == q_transform:
                transform_hits += 1

    import shutil
    shutil.rmtree(persist, ignore_errors=True)

    res = {
        "embedding": name,
        "model": model,
        "style_match@k": round(style_hits / max(total_retrieved, 1), 3),
        "transform_match@k": round(transform_hits / max(total_retrieved, 1), 3),
        "n_queries": len(test_df),
    }
    print(f"  style_match@{TOP_K}:     {res['style_match@k']}")
    print(f"  transform_match@{TOP_K}: {res['transform_match@k']}")
    return res


def main():
    if not os.path.exists(TEST_SET):
        print(f"❌ {TEST_SET} پیدا نشد. اول build_test_set.py را اجرا کن.")
        return

    train_path = os.path.join(HERE, "../data/train_set.csv")
    if not os.path.exists(train_path):
        print(f"❌ {train_path} پیدا نشد. اول build_test_set.py را اجرا کن.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(TEST_SET)
    if LIMIT:
        test_df = test_df.iloc[:LIMIT]

    print(f"Train: {len(train_df)} | Test (queries): {len(test_df)}")

    results = []
    for name, model in EMBEDDING_CANDIDATES.items():
        results.append(evaluate_embedding(name, model, train_df, test_df))

    rdf = pd.DataFrame(results)
    rdf.to_csv(RESULTS_OUT, index=False, encoding="utf-8-sig")
    print(f"\n✅ نتایج: {RESULTS_OUT}")
    print("\n" + rdf.to_string())
    print("\n⚠️ سند: اگر تفاوت کم بود، اصلاً تغییر نده. اول اندازه بگیر.")


if __name__ == "__main__":
    main()

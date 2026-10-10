"""
مورد ۷ سند — Retrieval روی AI Text در برابر Human در برابر ترکیب
سند نکته‌ی درستی می‌گوید:
  ممکن است AI1 و AI2 شبیه نباشند ولی Human1 و Human2 خیلی شبیه باشند.
  سیستم فعلی که فقط روی AI ایندکس می‌سازد، این را نمی‌فهمد.

این اسکریپت سه استراتژی page_content را مقایسه می‌کند:
  A) page_content = ai            (فعلی)
  B) page_content = human
  C) page_content = ai + " " + human  (ترکیب)

سنجه (بدون LLM، عینی):
  وقتی با متن AI کوئری می‌زنیم، آیا نمونه‌ای که بازیابی می‌شود
  از نظر style/transformation با کوئری همخوان است؟

نکته مهم: کوئری واقعی کاربر یک «متن AI/رسمی» است (نه human).
پس برای حالت B و C باید دقت کنیم که آیا embedding متن human
هنوز با کوئری AI همخوانی معنایی دارد یا نه.
"""

import os
import sys
import shutil
import pandas as pd
from typing import List, Dict

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "../backend"))

TEST_SET = os.path.join(HERE, "test_set.csv")
TRAIN_SET = os.path.join(HERE, "../data/train_set.csv")
RESULTS_OUT = os.path.join(HERE, "content_strategy_results.csv")

EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LIMIT = int(os.environ.get("CONTENT_AB_LIMIT", "50"))
TOP_K = 4


def make_docs(train_df: pd.DataFrame, strategy: str) -> List:
    from langchain_core.documents import Document
    docs = []
    for _, r in train_df.iterrows():
        ai = str(r.get("ai", "")).strip()
        human = str(r.get("human", "")).strip()
        if not ai or not human:
            continue
        if strategy == "ai":
            content = ai
        elif strategy == "human":
            content = human
        elif strategy == "combined":
            content = ai + " " + human
        else:
            raise ValueError(strategy)
        docs.append(Document(
            page_content=content,
            metadata={"style": str(r.get("style", "")),
                      "transformation": str(r.get("transformation", "")),
                      "ai": ai, "human": human},
        ))
    return docs


def eval_strategy(strategy: str, train_df: pd.DataFrame, test_df: pd.DataFrame, emb) -> Dict:
    from langchain_community.vectorstores import Chroma
    print(f"\n=== استراتژی: page_content = {strategy} ===")

    persist = os.path.join(HERE, f"_tmp_content_{strategy}")
    if os.path.exists(persist):
        shutil.rmtree(persist)

    docs = make_docs(train_df, strategy)
    vs = Chroma.from_documents(documents=docs, embedding=emb, persist_directory=persist)

    style_hits = transform_hits = total = 0
    for _, row in test_df.iterrows():
        query = str(row["ai"]).strip()   # کوئری همیشه متن AI/رسمی است
        if not query:
            continue
        q_style = str(row.get("style", ""))
        q_transform = str(row.get("transformation", ""))
        try:
            results = vs.similarity_search(query, k=TOP_K)
        except Exception:
            continue
        for doc in results:
            total += 1
            if doc.metadata.get("style") == q_style:
                style_hits += 1
            if doc.metadata.get("transformation") == q_transform:
                transform_hits += 1

    shutil.rmtree(persist, ignore_errors=True)
    res = {
        "strategy": strategy,
        "style_match@k": round(style_hits / max(total, 1), 3),
        "transform_match@k": round(transform_hits / max(total, 1), 3),
    }
    print(f"  style_match@{TOP_K}:     {res['style_match@k']}")
    print(f"  transform_match@{TOP_K}: {res['transform_match@k']}")
    return res


def main():
    if not (os.path.exists(TEST_SET) and os.path.exists(TRAIN_SET)):
        print("❌ اول build_test_set.py را اجرا کن.")
        return

    from langchain_huggingface import HuggingFaceEmbeddings
    emb = HuggingFaceEmbeddings(
        model_name=EMBEDDING,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    train_df = pd.read_csv(TRAIN_SET)
    test_df = pd.read_csv(TEST_SET)
    if LIMIT:
        test_df = test_df.iloc[:LIMIT]

    results = [eval_strategy(s, train_df, test_df, emb) for s in ["ai", "human", "combined"]]
    rdf = pd.DataFrame(results)
    rdf.to_csv(RESULTS_OUT, index=False, encoding="utf-8-sig")

    print(f"\n✅ نتایج: {RESULTS_OUT}")
    print("\n" + rdf.to_string())
    print("\n⚠️ اگر 'ai' بهترین یا تقریباً برابر بود، تغییر نده (سند: اول اندازه بگیر).")


if __name__ == "__main__":
    main()

"""
مورد ۱ سند — ساخت Test Set جدا
این اسکریپت ۲۰۰ نمونه از دیتاست را جدا می‌کند که هرگز در RAG استفاده نشوند.
خروجی: test_set.csv (held-out)  +  train_set.csv (برای ساخت ایندکس RAG)

نکته مهم: بعد از این، باید rag_engine را وادار کنیم که فقط روی train_set ایندکس بسازد،
نه روی کل دیتاست. (پایین‌تر در rag_engine یک متغیر محیطی اضافه شده.)
"""

import os
import pandas as pd

HERE = os.path.dirname(__file__)
FULL_DATASET = os.path.join(HERE, "../data/persian_humanizer_dataset.csv")
TEST_OUT     = os.path.join(HERE, "test_set.csv")
TRAIN_OUT    = os.path.join(HERE, "../data/train_set.csv")

N_TEST = 200
SEED = 42


def main():
    df = pd.read_csv(FULL_DATASET)
    print(f"کل دیتاست: {len(df)} نمونه")

    # نمونه‌گیری طبقه‌ای بر اساس transformation تا توزیع test متعادل بماند
    # هر گروه سهمی متناسب با اندازه‌اش در test می‌گیرد
    test_frac = N_TEST / len(df)
    test_parts = []
    for name, group in df.groupby("transformation"):
        n = max(1, round(len(group) * test_frac))
        n = min(n, len(group))
        test_parts.append(group.sample(n=n, random_state=SEED))

    test_df = pd.concat(test_parts).sample(frac=1, random_state=SEED)
    # دقیقاً N_TEST نمونه نگه دار
    if len(test_df) > N_TEST:
        test_df = test_df.iloc[:N_TEST]

    test_idx = set(test_df.index)
    train_df = df[~df.index.isin(test_idx)].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    test_df.to_csv(TEST_OUT, index=False, encoding="utf-8-sig")
    train_df.to_csv(TRAIN_OUT, index=False, encoding="utf-8-sig")

    print(f"✅ Test set:  {len(test_df)} نمونه → {TEST_OUT}")
    print(f"✅ Train set: {len(train_df)} نمونه → {TRAIN_OUT}")
    print("\nتوزیع transformation در test:")
    print(test_df["transformation"].value_counts().head(15).to_string())
    print("\n⚠️  حالا برای ساخت ایندکس RAG باید train_set استفاده شود نه کل دیتاست.")
    print("   متغیر محیطی USE_TRAIN_SET=1 را ست کن و /rebuild-index بزن.")


if __name__ == "__main__":
    main()

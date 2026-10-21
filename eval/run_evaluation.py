"""
مورد ۱ سند — Evaluation واقعی
این اسکریپت خروجی سیستم خودت را روی test_set تولید می‌کند و یک فایل می‌سازد
که آماده‌ی نمره‌دهی انسانی است (حفظ معنا / طبیعی بودن / دری بودن).

نکته سند: «امتیازها با کیفیت واقعی خروجی رابطه محکمی ندارند.»
راه حل: خروجی را کنار متن اصلی بگذار و از انسان نمره بگیر.

این اسکریپت دو خروجی می‌سازد:
  ۱. human_eval_sheet.csv  → برای نمره‌دهی دستی انسان (ستون‌های نمره خالی)
  ۲. auto_metrics.csv      → سنجه‌های خودکار (rule-based + dari benchmark) برای مقایسه

ChatGPT/Claude را نمی‌توان از اینجا صدا زد (نیاز به API key دارد). اگر خواستی،
ستون خالی gpt_output و claude_output گذاشته شده تا دستی پر کنی و مقایسه‌ی سه‌طرفه داشته باشی.
"""

import os
import sys
import csv
import pandas as pd

HERE = os.path.dirname(__file__)
BACKEND = os.path.join(HERE, "../backend")
sys.path.insert(0, BACKEND)
sys.path.insert(0, HERE)

TEST_SET = os.path.join(HERE, "test_set.csv")
HUMAN_SHEET = os.path.join(HERE, "human_eval_sheet.csv")
AUTO_METRICS = os.path.join(HERE, "auto_metrics.csv")

# محدودسازی تعداد برای اجرای سریع‌تر؛ None یعنی همه
LIMIT = int(os.environ.get("EVAL_LIMIT", "30"))


def run():
    if not os.path.exists(TEST_SET):
        print(f"❌ {TEST_SET} پیدا نشد. اول build_test_set.py را اجرا کن.")
        return

    df = pd.read_csv(TEST_SET)
    if LIMIT:
        df = df.iloc[:LIMIT]
    print(f"ارزیابی روی {len(df)} نمونه از test_set...")

    # وارد کردن سیستم (با USE_TRAIN_SET=1 تا leakage نباشد)
    os.environ["USE_TRAIN_SET"] = "1"
    try:
        from backend.humanizer_graph  import run_humanizer
    except Exception as e:
        print(f"❌ بارگذاری سیستم شکست خورد: {e}")
        print("   مطمئن شو Ollama و مدل llama3.1:8b در دسترس‌اند.")
        return

    from dari_benchmark import dari_score

    human_rows = []
    auto_rows = []

    for idx, row in df.iterrows():
        original = str(row["ai"]).strip()
        reference = str(row.get("human", "")).strip()  # خروجی مرجع دیتاست
        style = str(row.get("style", "casual")).strip()

        print(f"  [{idx+1}/{len(df)}] {original[:40]}...")
        try:
            result = run_humanizer(original)
            system_out = result.get("humanized_text", "").strip()
            rule_eval = result.get("evaluation", {})
        except Exception as e:
            system_out = f"[خطا: {e}]"
            rule_eval = {}

        d = dari_score(system_out)

        # برگه نمره‌دهی انسانی (ستون‌های نمره خالی برای پر کردن دستی)
        human_rows.append({
            "id": idx,
            "dataset_style": style,
            "متن_اصلی": original,
            "خروجی_سیستم": system_out,
            "خروجی_مرجع": reference,
            "gpt_output": "",      # دستی پر کن اگر خواستی
            "claude_output": "",   # دستی پر کن اگر خواستی
            "نمره_حفظ_معنا_1تا5": "",
            "نمره_طبیعی_بودن_1تا5": "",
            "نمره_دری_بودن_1تا5": "",
            "توضیح_ارزیاب": "",
        })

        # سنجه‌های خودکار
        auto_rows.append({
            "id": idx,
            "dataset_style": style,
            "rule_overall": rule_eval.get("overall_score", ""),
            "rule_meaning": rule_eval.get("meaning_preserved", ""),
            "rule_naturalness": rule_eval.get("naturalness", ""),
            "rule_formality": rule_eval.get("formality_reduced", ""),
            "dari_ratio": d["dari_ratio"],
            "dari_count": d["dari_count"],
            "iranian_count": d["iranian_count"],
        })

    # ذخیره
    pd.DataFrame(human_rows).to_csv(HUMAN_SHEET, index=False, encoding="utf-8-sig")
    pd.DataFrame(auto_rows).to_csv(AUTO_METRICS, index=False, encoding="utf-8-sig")

    print(f"\n✅ برگه نمره‌دهی انسانی: {HUMAN_SHEET}")
    print(f"✅ سنجه‌های خودکار:     {AUTO_METRICS}")

    # خلاصه سنجه خودکار
    adf = pd.DataFrame(auto_rows)
    print("\n── خلاصه سنجه‌های خودکار ──")
    print(f"میانگین dari_ratio:  {adf['dari_ratio'].mean():.3f}")
    nums = pd.to_numeric(adf["rule_overall"], errors="coerce")
    if nums.notna().any():
        print(f"میانگین rule_overall: {nums.mean():.2f}")
    print("\n⚠️ یادآوری سند: عدد rule_overall را خودت ساخته‌ای.")
    print("   نمره‌دهی انسانی در human_eval_sheet معیار واقعی است.")


if __name__ == "__main__":
    run()

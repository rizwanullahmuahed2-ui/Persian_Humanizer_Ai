"""
مورد ۵ سند — Benchmark دری
سند درست می‌گوید: یک لیست ساده dari_words معیار علمی نیست.

این ماژول یک سنجه‌ی شفاف و قابل‌دفاع می‌سازد:
  ۱. مجموعه‌ای از مارکرهای واژگانی/دستوری که در دری افغانستان رایج‌اند ولی در فارسی ایران کمتر.
  ۲. مجموعه‌ای از مارکرهای مختص فارسی ایران.
  ۳. یک تابع نمره‌دهی که نسبت مارکرهای دری به کل مارکرها را برمی‌گرداند (۰ تا ۱).

این هنوز یک سنجه‌ی واژگانیِ ناقص است (نه قضاوت انسانی)، ولی:
  - شفاف است (هر تصمیم قابل ردگیری است)
  - دوطرفه است (هم دری هم ایرانی را می‌سنجد، نه فقط حضور چند کلمه)
  - قابل گسترش است

⚠️ هشدار صداقت پژوهشی: این سنجه جایگزین ارزیابی انسانی نیست.
   برای ادعای واقعی «دری بودن» در مونوگراف، باید نتایج این benchmark را
   در کنار قضاوت چند ارزیاب انسانیِ دری‌زبان گزارش کنی.
"""

import re
from typing import Dict, List

# ── مارکرهای دری افغانستان ─────────────────────────────────────────────────────
# این‌ها صورت‌هایی هستند که در گفتار/نوشتار دری افغانستان رایج‌ترند.
DARI_MARKERS: List[str] = [
    # افعال کمکی و ربطی
    "استم", "استی", "استین", "استند", "می‌باشم",
    # ضمایر و اشارات
    "ازی", "ده", "دی",
    # واژگان رایج دری
    "بسیار", "خوب", "صنف", "پوهنتون", "پوهنحی", "فاکولته",
    "موتر", "بایسکل", "کمپیوتر", "تلویزون",
    "نل", "چاینک", "بوتل", "پطنوس",
    "روغن", "پیاز", "بادنجان", "کچالو",
    "شفاخانه", "دواخانه", "داکتر", "نرس",
    "مکتب", "معلم", "شاگرد", "کتابچه", "قلم",
    "اشتها", "نان چاشت", "نان شب", "ناشتا",
    "چطورستی", "خوب استی", "زنده باشی", "تشکر",
    "بچیم", "دخترم", "لالا", "آغه",
    "کته", "خورد", "کلان",
]

# ── مارکرهای فارسی ایران ────────────────────────────────────────────────────────
IRANIAN_MARKERS: List[str] = [
    # محاوره‌ای ایرانی
    "میشه", "نمیشه", "بریم", "بخوای", "میخوام", "میدم",
    "چطوری", "خوبی", "مرسی", "ممنون",
    "بچه", "پسر", "دختر",
    "ماشین", "دوچرخه", "کامپیوتر", "تلویزیون",
    "شیر آب", "قوری", "بطری", "سینی",
    "بیمارستان", "داروخانه", "دکتر", "پرستار",
    "مدرسه", "معلم", "دانش‌آموز", "دفتر", "خودکار",
    "دانشگاه", "دانشکده",
    "اشتها", "ناهار", "شام", "صبحانه",
    "خیلی", "زیاد", "کوچیک", "بزرگ",
    "گوجه", "پیاز", "بادمجان", "سیب‌زمینی",
]

# نرمال‌سازی برای حذف تفاوت نیم‌فاصله/فاصله
def _normalize(text: str) -> str:
    text = text.replace("\u200c", " ")  # نیم‌فاصله → فاصله
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_markers(text: str, markers: List[str]) -> int:
    norm = _normalize(text)
    count = 0
    for m in markers:
        m_norm = _normalize(m)
        # شمارش تعداد تکرار هر مارکر
        count += len(re.findall(re.escape(m_norm), norm))
    return count


def dari_score(text: str) -> Dict[str, float]:
    """
    نمره دری بودن متن.
    خروجی:
      dari_count    : تعداد مارکرهای دری
      iranian_count : تعداد مارکرهای ایرانی
      dari_ratio    : نسبت دری به کل مارکرها (۰ تا ۱). اگر هیچ مارکری نبود → 0.5 (نامشخص)
    """
    d = count_markers(text, DARI_MARKERS)
    i = count_markers(text, IRANIAN_MARKERS)
    total = d + i
    ratio = (d / total) if total > 0 else 0.5
    return {
        "dari_count": d,
        "iranian_count": i,
        "total_markers": total,
        "dari_ratio": round(ratio, 3),
    }


def evaluate_corpus(texts: List[str]) -> Dict[str, float]:
    """میانگین نمره دری روی یک مجموعه متن."""
    if not texts:
        return {"avg_dari_ratio": 0.0, "n": 0}
    scores = [dari_score(t)["dari_ratio"] for t in texts]
    return {
        "avg_dari_ratio": round(sum(scores) / len(scores), 3),
        "n": len(texts),
        "fully_dari": sum(1 for s in scores if s >= 0.8),
        "fully_iranian": sum(1 for s in scores if s <= 0.2),
        "mixed": sum(1 for s in scores if 0.2 < s < 0.8),
    }


# ── ارزیابی زنده‌ی یک جفت (متن اصلی → متن انسانی‌شده) ──────────────────────────
# این تابع قلب endpoint زنده /evaluate است. چند سنجه‌ی خودکار را یکجا برمی‌گرداند.

# واژه‌های رسمی که در محاوره معمولاً ساده می‌شوند (برای سنجش کاهش رسمیت)
_FORMAL_MARKERS = [
    "می‌باشد", "میباشد", "نمی‌باشد", "گردید", "گردد", "می‌گردد",
    "نمود", "نماید", "می‌نماید", "بایست", "می‌بایست", "مذکور",
    "فوق‌الذکر", "لذا", "بدین‌وسیله", "جهت", "درخصوص", "درراستای",
    "اتخاذ", "اعمال", "صورت می‌پذیرد", "می‌پذیرد", "واقع", "حائز",
]

# نشانه‌های محاوره‌ای فارسی (برای سنجش طبیعی/خودمانی شدن)
_CASUAL_MARKERS = [
    "میشه", "نمیشه", "بریم", "بخوای", "میخوام", "می‌خوام", "میدم",
    "میده", "میگه", "میگم", "اینجوری", "اونجوری", "چجوری", "یه ",
    "خیلی", "حالا", "دیگه", "آره", "نه ", "بابا", "خب",
]


def _count_any(text: str, markers) -> int:
    norm = _normalize(text)
    return sum(len(re.findall(re.escape(_normalize(m)), norm)) for m in markers)


def evaluate_humanization(original: str, humanized: str) -> Dict:
    """
    ارزیابی خودکار یک جفت (متن اصلی → خروجی).
    همه‌ی سنجه‌ها عینی و قابل‌محاسبه‌اند (بدون LLM).

    خروجی شامل:
      meaning_keep   : تخمین حفظ معنا بر اساس همپوشانی واژگانی (۰ تا ۱)
      formality_drop : چقدر واژه‌های رسمی کم شده‌اند (۰ تا ۱، بیشتر = بهتر)
      casualness     : چگالی نشانه‌های محاوره‌ای در خروجی
      dari           : نتیجه‌ی بنچمارک دری (dict)
      length_ratio   : نسبت طول خروجی به ورودی (نزدیک ۱ یعنی چیزی اضافه/حذف نشده)
      overall        : میانگین وزنیِ سنجه‌ها (۰ تا ۱) — فقط راهنما، نه قضاوت نهایی
    """
    orig = (original or "").strip()
    hum  = (humanized or "").strip()

    # ۱. حفظ معنا: همپوشانی کلمات محتوایی (Jaccard ساده روی توکن‌ها)
    def tokens(t):
        return set(re.findall(r"[آ-ی‌]+", _normalize(t)))
    to, th = tokens(orig), tokens(hum)
    if to:
        overlap = len(to & th) / len(to)
    else:
        overlap = 0.0
    meaning_keep = round(overlap, 3)

    # ۲. کاهش رسمیت
    f_orig = _count_any(orig, _FORMAL_MARKERS)
    f_hum  = _count_any(hum, _FORMAL_MARKERS)
    if f_orig > 0:
        formality_drop = round(max(0.0, (f_orig - f_hum) / f_orig), 3)
    else:
        formality_drop = 1.0 if f_hum == 0 else 0.0

    # ۳. محاوره‌ای بودن خروجی (چگالی نشانه‌ها بر حسب تعداد کلمات)
    n_words = max(1, len(hum.split()))
    casual_hits = _count_any(hum, _CASUAL_MARKERS)
    casualness = round(min(1.0, casual_hits / (n_words / 5)), 3)

    # ۴. دری
    dari = dari_score(hum)

    # ۵. نسبت طول
    lo, lh = len(orig.split()), len(hum.split())
    length_ratio = round(lh / max(1, lo), 3)

    # ۶. نمره‌ی کلی (وزنی) — صرفاً راهنما
    overall = round(
        0.40 * meaning_keep +
        0.25 * formality_drop +
        0.20 * casualness +
        0.15 * (1.0 - abs(1.0 - min(length_ratio, 2.0)))  # نزدیک ۱ بهتر
        , 3
    )

    return {
        "meaning_keep":   meaning_keep,
        "formality_drop": formality_drop,
        "casualness":     casualness,
        "dari_ratio":     dari["dari_ratio"],
        "dari_detail":    dari,
        "length_ratio":   length_ratio,
        "overall":        overall,
        "note": "سنجه‌های خودکار راهنما هستند، نه جایگزین قضاوت انسانی.",
    }


if __name__ == "__main__":
    # تست سریع
    o = "هوش مصنوعی در سال‌های اخیر رشد چشمگیری داشته است و می‌باشد."
    h = "هوش مصنوعی این سال‌ها خیلی رشد کرده."
    print("نمونه ارزیابی:")
    import json
    print(json.dumps(evaluate_humanization(o, h), ensure_ascii=False, indent=2))
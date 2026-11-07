"""Lexical dialect diagnostics, not a substitute for a Dari speaker reviewing the text.

Regional word lists are comparison data; shared words and substring matches
make these scores approximate. The Iranian list is not user-facing copy.
"""

import re
from typing import Dict, List

# این‌ها صورت‌هایی هستند که در گفتار/نوشتار دری افغانستان رایج‌ترند.
DARI_MARKERS: List[str] = [
    # افعال کمکی و ربطی
    "استم",
    "استی",
    "استین",
    "استند",
    "می‌باشم",
    # ضمایر و اشارات
    "ازی",
    "ده",
    "دی",
    # واژگان رایج دری
    "بسیار",
    "خوب",
    "صنف",
    "پوهنتون",
    "پوهنحی",
    "فاکولته",
    "موتر",
    "بایسکل",
    "کمپیوتر",
    "تلویزون",
    "نل",
    "چاینک",
    "بوتل",
    "پطنوس",
    "روغن",
    "پیاز",
    "بادنجان",
    "کچالو",
    "شفاخانه",
    "دواخانه",
    "داکتر",
    "نرس",
    "مکتب",
    "معلم",
    "شاگرد",
    "کتابچه",
    "قلم",
    "اشتها",
    "نان چاشت",
    "نان شب",
    "ناشتا",
    "چطورستی",
    "خوب استی",
    "زنده باشی",
    "تشکر",
    "بچیم",
    "دخترم",
    "لالا",
    "آغه",
    "کته",
    "خورد",
    "کلان",
]

IRANIAN_MARKERS: List[str] = [
    # محاوره‌ای ایرانی
    "میشه",
    "نمیشه",
    "بریم",
    "بخوای",
    "میخوام",
    "میدم",
    "چطوری",
    "خوبی",
    "مرسی",
    "ممنون",
    "بچه",
    "پسر",
    "دختر",
    "ماشین",
    "دوچرخه",
    "کامپیوتر",
    "تلویزیون",
    "شیر آب",
    "قوری",
    "بطری",
    "سینی",
    "بیمارستان",
    "داروخانه",
    "دکتر",
    "پرستار",
    "مدرسه",
    "معلم",
    "دانش‌آموز",
    "دفتر",
    "خودکار",
    "دانشگاه",
    "دانشکده",
    "اشتها",
    "ناهار",
    "شام",
    "صبحانه",
    "خیلی",
    "زیاد",
    "کوچیک",
    "بزرگ",
    "گوجه",
    "پیاز",
    "بادمجان",
    "سیب‌زمینی",
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


if __name__ == "__main__":
    # تست سریع
    samples = [
        "بسیار خوب استی؟ تشکر که آمدی.",  # دری
        "خیلی خوبی؟ مرسی که اومدی.",  # ایرانی
        "هوش مصنوعی رشد کرده است.",  # خنثی
    ]
    for s in samples:
        print(f"{s!r}")
        print("  →", dari_score(s))
    print("\nمجموعه:", evaluate_corpus(samples))

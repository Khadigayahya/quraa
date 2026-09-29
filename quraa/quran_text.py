"""Quran text index: how much of a transcript is literally Quran?

Works on Whisper output, so it must survive spelling noise: text is normalised
(no diacritics, unified alef/ya/ta-marbuta/hamza) and compared as word n-grams.
"""
import os
import re
from functools import lru_cache
from pathlib import Path

QURAN_FILE = Path(os.environ.get("QURAA_QURAN_FILE") or Path(__file__).parent / "assets" / "quran-simple-clean.txt")

_DIAC = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_NON_AR = re.compile("[^\u0621-\u064A ]+")
_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي", "ء": ""})


SURA_NAMES = (
    "الفاتحة البقرة آل_عمران النساء المائدة الأنعام الأعراف الأنفال التوبة يونس هود يوسف الرعد إبراهيم "
    "الحجر النحل الإسراء الكهف مريم طه الأنبياء الحج المؤمنون النور الفرقان الشعراء النمل القصص العنكبوت "
    "الروم لقمان السجدة الأحزاب سبأ فاطر يس الصافات ص الزمر غافر فصلت الشورى الزخرف الدخان الجاثية الأحقاف "
    "محمد الفتح الحجرات ق الذاريات الطور النجم القمر الرحمن الواقعة الحديد المجادلة الحشر الممتحنة الصف "
    "الجمعة المنافقون التغابن الطلاق التحريم الملك القلم الحاقة المعارج نوح الجن المزمل المدثر القيامة "
    "الإنسان المرسلات النبأ النازعات عبس التكوير الانفطار المطففين الانشقاق البروج الطارق الأعلى الغاشية "
    "الفجر البلد الشمس الليل الضحى الشرح التين العلق القدر البينة الزلزلة العاديات القارعة التكاثر العصر "
    "الهمزة الفيل قريش الماعون الكوثر الكافرون النصر المسد الإخلاص الفلق الناس"
)
SURA_NAMES = [n.replace("_", " ") for n in SURA_NAMES.split()]
assert len(SURA_NAMES) == 114


def normalize(text: str) -> list[str]:
    t = _DIAC.sub("", text).translate(_MAP)
    t = _NON_AR.sub(" ", t)
    return t.split()


def ngrams(words, n):
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


@lru_cache(maxsize=1)
def index():
    """-> (bigram set, trigram set, list of (sura, aya, normalised words))."""
    ayat = []
    for line in QURAN_FILE.read_text(encoding="utf-8").splitlines():
        s, a, txt = line.split("|", 2)
        ayat.append((int(s), int(a), normalize(txt)))
    bi, tri = set(), set()
    # n-grams also across consecutive ayat of the same sura (reciters don't pause at every aya)
    for i, (s, a, w) in enumerate(ayat):
        nxt = ayat[i + 1][2][:2] if i + 1 < len(ayat) and ayat[i + 1][0] == s else []
        ww = w + nxt
        bi |= ngrams(ww, 2)
        tri |= ngrams(ww, 3)
    return bi, tri, ayat


def quran_score(text: str) -> dict:
    """Fraction of the transcript's word bi/tri-grams that occur in the Quran."""
    w = normalize(text)
    bi, tri, _ = index()
    b, t = ngrams(w, 2), ngrams(w, 3)
    return {
        "words": len(w),
        "bigram": len(b & bi) / len(b) if b else 0.0,
        "trigram": len(t & tri) / len(t) if t else 0.0,
    }


def locate(text: str, top: int = 1):
    """Best-matching aya(s) for a transcript (by shared trigrams). Handy for showing *what* was recited."""
    w = normalize(text)
    t = ngrams(w, 3)
    if not t:
        return []
    _, _, ayat = index()
    scored = []
    for s, a, aw in ayat:
        k = len(t & ngrams(aw, 3))
        if k:
            scored.append((k, s, a))
    scored.sort(reverse=True)
    return [(s, a) for _, s, a in scored[:top]]

"""Non-acoustic hint: does the video's title / channel name mention a reciter?

Voice matching can only name reciters that are in the gallery. For links we also get the
video title, which very often says who is reading ("الشيخ المنشاوي - سورة الملك").
This is shown separately and labelled as coming from the title, never mixed with the voice score.
"""
import re
from collections import Counter

_DIAC = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u0640]")
_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})
_STOP = {"محمد", "احمد", "عبدالله", "عبدالرحمن", "علي", "بن", "ابن", "الشيخ", "القارئ", "abdul", "al", "muhammad",
         "mohamed", "mohammed", "ahmed", "sheikh", "shaikh", "the", "bin", "ibn"}


def _sura_words():
    from .quran_text import SURA_NAMES
    w = set()
    for n in SURA_NAMES:
        for t in _tokens(n):
            w |= {t, t[2:] if t.startswith("ال") else "ال" + t}
    return w | {"surah", "sura", "surat", "quran", "rahman", "yasin", "yaseen", "kahf", "mulk", "baqarah", "full"}


def _tokens(s: str) -> list[str]:
    s = _DIAC.sub("", s or "").translate(_MAP).lower()
    s = re.sub(r"عبد\s+ال", "عبدال", s)
    return [t for t in re.split(r"[^\w]+", s) if len(t) > 1]


def title_matches(text: str, gallery, top: int = 3) -> list[dict]:
    """People whose name is mentioned in `text`: all their distinctive name words must appear.
    A lone distinctive word (e.g. just 'المنشاوي') counts only if no other reciter shares it."""
    words = set(_tokens(text))
    if not words:
        return []
    weak = _STOP | _sura_words()                   # words that alone say nothing about the reciter
    cands = {}
    for pid in gallery.person_ids:
        for lang in ("ar", "en"):
            full = _tokens(gallery.name(pid, lang))
            if full:
                cands.setdefault(pid, []).append((full, [t for t in full if t not in weak]))
    df = Counter(t for names in cands.values() for _, toks in names for t in set(toks))
    out = []
    for pid, names in cands.items():
        score = 0.0
        for full, toks in names:
            hit = [t for t in toks if t in words]
            if len(full) >= 2 and all(t in words for t in full) and toks:
                score = max(score, 1.0)            # the whole name is written
            elif hit and len(hit) == len(toks) and len(toks) >= 2:
                score = max(score, 0.9)            # every distinctive word is written
            elif hit and all(df[t] == 1 for t in hit) and max(len(t) for t in hit) >= 4:
                score = max(score, 0.6 * len(hit) / max(1, len(toks)) + 0.3)
        if score:
            out.append({"person": pid, "name": gallery.name(pid, "ar"), "score": round(score, 2)})
    out.sort(key=lambda d: -d["score"])
    return out[:top]

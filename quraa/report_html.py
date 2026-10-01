"""Result card (HTML) for the web app. Styling lives in app.py (CSS classes prefixed `q-`)."""
from html import escape

from .quran_text import SURA_NAMES
from .reciters import style_of
from .report import STYLE, _mmss

VERDICT = {
    "quran": ("تلاوة قرآن", "q-ok"),
    "lecture": ("محاضرة / كلام عادي", "q-warn"),
    "mixed": ("تلاوة وكلام مع بعض", "q-mid"),
    "unclear": ("مش واضح", "q-mute"),
}
CHUNK = {"quran": "q-seg-quran", "speech": "q-seg-speech", "unclear": "q-seg-unclear"}


def _pct(sim, lo=0.25, hi=0.75):
    """Similarity -> 0..100 for the meter (cosine rarely leaves 0.25..0.75 here)."""
    return int(round(100 * min(1.0, max(0.0, (sim - lo) / (hi - lo)))))


def _timeline(c, duration):
    if not c or not c.get("chunks") or not duration:
        return ""
    segs = []
    for ch in c["chunks"]:
        left = 100 * ch["start"] / duration
        width = max(1.5, 100 * (ch["end"] - ch["start"]) / duration)
        tip = f"{_mmss(ch['start'])}–{_mmss(ch['end'])}: {escape(ch['text'][:80])}"
        segs.append(f'<span class="q-seg {CHUNK[ch["label"]]}" style="right:{left:.2f}%;width:{width:.2f}%" '
                    f'title="{tip}"></span>')
    return (f'<div class="q-timeline">{"".join(segs)}</div>'
            '<div class="q-legend"><span><i class="q-seg-quran"></i>قرآن</span>'
            '<span><i class="q-seg-speech"></i>كلام</span><span><i class="q-seg-unclear"></i>مش واضح</span>'
            f'<span class="q-dim">المدة {_mmss(duration)}</span></div>')


def _content_block(c, duration):
    if not c:
        return ""
    title, cls = VERDICT[c["verdict"]]
    ayat = ""
    if c.get("ayat"):
        chips = "".join(f'<span class="q-chip">سورة {SURA_NAMES[s - 1]} · {a}</span>' for s, a in c["ayat"][:5])
        ayat = f'<div class="q-row">{chips}</div>'
    return f"""
    <div class="q-panel">
      <div class="q-panel-head"><span class="q-kicker">نوع المحتوى</span>
        <span class="q-badge {cls}">{title}</span></div>
      {_timeline(c, duration)}
      {ayat}
    </div>"""


def _bars(rec):
    rows = []
    for i, t in enumerate(rec["top"]):
        rows.append(f"""
        <div class="q-bar {'q-bar-top' if i == 0 else ''}">
          <span class="q-bar-name">{escape(t['name'])}</span>
          <span class="q-bar-track"><span class="q-bar-fill" style="width:{_pct(t['similarity'])}%"></span></span>
          <span class="q-bar-val">{t['similarity']:.2f}</span>
        </div>""")
    return f'<div class="q-panel"><div class="q-kicker">أقرب الأصوات</div>{"".join(rows)}</div>'


def _hint_block(r, rec):
    hints = r.get("title_hints") or []
    voice = rec.get("person") if rec and rec.get("ok") and not rec.get("unknown") else None
    if hints and voice and hints[0]["person"] == voice:
        return '<div class="q-row q-center"><span class="q-chip q-chip-ok">✓ عنوان الفيديو كمان بيقول كده</span></div>'
    if hints:
        names = "، ".join(f"<b>{escape(h['name'])}</b>" for h in hints[:2])
        return (f'<div class="q-panel q-hint-box"><div class="q-kicker">📝 عنوان الملف/الفيديو بيذكر</div>'
                f'<div class="q-sub">{names}</div>'
                '<div class="q-dim">ده من الاسم المكتوب مش من الصوت — لو الصوت قال حاجة تانية، صدّق الصوت أكتر لو التطابق عالي.</div></div>')
    if rec and rec.get("ok") and rec.get("unknown") and r.get("source_text"):
        return (f'<div class="q-panel q-hint-box"><div class="q-kicker">📝 عنوان الفيديو</div>'
                f'<div class="q-sub">{escape(r["source_text"][:140])}</div>'
                '<div class="q-dim">القارئ مش في قاعدتنا، بس العنوان ممكن يساعدك تعرفه.</div></div>')
    return ""


def to_html(r: dict) -> str:
    c = r.get("content")
    rec = r.get("reciter")
    head = (f'<div class="q-file"><span>🎧 {escape(r["source"])}</span>'
            f'<span class="q-dim">{_mmss(r["duration_s"])} · تحليل في {r.get("elapsed_s", 0):.0f} ث</span></div>')

    if rec is None or not rec.get("ok"):
        msg = ("ده مش تلاوة، فمفيش قارئ نتعرف عليه." if rec is None
               else f"مقدرناش نسمع صوت واضح ({escape(rec['reason'])}).")
        hero = f'<div class="q-hero q-hero-empty"><div class="q-ornament">۞</div><div class="q-sub">{msg}</div></div>'
        return f'<div class="q-card">{head}{hero}{_content_block(c, r["duration_s"])}</div>'

    style = rec.get("style") or STYLE.get(style_of(rec["label"]), "")
    pct = _pct(rec["similarity"])
    if not rec["unknown"]:
        kicker, note, tone = "القارئ", "", "q-hero-ok"
        if rec.get("sounds_like"):
            kicker = "غالبًا القارئ"
            note = (f"صوته قريب جدًا من {escape('، '.join(rec['sounds_like']))} — لو تعرف القارئ، "
                    "قولنا الإجابة صح ولا غلط.")
        elif rec["margin"] < 0.03:
            note = "الفرق بينه وبين التاني صغير — النتيجة مش أكيدة."
    elif rec["margin"] >= 0.06:
        kicker, tone = "غالبًا القارئ", "q-hero-mid"
        note = "درجة التطابق أقل من العتبة، بس الصوت متميّز بوضوح — غالبًا فيه صدى أو ضجيج."
    else:
        kicker, tone = "أقرب صوت ليه", "q-hero-unk"
        note = "غالبًا القارئ ده مش موجود في قاعدة البيانات لسه."

    hero = f"""
    <div class="q-hero {tone}">
      <div class="q-ornament">۞</div>
      <div class="q-kicker">{kicker}</div>
      <div class="q-name">{escape(rec['name'])}</div>
      <div class="q-name-en">{escape(rec['name_en'])}</div>
      <div class="q-row q-center">
        {f'<span class="q-chip q-chip-gold">{style}</span>' if style else ''}
        <span class="q-chip">اتفاق {rec['votes']}/{rec['n_windows']} مقطع</span>
      </div>
      <div class="q-meter"><span style="width:{pct}%"></span></div>
      <div class="q-dim">درجة التطابق {rec['similarity']:.2f}{f" · العتبة {rec['threshold']:.2f}" if rec.get('threshold') else ''}</div>
      {f'<div class="q-note">{note}</div>' if note else ''}
    </div>"""
    return (f'<div class="q-card">{head}{hero}{_hint_block(r, rec)}{_content_block(c, r["duration_s"])}'
            f'{_bars(rec)}</div>')


def error_html(msg: str) -> str:
    return f'<div class="q-card"><div class="q-hero q-hero-unk"><div class="q-ornament">۞</div><div class="q-sub">{escape(msg)}</div></div></div>'


PLACEHOLDER = """
<div class="q-card q-placeholder">
  <div class="q-ornament">۞</div>
  <div class="q-sub">النتيجة هتظهر هنا</div>
  <div class="q-steps">
    <div><b>١</b><span>حط رابط أو ارفع ملف</span></div>
    <div><b>٢</b><span>بنسمع ونفرّق التلاوة عن الكلام</span></div>
    <div><b>٣</b><span>نقارن الصوت ببصمات القراء</span></div>
  </div>
</div>"""

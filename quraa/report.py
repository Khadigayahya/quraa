"""Human-readable (Arabic) summary of a pipeline result."""
from .quran_text import SURA_NAMES

VERDICTS = {
    "quran": "🕋 تلاوة قرآن",
    "lecture": "🎤 محاضرة / كلام عادي (مش تلاوة)",
    "mixed": "🔀 فيه تلاوة وكلام (محاضرة فيها آيات، أو تلاوة مش واضحة)",
    "unclear": "❔ مش واضح (صمت/موسيقى/صوت ضعيف)",
}
STYLE = {"mujawwad": "مجوّد", "murattal": "مرتّل", "muallim": "المصحف المعلّم", "warsh": "رواية ورش"}


def _mmss(s):
    return f"{int(s // 60)}:{int(s % 60):02d}"


def to_markdown(r: dict) -> str:
    from .reciters import style_of

    lines = [f"**الملف:** {r['source']} — المدة {_mmss(r['duration_s'])}"]
    c = r.get("content")
    if c:
        lines.append(f"\n### نوع المحتوى: {VERDICTS[c['verdict']]}")
        lines.append(f"نسبة المقاطع اللي طلعت قرآن: {c['quran_fraction']:.0%} من {len(c['chunks'])} مقطع")
        if c.get("ayat"):
            where = "، ".join(f"{SURA_NAMES[s - 1]} ({a})" for s, a in c["ayat"][:5])
            lines.append(f"أقرب آيات اتعرفت: {where}")
    rec = r.get("reciter")
    if rec is None:
        if c and c["verdict"] == "lecture":
            lines.append("\nمفيش تلاوة علشان نتعرف على القارئ.")
        return "\n".join(lines)
    if not rec["ok"]:
        lines.append(f"\n⚠️ {rec['reason']}")
        return "\n".join(lines)

    style = STYLE.get(style_of(rec["label"]), "")
    if rec["unknown"] and rec["margin"] >= 0.08:
        lines.append(f"\n## 🎙️ غالبًا: **{rec['name']}** (ثقة منخفضة)")
        lines.append(f"التشابه {rec['similarity']:.2f} أقل من العتبة {rec['threshold']:.2f}، بس متميّز بوضوح "
                     f"عن التاني (فرق {rec['margin']:.2f}) — غالبًا التسجيل فيه صدى أو ضجيج.")
    elif rec["unknown"]:
        lines.append(f"\n## ❓ قارئ غير موجود في القاعدة (غالبًا)")
        lines.append(f"أقرب صوت: **{rec['name']}** بتشابه {rec['similarity']:.2f} (أقل من العتبة {rec['threshold']:.2f})")
    else:
        lines.append(f"\n## 🎙️ القارئ: **{rec['name']}**" + (f" — {style}" if style else ""))
        lines.append(f"{rec['name_en']} · تشابه {rec['similarity']:.2f} · فرق عن التاني {rec['margin']:.2f} · "
                     f"أصوات النوافذ {rec['votes']}/{rec['n_windows']}")
        if rec["margin"] < 0.03:
            lines.append("⚠️ الفرق بين أول اتنين صغير — النتيجة مش أكيدة.")
    lines.append("\n| # | القارئ | التشابه | الأصوات |\n|---|---|---|---|")
    for i, t in enumerate(rec["top"], 1):
        lines.append(f"| {i} | {t['name']} | {t['similarity']:.3f} | {t['votes']} |")
    lines.append(f"\n<sub>وقت التحليل: {r['elapsed_s']:.1f} ثانية</sub>")
    return "\n".join(lines)

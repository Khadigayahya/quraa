"""Quraa web app (Gradio).

    python app.py              # http://127.0.0.1:7860
    python app.py --share      # public link (e.g. from Colab)
"""
import argparse
import os
import threading
import traceback

import gradio as gr

from quraa.pipeline import Quraa
from quraa.report_html import PLACEHOLDER, error_html, to_html

engine: Quraa | None = None
MAX_MINUTES = float(os.environ.get("QURAA_MAX_MINUTES", 30))   # longer files: only the first N minutes

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Tajawal:wght@400;500;700&display=swap" rel="stylesheet">
"""

# 8-point star lattice, drawn once and tiled behind everything
_PATTERN = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='80' height='80' viewBox='0 0 80 80'>"
            "<g fill='none' stroke='%23c9a24a' stroke-opacity='0.13' stroke-width='1'>"
            "<rect x='22' y='22' width='36' height='36'/><rect x='22' y='22' width='36' height='36' transform='rotate(45 40 40)'/>"
            "<circle cx='40' cy='40' r='6'/><path d='M0 40h14M66 40h14M40 0v14M40 66v14'/></g></svg>")

CSS = """
:root {
  --q-bg: #f5efe1; --q-ink: #14302a; --q-soft: #5b6b64; --q-card: #fffdf7; --q-line: #e6dcc3;
  --q-emerald: #0e5a47; --q-emerald-2: #137a5f; --q-gold: #b8912f; --q-gold-soft: #f3e6c2;
  --q-quran: #137a5f; --q-speech: #c8763a; --q-unclear: #b9b3a5;
  --q-shadow: 0 18px 40px -22px rgba(14, 58, 47, .45);
}
.dark {
  --q-bg: #0b1613; --q-ink: #efe8d6; --q-soft: #a9b3ad; --q-card: #11221d; --q-line: #24392f;
  --q-emerald: #1f9c79; --q-emerald-2: #27b58d; --q-gold: #d8b45a; --q-gold-soft: #3a3220;
  --q-shadow: 0 18px 40px -22px rgba(0, 0, 0, .8);
}
body, gradio-app, .gradio-container { background: var(--q-bg) url("__PATTERN__") repeat !important; }
.gradio-container { max-width: 980px !important; margin: auto; direction: rtl;
  font-family: 'Tajawal', system-ui, sans-serif !important; color: var(--q-ink); }
.gradio-container * { font-family: inherit; }
footer { display: none !important; }

/* ---------- header ---------- */
#q-header { text-align: center; padding: 38px 16px 30px; margin: 18px 0 8px; border-radius: 28px;
  background: url("__PATTERN_LIGHT__") repeat, radial-gradient(ellipse at top, #16735a 0%, #0c4a3b 55%, #08352a 100%);
  box-shadow: 0 24px 50px -28px rgba(8, 53, 42, .9); position: relative; overflow: hidden; }
#q-header::before, #q-header::after { content: "۞"; position: absolute; top: 50%; transform: translateY(-50%);
  font-size: 120px; color: #e6c878; opacity: .08; }
#q-header::before { right: 30px; } #q-header::after { left: 30px; }
#q-header .q-logo { font-family: 'Amiri', serif !important; font-size: 84px; line-height: 1.1; font-weight: 700;
  color: #ecd08a !important; text-shadow: 0 3px 0 rgba(0, 0, 0, .25), 0 0 30px rgba(236, 208, 138, .25); }
#q-header .q-tag { color: #e9f1ec !important; font-size: 19px; margin-top: 6px; opacity: .92; }
#q-header .q-divider { display: flex; align-items: center; gap: 14px; justify-content: center; margin: 16px auto 0;
  color: #ecd08a !important; max-width: 320px; }
#q-header .q-divider::before, #q-header .q-divider::after { content: ""; flex: 1; height: 1px;
  background: linear-gradient(90deg, transparent, #ecd08a, transparent); }
#q-header .q-stats { display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; margin-top: 18px; }
#q-header .q-stats span { color: #f5efe1 !important; border: 1px solid rgba(236, 208, 138, .45); border-radius: 999px;
  padding: 4px 14px; font-size: 14px; background: rgba(255, 255, 255, .06); }

/* ---------- input panel ---------- */
#q-input { background: var(--q-card); border: 1px solid var(--q-line); border-radius: 22px;
  padding: 10px 18px 18px; box-shadow: var(--q-shadow); }
#q-input .tab-nav, #q-input [role="tablist"] { justify-content: center; border: none !important; gap: 8px; }
#q-input [role="tab"] { border-radius: 999px !important; border: 1px solid var(--q-line) !important;
  padding: 8px 20px !important; font-weight: 500; font-size: 16px; }
#q-input [role="tab"][aria-selected="true"] { background: var(--q-emerald) !important; color: #fff !important;
  border-color: var(--q-emerald) !important; }
#q-input [role="tab"][aria-selected="true"]::after { display: none !important; }
.q-hint { text-align: center; color: var(--q-soft); font-size: 15px; }
.q-go { background: linear-gradient(135deg, var(--q-emerald), var(--q-emerald-2)) !important; color: #fff !important;
  border: none !important; border-radius: 14px !important; font-size: 18px !important; font-weight: 700 !important;
  padding: 14px !important; box-shadow: 0 10px 22px -12px var(--q-emerald); transition: transform .15s, box-shadow .15s; }
.q-go:hover { transform: translateY(-1px); box-shadow: 0 14px 26px -12px var(--q-gold); }

/* ---------- result card ---------- */
.q-card { background: var(--q-card); border: 1px solid var(--q-line); border-radius: 22px; padding: 22px;
  box-shadow: var(--q-shadow); direction: rtl; color: var(--q-ink); animation: q-in .45s ease-out; }
@keyframes q-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
.q-placeholder { text-align: center; padding: 40px 22px; color: var(--q-soft); border: 1.5px dashed var(--q-gold); }
.q-steps { display: grid; gap: 12px; margin: 26px auto 0; max-width: 340px; text-align: right; }
.q-steps div { display: flex; align-items: center; gap: 12px; padding: 10px 14px; border-radius: 14px;
  background: var(--q-bg); color: var(--q-ink); }
.q-steps b { display: inline-grid; place-items: center; width: 32px; height: 32px; border-radius: 50%; flex: none;
  background: var(--q-emerald); color: #fff; font-family: 'Amiri', serif; font-size: 18px; }
.q-file { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; font-size: 14px;
  padding-bottom: 12px; border-bottom: 1px dashed var(--q-line); }
.q-dim { color: var(--q-soft); font-size: 14px; }
.q-hero { text-align: center; padding: 26px 10px 18px; position: relative; }
.q-ornament { font-size: 30px; color: var(--q-gold); opacity: .85; }
.q-kicker { color: var(--q-soft); font-size: 15px; letter-spacing: .5px; margin-top: 4px; }
.q-name { font-family: 'Amiri', serif; font-size: 48px; font-weight: 700; line-height: 1.35; color: var(--q-emerald); }
.q-hero-mid .q-name { color: var(--q-gold); }
.q-hero-unk .q-name { color: var(--q-soft); font-size: 38px; }
.q-name-en { color: var(--q-soft); font-size: 15px; direction: ltr; }
.q-sub { font-size: 18px; color: var(--q-soft); margin-top: 8px; }
.q-row { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
.q-center { justify-content: center; }
.q-chip { border: 1px solid var(--q-line); border-radius: 999px; padding: 4px 12px; font-size: 14px; }
.q-chip-gold { background: var(--q-gold-soft); border-color: var(--q-gold); color: var(--q-ink); }
.q-meter { height: 10px; border-radius: 999px; background: var(--q-line); margin: 18px auto 8px; max-width: 360px; overflow: hidden; }
.q-meter span { display: block; height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--q-gold), var(--q-emerald)); margin-right: 0; margin-left: auto; }
.q-note { margin: 12px auto 0; max-width: 520px; font-size: 14px; padding: 8px 14px; border-radius: 12px;
  background: var(--q-gold-soft); }
.q-panel { border-top: 1px dashed var(--q-line); padding-top: 16px; margin-top: 16px; }
.q-panel-head { display: flex; justify-content: space-between; align-items: center; }
.q-badge { border-radius: 999px; padding: 4px 14px; font-weight: 700; font-size: 14px; color: #fff; }
.q-ok { background: var(--q-quran); } .q-warn { background: var(--q-speech); }
.q-mid { background: var(--q-gold); } .q-mute { background: var(--q-unclear); }
.q-timeline { position: relative; height: 16px; border-radius: 8px; background: var(--q-line); margin-top: 14px; overflow: hidden; }
.q-seg { position: absolute; top: 0; bottom: 0; border-radius: 4px; }
.q-seg-quran { background: var(--q-quran); } .q-seg-speech { background: var(--q-speech); } .q-seg-unclear { background: var(--q-unclear); }
.q-legend { display: flex; gap: 16px; flex-wrap: wrap; font-size: 13px; margin-top: 8px; color: var(--q-soft); }
.q-legend i { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-left: 5px; vertical-align: middle; }
.q-legend .q-dim { margin-right: auto; }
.q-bar { display: grid; grid-template-columns: minmax(110px, 190px) 1fr 42px; gap: 12px; align-items: center; margin-top: 10px; font-size: 15px; }
.q-bar-track { height: 8px; border-radius: 999px; background: var(--q-line); overflow: hidden; }
.q-bar-fill { display: block; height: 100%; border-radius: 999px; background: var(--q-unclear); margin-left: auto; }
.q-bar-top .q-bar-fill { background: linear-gradient(90deg, var(--q-gold), var(--q-emerald)); }
.q-bar-top .q-bar-name { font-weight: 700; color: var(--q-emerald); }
.q-bar-val { color: var(--q-soft); font-size: 13px; text-align: left; direction: ltr; }
.q-chip-ok { border-color: var(--q-quran); color: var(--q-quran); }
.q-hint-box { text-align: center; }
.q-hint-box .q-sub b { color: var(--q-ink); font-family: 'Amiri', serif; font-size: 24px; }
@media (max-width: 640px) {
  #q-header .q-logo { font-size: 58px; } .q-name { font-size: 36px; }
  .q-bar { grid-template-columns: 100px 1fr 38px; }
}
""".replace("__PATTERN_LIGHT__", _PATTERN.replace("stroke-opacity='0.13'", "stroke-opacity='0.16'")
                                .replace("%23c9a24a", "%23ecd08a")).replace("__PATTERN__", _PATTERN)

HEADER = """
<div id="q-header">
  <div class="q-logo">قُرّاء</div>
  <div class="q-tag">اسمع التلاوة… وإحنا نقولك مين القارئ</div>
  <div class="q-divider">۞</div>
  <div class="q-stats"><span>🎙️ {people} قارئ</span><span>🎬 يوتيوب وفيديو</span><span>🕋 قرآن ولا محاضرة؟</span></div>
</div>
"""

EXAMPLES = [
    "https://www.youtube.com/watch?v=jYQZAFEBNTw",
    "https://www.youtube.com/watch?v=dFl2ia_8VUk",
]


def _run(source, check_content, progress=gr.Progress()):
    if not source:
        return error_html("ارفع ملف أو حط رابط الأول.")
    try:
        progress(0.1, desc="بنجهّز الصوت…")
        return to_html(engine.analyze(source, check_content=check_content, max_minutes=MAX_MINUTES))
    except Exception as e:  # show the error in the page instead of a blank box
        traceback.print_exc()
        return error_html(f"حصل خطأ: {type(e).__name__}: {e}")


def from_video(url, file, check_content, progress=gr.Progress()):
    return _run((url or "").strip() or file, check_content, progress)


def from_audio(file, progress=gr.Progress()):
    return _run(file, True, progress)


def _style_kwargs():
    theme = gr.themes.Soft(primary_hue="emerald", secondary_hue="amber", radius_size="lg")
    return {"theme": theme, "css": CSS, "head": HEAD}


GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6


def build_ui():
    with gr.Blocks(title="قُرّاء — تعرّف على القارئ", **({} if GRADIO_6 else _style_kwargs())) as ui:
        gr.HTML(HEADER.format(people=len(engine.gallery.person_ids) if engine else "—"))
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_id="q-input"):
                with gr.Tab("🎬 فيديو أو رابط"):
                    gr.HTML('<div class="q-hint">رابط يوتيوب / فيسبوك… أو فيديو من جهازك</div>')
                    url = gr.Textbox(label="الرابط", placeholder="https://www.youtube.com/watch?v=…", rtl=False)
                    gr.Examples(EXAMPLES, inputs=url, label="جرّب")
                    vfile = gr.File(label="أو ارفع فيديو", file_types=["video", "audio"], type="filepath")
                    check = gr.Checkbox(label="اتأكد الأول إنه تلاوة مش محاضرة", value=False)
                    vbtn = gr.Button("تعرّف على القارئ", elem_classes="q-go")
                with gr.Tab("🎧 مقطع صوتي"):
                    gr.HTML('<div class="q-hint">هنعرف الأول: <b>قرآن ولا محاضرة؟</b> ولو قرآن نقولك مين القارئ</div>')
                    afile = gr.Audio(label="المقطع", type="filepath", sources=["upload", "microphone"])
                    abtn = gr.Button("حلّل المقطع", elem_classes="q-go")
            with gr.Column(scale=6):
                out = gr.HTML(PLACEHOLDER)
        vbtn.click(from_video, [url, vfile, check], out)
        abtn.click(from_audio, [afile], out)
    return ui


def launch(q: Quraa, **kw):
    """Start the UI around an already-loaded engine (used from the Colab notebook)."""
    global engine
    engine = q
    # load Whisper in the background so the first "audio" request doesn't wait for it
    threading.Thread(target=lambda: engine.content, daemon=True).start()
    kw.setdefault("max_file_size", "300mb")
    if GRADIO_6:
        kw = {**_style_kwargs(), **kw}
    return build_ui().queue().launch(**kw)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gallery")
    p.add_argument("--whisper")
    p.add_argument("--share", action="store_true")
    p.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
    a = p.parse_args()
    launch(Quraa(a.gallery, whisper_size=a.whisper), share=a.share, server_port=a.port,
           server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))

---
title: قُرّاء — Quraa
emoji: 🎙️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
short_description: مين القارئ ده؟ تعرّف على قارئ القرآن من صوته
---

# قُرّاء — مين القارئ ده؟

ارفع فيديو أو مقطع صوتي (أو سجّل من المايك)، والتطبيق يقولك:
1. **ده قرآن ولا محاضرة؟** (Whisper + مطابقة مع نص المصحف — وكمان يقولك أنهي سورة)
2. **مين القارئ؟** (بصمة صوت ECAPA-TDNN مقارنة ببصمات 199 قارئ)

## التشغيل على جهازك

```bash
pip install -r requirements.txt
python app.py                 # http://127.0.0.1:7860
python -m quraa <ملف-أو-رابط>  # من سطر الأوامر
```

> **ويندوز:** لو ظهر `WinError 1114 ... c10.dll` نزّلي أحدث
> [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

## النشر: كله في المتصفح (Vercel + Hugging Face Hub)

```
المتصفح ──► web/ على Vercel (الموقع)                         ← مجاني
        ──► huggingface.co/KhadijaYahya/quraa-models           ← ecapa.onnx + gallery (مجاني)
        ──► onnx-community/whisper-base (transformers.js)      ← قرآن ولا محاضرة
التحليل كله بيحصل على جهاز الزائر — مفيش سيرفر، والملفات مش بتترفع لأي حتة.
```

- **الموقع:** على Vercel: Import الريبو ← **Root Directory = `web`** ← Deploy.
- **تحديث البصمات** (بعد إضافة قراء):
  ```bash
  python build/make_gallery.py            # -> models/gallery_v2.npz
  python build/export_web_gallery.py      # -> web/models/gallery.{bin,json}
  HF_HUB_DISABLE_XET=1 hf upload KhadijaYahya/quraa-models web/models . --repo-type model
  ```
- **تحديث الموديل** (نادرًا): `python build/export_onnx.py --check ملف.mp3` (لازم cosine = 1.000000).
- `modal_app.py`: نسخة سيرفر اختيارية (بتدعم روابط يوتيوب) لو اتوفر كارت على Modal.

## الملفات

| | |
|---|---|
| `web/` | الموقع (HTML/CSS/JS) — بيتنشر على Vercel؛ `web/engine.js` = نسخة المتصفح من `quraa/` |
| `build/` | بناء البصمات (mp3quran على Modal)، وتصدير الموديل والقاعدة للمتصفح |
| `modal_app.py` | الـ API على Modal |
| `app.py` | واجهة Gradio (للتشغيل المحلي أو كولاب) |
| `quraa/` | الكود: تحميل الصوت، البصمات، قرآن/محاضرة، أسماء القراء |
| `models/gallery_*.npz` | بصمات القراء (`gallery_v1` لو موجودة، وإلا `gallery_ecapa_v0`) |
| `notebooks/reciter_v1_colab.ipynb` | التدريب والتقييم على كولاب — بيطلّع `gallery_v1.npz` (+ `ecapa_ft.pt`) |

## الإعدادات (متغيرات البيئة)

| المتغير | الاستخدام |
|---|---|
| `QURAA_MAX_MINUTES` | أقصى مدة بتتحلل من كل ملف (افتراضي 30 دقيقة) |
| `QURAA_YTDLP_COOKIES_TXT` | محتوى ملف cookies.txt لو يوتيوب طلب "sign in to confirm you're not a bot" |
| `QURAA_MODELS` | فولدر الموديلات (افتراضي `models/`) |

## ملاحظات

- التسجيلات المستخدمة في بناء البصمات من [everyayah.com](https://everyayah.com)، والحقوق لأصحابها — للبحث والتجربة.
  التطبيق بيحتفظ بالبصمات (أرقام) بس، مش التسجيلات.
- التطبيق بيعرف القراء اللي في القاعدة بس؛ غير كده بيقول «غالبًا مش موجود».

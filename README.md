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

ارفع فيديو أو مقطع صوتي أو حط رابط يوتيوب، والتطبيق يقولك:
1. **ده قرآن ولا محاضرة؟** (Whisper + مطابقة مع نص المصحف — وكمان يقولك أنهي سورة)
2. **مين القارئ؟** (بصمة صوت ECAPA-TDNN مقارنة ببصمات 38 قارئ)

## التشغيل على جهازك

```bash
pip install -r requirements.txt
python app.py                 # http://127.0.0.1:7860
python -m quraa <ملف-أو-رابط>  # من سطر الأوامر
```

> **ويندوز:** لو ظهر `WinError 1114 ... c10.dll` نزّلي أحدث
> [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe).

## النشر: الموقع على Vercel + الموديلات على Modal

```
المتصفح ──► web/ (Vercel، موقع ثابت)  ──POST /analyze──►  modal_app.py (Modal: ECAPA + Whisper)
```

1. **الـ API:** `modal deploy modal_app.py` ← بيطبع رابط زي `https://<workspace>--quraa-api-api-web.modal.run`
   - CPU افتراضيًا (مجاني). GPU أسرع بكتير: `QURAA_GPU=T4 modal deploy modal_app.py` (محتاج كارت على Modal).
   - يوتيوب بيمنع السيرفرات من غير cookies:
     `modal secret create quraa-youtube QURAA_YTDLP_COOKIES_TXT="$(cat cookies.txt)"` وبعدين `QURAA_YT_SECRET=1 modal deploy modal_app.py`
2. **الموقع:** حطي رابط الـ API في `web/config.js`، وعلى Vercel: Import الريبو ← **Root Directory = `web`** ← Deploy.

## الملفات

| | |
|---|---|
| `web/` | الموقع (HTML/CSS/JS) — بيتنشر على Vercel |
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

"""Quraa API on Modal (GPU, pay-per-second, scales to zero).

    modal serve modal_app.py     # temporary dev URL, reloads on save
    modal deploy modal_app.py    # permanent URL: https://<workspace>--quraa-api-web.modal.run

POST /analyze   multipart form:  file=<audio/video>  or  url=<link>,  check_content=true|false
                -> {"html": <result card>, "result": {...raw result...}}
GET  /health
"""
import os
from pathlib import Path

import modal

LOCAL = Path(__file__).parent
MAX_UPLOAD_MB = 150
MAX_MINUTES = 20            # longer files: only the first N minutes are analysed

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch", "torchaudio", "speechbrain>=1.0", "faster-whisper>=1.1", "yt-dlp", "numpy", "scipy",
                 "scikit-learn", "soundfile", "fastapi[standard]", "python-multipart")
    .env({"HF_HOME": "/cache/hf", "QURAA_MODELS": "/root/models",
          "QURAA_QURAN_FILE": "/root/assets/quran-simple-clean.txt"})
    .add_local_python_source("quraa")
    .add_local_dir(LOCAL / "quraa" / "assets", "/root/assets")            # Quran text (not a .py file)
    .add_local_dir(LOCAL / "models", "/root/models", ignore=["pretrained", "pretrained/**"])
)
cache = modal.Volume.from_name("quraa-cache", create_if_missing=True)
app = modal.App("quraa-api", image=image)


# QURAA_GPU=T4 modal deploy modal_app.py   -> GPU (needs a payment method on Modal; seconds per file)
# default                                  -> CPU (works on the free plan; ~1-2 min per file)
GPU = os.environ.get("QURAA_GPU") or None
HW = {"gpu": GPU} if GPU else {"cpu": 4.0, "memory": 8192}


# YouTube blocks datacenter IPs without cookies:  modal secret create quraa-youtube QURAA_YTDLP_COOKIES_TXT="$(cat cookies.txt)"
# then deploy with  QURAA_YT_SECRET=1 modal deploy modal_app.py
SECRETS = [modal.Secret.from_name("quraa-youtube")] if os.environ.get("QURAA_YT_SECRET") else []


# max_containers caps the bill if the site gets busy; scaledown_window = idle seconds before shutting down
@app.cls(**HW, volumes={"/cache": cache}, secrets=SECRETS, scaledown_window=300, max_containers=2, timeout=900)
@modal.concurrent(max_inputs=1)
class Api:
    @modal.enter()
    def load(self):
        from quraa.pipeline import Quraa

        self.q = Quraa(finetuned=None)
        self.q.embedder.clf.mods.eval()
        _ = self.q.content          # Whisper: large-v3-turbo on GPU, small on CPU
        cache.commit()              # keep downloaded models for the next cold start

    @modal.asgi_app()
    def web(self):
        import tempfile

        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware

        from quraa.report_html import to_html

        api = FastAPI(title="Quraa")
        api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])

        @api.get("/health")
        def health():
            return {"ok": True, "people": len(self.q.gallery.person_ids), "gallery": self.q.gallery.meta.get("source", "v1")}

        @api.post("/analyze")
        def analyze(file: UploadFile | None = File(None), url: str = Form(""), check_content: bool = Form(True)):
            src = url.strip()
            tmp = None
            if file is not None and file.filename:
                suffix = Path(file.filename).suffix[:10]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                size = 0
                while chunk := file.file.read(1 << 20):
                    size += len(chunk)
                    if size > MAX_UPLOAD_MB << 20:
                        tmp.close(); os.unlink(tmp.name)
                        raise HTTPException(413, f"الملف أكبر من {MAX_UPLOAD_MB}MB")
                    tmp.write(chunk)
                tmp.close()
                src = tmp.name
            if not src:
                raise HTTPException(400, "ارفع ملف أو حط رابط")
            try:
                r = self.q.analyze(src, check_content=check_content, max_minutes=MAX_MINUTES,
                                   display_name=file.filename if tmp else None)
                return {"html": to_html(r), "result": r}
            except Exception as e:
                raise HTTPException(422, f"{type(e).__name__}: {e}")
            finally:
                if tmp and os.path.exists(tmp.name):
                    os.unlink(tmp.name)

        return api

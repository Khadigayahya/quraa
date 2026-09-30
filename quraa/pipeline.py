"""End-to-end: file / link -> (Quran or lecture?) -> reciter name."""
import os
import threading
import time
from pathlib import Path

from . import audio
from .gallery import Gallery
from .identify import identify

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(os.environ.get("QURAA_MODELS", ROOT / "models"))


def default_gallery_path():
    for name in ("gallery_v3.npz", "gallery_v2.npz", "gallery_v1.npz", "gallery_ecapa_v0.npz"):
        if (MODELS / name).exists():
            return MODELS / name
    raise FileNotFoundError(f"No gallery found in {MODELS}. Run the v1 notebook or copy gallery_ecapa_v0.npz there.")


class Quraa:
    def __init__(self, gallery_path=None, finetuned=None, whisper_size=None, device=None):
        from .embedder import Embedder

        self.gallery = Gallery.load(gallery_path or default_gallery_path())
        ft = finetuned or self.gallery.meta.get("finetuned")
        if ft and not Path(ft).is_absolute():
            ft = MODELS / ft
        self.embedder = Embedder(finetuned=ft, device=device, cache_dir=str(MODELS / "pretrained"))
        if self.gallery.meta.get("finetuned") and not self.embedder.finetuned:
            raise FileNotFoundError(f"Gallery was built with fine-tuned weights {ft} but the file is missing.")
        self.whisper_size = whisper_size
        self.device = device
        self._content = None
        self._lock = threading.Lock()

    @property
    def content(self):
        with self._lock:  # may be first touched by a background preload and a request at once
            if self._content is None:
                from .content import ContentDetector
                self._content = ContentDetector(self.whisper_size, self.device)
        return self._content

    def analyze(self, source: str, check_content: bool = True, max_minutes: float | None = 60,
                display_name: str | None = None) -> dict:
        from .hints import title_matches

        t0 = time.time()
        w, name, text = audio.fetch_info(source, max_minutes, display_name)
        out = {"source": name, "duration_s": len(w) / audio.SR, "source_text": text,
               "title_hints": title_matches(text, self.gallery)}
        spans = None
        if check_content:
            c = self.content.analyze(w)
            out["content"] = c
            if c["verdict"] == "lecture":
                out["reciter"] = None
                out["elapsed_s"] = time.time() - t0
                return out
            if c["verdict"] == "mixed":         # only listen to the recited parts
                spans = [(ch["start"], ch["end"]) for ch in c["chunks"] if ch["label"] == "quran"]
        out["reciter"] = identify(w, self.embedder, self.gallery, spans=spans)
        out["elapsed_s"] = time.time() - t0
        return out

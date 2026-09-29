"""Quran recitation vs. lecture / talk.

Each ~30 s chunk is transcribed with Whisper (faster-whisper) and compared with the
Quran text. Recitation matches the Quran almost word-for-word; a lecture only matches
where it quotes an aya. Needs no training data, and also tells us *which* aya was read.
"""
import numpy as np

from . import quran_text
from .audio import SR


def _looping(text: str) -> bool:
    """Small Whisper models sometimes get stuck ("الجنة ولمين الجنة ولمين ..."): the word rate is then
    meaningless, so such a chunk should not vote for either side."""
    from collections import Counter
    w = quran_text.normalize(text)
    tri = Counter(" ".join(w[i:i + 3]) for i in range(len(w) - 2))
    return bool(tri) and max(tri.values()) >= 4


class ContentDetector:
    def __init__(self, model_size: str | None = None, device: str | None = None,
                 quran_threshold: float = 0.3, min_words: int = 5):
        import torch  # before faster_whisper: on Windows the other order breaks torch's DLL loading
        from faster_whisper import WhisperModel

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_size = model_size or ("large-v3-turbo" if device == "cuda" else "small")
        self.model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
        self.model_size = model_size
        self.device = device
        self.quran_threshold = quran_threshold
        self.min_words = min_words

    def transcribe(self, w: np.ndarray) -> str:
        segs, _ = self.model.transcribe(w, language="ar", beam_size=1, vad_filter=True,
                                        condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs)

    def classify_chunk(self, w: np.ndarray) -> dict:
        text = self.transcribe(w)
        sc = quran_text.quran_score(text)
        sc["wps"] = sc["words"] / max(len(w) / SR, 1.0)   # recitation is slow (~0.5-1 w/s), talk ~1.5-3 w/s
        if sc["words"] < self.min_words or _looping(text):
            label = "unclear"                       # silence/music/too short, or Whisper stuck repeating itself
        elif sc["trigram"] >= self.quran_threshold:
            label = "quran"
        elif sc["bigram"] >= 0.1 and sc["wps"] < 1.0:
            label = "quran"                         # slow + partly matching = recitation Whisper misheard
        elif sc["bigram"] >= 0.05 and sc["wps"] < 0.8:
            label = "quran"                         # very slow (tarteel pace) — small Whisper models garble the words
        else:
            label = "speech"
        return {"label": label, "text": text, **sc}

    def analyze(self, w: np.ndarray, chunk_s: float = 30, max_chunks: int | None = None) -> dict:
        """Classify evenly spread chunks of the file (bounded cost for long lectures)."""
        max_chunks = max_chunks or (12 if self.device == "cuda" else 6)
        n = int(chunk_s * SR)
        starts = np.arange(0, max(1, len(w) - n // 3), n)
        if len(starts) > max_chunks:
            starts = starts[np.linspace(0, len(starts) - 1, max_chunks).round().astype(int)]
        chunks = []
        for s in starts:
            r = self.classify_chunk(w[s:s + n])
            r.update(start=float(s / SR), end=float(min(len(w), s + n) / SR))
            chunks.append(r)
        judged = [c for c in chunks if c["label"] != "unclear"]
        q = sum(c["label"] == "quran" for c in judged)
        frac = q / len(judged) if judged else 0.0
        if not judged:
            verdict = "unclear"
        elif frac >= 0.6:
            verdict = "quran"
        elif frac <= 0.2:
            verdict = "lecture"
        else:
            verdict = "mixed"                        # e.g. a lecture with recited passages
        ayat = []
        for c in chunks:
            if c["label"] == "quran":
                c["aya"] = (quran_text.locate(c["text"], top=1) or [None])[0]
                if c["aya"] and c["aya"] not in ayat:
                    ayat.append(c["aya"])
        return {"verdict": verdict, "quran_fraction": frac, "chunks": chunks, "ayat": ayat}

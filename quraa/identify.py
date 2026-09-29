"""Who is reciting? Slide 6 s windows over the audio, embed, score against the gallery."""
import numpy as np

from .audio import SR, windows


def identify(w: np.ndarray, embedder, gallery, spans=None, win_s: float = 6, hop_s: float = 3,
             max_windows: int | None = None, topn: int = 5, threshold: float | None = None) -> dict:
    """w: 16 kHz mono audio. spans: optional [(start_s, end_s)] to restrict to (e.g. Quran parts only).

    Decision = mean similarity over windows (robust), votes shown as a sanity check.
    Accepted if the similarity clears the gallery threshold OR the winner is clearly ahead of the
    runner-up (margin): real-world recordings score lower than studio ones, but a known reciter
    still stands out, while an unknown voice sits close to several people (see build/make_gallery.py)."""
    if max_windows is None:
        max_windows = 80 if embedder.device.startswith("cuda") else 40
    if spans:
        pieces = [w[int(a * SR): int(b * SR)] for a, b in spans]
        w = np.concatenate([p for p in pieces if len(p)]) if pieces else w[:0]
    starts = windows(w, win_s, hop_s, max_windows)
    if not starts:
        return {"ok": False, "reason": "no usable audio (silence or too short)"}
    win = int(win_s * SR)
    E = embedder([w[s:s + win] for s in starts])
    S = gallery.score_people(E)                      # [n_windows, n_people]
    mean = S.mean(0)
    votes = np.bincount(S.argmax(1), minlength=S.shape[1])
    order = np.argsort(-mean)
    best = order[0]
    thr = gallery.threshold if threshold is None else threshold
    margin = float(mean[best] - mean[order[1]]) if len(order) > 1 else 1.0
    margin_thr = gallery.meta.get("margin_threshold", 0.10)

    # which recording set (style) of the winner matched best
    L = gallery.score_labels(E).mean(0)
    idx = [j for j, p in enumerate(gallery.people) if p == gallery.person_ids[best]]
    best_label = gallery.labels[max(idx, key=lambda j: L[j])]

    return {
        "ok": True,
        "person": gallery.person_ids[best],
        "name": gallery.name(gallery.person_ids[best], "ar"),
        "name_en": gallery.name(gallery.person_ids[best], "en"),
        "label": best_label,
        "style": gallery.style(best_label),
        "similarity": float(mean[best]),
        "margin": margin,
        "votes": int(votes[best]),
        "n_windows": len(starts),
        "unknown": bool(thr is not None and mean[best] < thr and margin < margin_thr),
        "threshold": thr,
        "top": [{"person": gallery.person_ids[i], "name": gallery.name(gallery.person_ids[i], "ar"),
                 "similarity": float(mean[i]), "votes": int(votes[i])} for i in order[:topn]],
    }

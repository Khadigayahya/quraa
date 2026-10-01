"""Gallery (.npz + .json) -> files the in-browser app reads.

    python build/export_web_gallery.py [--gallery models/gallery_v2.npz] [--out web/models]

gallery.bin  float32, little-endian: proj_mean[D] | proj_W[D x K] | C[N x K]   (no projection: C[N x D])
gallery.json everything else (labels, people, names, styles, thresholds, shapes)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quraa.gallery import Gallery  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=str(ROOT / "models" / "gallery_v2.npz"))
    ap.add_argument("--out", default=str(ROOT / "web" / "models"))
    a = ap.parse_args()
    g = Gallery.load(a.gallery)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    parts = []
    if g.proj_W is not None:
        parts += [g.proj_mean.astype("<f4").ravel(), g.proj_W.astype("<f4").ravel()]
    parts.append(g.C.astype("<f4").ravel())
    (out / "gallery.bin").write_bytes(np.concatenate(parts).tobytes())

    people = g.person_ids
    info = {
        "version": g.meta.get("source", "v?"),
        "dim": 192,
        "proj_k": int(g.proj_W.shape[1]) if g.proj_W is not None else 0,
        "n": len(g.labels),
        "labels": g.labels,
        "person_index": [people.index(p) for p in g.people],        # centroid -> person
        "people": people,
        "names": [g.name(p, "ar") for p in people],
        "names_en": [g.name(p, "en") for p in people],
        "styles": [g.style(l) for l in g.labels],
        "neighbours": [sorted(people.index(q) for q in g.neighbours().get(p, ())) for p in people],
        "threshold": g.threshold,
        "margin_threshold": g.meta.get("margin_threshold", 0.10),
    }
    (out / "gallery.json").write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    print(f"{len(people)} people, {len(g.labels)} centroids, proj_k={info['proj_k']} -> "
          f"{(out / 'gallery.bin').stat().st_size / 1e3:.0f} KB + {(out / 'gallery.json').stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()

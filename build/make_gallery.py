"""Turn the mp3quran embeddings (from build/mp3quran_modal.py) into models/gallery_v2.npz,
merged with the everyayah gallery, and measure accuracy on held-out surahs.

    modal volume get quraa-build emb build/emb
    python build/make_gallery.py [--method auto|B|C]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quraa import reciters                     # noqa: E402
from quraa.gallery import Gallery, _norm       # noqa: E402

_DIAC = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u0640]")
_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي"})


def name_tokens(s):
    s = _DIAC.sub("", s).translate(_MAP)
    s = re.sub(r"\b(الشيخ|القارئ|الدكتور|د\.)\s*", "", s)
    s = re.sub(r"عبد\s+ال", "عبدال", s)
    s = re.sub(r"\bبن\b|\bابن\b", " ", s)
    return [t for t in re.split(r"[\s\-]+", s) if t]


def same_person(a, b):
    ta, tb = name_tokens(a), name_tokens(b)
    if min(len(ta), len(tb)) < 2 or ta[-1] != tb[-1]:
        return False
    return set(ta) <= set(tb) or set(tb) <= set(ta)


def load_mp3q(emb_dir):
    rows = []
    for f in sorted(Path(emb_dir).glob("*.npz")):
        meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
        d = np.load(f)
        rows.append({**meta, "enroll": d["enroll"], "enroll_aug": d["enroll_aug"], "test": d["test"],
                     "test_surah": d["test_surah"], "label": f"mp3q/{meta['reciter_id']}/{meta['moshaf_id']}",
                     "person": f"mp3q_{meta['reciter_id']}"})
    return rows


def build(rows, method, people_ok=None, extra=None):
    """Gallery from enroll(+aug) windows. extra = (labels, centroids, people) from everyayah (raw space)."""
    use = [r for r in rows if people_ok is None or r["person"] in people_ok]
    X = np.vstack([np.vstack([r["enroll"], r["enroll_aug"]]) for r in use])
    L = np.concatenate([[r["label"]] * (len(r["enroll"]) + len(r["enroll_aug"])) for r in use])
    P = np.concatenate([[r["person"]] * (len(r["enroll"]) + len(r["enroll_aug"])) for r in use])
    proj = None
    if method == "C":
        Xn = _norm(X)
        proj = (Xn.mean(0), Gallery.fit_lda(Xn, P, n_components=150)[1])
    g = Gallery.build(X, L, proj=proj, people=P)
    if extra is not None:
        el, ec, ep = extra
        keep = [i for i, p in enumerate(ep) if people_ok is None or p in people_ok]
        C = np.vstack([g.C, g.transform(ec[keep])])
        g = Gallery(g.labels + [el[i] for i in keep], C, g.proj_mean, g.proj_W,
                    people=g.people + [ep[i] for i in keep])
    return g


def evaluate(g, rows, only=None, k=5):
    """Surah-level decision exactly like the app: mean window score per person."""
    hits1 = hits5 = n = 0
    best = []
    for r in rows:
        if only is not None and r["person"] not in only:
            continue
        for s in np.unique(r["test_surah"]):
            E = r["test"][r["test_surah"] == s]
            m = g.score_people(E).mean(0)
            order = np.argsort(-m)
            top = [g.person_ids[i] for i in order[:k]]
            hits1 += top[0] == r["person"]; hits5 += r["person"] in top; n += 1
            best.append(float(m[order[0]]))
    return (hits1 / n if n else 0.0), (hits5 / n if n else 0.0), n, np.array(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=str(ROOT / "build" / "emb"))
    ap.add_argument("--method", default="auto", choices=["auto", "B", "C"])
    ap.add_argument("--base", default=str(ROOT / "models" / "gallery_ecapa_v0.npz"))
    ap.add_argument("--out", default=str(ROOT / "models" / "gallery_v2.npz"))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rows = load_mp3q(a.emb)
    names = {r["person"]: r["name"] for r in rows}
    names_en = {r["person"]: r["name_en"] for r in rows if r["name_en"]}
    print(f"{len(rows)} recordings, {len(names)} reciters from mp3quran")

    # everyayah centroids: attach to the matching mp3quran person when the names agree
    base = Gallery.load(a.base)
    ea_people, merged = [], {}
    for lab, pid in zip(base.labels, base.people):
        ar = reciters.display_name(pid, "ar")
        hit = next((p for p, n in names.items() if same_person(ar, n)), None)
        if hit:
            merged[pid] = hit
        ea_people.append(hit or f"ea_{pid}")
        if not hit:
            names[f"ea_{pid}"] = ar
            names_en[f"ea_{pid}"] = reciters.display_name(pid, "en")
    print(f"everyayah: {len(set(base.people))} reciters, {len(merged)} matched to mp3quran by name:")
    for pid, hit in sorted(merged.items()):
        print(f"   {reciters.display_name(pid):<24} = {names[hit]}")
    extra = (["ea/" + l for l in base.labels], base.C, ea_people)

    # ---- closed-set accuracy on held-out surahs, per method ----
    res = {}
    for m in (["B", "C"] if a.method == "auto" else [a.method]):
        g = build(rows, m, extra=extra)
        t1, t5, n, _ = evaluate(g, rows)
        res[m] = t1
        print(f"method {m}: {len(g.person_ids)} people | held-out surahs: top-1 {t1:.1%} | top-5 {t5:.1%} | n={n}")
    method = max(res, key=res.get)

    # ---- open-set: hide 20% of people (also from LDA), calibrate on 1st test surah, measure on 2nd ----
    rng = np.random.default_rng(0)
    people = sorted({r["person"] for r in rows})
    hidden = set(rng.choice(people, len(people) // 5, replace=False))
    known = set(people) - hidden
    g = build(rows, method, people_ok=known, extra=extra)
    first = [dict(r, test=r["test"][r["test_surah"] == r["test_surah"][0]], test_surah=r["test_surah"][r["test_surah"] == r["test_surah"][0]])
             for r in rows if len(r["test_surah"])]
    second = [dict(r, test=r["test"][r["test_surah"] != r["test_surah"][0]], test_surah=r["test_surah"][r["test_surah"] != r["test_surah"][0]])
              for r in rows if len(r["test_surah"])]
    _, _, _, cal = evaluate(g, first, only=known)
    thr = float(np.percentile(cal, 5))
    t1k, _, nk, bk = evaluate(g, second, only=known)
    _, _, nu, bu = evaluate(g, second, only=hidden)
    print(f"open-set ({len(hidden)} hidden reciters): threshold {thr:.3f} | known accepted {(bk >= thr).mean():.1%} "
          f"| unknown rejected {(bu < thr).mean():.1%} | top-1 known {t1k:.1%}")

    # ---- final gallery: everything ----
    g = build(rows, method, extra=extra)
    g.threshold = thr
    g.names = names
    g.styles = {r["label"]: r["moshaf"] for r in rows}
    g.meta = {"source": "v2", "method": method, "margin_threshold": 0.10, "names_en": names_en, "people": len(g.person_ids),
              "recordings": len(g.labels), "closed_set_top1": res[method],
              "open_set": {"threshold": thr, "known_accepted": float((bk >= thr).mean()),
                           "unknown_rejected": float((bu < thr).mean())}}
    g.save(a.out)
    print(f"saved {a.out}: {len(g.person_ids)} reciters, {len(g.labels)} recording sets, method {method}")


if __name__ == "__main__":
    main()

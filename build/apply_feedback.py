"""Turn user feedback into a better gallery — carefully.

    # from the shared database (Supabase service key, never the public one):
    SUPABASE_URL=... SUPABASE_SERVICE_KEY=... python build/apply_feedback.py
    # or from an exported file:
    python build/apply_feedback.py --json feedback.json

What it does
  1. Labels every rating: 👍 → the predicted reciter; 👎 + a name → that reciter (matched to the gallery by
     name when possible); 👎 without a name → counted in the accuracy only.
  2. Keeps one rating per (file, reciter), and sets 20% of files aside as a test set (by file hash, so the
     same file never lands on both sides).
  3. Adds a "real-world" voiceprint per reciter from the remaining ratings (at least MIN_FILES different
     files), after dropping ratings whose voice is nowhere near that reciter (wrong or joke labels).
  4. New names (not in the gallery) become *proposals* in build/feedback_review.json. A person on the team
     sets "approve": true (and can fix the spelling) — only approved names are added.
  5. Builds models/gallery_v3.npz and prints accuracy before/after on the held-out ratings (and on the
     mp3quran held-out surahs when build/emb/ exists). Refuses to save if the studio accuracy drops.
Then: python build/export_web_gallery.py --gallery models/gallery_v3.npz  and upload web/models to the Hub.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "build"))
from quraa.gallery import Gallery, _norm      # noqa: E402
from make_gallery import name_tokens, same_person   # noqa: E402

MIN_FILES = 2           # different files before a reciter gets a real-world voiceprint
MIN_NEW_FILES = 3       # different files before a new name is even proposed
LABEL_FLOOR = 0.15      # a rating whose voice is below this vs. its reciter is treated as a wrong label
CONFLICT_GAP = 0.15     # app was sure of someone else by this much → needs CONFLICT_FILES agreeing files
CONFLICT_FILES = 3
NEW_CONSISTENCY = 0.40  # new name: its recordings must sound alike (mean pairwise similarity)
REVIEW = ROOT / "build" / "feedback_review.json"


# ---------------------------------------------------------------- loading
def from_supabase():
    url, key = os.environ.get("SUPABASE_URL", "").rstrip("/"), os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY (Dashboard → Project Settings → API → service_role).")
    rows, start, page = [], 0, 1000
    while True:
        req = urllib.request.Request(f"{url}/rest/v1/feedback?select=*&kind=eq.reciter&order=id",
                                     headers={"apikey": key, "Authorization": f"Bearer {key}",
                                              "Range": f"{start}-{start + page - 1}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            batch = json.loads(r.read().decode("utf-8"))
        rows += batch
        if len(batch) < page:
            return rows
        start += page


def norm_key(name):
    return "_".join(name_tokens(name or ""))


def windows_of(row):
    w = row.get("window_embeddings")
    if w and len(w) % 192 == 0:
        return np.asarray(w, np.float32).reshape(-1, 192)
    return np.asarray(row["embedding"], np.float32)[None, :]


def voiceprint(row):
    """One L2-normalised raw voiceprint per rating (same space as the gallery's inputs)."""
    return _norm(windows_of(row).mean(0))


def is_test(row):
    key = row.get("file_hash") or str(row.get("id"))
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % 5 == 0


# ---------------------------------------------------------------- labelling
def label_rows(rows, g):
    by_name = {}
    for pid in g.person_ids:
        by_name[norm_key(g.name(pid, "ar"))] = pid
    stats = defaultdict(int)
    out = []
    for r in rows:
        if r.get("kind", "reciter") != "reciter" or not r.get("embedding"):
            continue
        stats[r["verdict"]] += 1
        if r["verdict"] == "correct":
            pid = r.get("true_person") or r.get("predicted_person")
        elif r["verdict"] == "wrong" and (r.get("true_person") or r.get("true_name")):
            pid = r.get("true_person")
            if not pid or pid not in g.person_ids:
                name = r.get("true_name") or ""
                pid = by_name.get(norm_key(name)) or next(
                    (p for p in g.person_ids if same_person(name, g.name(p, "ar"))), None)
                if pid is None:
                    out.append({**r, "label": None, "new_key": norm_key(name), "new_name": name.strip()})
                    continue
        else:
            continue
        if pid and pid.startswith("user:"):          # a name only this user's browser knew
            out.append({**r, "label": None, "new_key": norm_key(r.get("true_name") or pid[5:]), "new_name": r.get("true_name") or pid[5:]})
            continue
        if pid in g.person_ids:
            out.append({**r, "label": pid})
    # one rating per (file, label): the latest wins
    seen, uniq = {}, []
    for r in sorted(out, key=lambda r: r.get("id") or 0):
        seen[(r.get("file_hash") or f"id{r.get('id')}", r["label"] or "new:" + r["new_key"])] = r
    uniq = list(seen.values())
    return uniq, dict(stats)


# ---------------------------------------------------------------- evaluation
def accuracy(g, rows):
    rows = [r for r in rows if r["label"] and r["label"] in g.person_ids]
    if not rows:
        return None, 0
    hits = sum(g.person_ids[int(np.argmax(g.score_people(windows_of(r)).mean(0)))] == r["label"] for r in rows)
    return hits / len(rows), len(rows)


def studio_accuracy(g):
    emb = ROOT / "build" / "emb"
    if not emb.exists():
        return None
    import make_gallery as mg
    return mg.evaluate(g, mg.load_mp3q(emb))[0]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="feedback rows exported as JSON (instead of Supabase)")
    ap.add_argument("--base", default=str(ROOT / "models" / "gallery_v2.npz"))
    ap.add_argument("--out", default=str(ROOT / "models" / "gallery_v3.npz"))
    ap.add_argument("--max-studio-drop", type=float, default=0.005)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rows = json.loads(Path(a.json).read_text(encoding="utf-8")) if a.json else from_supabase()
    base = Gallery.load(a.base)
    labelled, stats = label_rows(rows, base)
    rated = stats.get("correct", 0) + stats.get("wrong", 0)
    print(f"{len(rows)} ratings: 👍 {stats.get('correct', 0)} · 👎 {stats.get('wrong', 0)} · unsure {stats.get('unsure', 0)}"
          + (f" → users say {stats.get('correct', 0) / rated:.1%} correct" if rated else ""))

    train = [r for r in labelled if not is_test(r)]
    test = [r for r in labelled if is_test(r)]

    # ---- existing reciters: real-world voiceprints ----
    per = defaultdict(list)
    for r in train:
        if r["label"]:
            per[r["label"]].append(r)
    g = base
    added, dropped, held = [], 0, 0
    for pid, rs in per.items():
        E = np.vstack([voiceprint(r) for r in rs])
        S = base.score_people(E)
        own = S[:, base.person_ids.index(pid)]
        keep = own >= LABEL_FLOOR                                    # voice nowhere near this reciter → bad label
        # the app was confidently someone else: only trust it when several different files say the same
        conflict = (S.max(1) - own) >= CONFLICT_GAP
        conflict_files = {rs[i].get("file_hash") or rs[i].get("id") for i in np.where(conflict & keep)[0]}
        if conflict.any() and len(conflict_files) < CONFLICT_FILES:
            held += int((conflict & keep).sum()); keep &= ~conflict
        if keep.sum() >= 3:                                          # and drop outliers among the rest
            Z = base.transform(E[keep]); c = _norm(Z.mean(0, keepdims=True))
            ok = (Z @ c.T).ravel() >= 0.3
            idx = np.where(keep)[0]; keep[idx[~ok]] = False
        dropped += int((own < LABEL_FLOOR).sum())
        files = {rs[i].get("file_hash") or rs[i].get("id") for i in np.where(keep)[0]}
        if len(files) >= MIN_FILES:
            g = g.enroll(f"fb/{pid}", E[keep], person=pid, style="")
            added.append((pid, int(keep.sum())))
    print(f"real-world voiceprints added for {len(added)} reciters ({sum(n for _, n in added)} ratings used, "
          f"{dropped} dropped as unreliable, {held} held back: they contradict a confident result and too few files agree)")

    # ---- new names: propose, add only when approved ----
    review = json.loads(REVIEW.read_text(encoding="utf-8")) if REVIEW.exists() else {"names": {}}
    groups = defaultdict(list)
    for r in labelled:
        if not r["label"] and r["new_key"]:
            groups[r["new_key"]].append(r)
    for key, rs in groups.items():
        files = {r.get("file_hash") or r.get("id") for r in rs}
        E = np.vstack([voiceprint(r) for r in rs])
        Z = base.transform(E)
        cons = float((Z @ Z.T)[np.triu_indices(len(Z), 1)].mean()) if len(Z) > 1 else 0.0
        entry = review["names"].setdefault(key, {"name": rs[0]["new_name"], "approve": False})
        entry.update(files=len(files), ratings=len(rs), consistency=round(cons, 3),
                     ready=len(files) >= MIN_NEW_FILES and cons >= NEW_CONSISTENCY)
        if entry["approve"] and entry["ready"]:
            pid = f"fb_{key}"
            g = g.enroll(f"fb/new/{key}", E, display_name=entry["name"], person=pid, style="")
            print(f"  + new reciter: {entry['name']} ({len(files)} files)")
    REVIEW.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    pending = [v["name"] for v in review["names"].values() if v["ready"] and not v["approve"]]
    if pending:
        print(f"  {len(pending)} new names are ready for review in {REVIEW.name}: {', '.join(pending[:10])}")

    # ---- measure, then save ----
    b_acc, n = accuracy(base, test)
    a_acc, _ = accuracy(g, test)
    if n:
        print(f"held-out real-world ratings (n={n}): before {b_acc:.1%} → after {a_acc:.1%}")
    else:
        print("held-out real-world ratings: none yet (need more feedback)")
    s0, s1 = studio_accuracy(base), studio_accuracy(g)
    if s0 is not None:
        print(f"mp3quran held-out surahs: before {s0:.1%} → after {s1:.1%}")
        if s1 < s0 - a.max_studio_drop:
            sys.exit("studio accuracy dropped — not saving. Check the ratings (or raise --max-studio-drop).")
    if g is base:
        print("nothing to add yet — gallery unchanged.")
        return
    g.threshold = base.threshold
    used = sum(n for _, n in added)
    digest = hashlib.sha256(g.C.tobytes()).hexdigest()[:8]           # new content → new version → browsers refresh
    g.meta = {**base.meta, "source": f"v3.{digest}", "feedback": {"ratings": len(rows), "used": used,
              "reciters_improved": len(added), "heldout_before": b_acc, "heldout_after": a_acc, "heldout_n": n}}
    g.save(a.out)
    print(f"saved {a.out} ({len(g.person_ids)} reciters, {len(g.labels)} voiceprints, version {g.meta['source']})")


if __name__ == "__main__":
    main()

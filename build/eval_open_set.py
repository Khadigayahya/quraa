"""Why does the app sometimes name the wrong reciter with confidence? Compare decision rules.

Test material
  * real-world clips we have locally, labelled: reciter in the gallery (must be accepted with the right name)
    or not in the gallery (must be rejected as "unknown");
  * the mp3quran open-set experiment: 20% of reciters hidden from the gallery, their held-out surahs must be
    rejected, everyone else's accepted.

Rules compared (all use the same voiceprints; only the decision changes)
  current   sim >= thr  OR  margin >= 0.10
  strict    sim >= thr  AND margin >= m
  asnorm    adaptive score normalisation (top-K cohort on both sides) >= t
  combo     asnorm >= t  AND  margin >= m
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "build"))
from quraa.gallery import Gallery, _norm  # noqa: E402
import make_gallery as mg  # noqa: E402

K = 40   # cohort size for AS-norm


def remove(g, person):
    keep = [i for i, p in enumerate(g.people) if p != person]
    return Gallery([g.labels[i] for i in keep], g.C[keep], g.proj_mean, g.proj_W, g.threshold, g.names, g.meta,
                   [g.people[i] for i in keep], g.styles)


class Scorer:
    """Person scores for one clip (mean over windows), raw and AS-normalised."""

    def __init__(self, g):
        self.g = g
        # centroid-side cohort stats: each centroid against all other centroids
        S = g.C @ g.C.T
        np.fill_diagonal(S, -np.inf)
        top = -np.sort(-S, 1)[:, :K]
        self.c_mu, self.c_sd = top.mean(1), top.std(1) + 1e-6

    def __call__(self, E):
        g = self.g
        Z = g.transform(E)
        S = Z @ g.C.T                                        # windows x centroids
        s = S.mean(0)                                        # clip x centroids
        top = -np.sort(-s)[:K]
        t_mu, t_sd = top.mean(), top.std() + 1e-6
        a = 0.5 * ((s - t_mu) / t_sd + (s - self.c_mu) / self.c_sd)
        P = np.full(len(g.person_ids), -9.0); A = np.full(len(g.person_ids), -9.0)
        for j, p in enumerate(g._pidx):
            P[p] = max(P[p], s[j]); A[p] = max(A[p], a[j])
        o = np.argsort(-P)
        return {"best": g.person_ids[o[0]], "sim": P[o[0]], "margin": P[o[0]] - P[o[1]],
                "as_best": g.person_ids[int(np.argmax(A))], "as": float(A.max())}


RULES = {
    "current (sim≥thr OR margin≥0.10)": lambda r, thr: r["sim"] >= thr or r["margin"] >= 0.10,
    **{f"strict (sim≥thr AND margin≥{m:.2f})": (lambda m: lambda r, thr: r["sim"] >= thr and r["margin"] >= m)(m)
       for m in (0.05, 0.08, 0.10)},
    **{f"asnorm ≥ {t:.1f}": (lambda t: lambda r, thr: r["as"] >= t)(t) for t in (2.5, 3.0, 3.5, 4.0)},
    **{f"asnorm ≥ {t:.1f} AND margin≥0.05": (lambda t: lambda r, thr: r["as"] >= t and r["margin"] >= 0.05)(t)
       for t in (2.5, 3.0)},
}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    real = json.loads((ROOT / "build" / "eval_clips.json").read_text(encoding="utf-8"))
    g_full = Gallery.load(ROOT / "models" / "gallery_v2.npz")
    thr = g_full.threshold

    # ---------- real-world clips ----------
    print(f"real-world clips: {len(real)}")
    cases = []
    for c in real:
        g = remove(g_full, c["hide"]) if c.get("hide") else g_full
        r = Scorer(g)(np.asarray(c["E"], np.float32))
        cases.append((c, r))
        print(f"  {'IN ' if c['truth'] else 'OUT'} {c['name'][:34]:<34} best={g.name(r['best'])[:18]:<18} sim={r['sim']:.3f} "
              f"margin={r['margin']:.3f} asnorm={r['as']:.2f}")

    # ---------- studio open-set (mp3quran) ----------
    rows = mg.load_mp3q(ROOT / "build" / "emb")
    rng = np.random.default_rng(0)
    people = sorted({r["person"] for r in rows})
    hidden = set(rng.choice(people, len(people) // 5, replace=False))
    g_os = g_full
    for p in hidden:
        g_os = remove(g_os, p)
    sc = Scorer(g_os)
    studio = []
    for r in rows:
        for s in np.unique(r["test_surah"]):
            studio.append((r["person"] not in hidden, r["person"], sc(r["test"][r["test_surah"] == s])))

    print("\nrule                                   | real: right name | real: unknowns rejected | studio: known right | studio: unknown rejected")
    for name, rule in RULES.items():
        ok_in = [rule(r, thr) and r["best"] == c["truth"] for c, r in cases if c["truth"]]
        ok_out = [not rule(r, thr) for c, r in cases if not c["truth"]]
        s_in = [rule(r, thr) and r["best"] == p for known, p, r in studio if known]
        s_out = [not rule(r, thr) for known, p, r in studio if not known]
        print(f"{name:<39}| {sum(ok_in)}/{len(ok_in):<15}| {sum(ok_out)}/{len(ok_out):<22}| "
              f"{np.mean(s_in):6.1%}            | {np.mean(s_out):6.1%}")


if __name__ == "__main__":
    main()

"""Reference voiceprints ("gallery") and scoring.

Each recording set (folder label) keeps its own centroid; a person's score is the best
of their centroids. An optional linear projection (LDA trained on augmented data) is
applied to every embedding before cosine scoring.
"""
import json
from pathlib import Path

import numpy as np

from . import reciters


def _norm(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


class Gallery:
    def __init__(self, labels, centroids, proj_mean=None, proj_W=None, threshold=None, names=None, meta=None,
                 people=None, styles=None):
        self.labels = list(labels)                         # one per centroid (recording set)
        # person per centroid: given explicitly (mp3quran sets) or derived from everyayah folder names
        self.people = list(people) if people is not None else [reciters.person_of(l) for l in self.labels]
        self.C = _norm(np.asarray(centroids, dtype=np.float32))
        self.proj_mean = proj_mean
        self.proj_W = proj_W
        self.threshold = threshold
        self.names = dict(names or {})                     # person id -> display name overrides
        self.styles = dict(styles or {})                   # label -> recording style ("حفص عن عاصم - مرتل")
        self.meta = dict(meta or {})
        self.person_ids = sorted(set(self.people))
        self._pidx = np.array([self.person_ids.index(p) for p in self.people])

    # ---- building -------------------------------------------------------------------------
    @staticmethod
    def fit_lda(E, y, n_components=None):
        """Fit an LDA projection on (ideally augmented) embeddings. Returns (mean, W)."""
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        n_cls = len(set(y))
        k = min(n_components or n_cls - 1, n_cls - 1, E.shape[1])
        lda = LinearDiscriminantAnalysis(n_components=k, solver="eigen", shrinkage="auto").fit(E, y)
        return E.mean(0).astype(np.float32), lda.scalings_[:, :k].astype(np.float32)

    def transform(self, E):
        E = np.atleast_2d(E).astype(np.float32)
        if self.proj_W is not None:
            E = (E - self.proj_mean) @ self.proj_W
        return _norm(E)

    @classmethod
    def build(cls, E, labels, proj=None, **kw):
        """E: raw embeddings [N, D], labels: folder label per row. proj: (mean, W) or None."""
        labels = np.asarray(labels)
        g = cls([], np.zeros((0, 1)), *(proj or (None, None)), **kw)
        Z = g.transform(E)
        uniq = sorted(set(labels))
        C = np.stack([Z[labels == l].mean(0) for l in uniq])
        if kw.get("people") is not None:                   # people given per row -> one per centroid
            per = dict(zip(labels, kw["people"]))
            kw["people"] = [per[l] for l in uniq]
        return cls(uniq, C, *(proj or (None, None)), **kw)

    def enroll(self, label, E_raw, display_name=None, person=None, style=None):
        """Add a new reciter (or recording set) from a few clean clips' raw embeddings."""
        c = self.transform(E_raw).mean(0, keepdims=True)
        person = person or (self.people[self.labels.index(label)] if label in self.labels else reciters.person_of(label))
        labels, C, people = list(self.labels), self.C.copy(), list(self.people)
        if label in labels:
            C[labels.index(label)] = _norm(c)[0]
        else:
            labels.append(label); C = np.vstack([C, c]); people.append(person)
        new = Gallery(labels, C, self.proj_mean, self.proj_W, self.threshold, self.names, self.meta, people, self.styles)
        if display_name:
            new.names[person] = display_name
        if style:
            new.styles[label] = style
        return new

    def style(self, label):
        return self.styles.get(label) or reciters.STYLES_AR.get(reciters.style_of(label), "")

    # ---- scoring --------------------------------------------------------------------------
    def score_labels(self, E):
        return self.transform(E) @ self.C.T                              # [N, n_labels]

    def score_people(self, E):
        S = self.score_labels(E)
        P = np.full((S.shape[0], len(self.person_ids)), -1.0, dtype=np.float32)
        for j, p in enumerate(self._pidx):
            P[:, p] = np.maximum(P[:, p], S[:, j])
        return P                                                         # [N, n_people]

    def name(self, pid, lang="ar"):
        if lang != "ar":
            return self.meta.get("names_en", {}).get(pid) or reciters.display_name(pid, lang)
        return self.names.get(pid) or reciters.display_name(pid, lang)

    # ---- io -------------------------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        arrays = {"C": self.C, "labels": np.array(self.labels), "people": np.array(self.people)}
        if self.proj_W is not None:
            arrays.update(proj_mean=self.proj_mean, proj_W=self.proj_W)
        np.savez(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps(
            {"threshold": self.threshold, "names": self.names, "styles": self.styles, "meta": self.meta},
            ensure_ascii=False, indent=2),
            encoding="utf-8")

    @classmethod
    def load(cls, path):
        path = Path(path)
        d = np.load(path, allow_pickle=False)
        if "G" in d:                                     # baseline v0 file: G + classes, no projection
            return cls([str(c) for c in d["classes"]], d["G"], meta={"source": "v0"})
        info = {}
        if path.with_suffix(".json").exists():
            info = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        return cls([str(l) for l in d["labels"]], d["C"],
                   d["proj_mean"] if "proj_mean" in d else None,
                   d["proj_W"] if "proj_W" in d else None,
                   info.get("threshold"), info.get("names"), info.get("meta"),
                   [str(p) for p in d["people"]] if "people" in d else None, info.get("styles"))

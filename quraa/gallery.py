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
    def __init__(self, labels, centroids, proj_mean=None, proj_W=None, threshold=None, names=None, meta=None):
        self.labels = list(labels)                         # one per centroid (folder label)
        self.people = [reciters.person_of(l) for l in self.labels]
        self.C = _norm(np.asarray(centroids, dtype=np.float32))
        self.proj_mean = proj_mean
        self.proj_W = proj_W
        self.threshold = threshold
        self.names = dict(names or {})                     # person id -> display name overrides
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
        return cls(uniq, C, *(proj or (None, None)), **kw)

    def enroll(self, label, E_raw, display_name=None):
        """Add a new reciter (or recording set) from a few clean clips' raw embeddings."""
        c = self.transform(E_raw).mean(0, keepdims=True)
        if label in self.labels:
            self.C[self.labels.index(label)] = _norm(c)[0]
            new = Gallery(self.labels, self.C, self.proj_mean, self.proj_W, self.threshold, self.names, self.meta)
        else:
            new = Gallery(self.labels + [label], np.vstack([self.C, c]), self.proj_mean, self.proj_W,
                          self.threshold, self.names, self.meta)
        if display_name:
            new.names[reciters.person_of(label)] = display_name
        return new

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
        return self.names.get(pid) or reciters.display_name(pid, lang)

    # ---- io -------------------------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        arrays = {"C": self.C, "labels": np.array(self.labels)}
        if self.proj_W is not None:
            arrays.update(proj_mean=self.proj_mean, proj_W=self.proj_W)
        np.savez(path, **arrays)
        path.with_suffix(".json").write_text(json.dumps(
            {"threshold": self.threshold, "names": self.names, "meta": self.meta}, ensure_ascii=False, indent=2),
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
                   info.get("threshold"), info.get("names"), info.get("meta"))

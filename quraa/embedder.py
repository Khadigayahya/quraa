"""Voice embeddings: SpeechBrain ECAPA-TDNN (VoxCeleb), optionally with fine-tuned weights."""
from pathlib import Path

import numpy as np
import torch

ECAPA_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"


class Embedder:
    def __init__(self, finetuned: str | None = None, device: str | None = None, cache_dir: str = "pretrained_models"):
        from speechbrain.inference.speaker import EncoderClassifier

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        kw = {}
        try:  # symlinks need admin rights on Windows -> copy the files instead
            from speechbrain.utils.fetching import LocalStrategy
            kw["local_strategy"] = LocalStrategy.COPY
        except ImportError:
            pass
        self.clf = EncoderClassifier.from_hparams(
            source=ECAPA_SOURCE, savedir=str(Path(cache_dir) / "ecapa"), run_opts={"device": self.device}, **kw)
        self.clf.mods.eval()
        self.finetuned = None
        if finetuned and Path(finetuned).exists():
            state = torch.load(finetuned, map_location=self.device)
            self.clf.mods.embedding_model.load_state_dict(state["embedding_model"])
            self.finetuned = str(finetuned)

    @property
    def model(self):
        return self.clf.mods.embedding_model

    def forward(self, wavs: torch.Tensor, lens: torch.Tensor | None = None) -> torch.Tensor:
        """Differentiable path used for fine-tuning: [B, T] waveforms -> [B, 192] (not normalised)."""
        if lens is None:
            lens = torch.ones(wavs.shape[0], device=wavs.device)
        feats = self.clf.mods.compute_features(wavs)
        feats = self.clf.mods.mean_var_norm(feats, lens)
        return self.model(feats, lens).squeeze(1)

    @torch.no_grad()
    def __call__(self, wavs, batch_size: int = 32) -> np.ndarray:
        """list of equal-or-different-length float32 arrays (or one array) -> [N, 192] L2-normalised."""
        single = isinstance(wavs, np.ndarray) and wavs.ndim == 1
        if single:
            wavs = [wavs]
        self.model.eval()
        out = []
        for i in range(0, len(wavs), batch_size):
            chunk = wavs[i:i + batch_size]
            T = max(len(w) for w in chunk)
            x = torch.zeros(len(chunk), T)
            for j, w in enumerate(chunk):
                x[j, : len(w)] = torch.from_numpy(np.asarray(w, dtype=np.float32))
            lens = torch.tensor([len(w) / T for w in chunk])
            e = self.forward(x.to(self.device), lens.to(self.device)).cpu().numpy()
            out.append(e)
        E = np.concatenate(out)
        E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
        return E[0] if single else E

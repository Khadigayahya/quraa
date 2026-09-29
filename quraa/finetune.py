"""Fine-tune ECAPA on recitation with AAM-softmax + on-the-fly degradations.

Goal: close the clean-vs-real-world gap (baseline: 94% clean -> 73% noisy+reverb).
Labels are *people* (all recording sets of one reciter = one class), so the model
learns "same voice, different style/room/mic" instead of memorising recording sets.
"""
import math

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .augment import augment

SR = 16000


class CropDataset(Dataset):
    def __init__(self, paths, labels, crop_s=3.0, aug_p=0.8, babble_paths=None, seed=0):
        self.paths, self.labels = list(paths), np.asarray(labels)
        self.crop = int(crop_s * SR)
        self.aug_p = aug_p
        self.babble_paths = list(babble_paths or [])
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.paths)

    def _crop(self, w, rng):
        if len(w) >= self.crop:
            s = rng.integers(0, len(w) - self.crop + 1)
            return w[s:s + self.crop]
        return np.pad(w, (0, self.crop - len(w)), mode="wrap")

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, self.epoch, i))
        w, _ = sf.read(self.paths[i], dtype="float32")
        w = self._crop(w if w.ndim == 1 else w.mean(1), rng)
        if rng.random() < self.aug_p:
            babble = None
            if self.babble_paths:
                b, _ = sf.read(self.babble_paths[rng.integers(len(self.babble_paths))], dtype="float32")
                babble = b if b.ndim == 1 else b.mean(1)
            w = augment(w, rng, babble=babble)
        return torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32)), int(self.labels[i])


class AAMSoftmax(nn.Module):
    def __init__(self, dim, n_classes, margin=0.2, scale=30.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_classes, dim) * 0.01)
        self.m, self.s = margin, scale

    def forward(self, e, y):
        cos = F.linear(F.normalize(e), F.normalize(self.W)).clamp(-1 + 1e-7, 1 - 1e-7)
        theta = torch.acos(cos)
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(y, cos.shape[1]).bool()
        logits = torch.where(onehot, target, cos) * self.s
        return F.cross_entropy(logits, y), (cos.argmax(1) == y).float().mean()


def train(embedder, paths, labels, out_path, epochs=8, batch_size=48, lr=5e-5, head_lr=1e-3,
          crop_s=3.0, workers=2, eval_fn=None, log=print):
    """paths/labels: training wavs and integer person ids.
    eval_fn(embedder) -> float (higher is better) is called after each epoch; the best
    epoch's weights are saved to out_path. Returns the history list."""
    dev = embedder.device
    labels = np.asarray(labels)
    n_cls = int(labels.max()) + 1
    ds = CropDataset(paths, labels, crop_s, babble_paths=paths)
    batch_size = min(batch_size, len(ds))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True,
                    persistent_workers=False)
    model = embedder.model
    head = AAMSoftmax(192, n_cls).to(dev)
    opt = torch.optim.AdamW([{"params": model.parameters(), "lr": lr},
                             {"params": head.parameters(), "lr": head_lr}], weight_decay=1e-4)
    steps = max(1, epochs * len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / max(1, len(dl) // 2)) * 0.5 * (1 + math.cos(math.pi * k / steps)))

    best, hist = -1.0, []
    if eval_fn is not None:
        best = eval_fn(embedder)
        log(f"epoch 0 (pretrained): eval = {best:.4f}")
        torch.save({"embedding_model": model.state_dict()}, out_path)
    for ep in range(1, epochs + 1):
        ds.epoch = ep
        model.train()
        tot, acc, n = 0.0, 0.0, 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss, a = head(embedder.forward(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot, acc, n = tot + loss.item(), acc + a.item(), n + 1
        model.eval()
        score = eval_fn(embedder) if eval_fn is not None else -tot / n
        hist.append({"epoch": ep, "loss": tot / n, "train_acc": acc / n, "eval": score})
        log(f"epoch {ep}: loss = {tot / n:.3f} | train acc = {acc / n:.1%} | eval = {score:.4f}")
        if score > best:
            best = score
            torch.save({"embedding_model": model.state_dict()}, out_path)
            log("  saved (best so far)")
    state = torch.load(out_path, map_location=dev)
    model.load_state_dict(state["embedding_model"])
    model.eval()
    return hist

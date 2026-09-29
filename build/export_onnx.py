"""Export ECAPA (waveform -> 192-d voiceprint) to ONNX for the in-browser app, and check it
matches the PyTorch model on real audio.

    python build/export_onnx.py [--check some_audio_file ...]

The STFT is rewritten as two Conv1d layers (cos / sin kernels), which is numerically the same
thing but uses only ops every ONNX runtime (incl. onnxruntime-web) supports.
Input: float32 [batch, 96000] (6 s at 16 kHz) -> output: float32 [batch, 192] (NOT normalised).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch  # before onnxruntime (Windows DLL order)
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quraa.embedder import Embedder  # noqa: E402

WIN = 96000


class ConvSTFT(nn.Module):
    """== speechbrain STFT(n_fft=400, win 25 ms, hop 10 ms, hamming, center=True, constant pad)."""

    def __init__(self, n_fft=400, hop=160):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        win = torch.hamming_window(n_fft)
        n = torch.arange(n_fft).float()
        k = torch.arange(n_fft // 2 + 1).float()[:, None]
        ang = 2 * torch.pi * k * n / n_fft
        self.register_buffer("wr", (torch.cos(ang) * win)[:, None, :])
        self.register_buffer("wi", (-torch.sin(ang) * win)[:, None, :])

    def forward(self, x):                                   # x: [B, T]
        x = nn.functional.pad(x[:, None, :], (self.n_fft // 2, self.n_fft // 2))
        re = nn.functional.conv1d(x, self.wr, stride=self.hop)
        im = nn.functional.conv1d(x, self.wi, stride=self.hop)
        return (re ** 2 + im ** 2).transpose(1, 2)          # power spectrum [B, frames, 201]


class Voiceprint(nn.Module):
    def __init__(self, emb: Embedder):
        super().__init__()
        m = emb.clf.mods
        self.stft = ConvSTFT()
        self.fbank = m.compute_features.compute_fbanks     # mel filterbank + log (+ top_db)
        self.model = m.embedding_model

    def forward(self, wav):
        feats = self.fbank(self.stft(wav))
        feats = feats - feats.mean(dim=1, keepdim=True)     # InputNormalization(sentence, no std)
        # lengths=None == all windows full length (they are: fixed 6 s), and keeps batch size dynamic
        return self.model(feats, None).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "web" / "models" / "ecapa.onnx"))
    ap.add_argument("--check", nargs="*", default=[])
    a = ap.parse_args()

    emb = Embedder(device="cpu", cache_dir=str(ROOT / "models" / "pretrained"))
    net = Voiceprint(emb).eval()

    # 1) our re-implementation == speechbrain (features and embeddings)
    rng = np.random.default_rng(0)
    x = torch.from_numpy((rng.standard_normal((3, WIN)) * 0.1).astype(np.float32))
    with torch.no_grad():
        ref = emb.forward(x)
        ours = net(x)
    print("torch re-impl vs speechbrain: max |diff| =", float((ref - ours).abs().max()))

    # 2) export
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(net, (x,), a.out, input_names=["wav"], output_names=["emb"],
                      dynamic_axes={"wav": {0: "batch"}, "emb": {0: "batch"}}, opset_version=18, dynamo=True)
    print("saved", a.out, f"{Path(a.out).stat().st_size / 1e6:.1f} MB")

    # 3) onnxruntime == torch, on noise and on real recitation windows
    import onnxruntime as ort
    from quraa import audio

    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    tests = [x.numpy()]
    for f in a.check:
        w = audio.load(f, max_minutes=1)
        tests.append(np.stack([w[i:i + WIN] for i in audio.windows(w, 6, 6, max_windows=6)]).astype(np.float32))
    for t in tests:
        o = sess.run(None, {"wav": t})[0]
        r = emb.forward(torch.from_numpy(t)).detach().numpy()
        cos = (o * r).sum(1) / np.linalg.norm(o, axis=1) / np.linalg.norm(r, axis=1)
        print(f"onnx vs speechbrain on {len(t)} windows: min cosine = {cos.min():.6f}")


if __name__ == "__main__":
    main()

"""Degradations that imitate real-world recordings (mosque reverb, crowd/room noise,
phone/WhatsApp band-limiting, cheap mics). Used to train/evaluate robustness."""
import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 16000


def _colored_noise(n, rng, kind):
    x = rng.standard_normal(n)
    if kind == "white":
        return x
    f = np.fft.rfftfreq(n)
    f[0] = f[1]
    spec = np.fft.rfft(x) / (f ** (0.5 if kind == "pink" else 1.0))
    return np.fft.irfft(spec, n)


def reverb(w, rng, rt_s=None):
    rt_s = rt_s or rng.uniform(0.3, 1.5)
    n = int(rt_s * SR)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 6.9, n))   # -60 dB at rt_s
    pre = int(rng.uniform(0.005, 0.03) * SR)                        # early-reflection gap
    ir[:pre] = 0
    ir[0] = rng.uniform(0.5, 2.0) * np.linalg.norm(ir)             # direct-to-reverb ratio
    y = fftconvolve(w, ir)[: len(w)]
    return y / (np.abs(y).max() + 1e-9) * (np.abs(w).max() + 1e-9)


def add_noise(w, rng, snr_db=None, noise=None):
    snr_db = rng.uniform(3, 20) if snr_db is None else snr_db
    if noise is None:
        noise = _colored_noise(len(w), rng, rng.choice(["white", "pink", "brown"]))
    elif len(noise) < len(w):
        noise = np.tile(noise, int(np.ceil(len(w) / len(noise))))[: len(w)]
    else:
        s = rng.integers(0, len(noise) - len(w) + 1)
        noise = noise[s: s + len(w)]
    p_sig = np.mean(w ** 2) + 1e-12
    p_noise = np.mean(noise ** 2) + 1e-12
    return w + noise * np.sqrt(p_sig / (10 ** (snr_db / 10)) / p_noise)


def bandlimit(w, rng):
    lo = rng.uniform(80, 400)
    hi = rng.uniform(3000, 7000)
    sos = butter(4, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfilt(sos, w)


def clip(w, rng):
    g = rng.uniform(1.5, 4.0)
    return np.tanh(g * w / (np.abs(w).max() + 1e-9)) / np.tanh(g)


def augment(w, rng, babble=None):
    """Random chain of degradations. `babble` = optional pool of other voices (1-D array)
    mixed in at low level to imitate a crowd / another speaker."""
    y = w.astype(np.float64)
    if rng.random() < 0.7:
        y = reverb(y, rng)
    if rng.random() < 0.4:
        y = bandlimit(y, rng)
    if babble is not None and rng.random() < 0.3:
        y = add_noise(y, rng, snr_db=rng.uniform(8, 20), noise=babble)
    if rng.random() < 0.8:
        y = add_noise(y, rng)
    if rng.random() < 0.15:
        y = clip(y, rng)
    y *= rng.uniform(0.3, 1.0) / (np.abs(y).max() + 1e-9)
    return y.astype(np.float32)


def augment_fixed(w, rng, snr_db=10, rt_s=0.6):
    """The exact degradation of the baseline notebook (white noise 10 dB + 0.6 s reverb),
    kept so numbers stay comparable with v0."""
    n = int(rt_s * SR)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 8, n))
    ir[0] = 1.0
    y = fftconvolve(w, ir / np.linalg.norm(ir))[: len(w)]
    y = y / (np.abs(y).max() + 1e-9) * (np.abs(w).max() + 1e-9)
    noise = rng.standard_normal(len(y))
    p_noise = np.mean(y ** 2) / (10 ** (snr_db / 10))
    y = y + noise * np.sqrt(p_noise / np.mean(noise ** 2))
    return y.astype("float32")

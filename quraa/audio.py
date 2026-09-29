"""Getting audio in: local audio/video files, YouTube (or any yt-dlp) links -> 16 kHz mono float32."""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

SR = 16000
_URL = re.compile(r"^https?://", re.I)


def ffmpeg_exe() -> str:
    """System ffmpeg if available, otherwise the binary bundled with imageio-ffmpeg (handy on Windows)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError("ffmpeg not found. Install it or `pip install imageio-ffmpeg`.") from e


def is_url(s: str) -> bool:
    return bool(_URL.match(str(s).strip()))


def download(url: str, out_dir: str | None = None) -> str:
    return download_with_info(url, out_dir)[0]


def download_with_info(url: str, out_dir: str | None = None) -> tuple[str, dict]:
    """Download the best audio stream of a YouTube / social-media link with yt-dlp.

    If YouTube asks you to "sign in to confirm you're not a bot" (common on Colab and servers),
    export your browser cookies to a cookies.txt file and set QURAA_YTDLP_COOKIES=/path/cookies.txt,
    or put the file's *content* in QURAA_YTDLP_COOKIES_TXT (handy as a Hugging Face Space secret).
    """
    import yt_dlp

    if os.environ.get("QURAA_YTDLP_COOKIES_TXT") and not os.environ.get("QURAA_YTDLP_COOKIES"):
        p = Path(tempfile.gettempdir()) / "quraa_cookies.txt"
        p.write_text(os.environ["QURAA_YTDLP_COOKIES_TXT"], encoding="utf-8")
        os.environ["QURAA_YTDLP_COOKIES"] = str(p)
    out_dir = out_dir or tempfile.mkdtemp(prefix="quraa_")
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "ffmpeg_location": ffmpeg_exe(),
    }
    if os.environ.get("QURAA_YTDLP_COOKIES"):
        opts["cookiefile"] = os.environ["QURAA_YTDLP_COOKIES"]
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url.strip(), download=True)
        return ydl.prepare_filename(info), {k: info.get(k) for k in ("title", "uploader", "channel", "webpage_url")}


def load(path: str, max_minutes: float | None = None, start_s: float = 0) -> np.ndarray:
    """Decode any audio/video file ffmpeg understands to 16 kHz mono float32 in [-1, 1]."""
    cmd = [ffmpeg_exe(), "-nostdin", "-loglevel", "error"]
    if start_s:
        cmd += ["-ss", str(start_s)]
    cmd += ["-i", str(path)]
    if max_minutes:
        cmd += ["-t", str(max_minutes * 60)]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"]
    p = subprocess.run(cmd, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {p.stderr.decode(errors='replace')[-500:]}")
    w = np.frombuffer(p.stdout, dtype=np.float32).copy()
    if w.size == 0:
        raise RuntimeError(f"No audio track found in {path}")
    return w


def fetch(source: str, max_minutes: float | None = None) -> tuple[np.ndarray, str]:
    """source = local path or URL -> (waveform, human-readable name)."""
    w, name, _ = fetch_info(source, max_minutes)
    return w, name


def fetch_info(source: str, max_minutes: float | None = None, display_name: str | None = None):
    """Like fetch, plus a text describing the source (video title + channel, or file name) for hints."""
    if is_url(source):
        path, info = download_with_info(source)
        title = info.get("title") or Path(path).name
        text = " ".join(filter(None, [info.get("title"), info.get("uploader") or info.get("channel")]))
        return load(path, max_minutes), title, text
    name = display_name or Path(source).name
    return load(source, max_minutes), name, Path(name).stem.replace("_", " ")


def windows(w: np.ndarray, win_s: float = 6, hop_s: float = 3, max_windows: int | None = 80,
            rel_energy: float = 0.3, abs_energy: float = 1e-3) -> list[int]:
    """Start indices of analysis windows, dropping near-silent ones.

    Long files are subsampled evenly to at most `max_windows` windows so a 1-hour video
    costs the same as a 4-minute one."""
    win, hop = int(win_s * SR), int(hop_s * SR)
    if len(w) < win:
        return [0] if np.sqrt(np.mean(w ** 2)) > abs_energy else []
    starts = np.arange(0, len(w) - win + 1, hop)
    rms = np.array([np.sqrt(np.mean(w[s:s + win] ** 2)) for s in starts])
    keep = (rms > rel_energy * np.median(rms)) & (rms > abs_energy)
    starts = starts[keep]
    if max_windows and len(starts) > max_windows:
        starts = starts[np.linspace(0, len(starts) - 1, max_windows).round().astype(int)]
    return starts.tolist()

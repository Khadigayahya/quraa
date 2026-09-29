"""CLI:  python -m quraa <file-or-url> [--no-check]"""
import argparse
import json
import sys

from .pipeline import Quraa
from .report import to_markdown


def main():
    p = argparse.ArgumentParser(description="Identify the Quran reciter in an audio/video file or link.")
    p.add_argument("source", nargs="+", help="local file(s) or URL(s)")
    p.add_argument("--no-check", action="store_true", help="skip the Quran-vs-lecture step")
    p.add_argument("--gallery", help="path to gallery .npz (default: models/gallery_v1.npz)")
    p.add_argument("--whisper", help="faster-whisper model size (default: small on CPU, large-v3-turbo on GPU)")
    p.add_argument("--max-minutes", type=float, default=60, help="only analyse the first N minutes")
    p.add_argument("--json", action="store_true", help="print raw JSON")
    a = p.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    q = Quraa(a.gallery, whisper_size=a.whisper)
    for src in a.source:
        r = q.analyze(src, check_content=not a.no_check, max_minutes=a.max_minutes)
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str) if a.json else to_markdown(r))
        print()


if __name__ == "__main__":
    main()

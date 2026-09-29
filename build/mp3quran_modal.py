"""Build voiceprints for every reciter on mp3quran.net (~240 reciters, ~290 recordings), on Modal.

    modal run build/mp3quran_modal.py                 # all recordings
    modal run build/mp3quran_modal.py --limit 10      # quick test
    modal volume get quraa-build emb ./build/emb      # then: python build/make_gallery.py

Per recording (moshaf):
  * enroll: up to ENROLL_N short surahs (juz' 29-30), first MAX_MIN minutes of each
  * test:   up to TEST_N *different* surahs, kept apart to measure accuracy honestly
Audio is streamed with ffmpeg (-t), so only the first minutes are downloaded.
"""
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
API = "https://www.mp3quran.net/api/v3/reciters"
ENROLL = [67, 78, 55, 75, 76, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93]
TEST = [36, 56, 50, 68, 69, 70, 71, 72, 73, 74, 77, 51, 52, 53, 54, 57, 64, 65, 66]
ENROLL_N, TEST_N, MAX_MIN = 5, 2, 4
WIN_S = 6

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install("torch", "torchaudio", "speechbrain>=1.0", "numpy", "scipy", "soundfile", "requests")
    .env({"HF_HOME": "/cache/hf"})
    .add_local_python_source("quraa")
)
app = modal.App("quraa-build", image=image)
cache = modal.Volume.from_name("quraa-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("quraa-build", create_if_missing=True)


def _get(lang):
    import urllib.request

    with urllib.request.urlopen(f"{API}?language={lang}", timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))["reciters"]


def catalog():
    ar = _get("ar")
    en = {r["id"]: r["name"] for r in _get("eng")}
    jobs = []
    for r in ar:
        for m in r["moshaf"]:
            have = {int(x) for x in m["surah_list"].split(",") if x.strip()}
            enroll = [s for s in ENROLL if s in have][:ENROLL_N]
            test = [s for s in TEST if s in have][:TEST_N]
            if len(enroll) < 2:          # too little audio we can use -> skip
                continue
            jobs.append({"reciter_id": r["id"], "name": r["name"], "name_en": en.get(r["id"], ""),
                         "moshaf_id": m["id"], "moshaf": m["name"], "server": m["server"],
                         "enroll": enroll, "test": test})
    return jobs


@app.cls(cpu=4.0, memory=6144, volumes={"/cache": cache, "/out": out_vol}, timeout=1800, max_containers=8,
         retries=1)
class Builder:
    @modal.enter()
    def load(self):
        from quraa.embedder import Embedder

        self.emb = Embedder(cache_dir="/root/pretrained")

    @modal.method()
    def run(self, job):
        import numpy as np

        from quraa import audio
        from quraa.augment import augment

        key = f"{job['reciter_id']}_{job['moshaf_id']}"
        dst = Path(f"/out/emb/{key}.npz")
        if dst.exists():
            return {"key": key, "status": "cached"}
        rng = np.random.default_rng(job["moshaf_id"])
        win = WIN_S * audio.SR

        def surah_windows(s):
            url = f"{job['server'].rstrip('/')}/{s:03d}.mp3"
            w = audio.load(url, max_minutes=MAX_MIN)
            return [w[i:i + win] for i in audio.windows(w, WIN_S, WIN_S, max_windows=40)]

        from concurrent.futures import ThreadPoolExecutor

        def safe(s):
            try:
                return s, surah_windows(s), None
            except Exception as e:
                return s, [], f"{s}: {e}"[:200]

        with ThreadPoolExecutor(7) as ex:            # stream all surahs of this recording at once
            got = {s: (ws, err) for s, ws, err in ex.map(safe, job["enroll"] + job["test"])}
        enroll, enroll_aug, test, test_ids = [], [], [], []
        errors = [err for _, err in got.values() if err]
        for s in job["enroll"]:
            ws = got[s][0]
            enroll += ws
            enroll_aug += [augment(x, rng) for x in ws]
        for s in job["test"]:
            ws = got[s][0]
            test += ws
            test_ids += [s] * len(ws)
        if len(enroll) < 10:
            return {"key": key, "status": "too_little_audio", "errors": errors}
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, enroll=self.emb(enroll), enroll_aug=self.emb(enroll_aug),
                 test=self.emb(test) if test else np.zeros((0, 192), np.float32),
                 test_surah=np.array(test_ids, dtype=np.int16))
        dst.with_suffix(".json").write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
        out_vol.commit()
        return {"key": key, "status": "ok", "enroll": len(enroll), "test": len(test), "errors": errors}


@app.local_entrypoint()
def main(limit: int = 0):
    jobs = catalog()
    if limit:
        jobs = jobs[:limit]
    print(f"{len(jobs)} recordings from {len({j['reciter_id'] for j in jobs})} reciters")
    stats = {}
    for i, r in enumerate(Builder().run.map(jobs, order_outputs=False, return_exceptions=True), 1):
        st = r["status"] if isinstance(r, dict) else f"exception: {type(r).__name__}"
        stats[st] = stats.get(st, 0) + 1
        if not isinstance(r, dict) or r["status"] not in ("ok", "cached"):
            print("  !", r if not isinstance(r, dict) else {k: r.get(k) for k in ("key", "status", "errors")})
        if i % 20 == 0 or i == len(jobs):
            print(f"[{i}/{len(jobs)}] {stats}", flush=True)

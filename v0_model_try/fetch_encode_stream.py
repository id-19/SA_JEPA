# SA-CE v0 — stream-acquire dev-clean from HF datasets-server, encode, delete flac.
# WHY: disk-starved environment (0.5–1 GB free). Net cost per file = latents only (~15 KB),
# not the flac (~100 KB) — acquire/encode/delete in one pass.
# The validation split of openslr/librispeech_asr IS dev-clean (2703 utts, official audio).
# Usage: uv run python v0_model_try/fetch_encode_stream.py [--limit 1000] [--force]
import argparse
import csv
import json
import shutil
import socket
import sys
import time
import urllib.request
from pathlib import Path

socket.setdefaulttimeout(60)  # no network call may hang the job forever

sys.path.insert(0, str(Path(__file__).resolve().parent))
from encode_usm import DEVICE, load_usm, encode_chunk, verify_weights  # noqa: E402

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
DS = "https://datasets-server.huggingface.co/rows?dataset=openslr/librispeech_asr&config=clean&split=validation"


def disk_free_mb() -> int:
    return int(shutil.disk_usage(REPO).free / 2**20)


def iter_rows(offset: int = 0, length: int = 100):
    with urllib.request.urlopen(f"{DS}&offset={offset}&length={length}", timeout=120) as r:
        return [rr["row"] for rr in json.load(r)["rows"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=2703)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    verify_weights(REPO / "model_weights" / "usm.th")
    enc = load_usm(DEVICE)

    outdir = REPO / "data" / "latents" / "dev-clean"
    if REPO is None or not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
    man = outdir / "manifest.csv"
    seen_keys: set[str] = set()
    if man.exists() and not args.force:
        seen_keys = {f"{r['utt_id']}_{r['chunk']}" for r in csv.DictReader(open(man))}
        print(f"RESUME: {len(seen_keys)} chunks already cached — skipping them", flush=True)
    elif man.exists() and args.force:
        orphans = [p for p in outdir.glob("*.npy")]
        for p in orphans:
            p.unlink()
        print(f"--force: wiped {len(orphans)} npy, fresh manifest", flush=True)
        man.write_text("utt_id,chunk,n_frames,dur\n")
    else:
        outdir.mkdir(parents=True, exist_ok=True)
        man.write_text("utt_id,chunk,n_frames,dur\n")

    tmp = Path("/tmp/fes_flac")
    tmp.mkdir(exist_ok=True)

    t0, n, frames_total = time.time(), 0, 0
    offset = 0
    while n < args.limit:
        rows = iter_rows(offset, 100)
        if not rows:
            print(f"no more rows at offset {offset}")
            break
        for row in rows:
            if n >= args.limit:
                break
            utt, spk, text = row["id"], row["speaker_id"], row["text"]
            already = any(k.startswith(f"{utt}_") for k in seen_keys)
            if already:
                continue  # resume support: utterance fully cached
            url = row["audio"][0]["src"]
            flac = tmp / f"{utt}.flac"
            for attempt in range(3):
                try:
                    urllib.request.urlretrieve(url, flac)
                    break
                except Exception as e:
                    if attempt == 2:
                        print(f"SKIP {utt} after 3 attempts ({type(e).__name__})", flush=True)
                        flac = None
                        break
                    time.sleep(5 * (attempt + 1))
            if flac is None:
                continue
            if disk_free_mb() < 150:
                flac.unlink(missing_ok=True)
                sys.exit(f"DISK LOW: {disk_free_mb()} MB free — stopping cleanly after {n} files")
            wav, sr = sf.read(flac, dtype="float32", always_2d=True)
            if sr != 16000:
                flac.unlink(missing_ok=True)
                continue
            wav = wav.mean(axis=1)
            dur = len(wav) / sr
            n_chunks = max(1, int(dur) // 30 + (1 if (dur % 30) >= 2 else 0))
            for ci in range(n_chunks):
                cs = wav[ci * 30 * sr : (ci + 1) * 30 * sr]
                if len(cs) < 2 * sr:
                    continue
                lat = encode_chunk(enc, torch.from_numpy(cs))
                nf = lat.shape[0]
                rate = nf / (len(cs) / sr)
                if not (6.1 <= rate <= 6.9) or not torch.isfinite(lat.float()).all():
                    flac.unlink(missing_ok=True)
                    sys.exit(f"CONTRACT VIOLATION {utt}_{ci} (fps {rate:.2f}) — STOP")
                np.save(outdir / f"{utt}_{ci:03d}.npy", lat.numpy())
                with open(man, "a") as mf:
                    mf.write(f"{utt},{ci},{nf},{len(cs) / sr:.3f}\n")
                frames_total += nf
            flac.unlink(missing_ok=True)
            n += 1
            if n % 50 == 0:
                el = time.time() - t0
                print(f"{n}/{args.limit} | {el:.0f}s | pace {n / el * 3600:.0f} files/h | frames {frames_total} | free {disk_free_mb()} MB", flush=True)
        offset += 100

    (outdir / "acquisition_meta.json").write_text(json.dumps({
        "source": "HF datasets-server openslr/librispeech_asr validation (=dev-clean)",
        "n_files": n, "n_frames": frames_total, "elapsed_s": time.time() - t0,
        "text_pool": "metadata retained in this file only for utterances processed",
    }))
    print(f"DONE {n} files, {frames_total} frames in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
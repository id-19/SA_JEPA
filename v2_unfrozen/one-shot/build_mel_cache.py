# SA-JEPA v2 one-shot — build the 10 h mel cache from LibriSpeech train-clean-100.
# Source: HF parquet mirror (refs/convert/parquet), downloaded to /tmp by curl (probe_url.sh).
# NOTE: each parquet shard ≈ 6-7 h of audio; run once per shard (append-safe: manifest/windows
# append, meta aggregates, existing .pt files skipped). 0000 → 6.70 h, 0001 topped up to 13.70 h total.
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

HERE = Path(__file__).resolve().parent
CACHE = HERE / "mel_cache"
CACHE.mkdir(exist_ok=True)

SR, FPS, NMEL = 16000, 50, 80
T_WIN = 500  # 10 s window
TARGET_DUR_S = 10 * 3600  # 10 hours

PARQUET = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/ls100_train.parquet")
if not PARQUET.exists():
    sys.exit(f"parquet not at {PARQUET} — download it first (probe_url.sh has the URL pattern)")


def main():
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(PARQUET)
    schema_names = pf.schema_arrow.names
    print("schema:", schema_names, "| row groups:", pf.num_row_groups)

    mel_fn = torchaudio.transforms.MelSpectrogram(
        SR, n_fft=400, win_length=400, hop_length=SR // FPS, n_mels=NMEL, power=2.0
    )
    to_db = torchaudio.transforms.AmplitudeToDB("power", top_db=80)

    man = open(CACHE / "manifest.csv", "a" if (CACHE / "manifest.csv").exists() else "w", newline="")
    wins = open(CACHE / "windows.csv", "a" if (CACHE / "windows.csv").exists() else "w", newline="")
    appending = (CACHE / "manifest.csv").stat().st_size > 0
    man_w = csv.writer(man)
    if not appending:
        man_w.writerow(["utt", "dur_s", "speaker", "n_frames"])
    wins_w = csv.writer(wins)
    if not appending:
        wins_w.writerow(["utt", "t0", "n_frames"])
    # acquisition_meta.json is NOT opened here — the post-loop aggregation reads the PRIOR
    # meta, so opening (and truncating) it early would destroy the stats it needs to read.
    total_s, n_utts, n_wins = 0.0, 0, 0
    speakers = set()
    done = False
    for rg in range(pf.num_row_groups):
        if done:
            break
        tbl = pf.read_row_group(rg, columns=["file", "audio"])
        for i in range(tbl.num_rows):
            utt_rel = tbl["file"][i].as_py()  # e.g. "1001/134693/1001-134693-0000.flac"
            audio_struct = tbl["audio"][i].as_py()  # {'bytes', 'path', 'id'}
            utt = Path(utt_rel).stem
            out = CACHE / f"{utt}.pt"
            if out.exists():
                continue  # resume: already cached
            arr, sr = sf.read(io.BytesIO(audio_struct["bytes"]), dtype="float32", always_2d=True)
            assert sr == SR, sr
            wav = torch.from_numpy(arr.mean(axis=1))
            dur = wav.numel() / SR
            if dur < T_WIN / FPS:
                continue  # window needs 500 frames = 10 s; skip short utts entirely (v0 simplicity)
            with torch.no_grad():
                mel = to_db(mel_fn(wav.unsqueeze(0)))  # (1, 80, nf) @ 50 fps
            nf = mel.shape[-1]
            mel16 = mel.squeeze(0).to(torch.float16).T.contiguous()  # (nf, 80)
            torch.save(mel16, out)
            spk = utt.split("-")[0]
            man_w.writerow([utt, f"{dur:.3f}", spk, nf])
            # windows: non-overlapping, from the start
            k = int(nf // T_WIN)
            for j in range(k):
                wins_w.writerow([utt, j * T_WIN, T_WIN])
                n_wins += 1
            total_s += dur
            n_utts += 1
            speakers.add(spk)
            if n_utts % 250 == 0:
                print(f"{n_utts} utts | {total_s/3600:.2f} h | {n_wins} windows | free {total_s*FPS*NMEL*2/2**20:.0f} MB written", flush=True)
            if total_s >= TARGET_DUR_S:
                done = True
                break

    man.close(); wins.close()
    meta_p = CACHE / "acquisition_meta.json"
    prior = json.load(open(meta_p)) if (appending and meta_p.exists()) else {}
    total_s += prior.get("total_s", 0.0)
    n_utts += prior.get("n_utts", 0)
    n_wins += prior.get("n_windows", 0)
    speakers |= set(prior.get("speaker_ids", []))
    meta_f = open(meta_p, "w")
    meta_f.write(json.dumps({
        "source": "openslr/librispeech_asr clean/train (=train-clean-100), HF parquet mirror",
        "fps": FPS, "n_mels": NMEL, "total_s": total_s,
        "n_utts": n_utts, "n_windows": n_wins, "n_speakers": len(speakers),
        "speaker_ids": sorted(speakers),
    }))
    meta_f.close()
    print(f"DONE: {n_utts} utts, {total_s/3600:.2f} h, {n_wins} windows, {len(speakers)} speakers")


if __name__ == "__main__":
    main()
# SA-CE v0 — encode the corpus with the FROZEN USM audio tower, cache frame latents to disk.
# Boundary note: this is DATA TOOLING (Hermes-built). The model's forward pass is Ishaan's to write.
# Contract: 16 kHz input -> 30 s chunks -> (T_f, 1536) fp16 latents @ 6.27 fps (verified: 188 frames / 30 s).
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import soundfile as sf
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent  # script lives in v0_model_try/; repo paths are repo-root-relative
DEVICE = "cpu"  # MPS + audio towers + 3.7GB free disk/8GB RAM = a bad mix. CPU encode is fast enough for 5.4h.


def hf_sha256(repo: str, fname: str) -> str:
    url = f"https://huggingface.co/api/models/{repo}/tree/main"
    with urllib.request.urlopen(url, timeout=30) as r:
        for f in json.load(r):
            if f["path"] == fname:
                lfs = f.get("lfs") or {}
                return lfs.get("oid") or lfs.get("sha256") or f.get("oid")
    raise FileNotFoundError(fname)


def sha256_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_weights(wpath: Path, hf_repo: str = "n0mad-0/gemma3n-usm-rip", fname: str = "usm.th") -> None:
    exp = hf_sha256(hf_repo, fname)
    got = sha256_file(wpath)
    if got != exp:
        raise SystemExit(f"CHECKSUM MISMATCH for {wpath}: expected {exp[:12]}…, got {got[:12]}… — refusing to load.")
    print(f"weights sha256 OK ({exp[:12]}…)")


def load_usm(device: str) -> torch.nn.Module:
    from transformers import Gemma3nAudioConfig, Gemma3nAudioEncoder

    cfg = Gemma3nAudioConfig()
    enc = Gemma3nAudioEncoder(cfg)
    sd = torch.load(REPO / "model_weights" / "usm.th", map_location="cpu", weights_only=True)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    n_missing_loadable = sum(1 for k in missing if not any(p in k for p in ("embed_weights", "softembed")))
    if n_missing_loadable > 0:
        print(f"WARN {len(missing)} missing keys (first 5: {missing[:5]})")
        sys.exit("unexpected missing-loadable keys — stop, check")
    print(f"unexpected keys from checkpoint: {len(unexpected)} (expected: none)")
    if unexpected:
        sys.exit("unexpected keys — stop, check")
    return enc.to(device).eval()


def iter_flacs(root: Path):
    for p in sorted(root.rglob("*.flac")):
        yield p


def encode_chunk(enc: torch.nn.Module, wav_s: torch.Tensor) -> torch.Tensor:
    """wav_s: (samples,) float32 16 kHz -> (T_f, 1536) fp16 CPU. Passes the required
    audio_mel_mask, inverted per the n0mad-0 README (`encoder(audio_mel, ~audio_mel_mask)`)."""
    from transformers import Gemma3nAudioFeatureExtractor

    fe = Gemma3nAudioFeatureExtractor()
    feats = fe(wav_s.numpy(), sampling_rate=16000, return_tensors="pt")
    mel = feats["input_features"]  # (1, 80, T_mel)
    mel_mask = feats.get("input_features_mask")
    with torch.no_grad():
        if mel_mask is not None:
            out = enc(mel.to(DEVICE), ~mel_mask.to(DEVICE))
        else:
            out = enc(mel.to(DEVICE))
        if hasattr(out, "last_hidden_state"):
            lat = out.last_hidden_state
        elif isinstance(out, tuple):
            lat = out[0]
        else:
            lat = out
    return lat[0].to(torch.float16).cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="encode only first N files (smoke)")
    ap.add_argument("--outdir", default="data/latents/dev-clean")
    ap.add_argument("--force", action="store_true", help="overwrite manifest and restart")
    args = ap.parse_args()

    # verify the checkpoint BEFORE loading it (downloaded from HF — verify hash first)
    wpath = REPO / "model_weights" / "usm.th"
    verify_weights(wpath)
    enc = load_usm(DEVICE)

    root = REPO / "data" / "LibriSpeech" / "dev-clean"
    outdir = REPO / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    man = outdir / "manifest.csv"
    if man.exists() and not getattr(args, "force", False):
        sys.exit(f"{man} exists — refusing to append duplicates. Pass --force to restart.")
    man.write_text("utt_id,chunk,n_frames,dur\n")

    files = list(iter_flacs(root))
    if args.limit:
        files = files[: args.limit]
    t0 = time.time()
    n_done, n_frames_total = 0, 0
    t_last = t0
    for i, p in enumerate(files):
        utt = p.stem
        wav, sr = sf.read(p, dtype="float32", always_2d=True)
        assert sr == 16000, f"non-16k source: {p}"
        wav = wav.mean(axis=1)  # mono
        dur = len(wav) / sr
        n_chunks = int(dur) // 30 + (1 if (dur % 30) >= 2 else 0)
        for ci in range(n_chunks):
            cs = wav[ci * 30 * sr : (ci + 1) * 30 * sr]
            if len(cs) < 2 * sr:
                continue
            lat = encode_chunk(enc, torch.from_numpy(cs))
            nf = lat.shape[0]
            # CONTRACT GUARDS — a violation stops the whole job, never writes bad frames
            exp_rate = nf / (len(cs) / sr)
            if not (6.1 <= exp_rate <= 6.9):
                sys.exit(f"CONTRACT VIOLATION fps {exp_rate:.2f} at {utt}_{ci} — STOP")
            if not torch.isfinite(lat.float()).all():
                sys.exit(f"CONTRACT VIOLATION nonfinite latents at {utt}_{ci} — STOP")
            out = outdir / f"{utt}_{ci:03d}.npy"
            np.save(out, lat.numpy())
            with open(outdir / "manifest.csv", "a") as mf:
                mf.write(f"{utt},{ci},{nf},{len(cs) / sr:.3f}\n")
            n_frames_total += nf
        n_done += 1
        if n_done % 25 == 0:
            el = time.time() - t0
            print(f"{n_done}/{len(files)} flacs | {el:.0f}s | {n_frames_total} frames | pace {n_done / el * 3600:.0f}/h")
    print(f"DONE {n_done} files, {n_frames_total} frames, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
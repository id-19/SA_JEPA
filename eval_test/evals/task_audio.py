"""Log-mel frame prediction task for the arm eval.

Input = current 80-dim log-mel frame, target = next frame. The factory
returns the raw frame series; run_eval.py does the shift (x[:, :-1] ->
x[:, 1:]).

Real audio (default), cached ONCE outside the repo:

    /Users/id19/.claude/scratch/gdn_audio/sample.flac

This scratch sample is a single small speech clip extracted ONCE from the
user's LibriSpeech dev-clean tarball — extraction never lands in the repo.
Resolution order in make_data():
  1. explicit audio_path argument,
  2. the cached scratch sample (used as-is),
  3. extract the default small member from the tarball into the scratch
     dir (first run on a machine where step 2 has not happened yet),
  4. the in-memory random-tone synth — final fallback only, for a machine
     where BOTH the scratch file and the tarball are missing.

Loading uses soundfile (torchaudio.load in this env routes through
torchcodec, which cannot dlopen FFmpeg on this machine): sf.read ->
torch tensor (1, N), sample rate passed on. The v0 pipeline
(v0_model_try.data_pipeline_v0) is reused ONLY for the resample/mel/
normalise stages (cast_uniform -> get_log_mel -> get_normalised_log_mel),
so the 16 kHz LibriSpeech clip is resampled to the 24 kHz target by the
same code the v0 model trains against. Imports stay lazy and live inside
the functions: they pull in torch + torchaudio, so this file should stay
importable without them. v0_model_try sits at the repo root, three
directories up from this file, so we add the repo root to sys.path.
"""

import sys
from pathlib import Path

import numpy as np
import torch

NAME = "audio"
SAMPLE_RATE = 24000
DUR = 10.0
NOTE_DUR = 1.25
FADE = 0.05  # 50 ms linear fade at each note edge

# cached extraction target (outside the repo) and the tarball it came from
SCRATCH_DIR = Path("/Users/id19/.claude/scratch/gdn_audio")
SCRATCH_SAMPLE = SCRATCH_DIR / "sample.flac"
TARBALL = Path("/Users/id19/Programming/Data/dev-clean.tar.gz")
# small member chosen with tar -tzf on 2026-10-28 (104,850 bytes, ~5.6 s)
TARBALL_MEMBER = "LibriSpeech/dev-clean/2277/149896/2277-149896-0005.flac"


def _repo_root():
    root = Path(__file__).resolve().parents[2]  # eval_test/evals -> repo root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def synthesize_wav(seed):
    # 8 random tones, 1.25 s each, 50 ms fades -> (1, 240000) float32
    rng = np.random.default_rng(seed)
    sr = SAMPLE_RATE
    notes = rng.uniform(220.0, 880.0, size=8)
    fade = int(FADE * sr)
    per_note = int(NOTE_DUR * sr)
    out = np.zeros(int(DUR * sr), dtype=np.float32)
    for i, freq in enumerate(notes):
        t = np.arange(per_note, dtype=np.float32) / sr
        tone = 0.3 * np.sin(2 * np.pi * freq * t)
        env = np.ones(per_note, dtype=np.float32)
        env[:fade] = np.linspace(0.0, 1.0, fade)
        env[-fade:] = np.linspace(1.0, 0.0, fade)
        out[i * per_note:(i + 1) * per_note] = tone * env
    return torch.tensor(out).unsqueeze(0)  # (1, 240000) "wav"


def load_wave(path):
    """Local loader: soundfile instead of torchaudio.load (torchcodec
    backend in this env cannot open .flac without a system FFmpeg).

    Returns (wav (1, N) float32, sr int)."""
    import soundfile as sf

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(np.ascontiguousarray(data.mean(axis=1))).unsqueeze(0)
    return wav, sr


def _extract_default_member():
    # first-ever run on this machine: extract ONLY the default small member
    # into the scratch dir — never into the repo
    import tarfile

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)
    with tarfile.open(TARBALL, "r:gz") as tf:
        src = tf.extractfile(TARBALL_MEMBER)
        if src is None:
            raise RuntimeError(f"member not found in tarball: {TARBALL_MEMBER}")
        SCRATCH_SAMPLE.write_bytes(src.read())
    print(f"extracted {TARBALL_MEMBER} from tarball into {SCRATCH_SAMPLE}")
    return SCRATCH_SAMPLE


def resolve_audio():
    """audio_path=None default resolution: cached scratch sample, else
    extract from the tarball, else None (caller falls back to synth)."""
    if SCRATCH_SAMPLE.is_file():
        return SCRATCH_SAMPLE
    if TARBALL.is_file():
        try:
            _extract_default_member()
            return SCRATCH_SAMPLE
        except Exception as e:  # tarball present but unreadable -> synth
            print(f"warning: could not extract audio from {TARBALL}: {e}")
    return None


def make_data(audio_path=None, seed=0):
    # requires torch + torchaudio (v0_model_try builds a MelSpectrogram)
    _repo_root()
    from v0_model_try.data_pipeline_v0 import (
        cast_uniform,
        get_log_mel,
        get_normalised_log_mel,
    )

    source = None
    path = None
    if audio_path is not None:
        path = Path(audio_path)
        if not path.is_file():
            raise ValueError(f"--audio file not found: {path}")
        source = str(path)
    else:
        path = resolve_audio()
        source = str(path) if path is not None else "synth (tarball+cache missing)"

    frames_meta = {
        "source": source,
        "flac_seconds": None,
        "flac_frames": None,
    }

    if path is not None:
        wav, sr = load_wave(path)  # soundfile loader, NOT torchaudio.load
        frames_meta["flac_frames"] = wav.shape[1]
        frames_meta["flac_seconds"] = round(wav.shape[1] / sr, 2)
        wav = cast_uniform(wav, sr)  # 16 kHz LibriSpeech -> 24 kHz target
        frames = get_normalised_log_mel(get_log_mel(wav))
    else:
        print("warning: no real audio available; using in-memory tone synth")
        wav = synthesize_wav(seed)
        wav = cast_uniform(wav, SAMPLE_RATE)  # no-op at the right sample rate
        frames = get_normalised_log_mel(get_log_mel(wav))

    if frames.shape[0] < 100:
        raise ValueError(
            f"audio has only {frames.shape[0]} log-mel frames, need >= 100; "
            "pass --audio PATH with a longer recording"
        )

    # chronological split: first 80% of frames train, last 20% val
    n_train = int(frames.shape[0] * 0.8)

    return {
        "train": frames[:n_train].unsqueeze(0).float(),  # (1, Ttr, 80)
        "val": frames[n_train:].unsqueeze(0).float(),    # (1, Tva, 80)
        "d_in": 80,
        "d_out": 80,
        "loss": "mse",
        "source": source,
        "flac_seconds": frames_meta["flac_seconds"],
    }


if __name__ == "__main__":
    d = make_data()
    print(
        f"{NAME}: train {tuple(d['train'].shape)} val {tuple(d['val'].shape)} "
        f"d_in={d['d_in']} d_out={d['d_out']} loss={d['loss']} "
        f"source={d['source']} ({d['flac_seconds']} s of flac)"
    )
    print("first 100 normalised log-mel values of frame 0:")
    print(" ".join(f"{v:+.4f}" for v in d["train"][0, 0, :100].tolist()))
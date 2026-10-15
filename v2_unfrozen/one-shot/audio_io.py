# SA-JEPA v0 — audio <-> mel I/O. Bit-exact mel pipeline matching build_mel_cache.py, plus the
# inverse path (mel-dB -> power -> pseudo-inverse mel filter -> Griffin-Lim -> waveform) used
# to make the model's latents audible.
import io
import math
from pathlib import Path

import soundfile as sf
import torch
import torchaudio

SR, FPS, NMEL = 16000, 50, 80
N_FFT, WIN, HOP = 400, 400, 320

_mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
                                             n_mels=NMEL, power=2.0)
_db = torchaudio.transforms.AmplitudeToDB("power", top_db=80)


def wav_to_mel(wav: torch.Tensor | str | Path) -> torch.Tensor:
    """Waveform file/1-D tensor (any sample rate for files) -> mel dB (nf, 80) float32.
    Identical transform to build_mel_cache.py so the model sees exactly its training domain."""
    if isinstance(wav, (str, Path)):
        arr, sr = sf.read(str(wav), dtype="float32", always_2d=True)
        w = torch.from_numpy(arr.mean(axis=1))
        if sr != SR:
            w = torchaudio.functional.resample(w, sr, SR)
    else:
        w = wav if wav.dim() == 1 else wav.mean(dim=0)
    with torch.no_grad():
        mel = _db(_mel(w.unsqueeze(0)))  # (1, 80, nf)
    return mel.squeeze(0).T.contiguous()  # (nf, 80)


def mel_db_to_wav(mel_db: torch.Tensor, n_iter: int = 64, n_restarts: int = 1) -> torch.Tensor:
    """Mel dB (nf, 80) -> waveform via pseudo-inverse mel filter + Griffin-Lim phase recovery.
    Rough (GL is not a neural vocoder) but intelligible for speech."""
    power = torch.clamp(10.0 ** (mel_db.float().T / 10.0), min=0.0)  # (80, nf) power
    # Explicit pseudo-inverse of the mel filterbank. torchaudio's InverseMelScale uses an
    # lstsq driver that fails here: at 16 kHz the lowest slaney filters share FFT bins, so the
    # (201, 80) filterbank is rank-deficient. pinv (SVD) gives the minimum-norm solution.
    fb = torchaudio.functional.melscale_fbanks(N_FFT // 2 + 1, 0.0, SR / 2, NMEL, SR)  # (201, 80)
    lin = torch.linalg.pinv(fb.T) @ power              # (201, nf) linear power
    mag = torch.clamp(lin, min=1e-10).sqrt()           # magnitude
    gl = torchaudio.transforms.GriffinLim(
        n_fft=N_FFT, n_iter=n_iter, win_length=WIN, hop_length=HOP, power=1.0,
        momentum=0.99, rand_init=True,
    )
    with torch.no_grad():
        best, best_err = None, math.inf
        for r in range(n_restarts):
            torch.manual_seed(r)
            w = gl(mag)
            # consistency error: how well the reconstructed waveform's own mel matches the target
            err = torch.nn.functional.mse_loss(wav_to_mel(w).T, mel_db.T).item()
            if err < best_err:
                best, best_err = w, err
    return best


def save_wav(path, wav: torch.Tensor, sr: int = SR):
    w = wav.detach().cpu().float().flatten()
    peak = w.abs().max().item()
    if peak > 1e-6:
        w = w / max(1.0, peak)  # avoid clipping
    sf.write(str(path), w.numpy(), sr)
    return str(path)


if __name__ == "__main__":
    # Round-trip validation on a REAL cached utterance (no model involved): mel -> GL -> audio
    # -> mel. Reports mel correlation, so we know the audible path itself is sound before it is
    # used to demonstrate the model.
    import sys
    cache = Path("/Users/id19/Programming/ml_ai_and_data_science/SA-JEPA/v2_unfrozen/one-shot/mel_cache")
    pts = sorted(cache.glob("*.pt"))
    mel = torch.load(pts[0]).float()[:200]          # 4 s
    wav = mel_db_to_wav(mel)
    mel2 = wav_to_mel(wav)
    n = min(len(mel), len(mel2))
    a, b = mel[:n].flatten(), mel2[:n].flatten()
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    rms = wav.pow(2).mean().sqrt().item()
    out = cache.parent / "runs" / "roundtrip.wav"
    out.parent.mkdir(exist_ok=True)
    save_wav(out, wav)
    print(f"round-trip: mel corr {corr:.3f} | wav RMS {rms:.4f} | dur {len(wav)/SR:.2f}s -> {out}")
    print(f"source utt: {pts[0].stem}")

# Separate "the model is wrong" from "the vocoder is wrong", by ear.
#
# Same clean source, five renders:
#   1 clean source                         (never touched any vocoder)
#   2 GL 64 iters   <- what all earlier demos used
#   3 GL 512 iters  (best-of-4 restarts)
#   4 GL 512 of the TRUE mel, no model     <- the vocoder's ceiling
#   5 ORACLE PHASE of the model's mel      <- model magnitude + the SOURCE's phase
#   6 ORACLE PHASE of the true mel         <- sanity ceiling for method 5
#
# (5) is not a legitimate standalone vocoder (it borrows phase from the answer), but it is the
# standard way to judge mel fidelity: if the words are audible in (5) and not in (2), the model
# carried the content and the phase-recovery vocoder destroyed it.
import math
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audio_io import NMEL, N_FFT, HOP, SR, WIN, mel_db_to_wav, save_wav, wav_to_mel  # noqa: E402
from ckpt import load_ckpt  # noqa: E402
from model import SAJePAV0  # noqa: E402

OUT = HERE / "runs" / "demo_clean"
SRC = OUT / "Samantha_clean_input.wav"


def metrics(name, wav, true_mel):
    m = wav_to_mel(wav)
    n = min(len(m), len(true_mel))
    a, b = true_mel[:n].flatten(), m[:n].flatten()
    corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    snr = 10 * math.log10(float(a.var()) / F.mse_loss(b, a).item())
    det = F.mse_loss(m[:n, -20:], true_mel[:n, -20:]).item()
    print(f"{name:34s} corr {corr:.3f} | SNR {snr:5.2f} dB | top-band detail MSE {det:7.2f}")
    return snr


def mel_to_mag(mel_db):
    """mel dB (nf, 80) -> linear magnitude (201, nf), same pseudo-inverse as audio_io."""
    power = torch.clamp(10.0 ** (mel_db.float().T / 10.0), min=0.0)
    fb = torchaudio.functional.melscale_fbanks(N_FFT // 2 + 1, 0.0, SR / 2, NMEL, SR)
    lin = torch.linalg.pinv(fb.T) @ power
    return torch.clamp(lin, min=1e-10).sqrt()


def oracle_phase(mel_db, src_wav):
    """Render mel with the SOURCE waveform's phase."""
    w = src_wav if src_wav.dim() == 1 else src_wav.mean(0)
    spec = torch.stft(w, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                      window=torch.hann_window(WIN), return_complex=True)
    ph = spec / spec.abs().clamp(min=1e-10)
    mag = mel_to_mag(mel_db)
    n = min(ph.shape[1], mag.shape[1])
    X = mag[:, :n] * ph[:, :n]
    return torch.istft(X, n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                       window=torch.hann_window(WIN), length=w.numel())


def main():
    torch.set_grad_enabled(False)
    w, sr = sf.read(str(SRC), dtype="float32")
    src = torch.from_numpy(w)
    true_mel = wav_to_mel(src)

    m = SAJePAV0().eval()
    load_ckpt(m, str(HERE / "runs" / "v0"))
    e = m.encoder(true_mel.unsqueeze(0))
    dec_mel = m.decoder(e)[0].cpu()
    print(f"source {SRC.name}: {len(src)/sr:.2f}s -> {tuple(true_mel.shape)}\n")

    print("--- renders (mel fidelity vs the true mel) ---")
    save_wav(OUT / "r1_clean_source.wav", src)
    print(f"{'1 clean source':34s} (reference: this is the audio that went in)")
    metrics("2 GL 64 of model decode", mel_db_to_wav(dec_mel, n_iter=64), true_mel)
    save_wav(OUT / "r2_model_gl64.wav", mel_db_to_wav(dec_mel, n_iter=64))
    metrics("3 GL 512 x4 of model decode", mel_db_to_wav(dec_mel, n_iter=512, n_restarts=4), true_mel)
    save_wav(OUT / "r3_model_gl512.wav", mel_db_to_wav(dec_mel, n_iter=512, n_restarts=4))
    metrics("4 GL 512 of TRUE mel (no model)", mel_db_to_wav(true_mel, n_iter=512, n_restarts=4), true_mel)
    save_wav(OUT / "r4_true_gl512.wav", mel_db_to_wav(true_mel, n_iter=512, n_restarts=4))
    metrics("5 ORACLE phase, model mel", oracle_phase(dec_mel, src), true_mel)
    save_wav(OUT / "r5_model_oracle_phase.wav", oracle_phase(dec_mel, src))
    metrics("6 ORACLE phase, true mel", oracle_phase(true_mel, src), true_mel)
    save_wav(OUT / "r6_true_oracle_phase.wav", oracle_phase(true_mel, src))

    print("\nListen: 1 -> 2/3 (vocoder) -> 4 (vocoder ceiling) -> 5 (model content, clean phase)")
    print("If 5 is intelligible and 2 is not, the model carried the words; Griffin-Lim lost them.")


if __name__ == "__main__":
    main()

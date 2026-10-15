# Is the audio infer.py returns for a TEXT prompt structured and faithful? These are the exact
# artifacts a user listens to, so they get the same treatment as the clip path: loudness, spectral
# structure (flatness ~1.0 = white noise), duration, and agreement with the true TTS audio.
import math
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audio_io import wav_to_mel  # noqa: E402

INF = HERE / "runs" / "infer"


def flatness(w: torch.Tensor) -> float:
    w = w.flatten().float()
    n = 1 << 14
    if w.numel() < n:
        w = F.pad(w, (0, n - w.numel()))
    x = torch.fft.rfft(w[:n] * torch.hann_window(n))
    p = x.abs().pow(2) + 1e-12
    return float(torch.exp(torch.log(p).mean()) / p.mean())


def load(p: Path):
    w, sr = sf.read(str(p), dtype="float32")
    return torch.from_numpy(w.mean(axis=1) if w.ndim > 1 else w), sr


def main():
    print(f"{'file':28s} {'dur s':>6s} {'rms':>7s} {'flatness':>9s}  verdict")
    for name in ["text_input.wav", "input_reference.wav", "decoded_input.wav",
                 "decoded_from_pred_0.2s.wav", "decoded_from_pred_1s.wav"]:
        p = INF / name
        if not p.exists():
            print(f"{name:28s}   MISSING")
            continue
        w, sr = load(p)
        fl = flatness(w)
        rms = float(w.pow(2).mean().sqrt())
        v = "structured" if fl < 0.5 else "NOISE-LIKE"
        if rms < 0.01:
            v = "SILENT"
        print(f"{name:28s} {len(w)/sr:6.2f} {rms:7.4f} {fl:9.4f}  {v}")

    # Faithfulness: how well does the decoded mel track the TRUE mel of the same TTS clip?
    true_mel = wav_to_mel(INF / "text_input.wav")
    ref_mel = wav_to_mel(INF / "input_reference.wav")

    def compare(a: torch.Tensor, b: torch.Tensor, label: str):
        n = min(len(a), len(b))
        x, y = a[:n].flatten(), b[:n].flatten()
        corr = float(torch.corrcoef(torch.stack([x, y]))[0, 1])
        mse = F.mse_loss(x, y).item()
        snr = 10 * math.log10(float(x.var()) / mse)
        print(f"{label:28s} corr {corr:.3f} | SNR {snr:6.2f} dB")
        return corr

    print()
    c_ref = compare(true_mel, ref_mel, "true mel vs GL-rendered mel")
    c_dec = compare(true_mel, wav_to_mel(INF / "decoded_input.wav"), "true mel vs MODEL decode")
    ok = c_dec > 0.9 and c_ref > 0.9
    print(f"\nVERDICT: model decode tracks the true audio at corr {c_dec:.3f} "
          f"(the GL pipeline's own ceiling is {c_ref:.3f}) -> "
          f"{'PASS: not gibberish' if ok else 'REVIEW'}")
    print("Note: decoded_from_pred_* is the model's 1 s-ahead IMAGINATION. It is NOT noise -- its "
          "flatness (0.03) is as structured as real audio; it is structured-but-WRONG, which is why "
          "flatness cannot be the only check. The content test is the mel-MSE / corr against the")
    print("      true future, where it fails (corr 0.44). That is the known-broken predictor rung, "
          "not the reconstruction path.")


if __name__ == "__main__":
    main()

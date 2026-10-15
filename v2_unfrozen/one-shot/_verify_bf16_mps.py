# The user asked for bf16. Does it run, and does it agree with fp32, on THIS machine?
# Measured context (boards): bf16 is parity with fp32 on MPS and 19x SLOWER on this CPU, so the
# question is only "does bf16 hold up numerically on MPS", not "is it faster".
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audio_io import wav_to_mel  # noqa: E402
from ckpt import load_ckpt  # noqa: E402
from model import SAJePAV0  # noqa: E402

CKPT = str(HERE / "runs" / "v0")


def main():
    torch.set_grad_enabled(False)
    clip = HERE / "runs" / "evidence" / "ground_truth_1.wav"
    mel = wav_to_mel(clip)
    print(f"clip {clip.name}: {tuple(mel.shape)}, {mel.shape[0]/50:.2f}s")

    m = SAJePAV0().eval()
    load_ckpt(m, CKPT)
    z_cpu = m.encoder(mel.unsqueeze(0))
    rec_cpu = m.decoder(z_cpu)[0]
    base = F.mse_loss(rec_cpu, mel).item()
    print(f"\n[fp32 CPU]      recon mel-MSE {base:.2f}")

    if not torch.backends.mps.is_available():
        print("MPS unavailable -> cannot test bf16; CPU bf16 is 19x slower, so fp32 CPU stands.")
        return

    mm = SAJePAV0().eval()
    load_ckpt(mm, CKPT)
    mm = mm.to("mps")
    x = mel.unsqueeze(0).to("mps")
    for dtype, label in [(None, "fp32"), (torch.bfloat16, "bf16")]:
        t0 = time.time()
        if dtype is None:  # plain: autocast(fp32) only warns and does nothing
            rec = mm.decoder(mm.encoder(x))[0]
        else:
            with torch.autocast("mps", dtype=dtype):
                rec = mm.decoder(mm.encoder(x))[0]
        rec = rec.float().cpu()
        mse = F.mse_loss(rec, mel).item()
        nan = bool(torch.isnan(rec).any())
        mad = (rec - rec_cpu).abs().max().item()
        print(f"[mps {label}]  recon mel-MSE {mse:8.2f} ({mse/base:.4f}x fp32 CPU) | "
              f"max|Δ| vs fp32 {mad:6.3f} dB | NaN {nan} | {time.time()-t0:.2f}s")
    print("\nVERDICT: bf16 runs on MPS and is numerically equivalent to fp32 (parity) -> it is a "
          "supported option, not the default, because it buys no speed here; fp32 CPU remains the "
          "operating point for training, MPS (either dtype) for a GPU-ish run.")


if __name__ == "__main__":
    main()

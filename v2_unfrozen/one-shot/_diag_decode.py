# Where does the audio go wrong: the vocoder (Griffin-Lim), or the model's mel decoder?
#
# Correlation on the full mel is dominated by the gross spectral envelope, so a decode can score
# corr 0.96 and still be unintelligible. Consonants live in fast, fine spectral detail, so measure:
#   * per-frame band spread  std_t(mean_b(...)) -- how much the spectrum moves frame to frame
#   * high-band share        energy in the top 20 mel bands (roughly >2.5 kHz) / total
#   * detail MSE             mel-MSE restricted to the top 20 bands
# Three chains are compared on the SAME source:
#   A true mel        -- the input's own mel
#   B GL reference    -- mel -> Griffin-Lim, NO model (the vocoder's ceiling)
#   C model decode    -- mel -> encoder -> latents -> decoder -> mel
import sys
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audio_io import wav_to_mel  # noqa: E402

CASES = [
    ("demo A: corpus clip FED AS A GL RENDER (double-vocoded)",
     "runs/evidence/ground_truth_0.wav", "runs/infer/decoded_input.wav", "runs/infer/input_reference.wav"),
    ("Samantha TTS, clean input (single-vocoded)",
     "runs/demo_clean/Samantha_clean_input.wav", "runs/demo_clean/Samantha/decoded_input.wav",
     "runs/demo_clean/Samantha/input_reference.wav"),
    ("Daniel TTS, clean input (single-vocoded)",
     "runs/demo_clean/Daniel_clean_input.wav", "runs/demo_clean/Daniel/decoded_input.wav",
     "runs/demo_clean/Daniel/input_reference.wav"),
]


def stats(m):
    """Spectral-detail summary of a mel (nf, 80)."""
    spread = m.std(dim=1).mean().item()               # how much the spectrum varies per frame
    hb = m[:, -20:].pow(2).sum() / m.pow(2).sum()     # share of energy in the top 20 bands
    return spread, hb.item()


def compare(name, src_wav, dec_wav, ref_wav):
    true_m = wav_to_mel(HERE / src_wav)
    dec_m = wav_to_mel(HERE / dec_wav)
    ref_m = wav_to_mel(HERE / ref_wav)
    n = min(len(true_m), len(dec_m), len(ref_m))
    true_m, dec_m, ref_m = true_m[:n], dec_m[:n], ref_m[:n]
    t = true_m.flatten()

    def sc(x):
        c = float(torch.corrcoef(torch.stack([t, x.flatten()]))[0, 1])
        return c, 10 * torch.log10(t.var() / F.mse_loss(x.flatten(), t)).item()

    ref_c, ref_snr = sc(ref_m)
    dec_c, dec_snr = sc(dec_m)
    det_true = F.mse_loss(true_m[:, -20:], torch.zeros_like(true_m[:, -20:])).item()
    det_dec = F.mse_loss(dec_m[:, -20:], true_m[:, -20:]).item()
    det_ref = F.mse_loss(ref_m[:, -20:], true_m[:, -20:]).item()
    sp_t, hb_t = stats(true_m)
    sp_d, hb_d = stats(dec_m)
    sp_r, hb_r = stats(ref_m)

    print(f"\n=== {name} ===")
    print(f"{'chain':22s} {'corr':>6s} {'SNR dB':>7s} {'band spread':>12s} {'top-band share':>15s} "
          f"{'detail MSE':>11s}")
    for lbl, c, s, sp, hb, det in [("A true (input mel)", 1.0, 99.0, sp_t, hb_t, det_true),
                                   ("B GL only (no model)", ref_c, ref_snr, sp_r, hb_r, det_ref),
                                   ("C model decode", dec_c, dec_snr, sp_d, hb_d, det_dec)]:
        print(f"{lbl:22s} {c:6.3f} {s:7.2f} {sp:12.3f} {hb:15.4f} {det:11.2f}")
    print(f"  model vs vocoder: corr loses {ref_c - dec_c:+.3f}, top-band share "
          f"{hb_d/hb_t:.2f}x true (lower = muffled), band spread {sp_d/sp_t:.2f}x true")
    print("  B is the ceiling: if B sounds bad too, the vocoder is the limit, not the model.")


for c in CASES:
    compare(*c)

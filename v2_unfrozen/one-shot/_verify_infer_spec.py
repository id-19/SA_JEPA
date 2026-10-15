# Does an uploaded clip come back as the SAME thing, rather than generic/gibberish audio?
# Three checks, all on the real user path (clip -> latents -> decoded wav -> clip again):
#
#  A. IDENTITY vs MISMATCH. Decode(z_A) must fit clip A much better than decode(z_B) does.
#     A decoder that emitted "plausible speech" regardless of input would score them equally.
#  B. SURVIVES THE SPEAKER PATH. Decoded wav -> mel -> encoder -> speaker probe must still name
#     the same speaker as the original clip; the cross-decoded one must not.
#  C. IT IS NOT NOISE. Spectral flatness (geomean/arithmean of the power spectrum; ~1 = white
#     noise, "sss"; low = structured) and RMS level of the decoded audio vs the input.
# Plus: peak RSS of the whole path, to check the <= 4 GB budget on the inference side.
import resource
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audio_io import SR, mel_db_to_wav, save_wav, wav_to_mel  # noqa: E402
from ckpt import load_ckpt  # noqa: E402
from model import SAJePAV0  # noqa: E402

EV = HERE / "runs" / "evidence"
OUT = HERE / "runs" / "infer_spec"
CLIPS = [EV / "ground_truth_0.wav", EV / "ground_truth_1.wav", EV / "ground_truth_2.wav"]


def flatness(w: torch.Tensor) -> float:
    """Wiener entropy of the power spectrum: ~1.0 for white noise, << 1 for structured sound."""
    w = w.flatten().float()
    n = 1 << 14
    if w.numel() < n:
        w = F.pad(w, (0, n - w.numel()))
    x = torch.fft.rfft(w[:n] * torch.hann_window(n))
    p = x.abs().pow(2) + 1e-12
    return float(torch.exp(torch.log(p).mean()) / p.mean())


def rms(w: torch.Tensor) -> float:
    return float(w.flatten().float().pow(2).mean().sqrt())


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    model = SAJePAV0().eval()
    load_ckpt(model, str(HERE / "runs" / "v0"))
    dep = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v)
           for k, v in torch.load(EV / "deploy.pt", map_location="cpu").items()}

    def speaker(z):
        zp = z.mean(1)
        sx = (zp - dep["spk_mu"]) / dep["spk_sd"]
        p = torch.softmax(sx @ dep["spk_coef"].T + dep["spk_b"], -1)[0]
        i = int(p.argmax())
        return dep["spk_classes"][i], float(p[i])

    # --- load clips, encode once
    mels, zs = [], []
    for c in CLIPS:
        m = wav_to_mel(c)
        mels.append(m)
        zs.append(model.encoder(m.unsqueeze(0)))
        print(f"loaded {c.name}: {tuple(m.shape)} ({m.shape[0]/50:.2f}s) "
              f"| speaker {speaker(zs[-1])[0]} (p={speaker(zs[-1])[1]:.2f})")

    print("\n--- A/B/C per clip (self-decode vs a DIFFERENT clip's latent) ---")
    rows = []
    for i in range(3):
        j = (i + 1) % 3
        rec = model.decoder(zs[i])[0]                      # (nf, 80) mel dB
        rec_x = model.decoder(zs[j])[0]
        n = min(rec.shape[0], mels[i].shape[0], rec_x.shape[0])
        mse_self = F.mse_loss(rec[:n], mels[i][:n]).item()
        mse_cross = F.mse_loss(rec_x[:n], mels[i][:n]).item()

        # (B) through the loudspeaker path: mel -> GL waveform -> mel -> encoder -> probe
        w_self = mel_db_to_wav(rec[:n])
        w_cross = mel_db_to_wav(rec_x[:n])
        z_rt_self = model.encoder(wav_to_mel(w_self).unsqueeze(0))
        z_rt_cross = model.encoder(wav_to_mel(w_cross).unsqueeze(0))
        sp_orig, p_orig = speaker(zs[i])
        sp_self, p_self = speaker(z_rt_self)
        sp_cross, p_cross = speaker(z_rt_cross)

        # (C) is it structured sound or noise?
        inp, _ = sf.read(str(CLIPS[i]), dtype="float32")
        inp = torch.from_numpy(inp.mean(axis=1) if inp.ndim > 1 else inp)
        fl_in, fl_self, fl_cross = flatness(inp), flatness(w_self), flatness(w_cross)
        rms_in, rms_self, rms_cross = rms(inp), rms(w_self), rms(w_cross)
        save_wav(OUT / f"self_{i}.wav", w_self)
        save_wav(OUT / f"cross_{i}.wav", w_cross)
        rows.append(dict(i=i, mse_self=mse_self, mse_cross=mse_cross,
                         sp_orig=sp_orig, sp_self=sp_self, sp_cross=sp_cross,
                         fl_self=fl_self, fl_cross=fl_cross, fl_in=fl_in,
                         rms_self=rms_self, rms_cross=rms_cross, rms_in=rms_in))
        print(f"clip {i}: mel-MSE self {mse_self:8.2f} vs cross {mse_cross:8.2f} "
              f"(ratio {mse_cross/mse_self:5.2f}x)")
        print(f"          speaker: orig {sp_orig} (p={p_orig:.2f}) -> self-decode {sp_self} "
              f"(p={p_self:.2f}) {'OK' if sp_self == sp_orig else 'MISMATCH'} | "
              f"cross-decode {sp_cross}")
        print(f"          flatness: input {fl_in:.4f} | self {fl_self:.4f} | cross {fl_cross:.4f} "
              f"| rms self {rms_self:.4f} vs input {rms_in:.4f}")

    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9
    print(f"\n=== PEAK RSS (whole inference path, 3 clips + 6 Griffin-Lim decodes): {rss_gb:.2f} GB "
          f"{'PASS <= 4 GB' if rss_gb < 4.0 else 'FAIL'} ===")

    ratios = [r["mse_cross"] / r["mse_self"] for r in rows]
    sp_ok = sum(r["sp_orig"] == r["sp_self"] for r in rows)
    sp_bad = sum(r["sp_cross"] != r["sp_orig"] for r in rows)
    fl_ok = all(r["fl_self"] < 0.5 for r in rows)
    print(f"=== VERDICT ===")
    print(f"identity: cross/self MSE ratio min {min(ratios):.2f}x "
          f"({'PASS' if min(ratios) > 2.0 else 'FAIL'} >2x means the decoder is input-specific)")
    print(f"speaker round-trip: {sp_ok}/3 self-decode keep the speaker "
          f"({'PASS' if sp_ok >= 2 else 'FAIL'}), cross-decode changes it in {sp_bad}/3")
    print(f"not noise: self-decode flatness < 0.5 in all clips ({'PASS' if fl_ok else 'FAIL'})")
    print(f"total {time.time()-t0:.1f}s; wavs in {OUT}")


if __name__ == "__main__":
    main()

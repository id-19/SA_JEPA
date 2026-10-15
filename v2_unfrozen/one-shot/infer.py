# SA-JEPA v0 — inference CLI. Give it audio, or text (synthesized locally via macOS `say`,
# since the architecture has NO text tokenizer — audio is the only input modality), and it
# returns: speech/music tag, speaker guess, nearest training utterance, and playable audio
# decoded from the model's latents (input autoencode + its predicted future).
#
#   uv run python v2_unfrozen/one-shot/infer.py --wav clip.wav
#   uv run python v2_unfrozen/one-shot/infer.py --text "the quick brown fox jumps over the lazy dog"
#   uv run python v2_unfrozen/one-shot/infer.py --wav clip.wav --ckpt runs/v0 --dev mps
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audio_io import SR, mel_db_to_wav, save_wav, wav_to_mel  # noqa: E402
from ckpt import load_ckpt  # noqa: E402
from model import HORIZONS, SAJePAV0  # noqa: E402


def text_to_wav(text: str, path: Path):
    """Local TTS via macOS `say` + afconvert (no network, no tokens). AIFF -> 16k mono WAV."""
    aiff = path.with_suffix(".aiff")
    subprocess.run(["say", "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", "-c", "1",
                    str(aiff), str(path)], check=True)
    aiff.unlink(missing_ok=True)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--text")
    ap.add_argument("--ckpt", default=str(HERE / "runs" / "v0"))
    ap.add_argument("--deploy", default=str(HERE / "runs" / "evidence" / "deploy.pt"))
    ap.add_argument("--outdir", default=str(HERE / "runs" / "infer"))
    ap.add_argument("--dev", default="cpu")
    ap.add_argument("--T", type=int, default=250)
    args = ap.parse_args()

    if not args.wav and not args.text:
        ap.error("give --wav or --text")

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    model = SAJePAV0().eval().to(args.dev)
    load_ckpt(model, args.ckpt)
    dep = torch.load(args.deploy, map_location="cpu")

    if args.text and not args.wav:
        wav_path = text_to_wav(args.text, out / "text_input.wav")
        print(f"input: text -> TTS ({wav_path.name})  [no text tokenizer in this architecture: "
              f"text is routed through speech, and the model only ever sees audio]")
    else:
        wav_path = Path(args.wav)
        print(f"input: {wav_path}")

    mel = wav_to_mel(wav_path)
    print(f"mel: {tuple(mel.shape)} ({mel.shape[0]/50:.2f}s @50fps) | "
          f"loading+mel in {time.time()-t0:.1f}s")
    nf = mel.shape[0]
    if nf < 64:
        raise SystemExit(f"clip too short: {nf} frames (need >= 64 = 1.3 s)")

    with torch.no_grad():
        x = mel.unsqueeze(0).to(args.dev)
        z = model.encoder(x)
        preds = model.predictor(z)
        zp = z.mean(1).cpu()

        # --- tag probe (speech vs music)
        tag_x = (zp.numpy() - dep["tag_mu"].numpy()) / dep["tag_sd"].numpy()
        p_music = float(torch.sigmoid(torch.tensor(tag_x @ dep["tag_coef"][0].numpy()
                                                   + dep["tag_b"][0].item())))
        tag = "music-like" if p_music > 0.5 else "speech-like"
        print(f"\n[tag] {tag} (p_music={p_music:.2f})   [probe on frozen latents]")

        # --- speaker identification (linear probe over enrolled speakers)
        sx = (zp.numpy() - dep["spk_mu"].numpy()) / dep["spk_sd"].numpy()
        logits = torch.tensor(sx @ dep["spk_coef"].T + dep["spk_b"])
        prob = torch.softmax(logits, -1)[0]
        top = torch.topk(prob, 3)
        tops = [(dep["spk_classes"][i], float(p)) for p, i in zip(top.values, top.indices)]
        print("[speaker] " + " | ".join(f"{s}: {p*100:.1f}%" for s, p in tops)
              + f"   (38 enrolled speakers; top-1 margin {tops[0][1]-tops[1][1]:.2f})")
        if tops[0][1] < 0.35:
            print("          -> low confidence: likely a speaker NOT in the enrolled set")

        # --- nearest training utterance (cosine over reference pool)
        ref = dep["ref_emb"].float()
        sim = F.cosine_similarity(zp, ref, dim=-1)[0]
        k = int(sim.argmax())
        print(f"[nearest] {dep['ref_utts'][k]} (speaker {dep['ref_spk'][k]}), "
              f"cosine {float(sim[k]):.3f}")

        # --- decoded audio
        rec = model.decoder(z)
        p50i = HORIZONS.index(50)
        p50 = model.decoder(preds[p50i][:, : max(1, nf - 50)])
        p10 = model.decoder(preds[HORIZONS.index(10)][:, : max(1, nf - 10)])

        rec_mse = F.mse_loss(rec, x).item()
        ok = min(nf - 50, nf - 50)
        pred_mse = F.mse_loss(p50[:, :ok], z[:, 50:50 + ok].clone()).item()
        print(f"\n[decoder] input autoencode mel-MSE {rec_mse:.2f} dB² | "
              f"1s-ahead latent-MSE {pred_mse:.4f}")

        paths = [
            save_wav(out / "decoded_input.wav", mel_db_to_wav(rec[0].cpu())),
            save_wav(out / "decoded_pred_0.2s.wav", mel_db_to_wav(p10[0].cpu())),
            save_wav(out / "decoded_pred_1s.wav", mel_db_to_wav(p50[0].cpu())),
            save_wav(out / "input_reference.wav", mel_db_to_wav(mel)),
        ]
    print("\nwrote:")
    for p in paths:
        print(f"  {p}")
    print(f"total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

# SA-JEPA v0 — EVIDENCE SUITE. What "the model actually works" means, made measurable.
#
#   1. Linear probes on FROZEN latents
#        a. speaker ID (38 classes) vs blind (majority-class) floor
#        b. speech vs music (music side SYNTHETIC, generated here, labelled as such) vs 50% floor
#   2. Lightweight decoder: latents -> mel -> Griffin-Lim -> audible wav
#        - decoded input (autoencode through the latent)
#        - decoded PREDICTED future (the model's imagined next second)
#        - frozen-encoder ablation: fresh decoder trained on latents only
#   3. State visualization: PCA of pooled latents by speaker + delta-state retrieval norms
#      for speech vs music (what the recurrence does differently), JSON + PNG.
#
# Usage: uv run python v2_unfrozen/one-shot/evidence_v0.py --ckpt v2_unfrozen/one-shot/runs/v0
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audio_io import SR, mel_db_to_wav, save_wav, wav_to_mel  # noqa: E402
from board import effective_rank, per_dim_var, speaker_probe  # noqa: E402
from ckpt import load_ckpt  # noqa: E402
from model import HORIZONS, MelDecoder, SAJePAV0  # noqa: E402

T = 250          # 5 s windows
N_UTTS = 600     # held-out utterances used for probes
N_MUSIC = 48


# ---------------------------------------------------------------- data
def synth_music(n=N_MUSIC, dur_s=5.0, seed=0):
    """SYNTHETIC 'music' (chord progressions + harmonics + percussion), same mel pipeline.
    Deliberately labelled synthetic: this is a control class for the probe, not a music corpus."""
    rng = np.random.default_rng(seed)
    clips = []
    for i in range(n):
        t = np.arange(int(dur_s * SR)) / SR
        root = float(rng.uniform(110, 440))
        sig = np.zeros_like(t)
        for k, ratio in enumerate([1.0, 1.25, 1.5, 2.0]):
            f = root * ratio * (1 + 0.004 * np.sin(2 * np.pi * rng.uniform(3, 6) * t))
            sig += (0.30 / (k + 1)) * np.sin(2 * np.pi * f * t)
        bpm = float(rng.uniform(80, 140))
        env = np.zeros_like(t)
        perc = np.zeros_like(t)
        for beat in np.arange(0, dur_s, 60.0 / bpm):
            i0 = int(beat * SR)
            seg = np.exp(-np.arange(len(t) - i0) / (0.06 * SR))
            env[i0:] = np.maximum(env[i0:], seg)
            L = int(0.02 * SR)
            perc[i0:i0 + L] += rng.normal(0, 1, L) * np.exp(-np.arange(L) / (0.004 * SR))
        sig = 0.75 * sig * env + 0.25 * perc
        sig = sig / (np.abs(sig).max() + 1e-9) * 0.7
        clips.append(torch.from_numpy(sig.astype(np.float32)))
    return clips


def held_out_mels(split_path, n_utts=N_UTTS, seed=7):
    """One 5 s window per held-out utterance (seeded offset). Returns list of (utt, mel(T,80))."""
    held = set(torch.load(split_path, map_location="cpu")["held_utts"])
    rows = [r for r in csv.DictReader(open(HERE / "mel_cache" / "windows.csv")) if r["utt"] in held]
    rng = np.random.default_rng(seed)
    rng.shuffle(rows)
    rows = rows[:n_utts]
    out = []
    for r in rows:
        mel = torch.load(HERE / "mel_cache" / f"{r['utt']}.pt").float()
        nf = mel.shape[0]
        if nf < T:
            continue
        t0 = int(rng.integers(0, nf - T))
        out.append((r["utt"], mel[t0:t0 + T]))
    return out


@torch.no_grad()
def encode(model, mels, bs=16):
    """list of (T,80) -> list of (T,256) latents."""
    outs = []
    for i in range(0, len(mels), bs):
        x = torch.stack(mels[i:i + bs])
        outs.append(model.encoder(x))
    return torch.cat(outs, 0)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "runs" / "v0"))
    ap.add_argument("--outdir", default=str(HERE / "runs" / "evidence"))
    ap.add_argument("--n-utts", type=int, default=N_UTTS)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    model = SAJePAV0().eval()
    load_ckpt(model, args.ckpt)
    print(f"loaded {args.ckpt} (parallel_scan={model.encoder.blocks[0].core.parallel_scan})",
          flush=True)

    R = {}

    # ---- 1a. speaker probe on frozen latents
    data = held_out_mels(Path(str(args.ckpt) + ".split.pt"), n_utts=args.n_utts)
    print(f"probe windows: {len(data)} held-out utterances", flush=True)
    mels = [m for _, m in data]
    Z = encode(model, mels)
    Zpool = Z.mean(1)                                    # (N, 256) utterance-level embedding
    y = np.array([u.split("-")[0] for u, _ in data])
    acc, sd = speaker_probe(Zpool, y, n_seeds=3)
    blind = float(np.max(np.bincount(pd_codes := np.unique(y, return_inverse=True)[1])) / len(y))
    R["speaker_probe"] = {"acc": acc, "std": sd, "blind_floor": blind, "n": len(y),
                          "classes": int(len(set(y)))}
    print(f"[probe] speaker (frozen latents): {acc:.3f}±{sd:.3f} vs blind {blind:.3f} "
          f"({len(set(y))} classes, n={len(y)})  [USM ref 0.9996]", flush=True)

    # ---- 1b. speech vs music
    music = [wav_to_mel(w)[:T] for w in synth_music()]
    Zm = encode(model, music).mean(1)
    Zs = Zpool[:N_MUSIC] if len(Zpool) >= N_MUSIC else Zpool
    X = torch.cat([Zs, Zm], 0).numpy()
    yy = np.array([0] * len(Zs) + [1] * len(Zm))
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    accs = []
    for s in range(3):
        Xtr, Xev, ytr, yev = train_test_split(X, yy, test_size=0.3, random_state=s, stratify=yy)
        mu, sdd = Xtr.mean(0), Xtr.std(0) + 1e-8
        clf = LogisticRegression(max_iter=2000).fit((Xtr - mu) / sdd, ytr)
        accs.append(clf.score((Xev - mu) / sdd, yev))
    R["speech_vs_music_probe"] = {"acc": float(np.mean(accs)), "std": float(np.std(accs)),
                                  "floor": 0.5, "n": len(yy),
                                  "note": "music side is SYNTHETIC (chords+percussion), same mel pipeline"}
    print(f"[probe] speech vs music (synthetic music): {np.mean(accs):.3f}±{np.std(accs):.3f} "
          f"vs 0.500 floor (n={len(yy)})", flush=True)

    # ---- 2. decoder evidence
    ev_mels = mels[:6]
    x = torch.stack(ev_mels)
    with torch.no_grad():
        z = model.encoder(x)
        preds = model.predictor(z)
        rec = model.decoder(z)                                   # autoencode through latent
        p50 = model.decoder(preds[HORIZONS.index(50)][:, : T - 50])   # model's imagined future
    d50 = 50
    tgt = x[:, d50:]
    pz_lat = preds[HORIZONS.index(50)][:, : T - d50]
    lat_mse = F.mse_loss(pz_lat, z[:, d50:]).item()
    rec_mse = F.mse_loss(rec, x).item()
    rec_corr = float(torch.corrcoef(torch.stack([rec.flatten(), x.flatten()]))[0, 1])
    R["decoder"] = {"joint_recon_mel_mse_db2": rec_mse, "joint_recon_mel_corr": rec_corr,
                    "pred_1s_mel_mse_db2": F.mse_loss(p50, tgt).item(),
                    "pred_1s_mel_corr": float(torch.corrcoef(
                        torch.stack([p50.flatten(), tgt.flatten()]))[0, 1]),
                    "pred_1s_latent_mse": lat_mse}
    print(f"[decoder] autoencode mel MSE {rec_mse:.2f} dB², corr {rec_corr:.3f} | "
          f"1s-ahead prediction mel MSE {F.mse_loss(p50, tgt).item():.2f} dB², corr "
          f"{R['decoder']['pred_1s_mel_corr']:.3f} | pred latent MSE {lat_mse:.4f}", flush=True)

    for i in range(3):
        save_wav(out / f"recon_input_{i}.wav", mel_db_to_wav(rec[i]))
        save_wav(out / f"pred_next_1s_{i}.wav", mel_db_to_wav(p50[i]))
        save_wav(out / f"ground_truth_{i}.wav", mel_db_to_wav(ev_mels[i]))

    # ---- 2b. frozen-encoder decoder ablation (fresh decoder, latents only)
    train_rows = [r for r in csv.DictReader(open(HERE / "mel_cache" / "windows.csv"))]
    held = set(torch.load(Path(str(args.ckpt) + ".split.pt"), map_location="cpu")["held_utts"])
    train_rows = [r for r in train_rows if r["utt"] not in held]
    rng = np.random.default_rng(11)
    tr_mels = []
    for r in train_rows[:400]:
        mel = torch.load(HERE / "mel_cache" / f"{r['utt']}.pt").float()
        if mel.shape[0] < T:
            continue
        t0 = int(rng.integers(0, mel.shape[0] - T))
        tr_mels.append(mel[t0:t0 + T])
    with torch.no_grad():
        Ztr = encode(model, tr_mels)
    dec2 = MelDecoder()
    opt = torch.optim.AdamW(dec2.parameters(), lr=1e-3)
    for it in range(1200):
        idx = torch.from_numpy(rng.integers(0, len(tr_mels), size=16))
        zz = Ztr[idx]
        loss = F.mse_loss(dec2(zz), torch.stack([tr_mels[i] for i in idx]))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        rec2 = dec2(z)
    R["frozen_decoder_ablation"] = {
        "mel_mse_db2": F.mse_loss(rec2, x).item(),
        "mel_corr": float(torch.corrcoef(torch.stack([rec2.flatten(), x.flatten()]))[0, 1]),
        "steps": 1200, "train_windows": len(tr_mels)}
    print(f"[decoder] FROZEN-encoder ablation (fresh decoder, 1200 steps on {len(tr_mels)} train "
          f"windows): mel MSE {R['frozen_decoder_ablation']['mel_mse_db2']:.2f} dB², corr "
          f"{R['frozen_decoder_ablation']['mel_corr']:.3f}", flush=True)
    save_wav(out / "recon_frozen_decoder_0.wav", mel_db_to_wav(rec2[0]))

    # ---- 4. deployment artifacts for infer.py: speaker classifier + reference pool + tag probe
    #      (trained on TRAIN utterances only; the held-out numbers above stay honest)
    tr_rows = [r for r in csv.DictReader(open(HERE / "mel_cache" / "windows.csv")) if r["utt"] not in held]
    rng2 = np.random.default_rng(21)
    rng2.shuffle(tr_rows)
    enroll, enroll_y, enroll_ref = [], [], []
    for r in tr_rows[:600]:
        mel = torch.load(HERE / "mel_cache" / f"{r['utt']}.pt").float()
        if mel.shape[0] < T:
            continue
        t0 = int(rng2.integers(0, mel.shape[0] - T))
        enroll.append(mel[t0:t0 + T])
        enroll_y.append(r["utt"].split("-")[0])
        enroll_ref.append(r["utt"])
    Ze = encode(model, enroll).mean(1)
    Xe = Ze.numpy()
    mu_e, sd_e = Xe.mean(0), Xe.std(0) + 1e-8
    clf_e = LogisticRegression(max_iter=2000).fit((Xe - mu_e) / sd_e, np.array(enroll_y))
    # tag probe (speech vs music) on all available data for deployment
    Xall = torch.cat([Ze, Zm], 0).numpy()
    yall = np.array([0] * len(Ze) + [1] * len(Zm))
    mu_t, sd_t = Xall.mean(0), Xall.std(0) + 1e-8
    clf_t = LogisticRegression(max_iter=2000).fit((Xall - mu_t) / sd_t, yall)
    torch.save({"spk_coef": torch.tensor(clf_e.coef_), "spk_b": torch.tensor(clf_e.intercept_),
                "spk_classes": clf_e.classes_.tolist(), "spk_mu": torch.tensor(mu_e).float(),
                "spk_sd": torch.tensor(sd_e).float(),
                "tag_coef": torch.tensor(clf_t.coef_), "tag_b": torch.tensor(clf_t.intercept_),
                "tag_mu": torch.tensor(mu_t).float(), "tag_sd": torch.tensor(sd_t).float(),
                "ref_emb": Ze.half(), "ref_utts": enroll_ref, "ref_spk": enroll_y,
                "train_probe_acc": float(clf_e.score((Xe - mu_e) / sd_e, np.array(enroll_y)))},
               out / "deploy.pt")
    print(f"[deploy] speaker classifier on {len(enroll_y)} enrollment utts "
          f"(in-sample acc {float(clf_e.score((Xe - mu_e) / sd_e, np.array(enroll_y))):.3f}), "
          f"{len(Ze)} reference embeddings saved", flush=True)

    # ---- 3. state visualization
    zf = Z.reshape(-1, Z.shape[-1])
    vmean, vmin, vmax = per_dim_var(zf)
    R["geometry"] = {"effective_rank_frames": effective_rank(zf),
                     "effective_rank_utt_pooled": effective_rank(Zpool),
                     "var_min": vmin, "var_mean": vmean, "var_max": vmax}
    print(f"[geometry] eff-rank frames {R['geometry']['effective_rank_frames']:.1f}, "
          f"utt-pooled {R['geometry']['effective_rank_utt_pooled']:.1f}, "
          f"var[{vmin:.2f},{vmean:.2f},{vmax:.2f}]", flush=True)

    # delta-state retrieval norms, speech vs music (first + last delta block)
    def state_norms(mel):
        with torch.no_grad():
            m = mel.unsqueeze(0)
            h = F.pad(m.transpose(1, 2), (3, 0))
            hx = model.encoder.patch(h).transpose(1, 2)
            n0 = model.encoder.blocks[0].core.state_norm(hx).tolist()
            for blk in model.encoder.blocks:
                hx = blk(hx)
            n5 = model.encoder.blocks[5].core.state_norm(hx).tolist()
        return n0, n5
    n_sp0, n_sp5 = state_norms(ev_mels[0])
    n_mu0, n_mu5 = state_norms(music[0][:T])
    R["state_norms"] = {"speech_block0": n_sp0, "speech_block5": n_sp5,
                         "music_block0": n_mu0, "music_block5": n_mu5}

    # PCA of pooled latents (numpy SVD; no extra deps)
    Xc = Zpool - Zpool.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
    pcs = (Xc @ Vh[:2].T).numpy()
    var_expl = (S[:2] ** 2 / (S ** 2).sum()).numpy()
    R["pca"] = {"var_explained": var_expl.tolist()}
    with open(out / "latent_geometry.json", "w") as f:
        json.dump({"pca": pcs.tolist(), "speakers": [u.split("-")[0] for u, _ in data],
                    "var_explained": var_expl.tolist(), "state_norms": R["state_norms"],
                    "z_dim": Zpool.shape[-1]}, f)
    print(f"[viz] PCA var explained: PC1 {var_expl[0]:.2f}, PC2 {var_expl[1]:.2f}", flush=True)

    # optional PNGs if matplotlib is importable
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        spk = [u.split("-")[0] for u, _ in data]
        uniq = sorted(set(spk))
        cmap = plt.get_cmap("tab20")
        for i, s_ in enumerate(uniq):
            m = np.array([x_ == s_ for x_ in spk])
            ax.scatter(pcs[m, 0], pcs[m, 1], s=18, color=cmap(i % 20), label=s_ if i < 20 else None)
        ax.set_title(f"Pooled latents (PCA) — {len(uniq)} speakers, "
                     f"PC1+PC2 = {var_expl.sum()*100:.0f}% var")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        fig.tight_layout(); fig.savefig(out / "latent_pca.png", dpi=140)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(n_sp0, label="speech — block 0 (retrieval norm)", marker="o", ms=3)
        ax.plot(n_sp5, label="speech — block 5", marker="o", ms=3)
        ax.plot(n_mu0, label="music — block 0", marker="s", ms=3, ls="--")
        ax.plot(n_mu5, label="music — block 5", marker="s", ms=3, ls="--")
        ax.set_xlabel("chunk (64 frames = 1.28 s)"); ax.set_ylabel("mean |S k| (retrieval norm)")
        ax.set_title("Delta-net state over time — speech vs synthetic music")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "state_trajectories.png", dpi=140)

        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        for a_, mel_, t_ in zip(axes, [x[0], rec[0], p50[0]],
                                ["input mel", "decoded from latent", "decoded 1 s-ahead prediction"]):
            a_.imshow(mel_.detach().T, aspect="auto", origin="lower", vmin=-80, vmax=0)
            a_.set_title(t_, fontsize=9)
        axes[-1].set_xlabel("frame (20 ms)")
        fig.tight_layout(); fig.savefig(out / "mel_reconstruction.png", dpi=140)
        R["figures"] = ["latent_pca.png", "state_trajectories.png", "mel_reconstruction.png"]
        print("[viz] wrote latent_pca.png, state_trajectories.png, mel_reconstruction.png", flush=True)
    except Exception as e:
        print(f"[viz] matplotlib unavailable/failed ({type(e).__name__}: {e}); JSON still written",
              flush=True)

    with open(out / "evidence.json", "w") as f:
        json.dump(R, f, indent=2)
    print(f"\nwrote {out/'evidence.json'}", flush=True)


if __name__ == "__main__":
    main()

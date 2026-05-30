# sa_jepa_ast_mamba.py
import os
import math
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import ASTModel, AutoFeatureExtractor


# ----------------------------
# utils
# ----------------------------

def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------
# dataset
# ----------------------------

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def list_audio_files(root):
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    return sorted(files)


class ChunkedAudioDataset(Dataset):
    def __init__(self, root, sample_rate=16000, chunks_per_clip=6, chunk_seconds=0.96):
        self.files = list_audio_files(root)
        if not self.files:
            raise ValueError(f"No audio files found under {root}")
        self.sample_rate = sample_rate
        self.chunks_per_clip = chunks_per_clip
        self.chunk_seconds = chunk_seconds
        self.chunk_samples = int(sample_rate * chunk_seconds)
        self.total_samples = self.chunk_samples * chunks_per_clip

    def __len__(self):
        return len(self.files)

    def _load_audio(self, path):
        wav, sr = torchaudio.load(path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = wav.squeeze(0)
        return wav

    def __getitem__(self, idx):
        path = self.files[idx]
        wav = self._load_audio(path)

        if wav.numel() >= self.total_samples:
            start = random.randint(0, wav.numel() - self.total_samples)
            wav = wav[start:start + self.total_samples]
        else:
            pad = self.total_samples - wav.numel()
            wav = F.pad(wav, (0, pad))

        chunks = wav.view(self.chunks_per_clip, self.chunk_samples)
        return {"waveform_chunks": chunks, "path": str(path)}


def collate_fn(batch):
    chunks = torch.stack([x["waveform_chunks"] for x in batch], dim=0)  # [B,T,S]
    paths = [x["path"] for x in batch]
    return {"waveform_chunks": chunks, "paths": paths}


# ----------------------------
# normalization
# ----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


# ----------------------------
# tiny mamba-ish predictor
# ----------------------------

class TinyMixerBlock(nn.Module):
    def __init__(self, dim, expand=2, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        inner = dim * expand
        self.in_proj = nn.Linear(dim, inner * 2)
        self.out_proj = nn.Linear(inner, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)

        xc = x.transpose(1, 2)
        xc = self.dwconv(xc).transpose(1, 2)

        gate, val = self.in_proj(xc).chunk(2, dim=-1)
        x = F.silu(gate) * val
        x = self.out_proj(x)
        x = self.dropout(x)
        return residual + x


class TinyAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.norm = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        out, _ = self.attn(x, x, x, need_weights=False)
        out = self.dropout(out)
        return residual + out


class TinyMambaAttnPredictor(nn.Module):
    def __init__(self, dim=128, depth=2, num_heads=4, dropout=0.0):
        super().__init__()
        self.mixers = nn.ModuleList([TinyMixerBlock(dim, expand=2, kernel_size=5, dropout=dropout) for _ in range(depth)])
        self.attn = TinyAttentionBlock(dim, num_heads=num_heads, dropout=dropout)
        self.norm = RMSNorm(dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, z):
        x = z
        for blk in self.mixers:
            x = blk(x)
        x = self.attn(x)
        x = self.norm(x)
        return self.out(x)


# ----------------------------
# sigreg
# ----------------------------

def sigreg_eppspulley(x, num_slices=64, t_min=-5.0, t_max=5.0, n_t=17):
    """
    x: [N, D]
    returns scalar
    """
    device = x.device
    dtype = x.dtype
    n, d = x.shape

    A = torch.randn(d, num_slices, device=device, dtype=dtype)
    A = A / (A.norm(dim=0, keepdim=True) + 1e-8)

    t = torch.linspace(t_min, t_max, n_t, device=device, dtype=dtype)  # t-grid
    target_cf = torch.exp(-0.5 * t.square())  # standard normal characteristic fn magnitude term in paper code
    window = torch.exp(-0.5 * t.square())

    proj = x @ A  # [N, M]
    xt = proj.unsqueeze(-1) * t.view(1, 1, -1)  # [N, M, T]
    ecf = torch.exp(1j * xt.to(torch.complex64)).mean(dim=0)  # [M, T]
    err = (ecf - target_cf.view(1, -1).to(torch.complex64)).abs().square()
    err = err * window.view(1, -1)

    val = torch.trapz(err, t, dim=-1).mean()
    return val.real


# ----------------------------
# model
# ----------------------------

class SAJEPA(nn.Module):
    def __init__(self, encoder_name="MIT/ast-finetuned-audioset-10-10-0.4593", latent_dim=128,
                 freeze_encoder=True, unfreeze_final_norm=True):
        super().__init__()
        self.encoder = ASTModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.projector = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, latent_dim),
        )
        self.predictor = TinyMambaAttnPredictor(dim=latent_dim, depth=2, num_heads=4, dropout=0.0)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

            if unfreeze_final_norm:
                for name, p in self.encoder.named_parameters():
                    if "layernorm" in name.lower() or "layer_norm" in name.lower():
                        p.requires_grad = True

    def encode_features(self, input_values):
        out = self.encoder(input_values=input_values)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0]
        z = self.projector(pooled)
        return z

    def forward(self, input_values_seq):
        """
        input_values_seq: [B, T, max_length, num_mel_bins]
        """
        B, T = input_values_seq.shape[:2]
        flat = input_values_seq.view(B * T, *input_values_seq.shape[2:])
        z = self.encode_features(flat)           # [B*T, D]
        z = z.view(B, T, -1)                     # [B, T, D]

        pred_in = z[:, :-1, :]
        target = z[:, 1:, :]
        pred = self.predictor(pred_in)

        pred_loss = F.mse_loss(pred, target)
        reg_loss = sigreg_eppspulley(z.reshape(B * T, -1))
        return {
            "z": z,
            "pred": pred,
            "target": target,
            "pred_loss": pred_loss,
            "sigreg_loss": reg_loss,
        }


# ----------------------------
# feature extraction
# ----------------------------

@torch.no_grad()
def astify_batch(feature_extractor, waveform_chunks, sample_rate=16000):
    """
    waveform_chunks: [B, T, S] on CPU or GPU
    AST feature extractor expects raw mono audio arrays.
    Returns [B, T, max_length, num_mel_bins]
    """
    B, T, S = waveform_chunks.shape
    flat = waveform_chunks.reshape(B * T, S).detach().cpu().numpy().tolist()
    feats = feature_extractor(
        flat,
        sampling_rate=sample_rate,
        return_tensors="pt"
    )["input_values"]  # [B*T, max_length, num_mel_bins]
    return feats.view(B, T, feats.shape[1], feats.shape[2])


# ----------------------------
# train / eval
# ----------------------------

def run_epoch(model, loader, optimizer, feature_extractor, device, train=True,
              lambda_sig=0.05, grad_accum_steps=4, use_amp=True):
    model.train(train)
    total_loss = 0.0
    total_pred = 0.0
    total_sig = 0.0

    if train:
        optimizer.zero_grad(set_to_none=True)

    amp_device = "cuda" if device.type == "cuda" else "cpu"
    scaler = None  # keeping this simple

    for step, batch in enumerate(loader):
        wave = batch["waveform_chunks"]  # [B,T,S]
        feats = astify_batch(feature_extractor, wave)  # CPU tensor
        feats = feats.to(device)

        ctx = torch.autocast(device_type=amp_device, dtype=torch.float16, enabled=(use_amp and device.type == "cuda"))
        with ctx:
            out = model(feats)
            loss = out["pred_loss"] + lambda_sig * out["sigreg_loss"]
            loss_scaled = loss / grad_accum_steps

        if train:
            loss_scaled.backward()
            if (step + 1) % grad_accum_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_pred += out["pred_loss"].item()
        total_sig += out["sigreg_loss"].item()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "pred_loss": total_pred / n,
        "sigreg_loss": total_sig / n,
    }


# ----------------------------
# checkpoint
# ----------------------------

def save_ckpt(path, model, optimizer, epoch, args):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def load_ckpt(path, model, optimizer=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


# ----------------------------
# main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--encoder_name", type=str, default="MIT/ast-finetuned-audioset-10-10-0.4593")
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--chunks_per_clip", type=int, default=6)
    parser.add_argument("--chunk_seconds", type=float, default=0.96)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--lambda_sig", type=float, default=0.05)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="sajepa_ast_mamba.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = AutoFeatureExtractor.from_pretrained(args.encoder_name)
    train_ds = ChunkedAudioDataset(
        args.train_dir,
        sample_rate=args.sample_rate,
        chunks_per_clip=args.chunks_per_clip,
        chunk_seconds=args.chunk_seconds,
    )
    val_ds = ChunkedAudioDataset(
        args.val_dir,
        sample_rate=args.sample_rate,
        chunks_per_clip=args.chunks_per_clip,
        chunk_seconds=args.chunk_seconds,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )

    model = SAJEPA(
        encoder_name=args.encoder_name,
        latent_dim=args.latent_dim,
        freeze_encoder=True,
        unfreeze_final_norm=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total, trainable = count_params(model)
    enc_total, enc_train = count_params(model.encoder)
    pred_total, pred_train = count_params(model.predictor)
    proj_total, proj_train = count_params(model.projector)

    print(f"Model total params:       {total:,}")
    print(f"Model trainable params:   {trainable:,}")
    print(f"Encoder total/trainable:  {enc_total:,} / {enc_train:,}")
    print(f"Projector total/trainable:{proj_total:,} / {proj_train:,}")
    print(f"Predictor total/trainable:{pred_total:,} / {pred_train:,}")

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, optimizer, feature_extractor, device,
            train=True, lambda_sig=args.lambda_sig,
            grad_accum_steps=args.grad_accum_steps
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model, val_loader, optimizer, feature_extractor, device,
                train=False, lambda_sig=args.lambda_sig,
                grad_accum_steps=args.grad_accum_steps
            )

        print(
            f"epoch {epoch} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"(pred {train_metrics['pred_loss']:.4f}, sig {train_metrics['sigreg_loss']:.4f}) | "
            f"val loss {val_metrics['loss']:.4f} "
            f"(pred {val_metrics['pred_loss']:.4f}, sig {val_metrics['sigreg_loss']:.4f})"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            save_ckpt(args.save_path, model, optimizer, epoch, args)
            print(f"saved best checkpoint to {args.save_path}")

    print("done")


if __name__ == "__main__":
    main()
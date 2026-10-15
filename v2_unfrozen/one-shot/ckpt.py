# Checkpoint I/O that respects the repo's disk rule: no single file over ~9 MB.
# A checkpoint is ONE prefix with N shard files: <prefix>.part000.pt, .part001.pt, ...
# Each shard stores a list of (key, fp16 tensor) pairs; load_ckpt merges them in order.
import glob
from pathlib import Path

import torch

MAX_SHARD_BYTES = 9 * 2**20


def save_ckpt(model, prefix, extra: dict | None = None):
    """Save model.state_dict() as fp16 shards under `prefix`. `extra` (small metadata dict)
    is written to <prefix>.meta.pt (checked small: scalars/strings/dicts only)."""
    sd = {k: v.detach().to(torch.float16).cpu() for k, v in model.state_dict().items()}
    prefix = Path(prefix)
    shards, cur, cur_bytes = [], [], 0
    for k, v in sd.items():
        nbytes = v.numel() * 2
        if cur and cur_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append((k, v))
        cur_bytes += nbytes
    if cur:
        shards.append(cur)
    paths = []
    for i, sh in enumerate(shards):
        p = prefix.with_name(prefix.name + f".part{i:03d}.pt")
        torch.save(sh, p)
        paths.append(p)
    if extra is not None:
        torch.save(extra, prefix.with_name(prefix.name + ".meta.pt"))
    total_mb = sum(p.stat().st_size for p in paths) / 2**20
    sizes = ", ".join(f"{p.stat().st_size/2**20:.1f}MB" for p in paths)
    print(f"ckpt: {len(paths)} shard(s), {total_mb:.1f}MB total ({sizes})")
    return paths


def load_ckpt(model, prefix, strict=True):
    """Merge shards written by save_ckpt and load into `model` (fp16 -> fp32 upcast)."""
    prefix = Path(prefix)
    parts = sorted(glob.glob(str(prefix.with_name(prefix.name + ".part*.pt"))))
    if not parts:
        raise FileNotFoundError(f"no shards matching {prefix}.part*.pt")
    sd = {}
    for p in parts:
        for k, v in torch.load(p, map_location="cpu"):
            sd[k] = v.float()
    model.load_state_dict(sd, strict=strict)
    return model

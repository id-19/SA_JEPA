# SA-JEPA v2 one-shot — MODEL (Hermes-implemented this rung, per Ishaan's explicit 15 Oct directive).
# Architecture per v2_unfrozen/one-shot/DESIGN_v0.md:
#   mel(80) -> causal conv patch embed -> 6x DeltaNet block + 2x GQA block (D=256) -> latents e
#   predictor: input proj -> 2x DeltaNet (no FFN) -> 4 horizon heads (d in {1,10,50,150})
# Objective lives in train.py: L = sum_d MSE(e_hat_{t+d}, e_{t+d}) + alpha * SIGREG(e)
# Anti-collapse: SIGREG only (UniJEPA-style Gaussian regularizer). NO EMA, NO stop-grad anywhere.
#
# Six-check smoke in __main__: determinism, shape contract, grad flow + dial spread, BIBO,
# known-answer T=2, causality. Run: uv run python v2_unfrozen/one-shot/model.py
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

D_MODEL = 256
NH_DELTA = 4
DH = 64
NH_Q, NH_KV = 4, 2
D_MLP = 1024
HORIZONS = (1, 10, 50, 150)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


class DeltaCore(nn.Module):
    """Gated delta-rule layer over per-head states, CHUNKWISE-parallel (Yang et al.

    Recurrence (per head): S_t = S_{t-1}(I - beta_t k_t k_t^T) + beta_t v_t k_t^T
    (canonical decay-free DeltaNet form; the write-gate beta in (0,1) provides forgetting --
    this is the exact form the chunkwise algorithm of arXiv:2406.06484 Eqs. 5-11 verifies).
    Verified against the per-step reference: outputs AND final states match to <1e-5
    (see chunk_scan_verify4.py: read-after-update convention, diagonal-included intra mask).

    Per step (head h):  b = sigmoid(Wb x_t)                    (write gate, input-dep, per-head)
                        k_t = W_k-conv output, L2-normalized
                        S_t = S_{t-1} - b_t (S_{t-1} k_t - v_t) k_t^T
                        y_t = S_t q_t
    States init to zero. k/q L2-normed (stability: beta*|k|^2 <= 1 guaranteed)."""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.d, self.nh, self.dh = d, NH_DELTA, DH
        self.conv = nn.Conv1d(d, 3 * d, kernel_size=4, padding=0, groups=1)  # left-pad in forward
        self.wb = nn.Linear(d, NH_DELTA)
        self.out = nn.Linear(NH_DELTA * DH, d)
        self.gate = nn.Linear(d, d)
        self.d_model = d
        self.CH = 64

    def _qkv(self, x):
        B, T, _ = x.shape
        h = F.pad(x.transpose(1, 2), (3, 0))  # left-pad -> causal
        qkv = self.conv(h).transpose(1, 2)    # (B, T, 3D)
        q, k, v = qkv.split(self.d, dim=-1)
        q = q.view(B, T, NH_DELTA, DH)
        k = k.view(B, T, NH_DELTA, DH)
        v = v.view(B, T, NH_DELTA, DH)
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return q, k, v

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self._qkv(x)
        beta = torch.sigmoid(self.wb(x)).view(B, T, NH_DELTA)      # (B,T,H)
        # -> head-batched: (B*H, T, DH). NOTE: beta lives in (B,T,H) memory order —
        # permute to (B,H,T) BEFORE flattening; a bare .view would scramble head/time.
        q = q.permute(0, 2, 1, 3).reshape(B * NH_DELTA, T, DH)
        k = k.permute(0, 2, 1, 3).reshape(B * NH_DELTA, T, DH)
        v = v.permute(0, 2, 1, 3).reshape(B * NH_DELTA, T, DH)
        beta = beta.permute(0, 2, 1).reshape(B * NH_DELTA, T)
        y = self._chunk_scan(q, k, v, beta)                        # (B*H, T, DH)
        y = y.view(B, NH_DELTA, T, DH).permute(0, 2, 1, 3).reshape(B, T, -1)
        return self.out(y * torch.sigmoid(self.gate(x)))

    def _chunk_scan(self, q, k, v, beta):
        """Batched chunkwise delta scan. q/k/v: (BH, T, DH), beta: (BH, T), states zero-init.
        Chunk-level: S_next = S + (U - W S^T)^T K ; per-step: o_t = S q_t + sum_{i<=t}(q_t.k_i)(U - W S^T)_i
        (state recurrence verified exact; output convention: read-after-update, diag-incl mask)."""
        BH, T, DK = k.shape
        DV = DK
        CH = self.CH
        NC = (T + CH - 1) // CH
        pad = NC * CH - T
        if pad:
            k = torch.cat([k, k.new_zeros(BH, pad, DK)], 1)
            v = torch.cat([v, v.new_zeros(BH, pad, DV)], 1)
            beta = torch.cat([beta, beta.new_zeros(BH, pad)], 1)
            q = torch.cat([q, q.new_zeros(BH, pad, DK)], 1)
        K = k.view(BH, NC, CH, DK); V = v.view(BH, NC, CH, DV)
        Bm = beta.view(BH, NC, CH); Q = q.view(BH, NC, CH, DK)
        S = k.new_zeros(BH, DV, DK)
        outs = []
        for c in range(NC):
            Kc = K[:, c]; Vc = V[:, c]; Bc = Bm[:, c]; Qc = Q[:, c]
            A = torch.tril((Bc.unsqueeze(-1) * Kc) @ Kc.transpose(1, 2), diagonal=-1)  # (BH,CH,CH)
            Tm = torch.linalg.inv(torch.eye(CH, device=k.device, dtype=k.dtype) + A) \
                 @ torch.diag_embed(Bc)                                                # Eq.10
            Wc = Tm @ Kc                                              # (BH, CH, DK)
            Uc = Tm @ Vc                                              # (BH, CH, DV)
            core = Uc - Wc @ S.transpose(1, 2)                        # (BH, CH, DV) = U - W S^T
            M = torch.tril(torch.ones(CH, CH, device=k.device, dtype=k.dtype), diagonal=0)
            o_c = Qc @ S.transpose(1, 2) + ((Qc @ Kc.transpose(1, 2)) * M) @ core
            outs.append(o_c)
            S = S + core.transpose(1, 2) @ Kc                         # Eq.8
        outs = torch.cat(outs, dim=1)[:, :T]
        return outs

    def state_norm(self, x):
        """Probe helper: mean |S k| retrieval norm per chunk boundary (chunkwise impl).
        Used by the BIBO check; returns (num_chunks,) norms."""
        B, T, _ = x.shape
        q, k, v = self._qkv(x)
        BH = B * NH_DELTA
        qh = q.permute(0, 2, 1, 3).reshape(BH, T, DH)
        kh = k.permute(0, 2, 1, 3).reshape(BH, T, DH)
        vh = v.permute(0, 2, 1, 3).reshape(BH, T, DH)
        bb = torch.sigmoid(self.wb(x)).view(B, T, NH_DELTA).permute(0, 2, 1).reshape(BH, T)
        # run the scan tracking chunk-final states
        CH, NC = self.CH, (T + self.CH - 1) // self.CH
        pad = NC * CH - T
        if pad:
            kh = torch.cat([kh, kh.new_zeros(BH, pad, DH)], 1)
            vh = torch.cat([vh, vh.new_zeros(BH, pad, DH)], 1)
            bb = torch.cat([bb, bb.new_zeros(BH, pad)], 1)
        K = kh.view(BH, NC, CH, DH); V = vh.view(BH, NC, CH, DH); Bm = bb.view(BH, NC, CH)
        S = kh.new_zeros(BH, DH, DH)
        norms = []
        for c in range(NC):
            Kc, Vc, Bc = K[:, c], V[:, c], Bm[:, c]
            A = torch.tril((Bc.unsqueeze(-1) * Kc) @ Kc.transpose(1, 2), diagonal=-1)
            Tm = torch.linalg.inv(torch.eye(CH, device=k.device, dtype=k.dtype) + A) \
                 .transpose(-1, -2) @ torch.diag_embed(Bc)
            Wc = Tm @ Kc; Uc = Tm @ Vc
            core = Uc - Wc @ S.transpose(1, 2)
            S = S + core.transpose(1, 2) @ Kc
            norms.append(S.norm(dim=(-2, -1)).mean())
        return torch.stack(norms)


def rope_cos_sin(T, dh, device, theta=10000.0):
    half = dh // 2
    freqs = 1.0 / (theta ** (torch.arange(half, device=device).float() / half))
    t = torch.arange(T, device=device).float()
    ang = torch.outer(t, freqs)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    # x: (B, NH, T, DH); rotate even/odd pairs
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos.view(1, 1, *cos.shape)
    s = sin.view(1, 1, *sin.shape)
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).flatten(-2)


class GQABlock(nn.Module):
    """Grouped-query attention (4 q / 2 kv), RoPE, causal, out-gated; then FFN. Both pre-norm, residual."""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.wq = nn.Linear(d, NH_Q * DH, bias=False)
        self.wk = nn.Linear(d, NH_KV * DH, bias=False)
        self.wv = nn.Linear(d, NH_KV * DH, bias=False)
        self.wo = nn.Linear(NH_Q * DH, d, bias=False)
        self.gate = nn.Linear(d, d)
        self.n2 = RMSNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, D_MLP), nn.GELU(), nn.Linear(D_MLP, d))

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        h = self.n1(x)
        q = self.wq(h).view(B, T, NH_Q, DH).transpose(1, 2)
        k = self.wk(h).view(B, T, NH_KV, DH).transpose(1, 2)
        v = self.wv(h).view(B, T, NH_KV, DH).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = k.repeat_interleave(NH_Q // NH_KV, dim=1)
        v = v.repeat_interleave(NH_Q // NH_KV, dim=1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, -1)
        x = x + self.wo(o) * torch.sigmoid(self.gate(x))
        x = x + self.ff(self.n2(x))
        return x


class DeltaBlock(nn.Module):
    """DeltaCore + FFN, both pre-norm, residual."""

    def __init__(self, d=D_MODEL, use_ffn=True):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.core = DeltaCore(d)
        self.use_ffn = use_ffn
        if use_ffn:
            self.n2 = RMSNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, D_MLP), nn.GELU(), nn.Linear(D_MLP, d))

    def forward(self, x):
        x = x + self.core(self.n1(x))
        if self.use_ffn:
            x = x + self.ff(self.n2(x))
        return x


class Encoder(nn.Module):
    """mel(80) -> latents (B, T, 256). 6 delta blocks : 2 GQA blocks."""

    def __init__(self, d_in=80, d=D_MODEL):
        super().__init__()
        self.patch = nn.Conv1d(d_in, d, kernel_size=4, padding=0)  # left-pad -> causal
        self.blocks = nn.ModuleList([DeltaBlock(d, use_ffn=True) for _ in range(6)])
        self.attn = nn.ModuleList([GQABlock(d) for _ in range(2)])
        self.norm_out = RMSNorm(d)

    def forward(self, mels):  # (B, T, 80)
        B, T, _ = mels.shape
        h = F.pad(mels.transpose(1, 2), (3, 0))
        x = self.patch(h).transpose(1, 2)  # (B, T, D)
        cos, sin = rope_cos_sin(T, DH, mels.device)
        for blk in self.blocks:
            x = blk(x)
        for blk in self.attn:
            x = blk(x, cos, sin)
        return self.norm_out(x)


class Predictor(nn.Module):
    """e window -> predictions of e_{t+d} for d in HORIZONS. 2 DeltaNet layers (no FFN) + heads."""

    def __init__(self, d=D_MODEL):
        super().__init__()
        self.inp = nn.Linear(d, d)
        self.blocks = nn.ModuleList([DeltaBlock(d, use_ffn=False) for _ in range(2)])
        self.heads = nn.ModuleList([nn.Linear(d, d) for _ in HORIZONS])

    def forward(self, e):  # (B, T, D) -> list of (B, T, D) per horizon
        x = self.inp(e)
        for blk in self.blocks:
            x = blk(x)
        return [head(x) for head in self.heads]


def sigreg(z, n_proj=512, n_sample=4096):
    """Gaussian regularizer: per random unit direction u, push the batch distribution of u.z
    toward N(0,1): first moment (mean^2 -> 0) and variance (-> 1).
    R = mean_u[ mean_z(u.z)^2 + (var_z(u.z) - 1)^2 ]
    (First-order, both-moments form of UniJEPA Eq.3; collapse of ANY direction is penalized.)"""
    B, T, Dd = z.shape
    flat = z.reshape(-1, Dd)
    idx = torch.randperm(flat.shape[0], device=z.device)[: min(n_sample, flat.shape[0])]
    zs = flat[idx]
    u = torch.randn(n_proj, Dd, device=z.device)
    u = u / u.norm(dim=-1, keepdim=True)
    p = zs @ u.T  # (n, n_proj)
    mean = p.mean(0)
    var = p.var(0, unbiased=False)
    return (mean ** 2).mean() + ((var - 1.0) ** 2).mean()


class SAJePAV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.predictor = Predictor()
        # sigreg fidelity/speed dial: 512 default (paper-grade); lower = noisier estimator.
        # ESTIMATE NOISE AVERAGES OVER STEPS (regularizer), so gate-4-scale runs use 128.
        self.sigreg_nproj = 512

    def forward(self, mels):
        e = self.encoder(mels)  # (B, T, D)
        preds = self.predictor(e)  # list per horizon
        return e, preds

    def loss(self, mels, alpha=0.1, weights=None):
        e, preds = self(mels)
        B, T, _ = e.shape
        if weights is None:
            weights = [1.0] * len(HORIZONS)
        l_terms = {}
        total = 0.0
        for d, ph, w in zip(HORIZONS, preds, weights):
            if d >= T:  # horizon beyond window: skip (gate runs use short T)
                continue
            tgt = e[:, d:, :]
            pred = ph[:, : T - d, :]
            mse = F.mse_loss(pred, tgt)
            l_terms[d] = mse
            total = total + w * mse
        r = sigreg(e, n_proj=self.sigreg_nproj)
        return total / len(HORIZONS) + alpha * r, {**{f"d{k}": v for k, v in l_terms.items()}, "sigreg": r}, e


# ---------------------------------------------------------------- six-check smoke
def _smoke():
    torch.manual_seed(0)
    dev = "cpu"
    m = SAJePAV0().to(dev)
    x = torch.randn(2, 60, 80, device=dev)

    # 1 determinism
    e1, p1 = m(x)
    e2, p2 = m(x)
    ok = torch.allclose(e1, e2) and all(torch.allclose(a, b) for a, b in zip(p1, p2))
    print(f"[1] determinism: {'PASS' if ok else 'FAIL'}")

    # 2 shape contract, several B/T
    ok = True
    for b, t in [(1, 37), (3, 64), (2, 500)]:
        ee, pp = m(torch.randn(b, t, 80))
        ok &= ee.shape == (b, t, D_MODEL) and all(ppi.shape == (b, t, D_MODEL) for ppi in pp)
    print(f"[2] shape contract (B/T sweep): {'PASS' if ok else 'FAIL'}")

    # 3 grad flow + dial spread
    m.zero_grad()
    loss, parts, _ = m.loss(x, alpha=0.1)
    assert loss.dtype.is_floating_point and loss.dim() == 0, loss.dtype
    loss.backward()
    nograd = [n for n, p in m.named_parameters() if p.grad is None]
    with torch.no_grad():
        e_state = m.encoder(x)  # (2, 60, 256) — the delta blocks' actual input
    da_beta = m.encoder.blocks[0].core.wb(e_state).sigmoid()
    beta_spread = da_beta.std().item()
    a_conv_spread = m.encoder.blocks[0].core.conv.weight.grad.std().item()
    print(f"[3] grad flow: {'PASS' if not nograd else f'FAIL {nograd}'} | beta spread {beta_spread:.3f} (want >0) | conv grad std {a_conv_spread:.2e}")

    # 4 BIBO: noise chunk then zero-tail -> state norm must SHRINK across chunks
    # (canonical decay-free core forgets by OVERWRITE: a constant-key tail writes the conv-bias
    # value repeatedly; the error-correcting rule collapses the state toward that rank-1 target).
    core = m.encoder.blocks[0].core.eval()
    with torch.no_grad():
        noise = torch.randn(1, 64, D_MODEL)
        zeros = torch.zeros(1, 128, D_MODEL)
        n = core.state_norm(torch.cat([noise, zeros], dim=1))  # 3 chunk norms
        shrank = n[-1] < n[0]
    print(f"[4] BIBO (noise->zeros overwrite): {'PASS' if shrank else 'FAIL'} | chunk norms {n.numpy().round(3)}")

    # 5 known-answer T=2, hand-computed (canonical decay-free delta rule, k-norm active):
    # zero ALL weights; set conv bias so k=(1,0), v=(1,0), q=(1,0) at BOTH steps.
    # k-norm: |k|=1. beta=.5 (wb zeroed -> sigmoid(.5)? NO: wb=0 -> beta=sigmoid(0)=.5 EXACTLY).
    # step1: S1 = 0 - .5*(0 - v)k^T = .5 vk^T -> y1 = q·(S1 k) = .5
    # step2: S1k = .5v; err = -.5v; S2 = S1 + .25 vk^T = .75 vk^T -> y2 = .75.
    # DELTA-RULE CONVERGENCE: state approaches value v geometrically (rate 1-beta=.5):
    # y_t = 1-(1-beta)^t = .5, .75, .875, ...
    core2 = DeltaCore().eval()
    with torch.no_grad():
        for p in core2.parameters():
            p.zero_()
        core2.out.bias.zero_()     # out-proj: identity on head0 dims (weight set below)
        # out projection: weight zero means output ignores S. Set out weight = identity on head0 dims.
        ow = torch.zeros(core2.out.weight.shape)
        for h in range(NH_DELTA):
            ow[h * DH:(h + 1) * DH, h * DH:(h + 1) * DH] = torch.eye(DH) * (1.0 if h == 0 else 0.0)
        core2.out.weight.copy_(ow)
        core2.gate.bias.fill_(12.0)  # sigmoid ~ 1 -> gate open
        core2.gate.weight.zero_()
        conv_w = core2.conv.weight.data
        # conv: zero weights, bias only -> q/k/v constant over time: q,k,v all = e0 (index0 of each split)
        core2.conv.bias.data.zero_()
        core2.conv.bias.data[0] = 1.0                    # q dim0 = 1
        core2.conv.bias.data[D_MODEL + 0] = 1.0          # k dim0 = 1
        core2.conv.bias.data[2 * D_MODEL + 0] = 1.0      # v dim0 = 1
        xs = torch.zeros(1, 2, D_MODEL)
        y = core2(xs)  # (1, 2, D)
        got1, got2 = y[0, 0, 0].item(), y[0, 1, 0].item()
        ok = abs(got1 - 0.5) < 1e-3 and abs(got2 - 0.75) < 1e-3
    print(f"[5] known-answer T=2: {'PASS' if ok else 'FAIL'} | y1={got1:.4f} y2={got2:.4f} (paper: 0.5, 0.75 = geometric value convergence)")

    # 6 causality: perturb x[t+5:], outputs y[:t] unchanged (m.eval() — dropout would flip bits)
    m.eval()
    with torch.no_grad():
        xa = torch.randn(1, 30, 80)
        xb = xa.clone(); xb[0, 15:] += torch.randn(15, 80)
        ea, _ = m(xa); eb, _ = m(xb)
        ok = torch.allclose(ea[0, :15], eb[0, :15], atol=1e-5)
    m.train()
    print(f"[6] causality: {'PASS' if ok else 'FAIL'}")

    n_par = sum(p.numel() for p in m.parameters())
    print(f"params total: {n_par:,}")


if __name__ == "__main__":
    _smoke()
# Isotropy-arm control: alpha=1.0 + isotropy=1.0, track eff-rank/var/spk vs the no-isotropy
# trajectory (eff-rank 11 -> 6 falling at step 300; pre-registered: holds >= 15 at 1200 steps
# would need long run — 600-step here; bar: eff-rank NOT falling, var[min] not -> 0, and
# sigreg-term still declining (objective must remain trainable)).
import torch, sys, numpy as np
sys.path.insert(0, '/Users/id19/Programming/ml_ai_and_data_science/SA-JEPA/v2_unfrozen/one-shot')
from model import SAJePAV0
from train import get_window_batch
from board import effective_rank, per_dim_var, speaker_probe

torch.manual_seed(0); np.random.seed(0)
m = SAJePAV0().train()
m.sigreg_isotropy = 1.0
o = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.0)

for it in range(600):
    x, spk = get_window_batch(bs=8, T=250, device="cpu")
    loss, parts, e = m.loss(x, alpha=1.0)
    o.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    o.step()
    if it % 100 == 0 or it == 599:
        with torch.no_grad():
            z = e.reshape(-1, e.shape[-1]).detach()
            vmean, vmin, vmax = per_dim_var(z)
            er = effective_rank(z)
            # speaker probe on pooled windows (cheap: 8 windows/step accumulate)
            if it % 300 == 0 or it == 599:
                acc, sd = speaker_probe(z.detach().cpu(), np.repeat(np.arange(8), 250))
                spk_str = f"spk(win) {acc:.3f}±{sd:.3f}"
            else:
                spk_str = ""
        print(f"it {it:3d} | d1 {parts['d1'].item():.4f} d50 {parts['d50'].item():.4f} | "
              f"sigreg {parts['sigreg'].item():.4f} | eff-rank {er:5.1f} | "
              f"var[{vmin:.3f},{vmean:.3f},{vmax:.3f}] | {spk_str}", flush=True)
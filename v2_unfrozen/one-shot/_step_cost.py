# Realistic step-cost probe under CURRENT machine load, with the v0 objective (recon_w>0.3).
# Tests: thread count (torch.set_num_threads) x sigreg n_proj. Prints ms/step for each.
import sys
import time

import torch

sys.path.insert(0, '/Users/id19/Programming/ml_ai_and_data_science/SA-JEPA/v2_unfrozen/one-shot')
from model import SAJePAV0

BS, T = 8, 250

for nthreads in (4, 2):
    torch.set_num_threads(nthreads)
    for nproj in (512, 128):
        torch.manual_seed(0)
        m = SAJePAV0().train()
        m.sigreg_nproj = nproj
        x = torch.randn(BS, T, 80)
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)

        def one():
            loss, _, _ = m.loss(x, alpha=1.0, recon_w=0.3)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()

        one()  # warmup
        t0 = time.time()
        for _ in range(3):
            one()
        ms = (time.time() - t0) / 3 * 1000
        print(f'threads={nthreads} nproj={nproj}: {ms:6.0f} ms/step -> '
              f'2000 steps = {ms*2000/60000:.0f} min', flush=True)

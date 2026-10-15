# Does the log-depth scan remove what a device actually waits on? Two measurements:
#  (a) CPU op-event counts (forward+backward) for sequential vs parallel scan at T=100/250/500.
#      This is the device-independent launch-count proxy: every python-level step in a loop is one
#      more op the device must be handed and waited on.
#  (b) wall-clock per frame, bs=4, seq vs parallel, on cpu and mps, across T.
# Run AFTER training (CPU contention would corrupt both columns). Memory kept small: bs=4.
import sys
import time

import torch

sys.path.insert(0, '/Users/id19/Programming/ml_ai_and_data_science/SA-JEPA/v2_unfrozen/one-shot')
from model import SAJePAV0

BS = 4
print('T     scan      ops(fwd+bwd)  top-3 op kinds')
prof_rows = []
for T in (100, 250, 500):
    for par in (True, False):
        torch.manual_seed(0)
        m = SAJePAV0().train()
        for b in m.encoder.blocks:
            b.core.parallel_scan = par
        x = torch.randn(BS, T, 80)
        loss, _, _ = m.loss(x, alpha=1.0)
        loss.backward()  # warmup
        m.zero_grad()
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            loss, _, _ = m.loss(x, alpha=1.0)
            loss.backward()
        ev = prof.key_averages()
        total = sum(e.count for e in ev)
        tops = sorted(ev, key=lambda e: -e.count)[:3]
        print(f'{T:<5} {"parallel" if par else "seq     "}  {total:>10}   '
              + ', '.join(f'{e.key[:26]}x{e.count}' for e in tops), flush=True)
        prof_rows.append((T, par, total))

print('\nwall-clock per frame (ms), bs=4, fwd+bwd:')
print('T     device  seq      parallel  speedup')
for T in (100, 250, 500):
    for dev in ('cpu', 'mps') if torch.backends.mps.is_available() else ('cpu',):
        out = {}
        for par in (False, True):
            torch.manual_seed(0)
            m = SAJePAV0().to(dev).train()
            for b in m.encoder.blocks:
                b.core.parallel_scan = par
            x = torch.randn(BS, T, 80, device=dev)
            opt = torch.optim.AdamW(m.parameters(), lr=3e-4)

            def one():
                loss, _, _ = m.loss(x, alpha=1.0)
                opt.zero_grad()
                loss.backward()
                opt.step()
            one()
            if dev == 'mps':
                torch.mps.synchronize()
            t0 = time.time()
            for _ in range(3):
                one()
            if dev == 'mps':
                torch.mps.synchronize()
            out[par] = (time.time() - t0) / 3 / (BS * T) * 1000
        print(f'{T:<5} {dev:<7} {out[False]:7.4f}  {out[True]:7.4f}   '
              f'{out[False]/out[True]:5.2f}x', flush=True)

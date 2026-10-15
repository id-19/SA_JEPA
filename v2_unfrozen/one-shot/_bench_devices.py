# Device/precision throughput sweep for the SA-JEPA v2 core (unbuffered, per-config guards).
# Reports ms/step, kframe/s, peak RSS and swap delta so a thrashing config is visible, not hidden.
import subprocess
import sys
import time

import torch

sys.path.insert(0, '/Users/id19/Programming/ml_ai_and_data_science/SA-JEPA/v2_unfrozen/one-shot')

from model import SAJePAV0


def swap_used_mb():
    out = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout
    # "total = 2048.00M  used = 123.45M  free = ..."
    return float(out.split('used =')[1].split('M')[0].strip())


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def bench(dev, bs, amp, steps=6, T=250):
    m = SAJePAV0().to(dev).train()
    x = torch.randn(bs, T, 80, device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)

    def one():
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp):
            loss, _, _ = m.loss(x, alpha=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    for _ in range(2):
        one()
    if dev == 'mps':
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(steps):
        one()
    if dev == 'mps':
        torch.mps.synchronize()
    return (time.time() - t0) / steps * 1000


CONFIGS = [('cpu', 8, False), ('cpu', 16, False), ('cpu', 32, False),
           ('mps', 8, False), ('mps', 16, False), ('mps', 32, False),
           ('mps', 8, True), ('mps', 16, True), ('mps', 32, True)]

if __name__ == '__main__':
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for dev, bs, amp in CONFIGS:
        tag = f'{dev} bs={bs} {"bf16" if amp else "fp32"}'
        if only and only not in tag:
            continue
        sw0 = swap_used_mb()
        t0 = time.time()
        try:
            ms = bench(dev, bs, amp)
            kfps = bs * 250 / (ms / 1000) / 1000
            print(f'{tag}: {ms:7.0f} ms/step | {kfps:5.1f} kframe/s | '
                  f'peak RSS {peak_rss_gb():.2f} GB | swap {sw0:.0f}->{swap_used_mb():.0f} MB | '
                  f'wall {time.time()-t0:.0f}s', flush=True)
        except Exception as e:
            print(f'{tag}: FAILED {type(e).__name__}: {str(e)[:120]}', flush=True)

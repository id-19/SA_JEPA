# One-config throughput probe: _bench_one.py <dev> <bs> <amp 0|1> [steps] [T]
# Prints a single line with ms/step, kframe/s, peak RSS, swap delta. Run under `timeout`
# so a pathological config (MPS under memory pressure) cannot stall a sweep.
import subprocess
import sys
import time

import torch

sys.path.insert(0, '/Users/id19/Programming/ml_ai_and_data_science/SA-JEPA/v2_unfrozen/one-shot')
from model import SAJePAV0


def swap_used_mb():
    out = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout
    return float(out.split('used =')[1].split('M')[0].strip())


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30


def main():
    dev, bs, amp = sys.argv[1], int(sys.argv[2]), sys.argv[3] == '1'
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    T = int(sys.argv[5]) if len(sys.argv) > 5 else 250
    torch.manual_seed(0)
    sw0 = swap_used_mb()
    m = SAJePAV0().to(dev).train()
    x = torch.randn(bs, T, 80, device=dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)

    def one():
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=amp):
            loss, _, _ = m.loss(x, alpha=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()

    t0 = time.time()
    for _ in range(2):
        one()
    if dev == 'mps':
        torch.mps.synchronize()
    t1 = time.time()
    for _ in range(steps):
        one()
    if dev == 'mps':
        torch.mps.synchronize()
    t2 = time.time()
    ms = (t2 - t1) / steps * 1000
    kfps = bs * T / (ms / 1000) / 1000
    print(f'{dev:3s} bs={bs:2d} {"bf16" if amp else "fp32"}: {ms:7.0f} ms/step | '
          f'{kfps:5.1f} kframe/s | warmup {t1-t0:5.1f}s | peak RSS {peak_rss_gb():.2f} GB | '
          f'swap {sw0:.0f}->{swap_used_mb():.0f} MB | wall {t2-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()

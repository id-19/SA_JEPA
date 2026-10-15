# What the 15:21 training log SHOULD have printed in its spk(win) column.
#
# That run started ~14:53 (ended 15:21 after 1696 s) and the probe fix landed at 14:58, i.e. five
# minutes INTO the run -- so the process had the old `_quick_probe` loaded, the one that silently
# returned 0.0 whenever the stratified split raised. The "spk(win) 0.000+-0.000" at step 2000 is
# therefore a fabricated zero: it measured nothing. The checkpoint itself is unaffected (the probe
# is eval-only, never touches gradients or model selection), which the evidence run confirms by
# scoring 0.991 on these same weights.
#
# This script re-measures exactly that quantity, with the fixed code, on the same checkpoint -- the
# pool construction below mirrors train_v0.main() lines 95-107 verbatim.
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ckpt import load_ckpt  # noqa: E402
from model import SAJePAV0  # noqa: E402
from train_v0 import _quick_probe, split_utts  # noqa: E402  (main() is __main__-guarded)


def main():
    torch.set_grad_enabled(False)
    T, seed = 250, 0
    _, probe_rows = split_utts(seed=seed)

    by_spk: dict[str, list] = {}
    for r in probe_rows:
        by_spk.setdefault(r["utt"].split("-")[0], []).append(r)
    pool_rows = []
    for spk in sorted(by_spk):
        if len(by_spk[spk]) >= 4:
            pool_rows.extend(by_spk[spk][:4])
    xp = torch.zeros(len(pool_rows), T, 80)
    for bi, r in enumerate(pool_rows):
        xp[bi] = torch.load(HERE / "mel_cache" / f"{r['utt']}.pt").float()[:T]
    yp = [r["utt"].split("-")[0] for r in pool_rows]

    print(f"probe pool: {len(yp)} windows over {len(set(yp))} speakers "
          f"(min {min(yp.count(s) for s in set(yp))} windows/speaker)")
    print(f"blind chance level: {1/len(set(yp)):.3f}  <- the published '0.000' was BELOW chance, "
          f"which is the tell that it measured nothing")

    m = SAJePAV0().eval()
    load_ckpt(m, str(HERE / "runs" / "v0"))
    acc, sd = _quick_probe(m, xp, yp, "cpu")
    print(f"\nFIXED probe on runs/v0  ->  spk(win) {acc:.3f}+-{sd:.3f}")
    print("(what the training log should have printed; the log's pre-fix value was 0.000+-0.000)")


if __name__ == "__main__":
    main()

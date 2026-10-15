# Gate 1 rerun with the chunkwise core: overfit one real batch, alpha=0.01, CPU (147ms/step).
import torch, sys, time
sys.path.insert(0, '/Users/id19/Programming/ml_ai_and_data_science/SA-JEPA/v2_unfrozen/one-shot')
from model import SAJePAV0
torch.manual_seed(0)

torch.backends.mkldnn.enabled = True
m = SAJePAV0().train()
x = torch.randn(4, 120, 80)
if torch.mps.is_available():  # keep CPU per bench result (MPS slower at small batch)
    pass
opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
t0 = time.time()
for i in range(120):
    loss, parts, _ = m.loss(x, alpha=0.01)
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if i % 25 == 0 or i == 119:
        ds = {k: f"{v.item():.4f}" for k, v in parts.items()}
        print(f"it {i:3d}: loss {loss.item():.5f} | {ds} | {time.time()-t0:.0f}s elapsed")
verdict = "PASS" if loss.item() < 0.05 else "FAIL"
print(f"GATE-1 verdict: {verdict} (final loss {loss.item():.5f}, bar 0.05)")
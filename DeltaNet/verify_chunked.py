import torch

# from delta_net import UnifiedDeltaNet
from parallel_gdn import ParallelGatedDeltaNet


def check(decay, d_qk=8, chunk_sizes=(1, 2, 4, 8, 16, 32, 64)):
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        # model = UnifiedDeltaNet(d_input=7, d_output=3, d_qk=d_qk, decay=decay)
        model = ParallelGatedDeltaNet(D_input=7, D_output=3, D_qk=d_qk, D_v = d_qk)
        for (B, T) in ((1, 32), (1, 64), (2, 31), (3, 33)):
            torch.manual_seed(seed)
            x = torch.randn(B, T, 7)
            with torch.no_grad():
                ref = model.legacy_out(x)
            for c in chunk_sizes:
                got = model(x, chunk_size=c)
                # print(ref, got, sep="\n\n")
                assert torch.allclose(ref, got, atol=1e-5, rtol=1e-4), \
                    (f"decay={decay} seed={seed} B={B} T={T} c={c}: "
                     f"maxerr={(ref - got).abs().max().item():.3g}")
    print(f"OK decay={decay}")


if __name__ == "__main__":
    for dec in (None, "scalar", "vector"):
        check(dec)
    print("ALL PASS")

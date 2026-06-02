import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from 
import torch


def test_single_file():
    x = preprocess_audio_full(
        "Data/Mini-librispeech-asr/dev-clean-2/84/121550/84-121550-0000.flac"
    )
    assert x is not None
    assert x.shape == torch.Size([469, 80])
    assert abs(x.mean().item()) < 1e-5
    assert abs(x.std().item() - 1.0) < 1e-5


def test_dataloader():
    loader = get_dataloader(
        "Data/Mini-librispeech-asr/dev-clean-2",
        batch_size=8,
        shuffle=True,
    )
    for i, (xs, paths) in enumerate(loader):
        assert xs.shape == torch.Size([8, 469, 80])
        assert len(paths) == 8
        if i >= 2:
            break


if __name__ == "__main__":
    print("=== Single file test ===")
    test_single_file()
    print("Passed")

    print("\n=== DataLoader test ===")
    test_dataloader()
    print("Passed")

    print("\nAll tests passed!")

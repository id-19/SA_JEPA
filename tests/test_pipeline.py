import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from v0_model_try.data_pipeline_v0 import preprocess_audio_full, get_dataloader
import torch


def test_single_file():
    x = preprocess_audio_full(
        "Data/Mini-librispeech-asr/dev-clean-2/84/121550/84-121550-0000.flac"
    )
    assert x is not None
    assert x.shape == torch.Size([469, 80])
    assert abs(x.mean().item()) < 1e-5, "Mean is not near zero"
    assert abs(x.std().item() - 1.0) < 1e-5, "STD is not close enough to 1"


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
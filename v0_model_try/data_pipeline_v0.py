import torch
import torch.nn.functional as F
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

SAMPLE_RATE = 24000
CLIP_LENGTH_SECONDS = 5
CLIP_SAMPLES = SAMPLE_RATE * CLIP_LENGTH_SECONDS

N_FFT = 1024
HOP_LENGTH = 256
N_BINS = 80
EPS = 1e-6
BATCH_SIZE = 8

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_BINS,
)


def load_simple(path):
    wav, sr = torchaudio.load(path)
    return wav, sr


def cast_uniform(x, sample_sr, target_sr=SAMPLE_RATE):
    if x.dim() == 1:
        x = x.unsqueeze(0)

    if sample_sr != target_sr:
        x = torchaudio.functional.resample(x, sample_sr, target_sr)

    if x.shape[0] > 1:
        x = x.mean(dim=0, keepdim=True)
    return x


def trim_silence(waveform, threshold=0.01):
    x = waveform.abs().squeeze(0)
    mask = x > threshold

    if not mask.any():
        return None

    start = mask.nonzero()[0].item()
    end = mask.nonzero()[-1].item() + 1
    return waveform[:, start:end]


def cut_pad_clip(audio_wav, target_samples=CLIP_SAMPLES):
    num_samples = audio_wav.shape[1]
    if num_samples > target_samples:
        audio_wav = trim_silence(audio_wav)
        if audio_wav is None:
            return None

    # Ensure exact length: pad or truncate as needed
    num_samples = audio_wav.shape[1]
    if num_samples < target_samples:
        pad_amount = target_samples - num_samples
        audio_wav = F.pad(audio_wav, (0, pad_amount))
    elif num_samples > target_samples:
        audio_wav = audio_wav[:, :target_samples]
    return audio_wav


def get_log_mel(wav):
    mel = mel_transform(wav)
    log_mel = torch.log(mel + EPS)
    return log_mel


def get_normalised_log_mel(log_mel):
    mean = log_mel.mean()
    std = log_mel.std()
    x = (log_mel - mean) / (std + EPS)
    x = x.squeeze(0)
    x = x.transpose(0, 1)
    return x


def preprocess_audio_full(file_path: str):
    wav, sr = load_simple(file_path)
    wav = cast_uniform(wav, sr)
    wav = cut_pad_clip(wav)
    if wav is None:
        return None
    log_mel = get_log_mel(wav)
    return get_normalised_log_mel(log_mel)


def find_audio_files(root_dir):
    root = Path(root_dir)
    files = [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    return sorted(files)


class AudioDataset(Dataset):
    def __init__(self, root_dir):
        self.files = find_audio_files(root_dir)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = str(self.files[idx])
        x = preprocess_audio_full(path)
        return x, path


def collate_fn(batch):
    batch = [(x, p) for x, p in batch if x is not None]
    if not batch:
        return torch.empty(0), []
    xs, paths = zip(*batch)
    xs = torch.stack(xs, dim=0)
    return xs, list(paths)


def get_dataloader(root_dir, batch_size=BATCH_SIZE, shuffle=True, num_workers=0):
    dataset = AudioDataset(root_dir)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

if __name__ == "__main__":
    root_dir = "Data"# Parent directory containing audio files, ../__file__
    dataloader = get_dataloader(root_dir)
    for batch in dataloader:
        xs, paths = batch
        print(xs.shape)
        break
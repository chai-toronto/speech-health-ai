import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os

from src.data.util import exact_div, pad_or_truncate

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token

class AudioDataset(Dataset):
    def __init__(self, file_paths,
                 labels=None,
                 transform=None,
                 target_transform=None,
                 max_length=None,
                 sample_rate=8000):
        """
        Args:
            file_paths (list): List of audio file paths.
            labels (list, optional): Corresponding labels.
            transform (callable, optional): Transform applied on audio.
            target_transform (callable, optional): Transform applied on label.
            max_length (int, optional): Maximum length of audio in seconds. If None, use full audio.
            sample_rate (int): Target sample rate for resampling.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.sample_rate = sample_rate
        self.max_length = max_length

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path, normalize=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if self.max_length is not None:
            waveform = pad_or_truncate(waveform, self.max_length * self.sample_rate)

        if self.transform:
            waveform = self.transform(waveform)

        label = None
        if self.labels is not None:
            label = self.labels[idx]
            if self.target_transform:
                label = self.target_transform(label)

        return waveform, label


def create_dataloader(file_paths, labels=None, batch_size=32, shuffle=True, transform=None,
                      target_transform=None, sample_rate=16000, num_workers=4, pin_memory=True):
    """
    Utility to create DataLoader.
    """
    dataset = AudioDataset(
        file_paths=file_paths,
        labels=labels,
        transform=transform,
        target_transform=target_transform,
        sample_rate=sample_rate
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

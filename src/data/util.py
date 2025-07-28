import glob
import json
import os
from typing import Union, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio


def exact_div(x, y):
    assert x % y == 0
    return x // y

def get_sample_rate(path):
    """ Tested for FLAC.
    :param path: path to the audio file
    :return: sample rate of the audio file (int)
    """
    metadata = sf.info(path)
    return metadata.samplerate

def gather_flac_json_pairs(root_dir):
    """ Gather FLAC and JSON pairs in a directory. For LibriSpeech
    :return: list of (flac_path, json_path) tuples.
    """
    flac_files = glob.glob(os.path.join(root_dir, '**', '*.flac'), recursive=True)
    pairs = []

    for flac_path in flac_files:
        json_path = os.path.splitext(flac_path)[0] + '.json'
        if os.path.exists(json_path):
            pairs.append((flac_path, json_path))
        else:
            print(f"Warning: No JSON companion for {flac_path}")

    return pairs

def whisper_norm(log_spec: torch.Tensor, range = 8) -> torch.Tensor:
    """ Normalize log-spectrogram as used in Whisper,
    by mapping the range of magnitudes to [max(log_spec) - range, max(log_spec)], and then scaling to [0, 1]

    :param log_spec:
    :param range: minimum magnitude from max(log_spec) to be considered. Default 8.0.
    :return:
    """
    log_spec = torch.maximum(log_spec, log_spec.max() - range)
    range = range / 2
    log_spec = (log_spec + range) / range
    return log_spec

def z_score_norm(audio: torch.Tensor, mean = None, std = None) -> torch.Tensor:
    """ Normalize audio (log-spectrogram/waveform) by subtracting mean and dividing by std.
    Use the mean and std of the sample if mean and std are not provided,
    (as part of a dataset/batch perhaps).
    """
    if not (mean and std):
        mean = audio.mean()
        std = audio.std()
    return (audio - mean) / std

def batch_mean_std(audio: torch.Tensor):
    """ Get mean and std of a batch of log-spectrograms.
    :param audio: Tensor of shape (batch_size, channels, time)
    :return: Mean and std.
    """
    # Flatten across batch, channels, and time
    flattened = audio.flatten()
    mean = flattened.mean()
    std = flattened.std(unbiased=False)  # to match NumPy's ddof=0
    return mean, std

def min_max_norm(audio: torch.Tensor):
    return (audio - audio.min()) / (audio.max() - audio.min())

def log_compress(mel_spec):
    return torch.clamp(mel_spec, min=1e-10).log10()

def pad_or_truncate(waveform, max_length):
    length = waveform.shape[-1]
    if length > max_length:
        return waveform[..., :max_length]
    elif length < max_length:
        pad_amount = max_length - length
        return torch.nn.functional.pad(waveform, (0, pad_amount))
    return waveform

def load_jsons_as_dict(json_paths):
    """ key is filename, value is the dict of each json file."""
    result = {}
    for path in json_paths:
        filename = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'r') as f:
            data = json.load(f)  # each file contains one dict
        result[filename] = data
    return result

def save_json(file: dict, path: str):
    with open(path, 'w') as f:
        json.dump(file, f)


import pandas as pd
import io

def read_audioset_csv(filepath):
    # What a degenerate csv
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the last comment line as header
    header_line = ''
    data_start_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith('#'):
            header_line = line.lstrip('# ').strip()
            data_start_idx = idx + 1

    # Rebuild the CSV content: header + data
    data_str = header_line + '\n' + ''.join(lines[data_start_idx:])

    df = pd.read_csv(
        io.StringIO(data_str),
        quotechar='"',
        skipinitialspace=True
    )
    # Split the positive_labels column into a list of strings
    df['positive_labels'] = df['positive_labels'].str.split(',')

    return df

def resample(waveform, sr, target_sr):
    return torchaudio.transforms.Resample(sr, target_sr)(waveform)

def init_stat():
    return {
        'Silence trimmed': [],
        'Overlapped trimmed': [],
        'Not English': [],
        'English trimmed': [],
        'Trimmed': [],
    }
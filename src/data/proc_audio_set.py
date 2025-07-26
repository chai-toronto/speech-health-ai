import argparse
import glob
import json
import os
import random

import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm
import shutil

from src.data.lid import trim_non_target_language, infer_lid_distribution
from src.data.util import read_audioset_csv
from src.data.vad import find_overlapped_and_trim, find_speech_and_trim


# AUDIO_PATH = '/Volumes/LKieuData/AudioSet/data/extracted'
# OUTPUT_PATH = '/Volumes/LKieuData/AudioSet/data/processed'
drive = "/media/larry/55b84e27-f8a5-4823-9b85-197fc1c6075f/"
AUDIO_PATH = drive + '/AudioSet/extracted'
OUTPUT_PATH =  drive + '/AudioSet/processed_final'
os.makedirs(OUTPUT_PATH, exist_ok=True)

METADATA = [drive + x for x in
                ['/AudioSet/data/balanced_train_segments.csv',
                '/AudioSet/data/eval_segments.csv',
                '/AudioSet/data/unbalanced_train_segments.csv'
                ]
            ]

SAMPLE_RATE = 16_000
# For AudioSet, Speech and its subcategories
ALLOWED_TAGS = ["/m/09x0r", "/m/05zppz", "/m/02zsn", "/m/0ytgt", "/m/01h8n0", "/m/02qldy", "/m/0261r1", "/m/0brhx"]

def convert_audio_set(audio_path):
    flac_files = glob.glob(os.path.join(audio_path, '**', '*.flac'), recursive=True)
    return flac_files

def has_allowed_tag(labels):
    return any(tag in labels for tag in ALLOWED_TAGS)

def init_stat():
    return {
        'overlapped_trimmed': [],
        'num_overlapped': 0,
        'nonspeech_trimmed': [],
        'num_nonspeech': 0,
        'num_not_english': 0,
        'not_english': [],
        'eng_trimmed': [],
        'num_eng_trimmed': 0,
    }


def get_existing_files(output_path):
    return [f for f in os.listdir(output_path) if f.endswith('.flac')]


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', default=AUDIO_PATH)
    parser.add_argument('--output_path', default=OUTPUT_PATH)
    parser.add_argument('--metadata', default=METADATA, nargs='*')
    parser.add_argument('--verbose', default=True, action='store_true')

    args = parser.parse_args()

    existing_files = get_existing_files(args.output_path)

    paths = convert_audio_set(args.audio_path)

    # Filter for speech tags
    df = pd.concat([read_audioset_csv(file) for file in args.metadata])

    ytids_to_paths = {
        os.path.splitext(os.path.basename(p))[0]: p for p in paths
    }

    ytids = list(ytids_to_paths.keys())
    df_filtered = df[df['YTID'].isin(ytids)].copy()
    df_filtered['has_allowed_tag'] = df_filtered['positive_labels'].apply(has_allowed_tag)
    df_result = df_filtered[df_filtered['has_allowed_tag']].copy()

    filtered_ytids = df_result['YTID'].tolist()
    filtered_paths = [ytids_to_paths[ytid] for ytid in filtered_ytids if ytid in ytids_to_paths]

    # Uniformly sample 100 files
    filtered_paths = paths

    stats = init_stat()
    for path in tqdm(filtered_paths):
        # stat and save
        filename = os.path.basename(path)
        if filename in existing_files:
            existing_files.remove(filename)
            continue

        waveform, sr = torchaudio.load(path) # shape (num_channel, T)

        # Monochannel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # trimming silence
        silenced_trim = find_speech_and_trim(waveform, SAMPLE_RATE)
        if silenced_trim.shape[1] <= SAMPLE_RATE * 1:
            continue
        if waveform.shape[1] - silenced_trim.shape[1] >= SAMPLE_RATE:
            stats['nonspeech_trimmed'].append(path.split('/')[-1])
            stats['num_nonspeech'] += 1

        overlapped_trim = find_overlapped_and_trim(silenced_trim, SAMPLE_RATE)
        if overlapped_trim.shape[1] <= SAMPLE_RATE * 0.5:
            continue
        if silenced_trim.shape[1] - overlapped_trim.shape[1] >= SAMPLE_RATE:
            stats['overlapped_trimmed'].append(path.split('/')[-1])
            stats['num_overlapped'] += 1

        # Check English
        waveform_1d = waveform.squeeze() # Shape (T)
        prob_eng = infer_lid_distribution(waveform_1d, SAMPLE_RATE)['eng']
        if prob_eng < 0.4:
            stats['num_not_english'] += 1
            stats['not_english'].append(path.split('/')[-1])
            continue

        elif prob_eng < 0.9:
            waveform_1d, _ = trim_non_target_language(waveform_1d, SAMPLE_RATE, 'eng')
            if waveform_1d.shape[0] <= SAMPLE_RATE * 1:
                continue
            stats['num_eng_trimmed'] += 1
            stats['eng_trimmed'].append(path.split('/')[-1])
            waveform = waveform_1d.unsqueeze(0)

        output_path = os.path.join(args.output_path, filename)
        torchaudio.save(output_path, overlapped_trim, SAMPLE_RATE)

    with open(os.path.join(args.output_path, 'stat.json'), 'w') as f:
        json.dump(stats, f, indent=3)
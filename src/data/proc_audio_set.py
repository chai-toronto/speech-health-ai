import argparse
import glob
import os

import numpy as np
import pandas as pd
import torchaudio
from tqdm import tqdm


from src.data.lid import trim_non_target_language, infer_lid_distribution
from src.data.util import read_audioset_csv
from src.data.vad import find_overlapped_and_trim, find_speech_and_trim


# AUDIO_PATH = '/Volumes/LKieuData/AudioSet/data/extracted'
# OUTPUT_PATH = '/Volumes/LKieuData/AudioSet/data/processed'
AUDIO_PATH = '/Users/lkieu/Desktop/Audioset/audio'
OUTPUT_PATH = '/Users/lkieu/Desktop/Audioset/processed_final'
os.makedirs(OUTPUT_PATH, exist_ok=True)

METADATA = ['/Volumes/LKieuData/AudioSet/data/balanced_train_segments.csv',
            '/Volumes/LKieuData/AudioSet/data/eval_segments.csv',
            '/Volumes/LKieuData/AudioSet/data/unbalanced_train_segments.csv']

SAMPLE_RATE = 16_000
# For AudioSet, Speech and its subcategories
ALLOWED_TAGS = ["/m/09x0r", "/m/05zppz", "/m/02zsn", "/m/0ytgt", "/m/01h8n0", "/m/02qldy", "/m/0261r1", "/m/0brhx"]

def convert_audio_set(audio_path):
    flac_files = glob.glob(os.path.join(audio_path, '**', '*.flac'), recursive=True)
    return flac_files

def has_allowed_tag(labels):
    return any(tag in labels for tag in ALLOWED_TAGS)

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', default=AUDIO_PATH)
    parser.add_argument('--output_path', default=OUTPUT_PATH)
    parser.add_argument('--metadata', default=METADATA, nargs='*')

    args = parser.parse_args()

    # On the same subfolder of audio_path
    if args.output_path == '':
        args.output_path = os.path.dirname(args.audio_path)

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

    english = 0
    lang_trimmed = 0
    trimmed = []
    for path in tqdm(filtered_paths):
        waveform, sr = torchaudio.load(path) # shape (num_channel, T)

        # Monochannel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Check English
        waveform_1d = waveform.squeeze() # Shape (T)
        prob_eng = infer_lid_distribution(waveform_1d, SAMPLE_RATE)['eng']
        if prob_eng < 0.4:
            continue

        elif prob_eng < 0.9:
            waveform_1d, _ = trim_non_target_language(waveform_1d, SAMPLE_RATE, 'eng')
            if waveform_1d.shape[0] <= SAMPLE_RATE * 1:
                continue
            lang_trimmed += 1
            waveform = waveform_1d.unsqueeze(0)

        english += 1

        # trimming silence
        waveform = find_speech_and_trim(waveform, SAMPLE_RATE)
        if waveform.shape[1] <= SAMPLE_RATE * 1:
            continue
        waveform = find_overlapped_and_trim(waveform, SAMPLE_RATE)
        if waveform.shape[1] <= SAMPLE_RATE * 0.5:
            continue
        # stat and save
        trimmed.append(((SAMPLE_RATE * 10) - waveform.shape[-1]) / SAMPLE_RATE)
        filename = os.path.basename(path)
        output_path = os.path.join(args.output_path, filename)
        torchaudio.save(output_path, waveform, SAMPLE_RATE)

    print(f"Average trimmed duration: {sum(trimmed) / len(trimmed):.2f} seconds")
    print(f'Standard Deviation of trimmed: {np.std(trimmed):.2f} seconds')
    print(f"Language trimmed: {lang_trimmed} / {len(filtered_paths)}")
    print(f"English speechs found, resampled, mono and trimmed: {english} / {len(filtered_paths)}")

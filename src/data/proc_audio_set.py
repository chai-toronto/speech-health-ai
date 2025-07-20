import argparse
import glob
import os

import torchaudio
from datasets import Dataset
from tqdm import tqdm

from src.data import lid
from src.data.util import read_audioset_csv
from src.data.vad import vad_and_trim

AUDIO_PATH = '/Users/lkieu/Desktop/LibriSpeech'
OUTPUT_PATH = '/Users/lkieu/Desktop/LibriSpeech/processed'
METADATA = '/Users/lkieu/Desktop/Audioset/balanced_train_segments.csv'

SAMPLE_RATE = 16000
# For AudioSet, Speech and its subcategories
ALLOWED_TAGS = ["/m/09x0r", "/m/05zppz", "/m/02zsn", "/m/0ytgt", "/m/01h8n0", "/m/02qldy", "/m/0261r1", "/m/0brhx"]

def convert_audio_set(audio_path):
    flac_files = glob.glob(os.path.join(audio_path, '**', '*.flac'), recursive=True)
    return flac_files

def has_allowed_tag(labels):
    return any(tag in labels for tag in ALLOWED_TAGS)

if 'main' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', default=AUDIO_PATH)
    parser.add_argument('--output_path', default=OUTPUT_PATH)
    parser.add_argument('--metadata')

    args = parser.parse_args()

    # On the same subfolder of audio_path
    if args.output_path == '':
        args.output_path = os.path.dirname(args.audio_path)

    paths = convert_audio_set(args.audio_path)

    # Filter for speech tags
    df = read_audioset_csv(args.metadata)
    dataset = Dataset.from_pandas(df)

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
    trimmed = []
    for path in tqdm(filtered_paths):
        waveform, sr = torchaudio.load(path)

        # Monochannel
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)
        # Check if it has dominantly English speech
        if not lid.verify_english(waveform.squeeze()):
            continue
        english += 1

        # trimming silence
        waveform = vad_and_trim(waveform, SAMPLE_RATE)
        trimmed.append(((SAMPLE_RATE * 10) - waveform.shape[-1]) / SAMPLE_RATE)
        filename = os.path.basename(path)
        output_path = os.path.join(args.output_path, filename)
        torchaudio.save(output_path, waveform, SAMPLE_RATE)
    print(f"Average trimmed duration: {sum(trimmed) / len(trimmed):.2f} seconds")
    print(f"English speechs found, resampled, mono and trimmed: {english}")

import argparse
import glob
import os
from typing import Union

import pandas as pd
import torchaudio
import torchaudio.functional as F
from datasets import Dataset, Audio
from tqdm import tqdm

from src.data import lid
from src.data.util import load_jsons_as_dict, save_json, read_audioset_csv
from src.data.vad import vad_and_trim


AUDIO_PATH = '/Users/lkieu/Desktop/LibriSpeech'
OUTPUT_PATH = '/Users/lkieu/Desktop/LibriSpeech/processed'
DATASET_NAME = 'LibriSpeech'
METADATA = '/Users/lkieu/Desktop/Audioset/balanced_train_segments.csv'

SAMPLE_RATE = 16000
F_MIN = 60
F_MAX = 8000

# For AudioSet, Speech and its subcategories
ALLOWED_TAGS = ["/m/09x0r", "/m/05zppz", "/m/02zsn", "/m/0ytgt", "/m/01h8n0", "/m/02qldy", "/m/0261r1", "/m/0brhx"]


def convert_libri(audio_path):
    flac_files = glob.glob(os.path.join(audio_path, '**', '*.flac'), recursive=True)
    audios = []
    meta = []

    for flac_path in flac_files:
        json_path = os.path.splitext(flac_path)[0] + '.json'
        if os.path.exists(json_path):
            audios.append(flac_path)
            meta.append(json_path)
        else:
            print(f"Warning: No JSON companion for {flac_path}")

    return audios, load_jsons_as_dict(meta)



def convert_librispeech(audio_path):
    flac_files = glob.glob(os.path.join(audio_path, '**', '*.flac'), recursive=True)
    return flac_files



if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', default=AUDIO_PATH)
    parser.add_argument('--dataset_name', default=DATASET_NAME)
    parser.add_argument('--output_path', default=OUTPUT_PATH)
    parser.add_argument('--metadata')

    args = parser.parse_args()
    # On the same subfolder of audio_path
    if args.output_path == '':
        args.output_path = os.path.dirname(args.audio_path)

    if args.dataset_name.lower() == 'librilight':
        paths, metadata = convert_libri(args.audio_path)

        for path in paths:
            waveform, sr = torchaudio.load(path)
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
                waveform = resampler(waveform)

            filename = os.path.basename(path)
            torchaudio.save(os.path.join(args.output_path, filename), waveform, SAMPLE_RATE)
        print(os.getcwd())
        save_json(metadata, os.path.join(args.output_path, 'metadata.json'))
    # ........ Add more dataset here ..................
    # All dataset after conversion should have a list of paths to the audio file, and a dict of metadata where the key
    # is the filename and its metadata as the value.


    if args.dataset_name.lower() == 'librispeech':
        paths = convert_librispeech(args.audio_path)

        english = 0
        trimmed = []
        for path in tqdm(paths):
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














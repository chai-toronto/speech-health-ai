import argparse
import glob
import json
import os

import torchaudio
from tqdm import tqdm

from src.data.util import load_jsons_as_dict, init_stat
from src.data.vad import find_speech_and_trim

AUDIO_PATH = '/Users/lkieu/Desktop/LibriSpeech'
OUTPUT_PATH = '/Users/lkieu/Desktop/LibriSpeech/processed'
SAMPLE_RATE = 16000

def convert_libri_light(audio_path):
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


if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', default=AUDIO_PATH)
    parser.add_argument('--output_path', default=OUTPUT_PATH)
    args = parser.parse_args()

    paths, metadata = convert_libri_light(args.audio_path)
    stats = init_stat()
    for path in tqdm(paths):
        waveform, sr = torchaudio.load(path)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        silence_trim = find_speech_and_trim(waveform, SAMPLE_RATE)
        if silence_trim.shape[1] <= SAMPLE_RATE * 1:
            continue
        if waveform.shape[1] - silence_trim.shape[1] >= SAMPLE_RATE:
            stats['Silence trimmed'].append(os.path.basename(path))

        stats['Trimmed'].append(1 - (silence_trim.shape[1] / waveform.shape[1]))
        # TODO: split into short duration
    with open(os.path.join(args.output_path, 'stat.json'), 'w') as f:
        json.dump(stats, f, indent=3)


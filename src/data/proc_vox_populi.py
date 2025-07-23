import glob
import os
import random
from pathlib import Path

import torchaudio
from tqdm import tqdm

from src.data.lid import infer_lid_distribution, trim_non_target_language
from src.data.vad import find_speech_and_trim, find_overlapped_and_trim

random.seed(1234)
import argparse

drive = '/media/larry/LKieuData'
AUDIO_PATH = drive + '/VoxPopuli/unlabelled_data2/en/'
OUTPUT_PATH = drive + '/VoxPopuli/exploratory'
os.makedirs(OUTPUT_PATH, exist_ok=True)
SAMPLE_RATE = 16_000


def convert_vox_populi(audio_path):
    ogg_files = glob.glob(os.path.join(audio_path, '**', '*.ogg'), recursive=True)
    return ogg_files

if '__main__' == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', default=AUDIO_PATH)
    parser.add_argument('--output_path', default=OUTPUT_PATH)
    args = parser.parse_args()

    paths = convert_vox_populi(args.audio_path)

    #pick out 100 random
    random_paths = random.sample(paths, 100)

    stats = {}
    stats['Silence trimmed'] = []
    stats['Overlapped trimmed'] = []
    stats['Not English'] = []

    for path in tqdm(random_paths):
        filename = os.path.basename(path)
        waveform, sr = torchaudio.load(path)

        # Monochannel
        if waveform.shape[0] > 1:
            stats['Non-mono'] = 1 if 'Non-mono' not in stats else stats['Non-mono'] + 1
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != SAMPLE_RATE:
            stats['Resampled'] = 1 if 'Resampled' not in stats else stats['Resampled'] + 1
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = resampler(waveform)

        # Check if in English
        waveform_1d = waveform.squeeze()
        prob_eng = infer_lid_distribution(waveform_1d, SAMPLE_RATE)['eng']
        if prob_eng < 0.4:
            stats['Not English'].append(filename)
            continue

        elif prob_eng < 0.9:
            waveform_1d, _ = trim_non_target_language(waveform_1d, SAMPLE_RATE, 'eng')
            if waveform_1d.shape[0] <= SAMPLE_RATE * 1:
                continue
            waveform = waveform_1d.unsqueeze(0)

        silence_trimmed = find_speech_and_trim(waveform, SAMPLE_RATE)
        stats['Silence trimmed'].append((waveform.shape[1] - silence_trimmed.shape[1])/waveform.shape[1])

        overlapped_trimmed = find_overlapped_and_trim(silence_trimmed, SAMPLE_RATE)
        stats['Overlapped trimmed'].append((waveform.shape[1] - overlapped_trimmed.shape[1] + silence_trimmed.shape[1])
                                           /waveform.shape[1])

        filename = os.path.basename(path)
        output_path = os.path.join(args.output_path, filename)
        torchaudio.save(output_path, waveform, SAMPLE_RATE)

    stats['Silence_avg'] = sum(stats['Silence trimmed']) / len(stats['Silence trimmed'])
    stats['Overlapped_avg'] = sum(stats['Overlapped trimmed']) / len(stats['Overlapped trimmed'])
    stats['Not_English_num'] = len(stats['Not English'])
    stats['Non-mono'] = stats['Non-mono'] if 'Non-mono' in stats else 0
    stats['Resampled'] = stats['Resampled'] if 'Resampled' in stats else 0

    # Save to json
    with open(os.path.join(args.output_path, 'stats_librivox_pass1.json'), 'w') as f:
        f.write(str(stats))

    print('done')










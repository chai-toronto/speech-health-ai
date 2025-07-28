import argparse
import json
import os

import torch
from datasets import load_dataset, Audio
import numpy as np
from tqdm import tqdm
import torchaudio

from src.data.lid import infer_lid_distribution, trim_non_target_language
from src.data.util import init_stat
from src.data.vad import find_speech_and_trim, find_overlapped_and_trim

INPUT_DIR = ''
OUTPUT_DIR = '/Users/lkieu/Desktop/gigaspeech/processed100/'
os.makedirs(OUTPUT_DIR, exist_ok=True)
SAMPLE_RATE = 16_000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default=INPUT_DIR)
    parser.add_argument('--output_path', default=OUTPUT_DIR)
    args = parser.parse_args()

    if args.input_path == '':
        gs = load_dataset("speechcolab/gigaspeech", "xl", split="train", streaming=True).shuffle()
    else:
        gs = load_dataset("speechcolab/gigaspeech", "xl", split="train", cache_dir=args.input_dir, streaming=True).shuffle()

    stats = init_stat()

    num_samples = np.inf
    progress = tqdm(total=num_samples)
    i = 0
    cur = next(iter(gs))
    while i < num_samples and cur is not None:
        audio = Audio(cur['audio']).sampling_rate
        waveform, sr = audio['array'], audio['sampling_rate']
        filename = os.path.basename(audio['path'])

        waveform = torch.from_numpy(waveform.astype(np.float32)).unsqueeze(0)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != SAMPLE_RATE:
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

        silence_trim = find_speech_and_trim(waveform, SAMPLE_RATE)
        if silence_trim.shape[1] <= SAMPLE_RATE * 1:
            continue
        if waveform.shape[1] - silence_trim.shape[1] >= SAMPLE_RATE:
            stats['Silence trimmed'].append(filename)

        overlap_trim = find_overlapped_and_trim(silence_trim, SAMPLE_RATE)

        if overlap_trim.shape[1] <= SAMPLE_RATE * 1:
            continue
        if silence_trim.shape[1] - overlap_trim.shape[1] >= SAMPLE_RATE:
            stats['Overlapped trimmed'].append(filename)
        #TODO: silence first, then lang trimmed. Document all the hyperparam.
        stats['Trimmed'].append(1 - (overlap_trim.shape[1] / waveform.shape[1]))
        output_path = os.path.join(args.output_path, filename)
        torchaudio.save(output_path, overlap_trim, SAMPLE_RATE)

        i += 1
        progress.update(1)
        cur = next(iter(gs))

    # Save to json
    with open(os.path.join(args.output_path, 'stat.json'), 'w') as f:
        json.dump(stats, f, indent=3)

    print('done')
    progress.close()





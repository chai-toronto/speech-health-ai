import argparse
import glob
import json
import os
import traceback

import torchaudio
import ray
from tqdm import tqdm

SAMPLE_RATE = 16_000

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

def merge_stats(stats):
    final_stat = init_stat()
    for stat in stats:
        for k in stat:
            if isinstance(stat[k], list):
                final_stat[k].extend(stat[k])
            else:
                final_stat[k] += stat[k]
    return final_stat

def should_skip(path, output_root):
    return os.path.exists(os.path.join(output_root, os.path.basename(path)))

@ray.remote
def process_file_cpu(path, output_root):
    import torch
    from src.data.vad import VAD
    from src.data.lid import LID
    device = torch.device("cpu")
    try:
        vad = VAD(device)
        lid = LID(device)
        waveform, sr = torchaudio.load(path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

        waveform = vad.find_speech_and_trim(waveform, SAMPLE_RATE)
        if waveform.shape[1] <= SAMPLE_RATE * 1:
            return None

        waveform = vad.find_overlapped_and_trim(waveform, SAMPLE_RATE)
        if waveform.shape[1] <= SAMPLE_RATE * 0.5:
            return None

        prob_eng = lid.infer_lid_distribution(waveform.squeeze(), SAMPLE_RATE)['eng']
        if prob_eng < 0.4:
            return None
        elif prob_eng < 0.9:
            trimmed, _ = lid.trim_non_target_language(waveform.squeeze(), SAMPLE_RATE, 'eng')
            if trimmed.shape[0] <= SAMPLE_RATE * 1:
                return None
            waveform = trimmed.unsqueeze(0)

        out_path = os.path.join(output_root, os.path.basename(path))
        torchaudio.save(out_path, waveform, SAMPLE_RATE)
    except Exception as e:
        print(f"[CPU] Error processing {path}: {e}")
        # traceback.print_exc()
        return None

    return

@ray.remote(num_gpus=0.25)
def process_file_gpu(path, output_root):
    import torch
    from src.data.vad import VAD
    from src.data.lid import LID

    device = torch.device("cuda")

    try:
        vad = VAD(device)
        lid = LID(device)

        waveform, sr = torchaudio.load(path)
        waveform = waveform.to(device)

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            resample = torchaudio.transforms.Resample(sr, SAMPLE_RATE).to(device)
            waveform = resample(waveform)

        waveform = vad.find_speech_and_trim(waveform, SAMPLE_RATE)
        if waveform.shape[1] <= SAMPLE_RATE * 1:
            return None

        waveform = vad.find_overlapped_and_trim(waveform, SAMPLE_RATE)
        if waveform.shape[1] <= SAMPLE_RATE:
            return None

        prob_eng = lid.infer_lid_distribution(waveform.squeeze(), SAMPLE_RATE)['eng']
        if prob_eng < 0.4:
            return None
        elif prob_eng < 0.9:
            trimmed, _ = lid.trim_non_target_language(waveform.squeeze(), SAMPLE_RATE, 'eng')
            if trimmed.shape[0] <= SAMPLE_RATE * 1:
                return None
            waveform = trimmed.unsqueeze(0)

        out_path = os.path.join(output_root, os.path.basename(path))
        waveform = waveform.to(torch.device('cpu'))
        torchaudio.save(out_path, waveform, SAMPLE_RATE)

    except Exception as e:
        print(f"[GPU] Error: {e}")
        traceback.print_exc()
        return None

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path',
                        default="/media/larry/55b84e27-f8a5-4823-9b85-197fc1c6075f//AudioSet/extracted")
    parser.add_argument('--output_path',
                        default="/media/larry/55b84e27-f8a5-4823-9b85-197fc1c6075f//AudioSet/processed_final")
    parser.add_argument('--num_gpu_tasks', type=int, default=4)
    parser.add_argument('--ray_cpus', type=int, default=26)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    ray.init(num_cpus=args.ray_cpus)

    all_paths = glob.glob(os.path.join(args.audio_path, '**', '*.flac'), recursive=True)
    to_process = [p for p in all_paths if not should_skip(p, args.output_path)]

    num_process = len(to_process)
    print(f"Found {len(all_paths)} total files, {num_process} to process (resume enabled)")

    gpu_tasks = [process_file_gpu.remote(p, args.output_path) for p in to_process[:num_process//3]]
    cpu_tasks = [process_file_cpu.remote(p, args.output_path) for p in to_process[num_process//3:]]
    futures = gpu_tasks + cpu_tasks

    pbar = tqdm(total=len(futures))
    pending = set(futures)

    while pending:
        done, pending = ray.wait(list(pending), num_returns=1)
        for obj_ref in done:
            pbar.update(1)

    pbar.close()


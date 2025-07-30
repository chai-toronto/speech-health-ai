import argparse
import os
import traceback
from datetime import datetime
import csv
import math
from pathlib import Path
import multiprocessing as mp
import torchaudio
import time

# VAD Configuration
BETWEEN_SEGMENT = 0.15
HYPER_PARAMETERS = {
    "min_duration_on": 0.0,
    "min_duration_off": 0.0
}


class VAD:
    def __init__(self, device):
        from pyannote.audio.pipelines import VoiceActivityDetection, OverlappedSpeechDetection
        from pyannote.audio import Model

        self.model = Model.from_pretrained("pyannote/segmentation-3.0")
        self.model.to(device)
        self.speech_pipeline = VoiceActivityDetection(segmentation=self.model)
        self.speech_pipeline.instantiate(HYPER_PARAMETERS)

        self.overlapped_pipeline = OverlappedSpeechDetection(segmentation=self.model)
        self.overlapped_pipeline.instantiate(HYPER_PARAMETERS)

    def find_speech(self, waveform, sample_rate):
        vad = self.speech_pipeline({'waveform': waveform, 'sample_rate': sample_rate})
        return str(vad)

    def find_overlapped(self, waveform, sample_rate):
        overlapped = self.overlapped_pipeline({'waveform': waveform, 'sample_rate': sample_rate})
        return str(overlapped)


def worker_process(worker_id, file_paths, output_dir, device='cpu', gpu_id=0):
    """
    Worker process that handles its own share of files and writes its own CSV.
    """
    print(f"[Worker-{worker_id}] Starting with {len(file_paths)} files on {device}")

    # Initialize VAD model in this process
    try:
        if device == 'cuda':
            import torch
            torch.cuda.set_device(gpu_id)

        vad = VAD(device)
        print(f"[Worker-{worker_id}] VAD model initialized")
    except Exception as e:
        print(f"[Worker-{worker_id}] Failed to initialize VAD: {e}")
        return

    # Set up output CSV for this worker
    output_csv = Path(output_dir) / f"worker_{worker_id}_annotations.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    successful_count = 0

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow(['file_path', 'speech_annotation', 'overlap_annotation'])

        for file_path in file_paths:
            try:
                # Load audio
                waveform, sample_rate = torchaudio.load(file_path)

                if device == 'cuda':
                    waveform = waveform.to('cuda', non_blocking=True)

                # Get annotations
                speech_annotation = vad.find_speech(waveform, sample_rate)
                overlap_annotation = vad.find_overlapped(waveform, sample_rate)

                # Write to CSV immediately
                writer.writerow([
                    str(file_path),
                    speech_annotation,
                    overlap_annotation
                ])

                successful_count += 1

            except Exception as e:
                print(f"[Worker-{worker_id}] Error processing {file_path}: {e}")
                # Continue with next file

            finally:
                # Clean up GPU memory
                if 'waveform' in locals():
                    del waveform
                if device == 'cuda':
                    import torch
                    torch.cuda.empty_cache()

            processed_count += 1

            # Progress reporting
            if processed_count % 50 == 0:
                print(f"[Worker-{worker_id}] Processed {processed_count}/{len(file_paths)} files")

    print(f"[Worker-{worker_id}] Completed! Processed {processed_count} files, {successful_count} successful")
    print(f"[Worker-{worker_id}] Results saved to: {output_csv}")


def split_work(file_paths, num_workers):
    """Split file paths evenly among workers"""
    chunk_size = math.ceil(len(file_paths) / num_workers)
    chunks = []

    for i in range(num_workers):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(file_paths))
        chunk = file_paths[start_idx:end_idx]
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)

    return chunks


def get_files(root_path, extension):
    """Get all audio files with specified extension from root directory"""
    root = Path(root_path)

    # Ensure extension starts with dot
    if not extension.startswith('.'):
        extension = '.' + extension

    files = list(root.glob(f'**/*{extension}'))
    return files


def main():
    parser = argparse.ArgumentParser("Process-based VAD annotation pipeline")
    parser.add_argument("--root", "-r", type=str, required=False,
                        help="Root directory containing audio files",
                        default='/Users/lkieu/Desktop/VoxPopuli/unlabelled_data')
    parser.add_argument("--output_dir", "-o", type=str, required=False,
                        help="Output CSV file path",
                        default='/Users/lkieu/Desktop/VoxPopuli/vad_annotation')
    parser.add_argument("--extension_in", "-e", type=str, required=False,
                        help="Audio file extension (e.g., 'ogg', 'wav', 'mp3')",
                        default='ogg')
    parser.add_argument("--workers", "-w", type=int, default=14,
                        help="Number of worker processes")
    parser.add_argument("--device", "-d", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device to use for VAD (cpu or cuda)")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use (when device=cuda)")

    args = parser.parse_args()

    print("🚀 SIMPLE PROCESS-BASED VAD PIPELINE")
    print(f"  Root directory: {args.root}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Extension: {args.extension_in}")
    print(f"  Workers: {args.workers}")
    print(f"  Device: {args.device}")

    # Get all audio files with specified extension
    print(f"📁 Scanning for *.{args.extension_in} files...")
    all_files = get_files(args.root, args.extension_in)
    print(f"Found {len(all_files)} audio files")

    if not all_files:
        print(f"❌ No *.{args.extension_in} files found!")
        return

    # Split work among processes
    work_chunks = split_work(all_files, args.workers)
    print(f"📦 Work distribution:")
    for i, chunk in enumerate(work_chunks):
        print(f"  Worker-{i}: {len(chunk)} files")

    # Start processing
    print("⚡ Starting parallel processing...")
    start_time = time.time()

    # Create and start processes
    processes = []
    for i, chunk in enumerate(work_chunks):
        # For CUDA, you might want to assign different GPU IDs
        worker_gpu_id = args.gpu_id if args.device == 'cpu' else (args.gpu_id + (i % 1))

        p = mp.Process(
            target=worker_process,
            args=(i, chunk, args.output_dir, args.device, worker_gpu_id),
            name=f"Worker-{i}"
        )
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for i, p in enumerate(processes):
        print(f"⏳ Waiting for Worker-{i}...")
        p.join()

        if p.exitcode == 0:
            print(f"✅ Worker-{i} completed successfully")
        else:
            print(f"❌ Worker-{i} failed with exit code {p.exitcode}")

    total_time = time.time() - start_time

    print(f"⚡ Processing complete!")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Rate: {len(all_files) / total_time:.1f} files/sec")

    # List output files
    output_dir = Path(args.output_dir)
    csv_files = list(output_dir.glob("worker_*_annotations.csv"))
    print(f"📊 Output CSV files:")
    for csv_file in sorted(csv_files):
        print(f"  {csv_file}")

    print("🎉 Pipeline complete!")
    print(f"💡 To combine results: cat {args.output_dir}/worker_*_annotations.csv > combined_annotations.csv")


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    main()
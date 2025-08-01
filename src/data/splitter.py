#!/usr/bin/env python3
"""
Simple concurrent audio splitter: splits files into 5-second chunks at 16kHz
"""
import os
import time
import threading
from pathlib import Path
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import torchaudio
import torchaudio.transforms as T

from src.data.util import get_files, copy_folder_structure

# Global queue for chunks to be saved
SAVE_QUEUE = Queue(maxsize=28000)
saver_threads = []
shutdown_event = threading.Event()


def saver_worker(worker_id):
    """Worker that saves audio chunks from the queue"""
    print(f"[Saver-{worker_id}] Started")
    saved_count = 0

    while not shutdown_event.is_set():
        try:
            item = SAVE_QUEUE.get(timeout=2.0)
            if item is None:  # Shutdown signal
                break

            chunk, sample_rate, output_path = item

            # Save the chunk
            Path(output_path).parent.mkdir(exist_ok=True, parents=True)
            torchaudio.save(output_path, chunk, sample_rate)

            saved_count += 1
            if saved_count % 50 == 0:
                print(f"[Saver-{worker_id}] Saved {saved_count} chunks")

            SAVE_QUEUE.task_done()

        except Exception as e:
            print(f"[Saver-{worker_id}] Error: {e}")
            if 'item' in locals():
                SAVE_QUEUE.task_done()


def start_savers(num_savers=8):
    """Start saver threads"""
    global saver_threads
    saver_threads = []

    for i in range(num_savers):
        saver = threading.Thread(target=saver_worker, args=(i,), daemon=True)
        saver.start()
        saver_threads.append(saver)

    print(f"Started {num_savers} saver threads")
    return saver_threads


def process_audio_file(input_path, output_dir, input_dir, target_sr=16000, chunk_duration=5.0):
    """
    Process one audio file: resample if needed and split into chunks
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    input_dir = Path(input_dir)
    output_subdir = output_dir / input_path.relative_to(input_dir).parent

    try:
        # Load audio file
        waveform, orig_sr = torchaudio.load(input_path)


        # Resample if needed
        if orig_sr != target_sr:
            print(f"[Producer] Processing {input_path.name}: {orig_sr}Hz → {target_sr}Hz")
            resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
            waveform = resampler(waveform)

        # Calculate chunk parameters
        chunk_samples = int(target_sr * chunk_duration)  # 5 seconds worth of samples
        total_samples = waveform.size(1)
        num_chunks = (total_samples + chunk_samples - 1) // chunk_samples  # Ceiling division

        # Split into chunks and queue for saving
        length_sec = total_samples / target_sr
        chunks_queued = 0
        for i in range(num_chunks):
            start_sample = i * chunk_samples
            end_sample = min(start_sample + chunk_samples, total_samples)

            # Extract chunk
            chunk = waveform[:, start_sample:end_sample]

            # Create output filename
            output_filename = f"{input_path.stem}_chunk_{i:03d}.wav"
            output_path = output_subdir / output_filename

            # Queue for saving
            SAVE_QUEUE.put((chunk.clone(), target_sr, str(output_path)))
            chunks_queued += 1

        return chunks_queued, length_sec

    except Exception as e:
        print(f"[Producer] Error processing {input_path}: {e}")
        return 0


def split_audio_files(input_files, output_dir, input_dir, num_producers=4, num_savers=8,
                      target_sr=16000, chunk_duration=5.0):
    """
    Main function to split multiple audio files concurrently

    Args:
        input_files: List of input file paths
        output_dir: Output directory path
        input_dir: Input directory path (for path replacement)
        num_producers: Number of producer threads
        num_savers: Number of saver threads
        target_sr: Target sample rate
        chunk_duration: Duration of each chunk in seconds
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"🎵 Starting Audio Splitter:")
    print(f"  Input files: {len(input_files)}")
    print(f"  Input dir: {input_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Target sample rate: {target_sr}Hz")
    print(f"  Chunk duration: {chunk_duration}s")
    print(f"  Producers: {num_producers}")
    print(f"  Savers: {num_savers}")

    # Start saver threads
    start_savers(num_savers)

    # Process files with producer threads
    start_time = time.time()
    total_chunks = 0
    total_times = 0.0

    with ThreadPoolExecutor(max_workers=num_producers) as executor:
        # Submit all files for processing
        futures = [
            executor.submit(process_audio_file, input_file, output_dir, input_dir, target_sr, chunk_duration)
            for input_file in input_files
        ]

        # Collect results
        for i, future in enumerate(futures):
            try:
                chunks_count, length_sec = future.result()
                total_times += length_sec
                total_chunks += chunks_count
                print(f"Completed {i + 1}/{len(input_files)} files")
            except Exception as e:
                print(f"Producer failed on file {i + 1}: {e}")

    # Wait for all saves to complete
    print("Waiting for all chunks to be saved...")
    SAVE_QUEUE.join()

    # Shutdown savers
    shutdown_event.set()
    for _ in saver_threads:
        SAVE_QUEUE.put(None)
    for saver in saver_threads:
        saver.join(timeout=5)

    # Summary
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n✅ Audio Splitting Complete!")
    print(f"  Files processed: {len(input_files)}")
    print(f"  Total chunks created: {total_chunks}")
    print(f"  Recording processed: {total_times / 3600:.2f}h")
    print(f"  Time spent splitting: {total_time:.1f}s")
    print(f"  Rate: {len(input_files) / total_time:.1f} files/sec")
    print(f"  Chunk rate: {total_chunks / total_time:.1f} chunks/sec")


def main():
    """Simple command line interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Split audio files into 5-second chunks")
    parser.add_argument("--input", help="Input file or directory",
                        default=Path(r"D:\medium_trimmed"))
    parser.add_argument("--output", help="Output directory",
                        default=Path(r"D:\medium_split"))
    parser.add_argument("--duration", "-d", type=float, default=5.0,
                        help="Chunk duration in seconds (default: 5.0)")
    parser.add_argument("--sample-rate", "-sr", type=int, default=16000,
                        help="Target sample rate (default: 16000)")
    parser.add_argument("--producers", "-p", type=int, default=512,
                        help="Number of producer threads (default: 4)")
    parser.add_argument("--savers", "-s", type=int, default=256,
                        help="Number of saver threads (default: 8)")
    parser.add_argument("--ext_in", type=str, default=".flac")

    args = parser.parse_args()

    # Find input files
    input_path = Path(args.input)

    if input_path.is_file():
        input_files = [input_path]
        input_dir = input_path.parent
    elif input_path.is_dir():
        # Copy the structure of input dir.
        copy_folder_structure(args.input, args.output)
        input_files = get_files(input_path, args.ext_in)
        if not input_files:
            print(f"No audio files found in {input_path}")
            return
        input_dir = input_path
    else:
        print(f"Input path {input_path} does not exist")
        return

    print(f"Found {len(input_files)} audio files to process")

    # Process files
    split_audio_files(
        input_files=input_files,
        output_dir=args.output,
        input_dir=input_dir,
        num_producers=args.producers,
        num_savers=args.savers,
        target_sr=args.sample_rate,
        chunk_duration=args.duration,
    )


# Example usage as a module
def quick_split(input_file, output_dir, chunk_seconds=5.0):
    """
    Quick function to split a single file
    """
    split_audio_files(
        input_files=[input_file],
        output_dir=output_dir,
        num_producers=1,
        num_savers=4,
        chunk_duration=chunk_seconds
    )


if __name__ == "__main__":
    main()
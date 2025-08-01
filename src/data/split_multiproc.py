#!/usr/bin/env python3
"""
Simple multiprocess audio splitter: splits files into 5-second chunks at 16kHz
"""
import multiprocessing
import time
from pathlib import Path
import tqdm
import torchaudio
import torchaudio.transforms as T

from src.data.util import get_files, copy_folder_structure


def save(chunk, sample_rate, output_path):
    """Save a single audio chunk"""
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    torchaudio.save(output_path, chunk, sample_rate)


def cut(input_path, output_dir, input_dir, target_sr=16000, chunk_duration=5.0):
    """
    Cut a single audio file into chunks
    Returns (chunks_created, audio_length_seconds)
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    input_dir = Path(input_dir)

    # Calculate output subdirectory by replacing input_dir with output_dir
    output_subdir = output_dir / input_path.relative_to(input_dir).parent

    try:
        # Load audio
        waveform, orig_sr = torchaudio.load(input_path)

        # Resample if needed
        if orig_sr != target_sr:
            resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
            waveform = resampler(waveform)

        # Calculate chunks
        chunk_samples = int(target_sr * chunk_duration)
        total_samples = waveform.size(1)
        num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

        # Cut and save chunks
        for i in range(num_chunks):
            start_sample = i * chunk_samples
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = waveform[:, start_sample:end_sample]

            output_filename = f"{input_path.stem}_chunk_{i:03d}.wav"
            output_path = output_subdir / output_filename

            save(chunk, target_sr, str(output_path))

        audio_length = total_samples / target_sr
        return num_chunks, audio_length

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return 0, 0.0


def cut_many(input_files, output_dir, input_dir, num_workers=4, target_sr=16000, chunk_duration=5.0):
    """
    Cut multiple audio files using multiprocessing
    """
    print(f"🎵 Cutting {len(input_files)} files with {num_workers} workers...")

    start_time = time.time()

    # Prepare arguments for multiprocessing
    args = [(f, output_dir, input_dir, target_sr, chunk_duration) for f in input_files]

    # Process files
    with multiprocessing.Pool(processes=num_workers) as pool:
        for _ in tqdm.tqdm(pool.starmap_async(cut, args).get(), total=len(args)):
            pass

    # Calculate stats

    processing_time = time.time() - start_time

    print(f"✅ Complete!")
    print(f"  Files: {len(input_files)}")
    print(f"  Processing time: {processing_time:.1f}s")
    print(f"  Speed: {len(input_files) / processing_time:.1f} files/sec")


def parseargs():
    """Parse command line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description="Split audio files into chunks")
    parser.add_argument("--input", default=Path(r"D:\small_trimmed"),
                        help="Input file or directory")
    parser.add_argument("--output", default=Path(r"D:\small_split"),
                        help="Output directory")
    parser.add_argument("--duration", "-d", type=float, default=5.0,
                        help="Chunk duration in seconds")
    parser.add_argument("--sample-rate", "-sr", type=int, default=16000,
                        help="Target sample rate")
    parser.add_argument("--workers", "-w", type=int, default=32,
                        help="Number of worker processes")
    parser.add_argument("--ext_in", type=str, default=".flac",
                        help="Input file extension")

    return parser.parse_args()


def main():
    """Main function"""
    args = parseargs()

    # Find input files
    input_path = Path(args.input)

    if input_path.is_file():
        input_files = [input_path]
        input_dir = input_path.parent
    elif input_path.is_dir():
        input_files = get_files(input_path, args.ext_in)
        if not input_files:
            print(f"No audio files found in {input_path}")
            return
        input_dir = input_path
        copy_folder_structure(input_dir, args.output)
    else:
        print(f"Input path does not exist: {input_path}")
        return

    print(f"Found {len(input_files)} audio files")

    # Process files
    cut_many(
        input_files=input_files,
        output_dir=args.output,
        input_dir=input_dir,
        num_workers=args.workers,
        target_sr=args.sample_rate,
        chunk_duration=args.duration
    )


if __name__ == "__main__":
    main()
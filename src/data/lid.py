import argparse

import csv

from pathlib import Path
import multiprocessing as mp
import torchaudio
import time

from src.data.util import get_files, split_work


class LID:
    def __init__(self, device):
        from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
        import torch

        model_id = "facebook/mms-lid-256"

        self.processor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)
        self.device = device
        self.model.to(device)


    def verify_english(self, waveform, sample_rate):
        inputs = self.processor(waveform, sampling_rate=sample_rate, return_tensors="pt")
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        lang_id = torch.argmax(outputs, dim=-1)[0].item()
        return self.model.config.id2label[lang_id] == 'eng'

    def infer_lid_distribution(self, waveform_chunk, sample_rate, threshold=1):
        """
        Returns the language probability distribution for a waveform chunk.

        Args:
            waveform_chunk (Tensor): 1D tensor of audio samples.
            sample_rate (int): Sampling rate of the audio.
            threshold (float): top languages that explain certain percentages of the distribution. Default as 1 to return all languages.
        Returns:
            Dict[str, float]: Mapping from language code to probability.
        """
        inputs = self.processor(waveform_chunk, sampling_rate=sample_rate, return_tensors="pt")
        inputs.to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze()

        # Map probabilities to language labels
        lang_distribution = {
            self.model.config.id2label[idx]: prob.item()
            for idx, prob in enumerate(probs)
        }

        sorted_langs = sorted(lang_distribution.items(), key=lambda x: x[1], reverse=True)
        cumulative_prob = 0.0
        selected_langs = {}

        for lang, prob in sorted_langs:
            selected_langs[lang] = prob
            cumulative_prob += prob
            if cumulative_prob >= threshold:
                break

        return selected_langs


    def trim_non_target_language(
        self,
        waveform,
        sample_rate,
        target_lang='eng',
        window_sec=3.0,
        hop_sec=1.0,
        chunk_threshold=0.7
    ):
        assert waveform.ndim == 1, f"Expected 1D Tensor, got a Tensor of shape: {waveform.shape}"

        num_samples = waveform.shape[0]

        window_samples = int(window_sec * sample_rate)
        hop_samples = int(hop_sec * sample_rate)

        retained_segments = []

        for start_sample in range(0, num_samples - window_samples + 1, hop_samples):
            chunk = waveform[start_sample: start_sample + window_samples]
            target_prob = self.infer_lid_distribution(chunk, sample_rate)[target_lang]

            if target_prob >= chunk_threshold:
                start_time = start_sample / sample_rate
                end_time = (start_sample + window_samples) / sample_rate
                retained_segments.append((start_time, end_time))

        # Merge adjacent or overlapping segments
        merged_segments = []
        for seg in retained_segments:
            if not merged_segments or seg[0] > merged_segments[-1][1]:
                merged_segments.append(seg)
            else:
                merged_segments[-1] = (merged_segments[-1][0], max(merged_segments[-1][1], seg[1]))

        # Concatenate retained audio
        trimmed_audio = []
        for start_time, end_time in merged_segments:
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            trimmed_audio.append(waveform[start_sample:end_sample])

        if trimmed_audio:
            trimmed_audio = torch.cat(trimmed_audio)
        else:
            trimmed_audio = torch.empty(0)

        return trimmed_audio, merged_segments

def worker_verify(worker_id, file_paths, output_dir, device='cpu', gpu_id=0):
    """
        Worker process that handles its own share of files and writes its own CSV.
        """
    print(f"[Worker-{worker_id}] Starting with {len(file_paths)} files on {device}")

    # Initialize VAD model in this process
    try:
        if device == 'cuda':
            import torch
            torch.cuda.set_device(gpu_id)

        lid = LID(device)
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
        writer.writerow(['file_path'])

        for file_path in file_paths:
            try:
                # Load audio
                waveform, sample_rate = torchaudio.load(file_path)

                if device == 'cuda':
                    waveform = waveform.to('cuda', non_blocking=True)

                waveform = waveform.squeeze()

                # Get annotations
                if lid.verify_english(waveform, sample_rate):
                    writer.writerow([str(file_path)])

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
            target=worker_verify(),
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
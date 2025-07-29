import argparse
import os
import traceback
from datetime import datetime
import glob
import gzip
import csv
import re

import pandas as pd
from sympy.codegen import Print

import threading
import time
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import torchaudio
import torch

# Optimized configuration for smooth CPU load
WORK_QUEUE_SIZE = 500  # Smaller, faster cycling
AUDIO_BUFFER_SIZE = 200  # Pre-loaded audio buffer
SAVE_QUEUE_SIZE = 1000  # Reduced from 16000
BATCH_SIZE = 4  # Process multiple files per worker cycle

# Queues
WORK_QUEUE = queue.Queue(maxsize=WORK_QUEUE_SIZE)
AUDIO_BUFFER = queue.Queue(maxsize=AUDIO_BUFFER_SIZE)  # Pre-loaded audio
SAVE_QUEUE = queue.Queue(maxsize=SAVE_QUEUE_SIZE)

# Threading
saver_threads = []
loader_threads = []
vad_workers = []
scaling_lock = threading.Lock()
active_savers = 0
shutdown_event = threading.Event()
stats_lock = threading.Lock()

# Stats tracking
processing_stats = {
    'files_loaded': 0,
    'files_processed': 0,
    'files_saved': 0,
    'last_update': time.time()
}

# Updated VAD module integration
BETWEEN_SEGMENT = 0.15

HYPER_PARAMETERS = {
    # remove speech regions shorter than that many seconds.
    "min_duration_on": 0.0,
    # fill non-speech regions shorter than that many seconds.
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


def trim_speech(waveform, sample_rate, annotation_str, to_annotation=True):
    """If to_annotation is True, keep the annotated segments. Otherwise, remove them."""
    time_ranges = parse_annotation(annotation_str)

    T = waveform.shape[1]
    mask = torch.zeros(T, dtype=torch.bool) if to_annotation else torch.ones(T, dtype=torch.bool)

    for start_sec, end_sec in time_ranges:
        start_sample = max(0, int(start_sec * sample_rate))
        end_sample = min(T, int(end_sec * sample_rate))

        if to_annotation:
            mask[start_sample:end_sample] = True
        else:
            mask[start_sample:end_sample] = False

    trimmed_waveform = waveform[:, mask]
    return trimmed_waveform


def parse_annotation(annotation_str):
    # Extract start and end times in seconds
    pattern = r'\[\s*(\d+:\d+:\d+\.\d+)\s*-->\s*(\d+:\d+:\d+\.\d+)\]'
    matches = re.findall(pattern, annotation_str)

    def time_to_seconds(t):
        h, m, s = t.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    time_ranges = [(time_to_seconds(start), time_to_seconds(end) + BETWEEN_SEGMENT) for start, end in matches]
    return time_ranges


class AudioLoader:
    """Dedicated audio loading worker to prevent I/O blocking"""

    def __init__(self, loader_id):
        self.loader_id = loader_id
        self.loaded_count = 0

    def run(self):
        """Load audio files asynchronously"""
        print(f"[Loader-{self.loader_id}] Started")

        while not shutdown_event.is_set():
            try:
                # Get work item (file paths)
                item = WORK_QUEUE.get(timeout=1.0)
                if item is None:  # Shutdown signal
                    break

                in_path, out_path = item

                # Load audio
                waveform, sr = torchaudio.load(in_path)

                # Put loaded audio in buffer for VAD workers
                AUDIO_BUFFER.put((waveform, sr, out_path), timeout=5.0)

                self.loaded_count += 1

                with stats_lock:
                    processing_stats['files_loaded'] += 1

                WORK_QUEUE.task_done()

            except queue.Empty:
                continue
            except queue.Full:
                print(f"[Loader-{self.loader_id}] Audio buffer full, waiting...")
                time.sleep(0.1)
            except Exception as e:
                print(f"[Loader-{self.loader_id}] Error loading: {e}")
                if not WORK_QUEUE.empty():
                    WORK_QUEUE.task_done()

        print(f"[Loader-{self.loader_id}] Finished. Loaded {self.loaded_count} files")


class VADWorker:
    """VAD worker that only does detection, trimming moved to savers"""

    def __init__(self, worker_id, device='cpu', gpu_id=0):
        self.worker_id = worker_id
        self.device = device
        self.gpu_id = gpu_id
        self.vad_model = None
        self.processed_count = 0

    def initialize_model(self):
        """Initialize VAD model once per worker"""
        if self.device == 'cuda':
            torch.cuda.set_device(self.gpu_id)

        try:
            self.vad_model = VAD(self.device)
            print(f"[VAD-Worker-{self.worker_id}] Model initialized on {self.device}")
        except Exception as e:
            print(f"[VAD-Worker-{self.worker_id}] Failed to initialize model: {e}")
            raise e

    def process_batch(self, batch):
        """Process a batch of audio files - detection only"""
        for waveform, sr, out_path in batch:
            try:
                if self.device == 'cuda':
                    waveform = waveform.to('cuda', non_blocking=True)

                # Step 1: Find speech segments (do this first as per design)
                speech_annotation = self.vad_model.find_speech(waveform, sr)

                # # Step 2: Find overlapped speech segments
                # overlapped_annotation = self.vad_model.find_overlapped(waveform, sr)

                # Move waveform back to CPU for saving
                if self.device == 'cuda':
                    waveform = waveform.cpu()

                # Send to savers with annotations for trimming
                save_item = {
                    'waveform': waveform,
                    'sample_rate': sr,
                    'out_path': out_path.as_posix(),
                    'speech_annotation': speech_annotation,
                    # 'overlapped_annotation': overlapped_annotation
                }

                # Non-blocking save queue put
                try:
                    SAVE_QUEUE.put(save_item, timeout=0.1)
                    self.processed_count += 1

                    with stats_lock:
                        processing_stats['files_processed'] += 1

                except queue.Full:
                    # If save queue is full, wait briefly and retry
                    time.sleep(0.05)
                    SAVE_QUEUE.put(save_item, timeout=2.0)
                    self.processed_count += 1

                    with stats_lock:
                        processing_stats['files_processed'] += 1

            except Exception as e:
                print(f"[VAD-Worker-{self.worker_id}] Error processing: {e}")
                print(traceback.format_exc())
            finally:
                if 'waveform' in locals():
                    del waveform

        # GPU cleanup after batch
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def run(self):
        """Main worker loop with batching"""
        print(f"[VAD-Worker-{self.worker_id}] Starting on {self.device}")

        # Initialize model once
        self.initialize_model()

        # Process batches from audio buffer
        while not shutdown_event.is_set():
            batch = []

            # Collect a batch
            try:
                # Get first item (blocking)
                item = AUDIO_BUFFER.get(timeout=2.0)
                if item is None:  # Shutdown signal
                    break
                batch.append(item)

                # Get additional items (non-blocking)
                for _ in range(BATCH_SIZE - 1):
                    try:
                        item = AUDIO_BUFFER.get_nowait()
                        if item is None:
                            break
                        batch.append(item)
                    except queue.Empty:
                        break

                # Process the batch
                if batch:
                    self.process_batch(batch)

                    # Mark all items as done
                    for _ in batch:
                        AUDIO_BUFFER.task_done()

                # Progress reporting
                if self.processed_count % 50 == 0 and self.processed_count > 0:
                    print(f"[VAD-Worker-{self.worker_id}] Processed {self.processed_count} files")

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VAD-Worker-{self.worker_id}] Worker error: {e}")
                # Mark batch items as done even on error
                for _ in batch:
                    if not AUDIO_BUFFER.empty():
                        AUDIO_BUFFER.task_done()

        print(f"[VAD-Worker-{self.worker_id}] Finished. Processed {self.processed_count} files total")


class AdaptiveScaler:
    def __init__(self, min_savers=8, max_savers=24):
        self.min_savers = min_savers
        self.max_savers = max_savers
        self.current_savers = min_savers
        self.scale_history = []

    def start_monitoring(self):
        """Start adaptive scaling monitor with detailed stats"""
        monitor = threading.Thread(target=self._adaptive_monitor, daemon=True)
        monitor.start()
        return monitor

    def _adaptive_monitor(self):
        """Monitor queues and performance with detailed reporting"""
        print("🔍 Adaptive scaling monitor started")
        last_processed = 0

        while not shutdown_event.is_set():
            work_queue_size = WORK_QUEUE.qsize()
            audio_buffer_size = AUDIO_BUFFER.qsize()
            save_queue_size = SAVE_QUEUE.qsize()

            with stats_lock:
                current_processed = processing_stats['files_processed']
                processing_rate = (current_processed - last_processed) / 15  # files/sec
                last_processed = current_processed

                print(f'[{datetime.now().strftime("%H:%M:%S")}] '
                      f'Work: {work_queue_size:3d} | '
                      f'Audio: {audio_buffer_size:3d} | '
                      f'Save: {save_queue_size:3d} | '
                      f'Rate: {processing_rate:.1f}/s | '
                      f'Processed: {current_processed:,}')

            save_queue_pct = (save_queue_size / SAVE_QUEUE_SIZE) * 100

            # Scale up conditions
            if (save_queue_pct > 75 and self.current_savers < self.max_savers):
                self._scale_up()

            # Scale down conditions
            elif (save_queue_pct < 25 and self.current_savers > self.min_savers):
                self._scale_down()

            time.sleep(15)  # Check every 15 seconds

    def _scale_up(self):
        """Add more saver threads"""
        with scaling_lock:
            add_count = min(4, self.max_savers - self.current_savers)
            if add_count > 0:
                print(
                    f"🚀 SCALE UP: Adding {add_count} savers ({self.current_savers}→{self.current_savers + add_count})")

                for i in range(add_count):
                    saver_id = len(saver_threads)
                    saver = threading.Thread(target=adaptive_saver_worker, args=(saver_id,), daemon=True)
                    saver.start()
                    saver_threads.append(saver)

                self.current_savers += add_count
                self.scale_history.append(f"UP to {self.current_savers}")

    def _scale_down(self):
        """Remove saver threads"""
        with scaling_lock:
            remove_count = min(2, self.current_savers - self.min_savers)
            if remove_count > 0:
                print(
                    f"📉 SCALE DOWN: Removing {remove_count} savers ({self.current_savers}→{self.current_savers - remove_count})")

                global active_savers
                active_savers = self.current_savers - remove_count
                self.current_savers -= remove_count
                self.scale_history.append(f"DOWN to {self.current_savers}")


def adaptive_saver_worker(worker_id):
    """Saver worker that handles trimming based on annotations"""
    global active_savers
    print(f"[Saver-{worker_id}] Started with trimming capability")
    saved_count = 0
    idle_time = 0

    while not shutdown_event.is_set():
        # Check if we should exit (scaled down)
        if worker_id >= active_savers and idle_time > 30:
            print(f"[Saver-{worker_id}] Exiting (scaled down)")
            break

        try:
            item = SAVE_QUEUE.get(timeout=1.0)
            if item is None:  # Shutdown signal
                break

            # Extract data from the item dict
            waveform = item['waveform']
            sample_rate = item['sample_rate']
            out_path = item['out_path']
            speech_annotation = item['speech_annotation']
            # overlapped_annotation = item['overlapped_annotation']

            # Step 1: Trim to speech segments (keep speech)
            if speech_annotation and speech_annotation.strip():
                waveform = trim_speech(waveform, sample_rate, speech_annotation, to_annotation=True)

            # Step 2: Remove overlapped speech segments (remove overlaps)
            # if overlapped_annotation and overlapped_annotation.strip():
            #     waveform = trim_speech(waveform, sample_rate, overlapped_annotation, to_annotation=False)

            # Skip saving if waveform is too short after trimming
            min_duration = 1.0  # 1 second minimum
            if waveform.shape[1] < min_duration * sample_rate:
                print(f"[Saver-{worker_id}] Skipping {Path(out_path).name} - too short after trimming")
                SAVE_QUEUE.task_done()
                continue

            # Save the trimmed audio
            torchaudio.save(out_path, waveform, sample_rate)
            saved_count += 1
            idle_time = 0

            with stats_lock:
                processing_stats['files_saved'] += 1

            if saved_count % 100 == 0:
                print(f"[Saver-{worker_id}] Saved {saved_count} files")

            SAVE_QUEUE.task_done()

        except queue.Empty:
            idle_time += 1
            continue
        except Exception as e:
            print(f"[Saver-{worker_id}] Error processing/saving: {e}")
            print(traceback.format_exc())
            SAVE_QUEUE.task_done()


def get_adaptive(args):
    """Main function with optimized pipeline"""
    global active_savers

    # Setup
    audio_root = Path(args.root)
    out_root = Path(args.rootout) / "vad_proc"
    out_root.mkdir(exist_ok=True, parents=True)

    # Load manifest
    print("Loading manifest...")
    manifest = get_files(audio_root)

    if args.cont:
        print("Checking existing files...")
        existing_files = set(Path(p).stem for p in
                             out_root.glob('**/*.wav'))
        print(f"Found {len(existing_files)} existing files")

    items = []
    for file in manifest:
        name = file.stem
        if args.cont and name in existing_files:
            continue
        else:
            items.append((file, out_root / Path(name).with_suffix(".wav")))

    print(f"🚀 OPTIMIZED VAD PIPELINE CONFIGURATION:")
    print(f"  Work Queue: {WORK_QUEUE_SIZE} items")
    print(f"  Audio Buffer: {AUDIO_BUFFER_SIZE} items")
    print(f"  Save Queue: {SAVE_QUEUE_SIZE} items")
    print(f"  Batch Size: {BATCH_SIZE} files/worker")
    print(f"  Loaders: {args.loaders}")
    print(f"  VAD workers: {args.workers} (detection only)")
    print(f"  Initial savers: 8 (with trimming)")
    print(f"  Files to process: {len(items):,}")

    # Start adaptive saver system
    scaler = AdaptiveScaler(min_savers=1, max_savers=24)
    active_savers = 1

    # Start initial savers
    for i in range(8):
        saver = threading.Thread(target=adaptive_saver_worker, args=(i,), daemon=True)
        saver.start()
        saver_threads.append(saver)

    # Start audio loaders
    for i in range(args.loaders):
        loader = AudioLoader(loader_id=i)
        loader_thread = threading.Thread(target=loader.run, daemon=True)
        loader_thread.start()
        loader_threads.append((loader, loader_thread))

    # Start VAD workers
    device = 'cuda' if getattr(args, 'cuda', False) else 'cpu'
    gpu_id = getattr(args, 'gpu_id', 0)

    for i in range(args.workers):
        worker_gpu_id = gpu_id if device == 'cpu' else (gpu_id + (i % 1))
        worker = VADWorker(worker_id=i, device=device, gpu_id=worker_gpu_id)
        worker_thread = threading.Thread(target=worker.run, daemon=True)
        worker_thread.start()
        vad_workers.append((worker, worker_thread))

    # Start monitoring
    monitor = scaler.start_monitoring()

    # Feed work queue with file paths
    print("Starting pipeline...")
    start_time = time.time()

    # Queue all work items
    for item in items:
        WORK_QUEUE.put(item)

    print(f"All {len(items)} items queued for processing")

    # Wait for all work to complete
    print("Waiting for processing to complete...")
    WORK_QUEUE.join()  # All files loaded
    AUDIO_BUFFER.join()  # All audio processed
    SAVE_QUEUE.join()  # All files saved

    # Shutdown sequence
    print("Shutting down workers...")
    shutdown_event.set()

    # Shutdown loaders
    for _ in loader_threads:
        WORK_QUEUE.put(None)
    for loader, thread in loader_threads:
        thread.join(timeout=5)

    # Shutdown VAD workers
    for _ in vad_workers:
        AUDIO_BUFFER.put(None)
    for worker, thread in vad_workers:
        thread.join(timeout=10)

    # Shutdown savers
    for _ in saver_threads:
        SAVE_QUEUE.put(None)
    for t in saver_threads:
        t.join(timeout=5)

    total_time = time.time() - start_time
    final_files = sum(1 for _ in out_root.glob('**/*.wav'))

    print(f"🎉 OPTIMIZED VAD PIPELINE COMPLETE:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Files created: {final_files:,}")
    print(f"  Rate: {final_files / total_time:.1f} files/sec")
    print(f"  Final stats: {processing_stats}")
    print(f"  Scaling events: {scaler.scale_history}")


def get_files(root):
    return set(Path(p) for p in root.glob('**/*.ogg'))


def get_args():
    parser = argparse.ArgumentParser("Optimized VAD pipeline with separated detection and trimming")
    parser.add_argument(
        "--root", "-r", type=str, help="data root path",
        default='/Users/lkieu/Desktop/VoxPopuli/unlabelled_data'
    )
    parser.add_argument("--rootout", "-o", type=str, help="output root path",
                        default='/Users/lkieu/Desktop/VoxPopuli/')

    parser.add_argument("--workers", "-w", type=int, default=12,
                        help="Number of VAD workers (detection only)")
    parser.add_argument("--loaders", "-l", type=int, default=1,
                        help="Number of audio loading threads")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="Use CUDA for VAD processing")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use (default: 0)")
    parser.add_argument("--cont", '-c', default=False, action="store_true",
                        help="Continue from existing segmented files")
    return parser.parse_args()


def main():
    args = get_args()
    get_adaptive(args)


if __name__ == "__main__":
    main()
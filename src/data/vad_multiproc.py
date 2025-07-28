import argparse
import traceback
from datetime import datetime
import threading
import time
import queue
from pathlib import Path

import torchaudio

# Adaptive configuration
QUEUE_SIZE = 16000
SAVE_QUEUE = queue.Queue(maxsize=QUEUE_SIZE)  # ~150GB RAM capacity
WORK_QUEUE = queue.Queue(maxsize=1000)  # Input work queue
saver_threads = []
vad_workers = []
scaling_lock = threading.Lock()
active_savers = 0
shutdown_event = threading.Event()


class VADWorker:
    """Persistent VAD worker with model loaded once"""

    def __init__(self, worker_id, device='cpu', gpu_id=0):
        self.worker_id = worker_id
        self.device = device
        self.gpu_id = gpu_id
        self.vad_model = None
        self.processed_count = 0

    def initialize_model(self):
        """Initialize VAD model once per worker"""
        if self.device == 'cuda':
            import torch
            torch.cuda.set_device(self.gpu_id)

        try:
            from src.data.vad import VAD
            self.vad_model = VAD(self.device)
            print(f"[VAD-Worker-{self.worker_id}] Model initialized on {self.device}")
        except Exception as e:
            print(f"[VAD-Worker-{self.worker_id}] Failed to initialize model: {e}")
            raise e

    def process_item(self, item):
        """Process single item with persistent model"""
        in_path, out_path = item

        try:
            # Load audio
            waveform, sr = torchaudio.load(in_path)
            if self.device == 'cuda':
                waveform = waveform.to('cuda', non_blocking=True)

            # Process with persistent VAD model
            waveform = self.vad_model.find_speech_and_trim(waveform, sr)
            waveform = self.vad_model.find_overlapped_and_trim(waveform, sr)

            # Queue for saving
            SAVE_QUEUE.put((waveform.cpu() if self.device == 'cuda' else waveform, sr, out_path.as_posix()))

            self.processed_count += 1
            if self.processed_count % 50 == 0:
                print(f"[VAD-Worker-{self.worker_id}] Processed {self.processed_count} files")

        except Exception as e:
            print(f"[VAD-Worker-{self.worker_id}] Error processing {in_path}: {e}")
            print(traceback.format_exc())
        finally:
            # Clean up GPU memory
            if 'waveform' in locals():
                del waveform
            if self.device == 'cuda':
                import torch
                torch.cuda.empty_cache()

    def run(self):
        """Main worker loop"""
        print(f"[VAD-Worker-{self.worker_id}] Starting on {self.device}")

        # Initialize model once
        self.initialize_model()

        # Process items from queue
        while not shutdown_event.is_set():
            try:
                item = WORK_QUEUE.get(timeout=2.0)
                if item is None:  # Shutdown signal
                    break

                self.process_item(item)
                WORK_QUEUE.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[VAD-Worker-{self.worker_id}] Worker error: {e}")
                if not WORK_QUEUE.empty():
                    WORK_QUEUE.task_done()

        print(f"[VAD-Worker-{self.worker_id}] Finished. Processed {self.processed_count} files total")


class AdaptiveScaler:
    def __init__(self, min_savers=16, max_savers=48):
        self.min_savers = min_savers
        self.max_savers = max_savers
        self.current_savers = min_savers
        self.scale_history = []

    def start_monitoring(self):
        """Start adaptive scaling monitor"""
        monitor = threading.Thread(target=self._adaptive_monitor, daemon=True)
        monitor.start()
        return monitor

    def _adaptive_monitor(self):
        """Monitor and adapt saver count"""
        print("🔍 Adaptive scaling monitor started")

        while not shutdown_event.is_set():
            save_queue_size = SAVE_QUEUE.qsize()
            work_queue_size = WORK_QUEUE.qsize()
            print(f'[{datetime.now()}] Save Queue: {save_queue_size:,}, Work Queue: {work_queue_size:,}')
            queue_pct = (save_queue_size / QUEUE_SIZE) * 100

            # Scale up conditions
            if (queue_pct > 85 and self.current_savers < self.max_savers):
                self._scale_up()

            # Scale down conditions
            elif (queue_pct < 30 and self.current_savers > self.min_savers):
                self._scale_down()

            time.sleep(15)  # Check every 15 seconds

    def _scale_up(self):
        """Add more saver threads"""
        with scaling_lock:
            add_count = min(8, self.max_savers - self.current_savers)
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
        """Remove saver threads (by letting them timeout)"""
        with scaling_lock:
            remove_count = min(4, self.current_savers - self.min_savers)
            if remove_count > 0:
                print(
                    f"📉 SCALE DOWN: Removing {remove_count} savers ({self.current_savers}→{self.current_savers - remove_count})")

                # Signal threads to exit by reducing active count
                global active_savers
                active_savers = self.current_savers - remove_count
                self.current_savers -= remove_count
                self.scale_history.append(f"DOWN to {self.current_savers}")


def adaptive_saver_worker(worker_id):
    """Adaptive saver worker that can scale down"""
    global active_savers
    print(f"[Saver-{worker_id}] Started")
    saved_count = 0
    idle_time = 0

    while not shutdown_event.is_set():
        # Check if we should exit (scaled down)
        if worker_id >= active_savers and idle_time > 30:
            print(f"[Saver-{worker_id}] Exiting (scaled down)")
            break

        try:
            item = SAVE_QUEUE.get(timeout=2.0)
            if item is None:  # Shutdown signal
                break

            waveform, sr, out_path = item

            # Fast save (directories pre-created)
            # import pickle
            # with open(out_path + '.pkl', 'wb') as f:
            #     pickle.dump((chunk.numpy(), sr), f)
            torchaudio.save(out_path, waveform, sr)
            saved_count += 1
            idle_time = 0

            if saved_count % 200 == 0:
                queue_size = SAVE_QUEUE.qsize()
                print(f"[Saver-{worker_id}] {saved_count} saves, Queue: {queue_size}")

            SAVE_QUEUE.task_done()

        except queue.Empty:
            idle_time += 2
            continue
        except Exception as e:
            print(f"[Saver-{worker_id}] Error: {e}")
            SAVE_QUEUE.task_done()


def get_adaptive(args):
    """Main function with persistent VAD workers"""
    global active_savers

    # Setup
    audio_root = Path(args.root)
    out_root = Path(args.rootout) / "vad_proc"
    out_root.mkdir(exist_ok=True, parents=True)
    in_ext = args.in_ext
    out_ext = args.out_ext

    # Load manifest
    print("Loading manifest...")
    manifest = get_files(audio_root, in_ext)

    if args.cont:
        print("Checking existing files...")
        existing_files = set(Path(p).stem for p in
                             out_root.glob(f'**/*{out_ext}'))
        print(f"Found {len(existing_files)} existing files")

    items = []
    for file in manifest:
        name = file.stem
        if args.cont and name in existing_files:
            continue
        else:
            items.append((file, out_root / Path(name).with_suffix(out_ext)))

    print(f"  Save Queue: {QUEUE_SIZE} items (~150GB RAM)")
    print(f"  Work Queue: 1000 items")
    print(f"  Initial savers: 16")
    print(f"  Max savers: 16")
    print(f"  VAD workers: {args.workers}")
    print(f"  Files to process: {len(items):,}")

    init_savers = 4
    # Start adaptive system
    scaler = AdaptiveScaler(min_savers=init_savers, max_savers=init_savers)
    active_savers = init_savers

    # Start initial savers
    for i in range(init_savers):
        saver = threading.Thread(target=adaptive_saver_worker, args=(i,), daemon=True)
        saver.start()
        saver_threads.append(saver)

    # Start VAD workers with persistent models
    device = 'cuda' if getattr(args, 'cuda', False) else 'cpu'
    gpu_id = getattr(args, 'gpu_id', 0)

    for i in range(args.workers):
        # If using CUDA, you might want to distribute workers across GPUs
        worker_gpu_id = gpu_id if device == 'cpu' else (gpu_id + (i % 1))  # Modify for multi-GPU
        worker = VADWorker(worker_id=i, device=device, gpu_id=worker_gpu_id)
        worker_thread = threading.Thread(target=worker.run, daemon=True)
        worker_thread.start()
        vad_workers.append((worker, worker_thread))

    # Start adaptive monitor
    monitor = scaler.start_monitoring()

    # Feed work queue
    print("Feeding work queue...")
    start_time = time.time()

    for item in items:
        WORK_QUEUE.put(item)

    print(f"All {len(items)} items queued for processing")

    # Wait for work queue to be processed
    print("Waiting for VAD processing to complete...")
    WORK_QUEUE.join()

    # Shutdown VAD workers
    print("Shutting down VAD workers...")
    for _ in vad_workers:
        WORK_QUEUE.put(None)

    for worker, thread in vad_workers:
        thread.join(timeout=10)

    # Wait for saves to complete
    print("🏁 VAD processing finished. Waiting for saves...")
    SAVE_QUEUE.join()

    # Shutdown savers
    shutdown_event.set()
    for _ in saver_threads:
        SAVE_QUEUE.put(None)
    for t in saver_threads:
        t.join(timeout=5)

    total_time = time.time() - start_time
    final_files = sum(1 for _ in out_root.glob(f'**/*{out_ext}'))

    print(f"🎉 PERSISTENT VAD PIPELINE COMPLETE:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Files created: {final_files:,}")
    print(f"  Rate: {final_files / total_time:.1f} files/sec")
    print(f"  Scaling events: {scaler.scale_history}")


def get_files(root, in_ext):
    return set(Path(p) for p in root.glob(f'**/*{in_ext}'))


def get_args():
    parser = argparse.ArgumentParser("Prepare unlabelled data with persistent VAD models")
    parser.add_argument(
        "--root", "-r", type=str, help="data root path",
        default='/Users/lkieu/Desktop/VoxPopuli/unlabelled_data'
    )
    parser.add_argument("--rootout", "-o", type=str, help="output root path",
                        default='/Users/lkieu/Desktop/VoxPopuli/')
    parser.add_argument("--workers", "-w", type=int, default=14,  # Reduced for persistent models
                        help="Number of VAD workers with persistent models")
    parser.add_argument("--cuda", action="store_true", default=False,
                        help="Use CUDA for VAD processing")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use (default: 0)")
    parser.add_argument("--cont", '-c', default=False, action="store_true",
                        help="Continue from existing segmented files (skip already processed ones).")
    parser.add_argument("--in_ext", type=str, help="input file extension",
                        default='.ogg')
    parser.add_argument(
        "--out_ext", type=str, help="output file extension (will be appended to file name),"
        "e.g. '.wav' for a 16kHz 16-bit PCM WAV file",
        default='.wav'
    )
    return parser.parse_args()


def main():
    args = get_args()
    get_adaptive(args)


if __name__ == "__main__":
    main()
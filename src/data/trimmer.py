import argparse
from datetime import datetime
import pandas as pd
import threading
import time
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import torchaudio

from src.data.util import download_file, get_files

# Adaptive configuration
QUEUE_SIZE = 16000
SAVE_QUEUE = queue.Queue(maxsize=QUEUE_SIZE)  # ~150GB RAM capacity
saver_threads = []
scaling_lock = threading.Lock()
active_savers = 0
shutdown_event = threading.Event()


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
            queue_size = SAVE_QUEUE.qsize()
            print(f'[{datetime.now()}] Queue: {queue_size:,} items')
            queue_pct = (queue_size / QUEUE_SIZE) * 100

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

            chunk, sr, out_path = item

            torchaudio.save(out_path, chunk, sr)
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



def enhanced_segment(item, cuda=False, gpu_id=0):
    """Enhanced segment function with queue monitoring"""
    in_path, segments, out_root = item
    file_name = Path(in_path).stem

    if cuda:
        import torch
        torch.cuda.set_device(gpu_id)

    try:
        waveform, sr = torchaudio.load(in_path)
        if cuda:
            waveform = waveform.to('cuda', non_blocking=True)

        for i, s, e in segments:
            start, end = int(s * sr), min(waveform.size(1), int(e * sr))
            chunk = waveform[:, start:end].clone()

            if cuda:
                chunk = chunk.cpu()

            out_path = Path(out_root) / f'{file_name}_{i}.ogg'
            SAVE_QUEUE.put((chunk, sr, out_path.as_posix()))

    except Exception as e:
        print(f"Error processing {in_path}: {e}")
    finally:
        if 'waveform' in locals():
            del waveform
        if cuda:
            torch.cuda.empty_cache()


def get_adaptive(args):
    """Main function with full adaptive system"""
    global active_savers

    # Setup
    audio_root = Path(args.root)
    out_root = Path(args.rootout) / "unlabelled_data"
    out_root.mkdir(exist_ok=True, parents=True)

    # Load manifest
    print("Loading manifest...")

    manifest = get_metadata_vox_pop(out_root, args.subset)

    # Build items list
    items = defaultdict(list)
    if args.cont:
        print("Checking existing files...")
        existing_files = set(Path(p).name for p in
                             get_files(out_root, args.ext_out))
        print(f"Found {len(existing_files)} existing files")

    for event_id, seg_no, start, end in manifest:
        segment = f"{event_id}_{seg_no}{args.ext_out.lower()}"
        if args.cont and segment in existing_files:
            continue

        lang, year = event_id.rsplit("_", 1)[1], event_id[:4]
        path = audio_root / lang / year / f"{event_id}{args.ext_in.lower()}"
        items[path.as_posix()].append((seg_no, float(start), float(end)))

    items_list = [(k, v, out_root.as_posix()) for k, v in items.items() if v]


    print(f"🚀 ADAPTIVE BEAST MODE:")
    print(f"  Queue: {QUEUE_SIZE} items (~150GB RAM)")
    print(f"  Initial savers: 16")
    print(f"  Max savers: 48")
    print(f"  Producers: {args.workers}")
    print(f"  Files to process: {len(items_list):,}")

    init_savers = 16
    # Start adaptive system
    scaler = AdaptiveScaler(min_savers=init_savers, max_savers=init_savers)
    active_savers = init_savers

    # Start initial savers
    for i in range(init_savers):
        saver = threading.Thread(target=adaptive_saver_worker, args=(i,), daemon=True)
        saver.start()
        saver_threads.append(saver)

    # Run producers
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(enhanced_segment, item,
                                   getattr(args, 'cuda', False),
                                   getattr(args, 'gpu_id', 0))
                   for item in items_list]

        for i, future in enumerate(futures):
            try:
                future.result()
                if i % 100 == 0:
                    print(f"Producers: {i}/{len(items_list)} complete")
            except Exception as e:
                print(f"Producer error: {e}")

    # Shutdown sequence
    print("🏁 Producers finished. Waiting for saves...")
    SAVE_QUEUE.join()

    shutdown_event.set()
    for _ in saver_threads:
        SAVE_QUEUE.put(None)
    for t in saver_threads:
        t.join(timeout=5)

    total_time = time.time() - start_time
    final_files = sum(1 for _ in out_root.glob('**/*.ogg'))

    print(f"🎉 ADAPTIVE BEAST MODE COMPLETE:")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Files created: {final_files:,}")
    print(f"  Rate: {final_files / total_time:.1f} files/sec")
    print(f"  Scaling events: {scaler.scale_history}")


def get_metadata_vox_pop(out_root, subset):
    """Only known to work with en_v2 subset"""
    def predicate(id_):
        return id_.endswith(subset.split("_")[0])

    DOWNLOAD_BASE_URL = "https://dl.fbaipublicfiles.com/voxpopuli"
    filename = "unlabelled_v2"

    url = f"{DOWNLOAD_BASE_URL}/annotations/{filename}.tsv.gz"
    tsv_path = out_root / f'{filename}.tsv'
    if not tsv_path.exists():
        download_file(url, out_root.as_posix(), Path(url).name)

    df = pd.read_csv(tsv_path, sep='\t', usecols=["event_id", "segment_no", "start", "end"])
    rows = df[df["event_id"].apply(predicate)][["event_id", "segment_no", "start", "end"]]
    rows = list(rows.itertuples(index=False, name=None))

    return rows


def get_args():
    parser = argparse.ArgumentParser("Prepare unlabelled data")
    parser.add_argument(
        "--root", "-r", type=str, help="data root path",
        default='/Volumes/Kieu4TB/gigaspeech/gigaspeech/data/extracted/xl'
    )
    parser.add_argument("--rootout", "-o", type=str, help="output root path",
                        default='/Volumes/Kieu4TB/gigaspeech/gigaspeech/data/trimmed/xl')

    parser.add_argument("--workers", "-w", type=int, default=47,
                        help="Total number of workers")
    parser.add_argument("--gpu", "-g", type=int, default=None,
                        help="GPU ID to dedicate to one worker (default: 0)")
    parser.add_argument("--cont", '-c', default=False, action="store_true",
                        help="Continue from existing segmented files (skip already processed ones).")
    parser.add_argument("--ext-in", default='.wav')
    parser.add_argument("--ext-out", default='.wav')
    return parser.parse_args()


def main():
    args = get_args()
    get_adaptive(args)


if __name__ == "__main__":
    main()
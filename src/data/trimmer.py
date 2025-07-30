import threading
import time
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import torchaudio

# Adaptive configuration
SAVE_QUEUE = queue.Queue(maxsize=12000)  # ~150GB RAM capacity
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
            queue_pct = (queue_size / 12000) * 100

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

            # Fast save (directories pre-created)
            torchaudio.save(out_path, chunk.cpu(), sr)
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
    name = Path(in_path).stem

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

        out_path = Path(out_root) / f'{name}.ogg'
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
    out_root = Path(args.rootout) / ""
    out_root.mkdir(exist_ok=True, parents=True)

    # Load manifest
    print("Loading manifest...")
    try:
        manifest = get_metadata(out_root, args.subset)
    except Exception:
        manifest = get_metadata2(out_root, args.subset)


    # Build items list
    items = defaultdict(list)
    if args.cont:
        existing_files = set(Path(p).name for p in
                             out_root.glob('**/*.ogg'))
        print(f"Found {len(existing_files)} existing files")

    for event_id, seg_no, start, end in manifest:
        segment = f"{event_id}_{seg_no}.ogg"
        if args.cont and segment in existing_files:
            continue

        lang, year = event_id.rsplit("_", 1)[1], event_id[:4]
        path = audio_root / lang / year / f"{event_id}.ogg"
        items[path.as_posix()].append((seg_no, float(start), float(end)))

    items_list = [(k, v, out_root.as_posix()) for k, v in items.items() if v]

    print(f"🚀 ADAPTIVE BEAST MODE:")
    print(f"  Queue: 12000 items (~150GB RAM)")
    print(f"  Initial savers: 16")
    print(f"  Max savers: 48")
    print(f"  Producers: {args.workers}")
    print(f"  Files to process: {len(items_list):,}")

    # Start adaptive system
    scaler = AdaptiveScaler(min_savers=16, max_savers=48)
    active_savers = 16

    # Start initial savers
    for i in range(16):
        saver = threading.Thread(target=adaptive_saver_worker, args=(i,), daemon=True)
        saver.start()
        saver_threads.append(saver)

    # Start adaptive monitor
    monitor = scaler.start_monitoring()

    # Progress monitoring
    def progress_monitor():
        while not shutdown_event.is_set():
            queue_size = SAVE_QUEUE.qsize()
            files_done = sum(1 for _ in out_root.glob('**/*.ogg'))
            print(f"📊 Queue: {queue_size:,}/12000 ({queue_size / 120:.1f}%), "
                  f"Files: {files_done:,}, Savers: {active_savers}")
            time.sleep(30)

    progress_thread = threading.Thread(target=progress_monitor, daemon=True)
    progress_thread.start()

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


# Usage
if __name__ == "__main__":
    args = get_args()
    get_adaptive(args)
"""Dedicated GPU worker: hot loop only (erosion+reconstruction+mean × 16).

CPU workers do background removal + texture normalization, then send
pre-processed (mask, pixels) to this worker via queue. The GPU worker
only does the fast cupy/cucim hot loop — no CPU-bound preprocessing.
"""

import multiprocessing as mp
import time


def gpu_worker_loop(gran_queue, result_store, result_lock, batch_size=500):
    """Pull pre-processed (req_id, mask, pixels) and run GPU hot loop only."""
    import numpy, warnings
    import cupy as cp
    import skimage.morphology
    import scipy.ndimage
    from cucim.skimage.morphology import erosion as gpu_erosion, reconstruction as gpu_reconstruction
    from cupyx.scipy.ndimage import mean as gpu_mean
    from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix

    cp.get_default_memory_pool().set_limit(size=2 * 1024**3)

    n_processed = 0
    t_start = time.perf_counter()
    ng = 16
    fp_gpu = cp.asarray(skimage.morphology.disk(1, dtype=bool))

    while True:
        try:
            item = gran_queue.get(timeout=0.5)
        except Exception:
            continue

        if item is None:
            elapsed = time.perf_counter() - t_start
            rate = n_processed / elapsed if elapsed > 0 else 0
            print(f"    [GPU worker] shutting down. {n_processed} processed ({rate:.0f}/s)", flush=True)
            return

        req_id, mask, pixels = item

        # GPU hot loop — pixels are already bg-removed + texture-normalized
        try:
            unique_labels = numpy.unique(mask)
            unique_labels = unique_labels[unique_labels > 0]

            if not unique_labels.any():
                result = {f"Granularity_{i}": numpy.zeros((0,)) for i in range(1, ng + 1)}
            else:
                range_ = numpy.arange(1, numpy.max(mask) + 1)
                current_mean = fix(scipy.ndimage.mean(pixels, mask, range_))
                start_mean = numpy.maximum(current_mean, numpy.finfo(float).eps)

                pixels_gpu = cp.asarray(pixels)
                ero_gpu = pixels_gpu.copy()
                labels_gpu = cp.asarray(mask)
                range_gpu = cp.asarray(range_)

                result = {}
                for gid in range(1, ng + 1):
                    ero_gpu = gpu_erosion(ero_gpu.copy(), footprint=fp_gpu)
                    rec_gpu = gpu_reconstruction(ero_gpu, pixels_gpu, footprint=fp_gpu)
                    new_mean = fix(cp.asnumpy(gpu_mean(rec_gpu, labels_gpu, range_gpu)))
                    result[f"Granularity_{gid}"] = (current_mean - new_mean) * 100 / start_mean

                del pixels_gpu, ero_gpu, labels_gpu, range_gpu
                cp.get_default_memory_pool().free_all_blocks()

        except (ValueError, IndexError):
            result = {}

        with result_lock:
            result_store[req_id] = result

        n_processed += 1
        if n_processed % 200 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"    [GPU worker] {n_processed} processed ({n_processed/elapsed:.0f}/s)", flush=True)


def _get_gpu_vram_mb():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True)
        return int(out.strip().split("\n")[0])
    except Exception:
        return 48000


def start_gpu_workers(n_workers=None, batch_size=500):
    """Start GPU worker processes. 2-3 workers is optimal (avoids CUDA context contention)."""
    if n_workers is None:
        n_workers = min(3, max(1, _get_gpu_vram_mb() // 2500))

    manager = mp.Manager()
    result_store = manager.dict()
    result_lock = manager.Lock()
    gran_queue = mp.Queue(maxsize=500)

    gpu_procs = []
    for i in range(n_workers):
        proc = mp.Process(
            target=gpu_worker_loop,
            args=(gran_queue, result_store, result_lock, batch_size),
            daemon=True,
        )
        proc.start()
        gpu_procs.append(proc)
    return gpu_procs, gran_queue, result_store, result_lock


def stop_gpu_workers(gpu_procs, gran_queue):
    for _ in gpu_procs:
        gran_queue.put(None)
    for proc in gpu_procs:
        proc.join(timeout=60)
        if proc.is_alive():
            proc.terminate()

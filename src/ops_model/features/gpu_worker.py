"""Dedicated GPU worker: bucketed batched hot loop.

Images are grouped by size (rounded to 32px) so each batch has minimal
padding. Edge-padded with 31px max margin prevents erosion boundary artifacts
over 16 iterations.

Architecture: N CPU workers → gran_queue → M GPU workers → result_store
"""

import multiprocessing as mp
import time
from collections import defaultdict


def gpu_worker_loop(gran_queue, result_store, result_lock, batch_size=8):
    """Accumulate images, group by size bucket, flush same-size batches to GPU."""
    import numpy, warnings
    import cupy as cp
    import skimage.morphology
    import scipy.ndimage
    from cucim.skimage.morphology import erosion as gpu_erosion, reconstruction as gpu_reconstruction
    from cupyx.scipy.ndimage import mean as gpu_mean
    from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix

    n_processed = 0
    t_start = time.perf_counter()
    ng = 16
    BUCKET = 32  # round dimensions to nearest multiple

    # Buckets: {(bucket_h, bucket_w): [(req_id, mask, pixels), ...]}
    buckets = defaultdict(list)

    def _bucket_key(h, w):
        return (((h + BUCKET - 1) // BUCKET) * BUCKET,
                ((w + BUCKET - 1) // BUCKET) * BUCKET)

    def _flush_bucket(items, bh, bw):
        """Process a same-size batch on GPU."""
        nonlocal n_processed
        if not items:
            return

        n = len(items)
        meta = []
        valid = []

        for i, (req_id, mask, pixels) in enumerate(items):
            unique_labels = numpy.unique(mask)
            unique_labels = unique_labels[unique_labels > 0]
            if not unique_labels.any():
                meta.append(None)
                with result_lock:
                    result_store[req_id] = {f"Granularity_{g}": numpy.zeros((0,)) for g in range(1, ng + 1)}
                continue
            range_ = numpy.arange(1, numpy.max(mask) + 1)
            current_mean = fix(scipy.ndimage.mean(pixels, mask, range_))
            start_mean = numpy.maximum(current_mean, numpy.finfo(float).eps)
            meta.append({"req_id": req_id, "range_": range_,
                         "current_mean": current_mean, "start_mean": start_mean,
                         "h": pixels.shape[0], "w": pixels.shape[1]})
            valid.append((len(meta) - 1, mask, pixels))

        if not valid:
            n_processed += n
            return

        # Pad to bucket size with edge values — max 31px padding
        nv = len(valid)
        pix_batch = numpy.zeros((nv, bh, bw), dtype=numpy.float64)
        mask_batch = numpy.zeros((nv, bh, bw), dtype=numpy.int32)
        for bi, (mi, mask, pixels) in enumerate(valid):
            h, w = pixels.shape
            pix_batch[bi] = numpy.pad(pixels, ((0, bh - h), (0, bw - w)), mode='edge')
            mask_batch[bi, :h, :w] = mask

        # Upload once
        pix_gpu = cp.asarray(pix_batch)
        ero_gpu = pix_gpu.copy()
        mask_gpu = cp.asarray(mask_batch)

        fp_3d = cp.zeros((1, 3, 3), dtype=bool)
        fp_3d[0] = cp.asarray(skimage.morphology.disk(1, dtype=bool))

        gpu_ranges = [cp.asarray(meta[mi]["range_"]) for bi, (mi, _, _) in enumerate(valid)]

        # Batched GPU loop with 2-thread overlap:
        # Thread 1 (main): GPU kernels (erosion+reconstruction+mean)
        # Thread 2: result assembly from previous iteration (overlaps with GPU)
        import threading
        all_results = [{} for _ in range(len(meta))]
        prev_means = None
        prev_gid = None
        assemble_done = threading.Event()
        assemble_done.set()  # initially "done"

        def _assemble(gid, means_list):
            for bi, (mi, _, _) in enumerate(valid):
                m = meta[mi]
                all_results[mi][f"Granularity_{gid}"] = (m["current_mean"] - means_list[bi]) * 100 / m["start_mean"]

        for gid in range(1, ng + 1):
            ero_gpu = gpu_erosion(ero_gpu.copy(), footprint=fp_3d)
            rec_gpu = gpu_reconstruction(ero_gpu, pix_gpu, footprint=fp_3d)

            # Compute means on GPU, download small results
            means = []
            for bi, (mi, _, _) in enumerate(valid):
                means.append(fix(cp.asnumpy(gpu_mean(rec_gpu[bi], mask_gpu[bi], gpu_ranges[bi]))))

            # Wait for previous assembly to finish, then start new one
            assemble_done.wait()
            if prev_means is not None:
                _assemble(prev_gid, prev_means)
            prev_means = means
            prev_gid = gid

        # Final assembly
        if prev_means is not None:
            _assemble(prev_gid, prev_means)

        # Store results
        with result_lock:
            for mi, m in enumerate(meta):
                if m is not None:
                    result_store[m["req_id"]] = all_results[mi]

        del pix_gpu, ero_gpu, mask_gpu
        for r in gpu_ranges:
            del r
        cp.get_default_memory_pool().free_all_blocks()

        n_processed += n
        if n_processed % 200 == 0 or n_processed < 50:
            elapsed = time.perf_counter() - t_start
            print(f"    [GPU worker] {n_processed} processed ({n_processed/elapsed:.0f}/s)", flush=True)

    def _flush_all():
        """Flush all non-empty buckets."""
        for (bh, bw), items in list(buckets.items()):
            if items:
                _flush_bucket(items, bh, bw)
        buckets.clear()

    # Main loop
    while True:
        try:
            item = gran_queue.get(timeout=0.1)
        except Exception:
            # Timeout — flush any full buckets
            for (bh, bw), items in list(buckets.items()):
                if items:
                    _flush_bucket(items, bh, bw)
                    buckets[(bh, bw)] = []
            continue

        if item is None:
            _flush_all()
            elapsed = time.perf_counter() - t_start
            rate = n_processed / elapsed if elapsed > 0 else 0
            print(f"    [GPU worker] shutting down. {n_processed} processed ({rate:.0f}/s)", flush=True)
            return

        req_id, mask, pixels = item
        h, w = pixels.shape
        key = _bucket_key(h, w)
        buckets[key].append(item)

        # Flush bucket when full
        if len(buckets[key]) >= batch_size:
            _flush_bucket(buckets[key], key[0], key[1])
            buckets[key] = []


def _get_gpu_vram_mb():
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True)
        return int(out.strip().split("\n")[0])
    except Exception:
        return 48000


def start_gpu_workers(n_workers=None, batch_size=8):
    """Start GPU worker processes."""
    if n_workers is None:
        n_workers = min(8, max(1, _get_gpu_vram_mb() // 2500))

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

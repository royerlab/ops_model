"""GPU granularity support functions.

background_remove_cpu: CPU background removal (identical to cp_measure's).
    Called by CPU workers before sending to GPU queue.

get_granularity_gpu: Full GPU granularity (bg removal on CPU + hot loop on GPU).
    Used for standalone/validation runs.
"""

import numpy
import scipy.ndimage
import skimage.morphology
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix

try:
    import cupy as cp
    from cucim.skimage.morphology import erosion as gpu_erosion, reconstruction as gpu_reconstruction
    from cupyx.scipy.ndimage import mean as gpu_mean
    HAS_GPU = True
except ImportError:
    HAS_GPU = False


def background_remove_cpu(pixels, mask, image_sample_size=0.25, element_size=10):
    """CPU background removal — identical to cp_measure's get_granularity.

    Called by CPU workers to pre-process images before sending to GPU queue.
    """
    new_shape = numpy.array(pixels.shape)
    pixels = pixels.copy()
    mask = mask.copy()

    if image_sample_size < 1:
        back_shape = new_shape * image_sample_size
        i, j = (
            numpy.mgrid[0 : back_shape[0], 0 : back_shape[1]].astype(float)
            / image_sample_size
        )
        back_pixels = scipy.ndimage.map_coordinates(pixels, (i, j), order=1)
        back_mask = scipy.ndimage.map_coordinates(mask.astype(float), (i, j)) > 0.9
    else:
        back_pixels = pixels
        back_mask = mask
        back_shape = new_shape

    fp = skimage.morphology.disk(element_size, dtype=bool)
    bpm = numpy.zeros_like(back_pixels)
    bpm[back_mask == 1] = back_pixels[back_mask == 1]
    bp = skimage.morphology.erosion(bpm, footprint=fp)
    bpm2 = numpy.zeros_like(bp)
    bpm2[back_mask == 1] = bp[back_mask == 1]
    bp = skimage.morphology.dilation(bpm2, footprint=fp)

    if image_sample_size < 1:
        i, j = numpy.mgrid[0 : new_shape[0], 0 : new_shape[1]].astype(float)
        i *= float(back_shape[0] - 1) / float(new_shape[0] - 1)
        j *= float(back_shape[1] - 1) / float(new_shape[1] - 1)
        bp = scipy.ndimage.map_coordinates(bp, (i, j), order=1)

    pixels -= bp
    pixels[pixels < 0] = 0
    return pixels


def get_granularity_gpu(
    mask: numpy.ndarray,
    pixels: numpy.ndarray,
    subsample_size: float = 0.25,
    image_sample_size: float = 0.25,
    element_size: int = 10,
    granular_spectrum_length: int = 16,
) -> dict[str, float]:
    """Full GPU granularity: CPU background removal + GPU hot loop.

    For standalone/validation use. In production, CPU workers call
    background_remove_cpu separately and GPU workers only run the hot loop.
    """
    # Background removal on CPU
    pixels = background_remove_cpu(pixels, mask, image_sample_size, element_size)

    # Hot loop on GPU
    ng = granular_spectrum_length
    unique_labels = numpy.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]

    if not unique_labels.any():
        return {f"Granularity_{i}": numpy.zeros((0,)) for i in range(1, ng + 1)}

    range_ = numpy.arange(1, numpy.max(mask) + 1)
    current_mean = fix(scipy.ndimage.mean(pixels, mask, range_))
    start_mean = numpy.maximum(current_mean, numpy.finfo(float).eps)

    pixels_gpu = cp.asarray(pixels)
    ero_gpu = pixels_gpu.copy()
    labels_gpu = cp.asarray(mask)
    range_gpu = cp.asarray(range_)
    fp_gpu = cp.asarray(skimage.morphology.disk(1, dtype=bool))

    results = {}
    for gid in range(1, ng + 1):
        ero_gpu = gpu_erosion(ero_gpu.copy(), footprint=fp_gpu)
        rec_gpu = gpu_reconstruction(ero_gpu, pixels_gpu, footprint=fp_gpu)
        new_mean = fix(cp.asnumpy(gpu_mean(rec_gpu, labels_gpu, range_gpu)))
        results[f"Granularity_{gid}"] = (current_mean - new_mean) * 100 / start_mean

    del pixels_gpu, ero_gpu, labels_gpu, range_gpu, fp_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return results

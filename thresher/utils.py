__all__ = ["load_image", "trim_image", "centroid_image", "unravel_scene",
            "unravel_psf", "timer"]

import time
import logging

import numpy as np
from scipy.signal import fftconvolve as convolve
import pyfits


#
# Image/filesystem utilities
#
def load_image(fn, hdu=0, dtype=float):
    """
    Get the image data from a FITS file.

    """
    logging.info("Loading data file: {0}".format(fn))
    f = pyfits.open(fn)
    data = np.array(f[hdu].data, dtype=dtype)
    f.close()
    return data


def trim_image(image, size):
    """
    Trim an image to be square with shape `(size, size)`.

    """
    shape = np.array(image.shape)
    assert np.all(shape >= size), \
            "You can't 'trim' an image to be larger than it was!"
    mn = (0.5 * (shape - np.array([size, size]))).astype(int)
    return image[mn[0]:mn[0] + size, mn[1]:mn[1] + size]


def centroid_image(image, size, scene=None, coords=None, mask=None):
    """
    Centroid an image based on the current scene by projecting and
    convolving.

    """
    if coords is None:
        assert scene is not None

        # Projected convolutions.
        x0 = np.argmax(convolve(np.sum(image, axis=1), np.sum(scene, axis=1),
            mode="valid"))
        y0 = np.argmax(convolve(np.sum(image, axis=0), np.sum(scene, axis=0),
            mode="valid"))
        center = (x0, y0)

        # Deal with shapes.
        s_dim = (np.array(scene.shape) - 1) / 2
        center = np.array(center) + s_dim
        logging.info("Got image center: {0}".format(center))
    else:
        center = np.array(coords)

    center = center.astype(int)

    # Deal with the edges of the images.
    mn = np.floor(center - 0.5 * size).astype(int)
    mn_r = np.zeros_like(center)
    mn_r[mn < 0] = -mn[mn < 0]
    mn[mn < 0] = 0

    mx = np.floor(center + 0.5 * size).astype(int)
    m = mx > np.array(image.shape)
    mx_r = size * np.ones_like(center)
    mx_r[m] -= mx[m] - np.array(image.shape)[m]
    mx[m] = np.array(image.shape)[m]

    # Build the mask for the output.
    final_mask = np.zeros((size, size))
    final_mask[mn_r[0]:mx_r[0], mn_r[1]:mx_r[1]] = 1.0
    if mask is not None:
        final_mask[mn_r[0]:mx_r[0], mn_r[1]:mx_r[1]] *= \
                mask[mn[0]:mx[0], mn[1]:mx[1]]

    # Build the result.
    result = np.zeros((size, size))
    result[mn_r[0]:mx_r[0], mn_r[1]:mx_r[1]] = image[mn[0]:mx[0], mn[1]:mx[1]]

    return center, result, final_mask.astype(float)


#
# Index gymnastics
#
def xy2index(shape, x, y):
    """
    Go from x,y position in a 2-d numpy array to one-d index in the
    flattened 1-d array.

    """
    return (x * shape[1] + y)


def index2xy(shape, i):
    """Inverse of `xy2index()`."""
    return ((i / shape[1]), (i % shape[1]))


def unravel_scene(S, P):
    """
    Unravel the scene object to prepare for the least squares problem.

    ## Arguments

    * `scene` (numpy.ndarray): The 2-D scene object.
    * `P` (int): The half-width of the PSF object.

    """
    # Work out all the dimensions first.
    D = S - 2 * P

    psf_shape = 2 * P + 1
    psf_size = psf_shape ** 2

    data_shape = (D, D)
    data_size = D ** 2

    # The layout of the scene.
    scene = np.arange(S ** 2).reshape((S, S))

    # Build the output array.
    result = np.empty((data_size, psf_size), dtype=int)

    # Loop over the valid data region.
    for k in xrange(data_size):
        dx, dy = index2xy(data_shape, k)
        grid = np.meshgrid(dy + np.arange(psf_shape),
                           dx + np.arange(psf_shape))[::-1]
        result[k, :] = scene[grid].flatten()

    return result


def unravel_psf(S, P):
    D = S - 2 * P
    data_size = D ** 2
    psf_size = (2 * P + 1) ** 2

    psfX, psfY = index2xy((2 * P + 1,) * 2, np.arange(psf_size))

    rows = np.zeros(data_size * psf_size, dtype=int)
    cols = np.zeros_like(rows)

    for k in range(data_size):
        dx, dy = index2xy((D, D), k)
        s = slice(k * psf_size, (k + 1) * psf_size)
        rows[s] = k
        cols[s] = xy2index((S, S), psfX + dx, psfY + dy)

    return rows, cols


def timer(f, lf=None):
    """
    A decorator used for some simple profiling.

    """
    if lf is None:
        lf = logging.info

    def _func(*args, **kwargs):
        t = time.time()
        r = f(*args, **kwargs)
        dt = time.time() - t
        txt = "{0} took {1} seconds".format(f.__name__, dt)
        lf(txt)
        return r

    return _func

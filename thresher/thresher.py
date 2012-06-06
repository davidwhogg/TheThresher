"""
This file is part of The Thresher.

"""

__all__ = ["Scene", "load_image", "centroid_image"]

import os
import logging
import gc
import glob

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.signal import fftconvolve as convolve
import scipy.optimize as op

import pyfits

from utils import dfm_time


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
    scene_size = S ** 2
    psf_size = (2 * P + 1) ** 2

    psfX, psfY = index2xy((2 * P + 1,) * 2, np.arange(psf_size))

    rows = np.zeros(data_size * psf_size, dtype=int)
    cols = np.zeros_like(rows)

    for k in range(data_size):
        dx, dy = index2xy((D, D), k)
        s = slice(k * psf_size, (k + 1) * psf_size)
        rows[s] = k
        cols[s] = xy2index((S, S), psfX + dx, psfY + dy)

    # add entries for old-scene-based regularization
    rows = np.append(rows, np.arange(data_size, data_size + scene_size))
    cols = np.append(cols, np.arange(scene_size))

    return rows, cols


def load_image(fn):
    """
    Get the image data from a FITS file.

    """
    logging.info("Loading data file: {0}".format(fn))
    f = pyfits.open(fn)
    data = np.array(f[0].data, dtype=float)
    f.close()
    return data


def centroid_image(image, scene, size, coords=None):
    """
    Centroid an image based on the current scene by projecting and
    convolving.

    """
    if coords is None:
        convolved = convolve(image, scene, mode="valid")
        center = np.unravel_index(convolved.argmax(), convolved.shape)

        # Deal with shapes.
        s_dim = (np.array(scene.shape) - 1) / 2
        center = np.array(center) + s_dim
    else:
        center = np.array(coords)

    logging.info("Got image center: {0}".format(center))

    # Deal with the edges of the images.
    mn = center - size / 2
    mn_r = np.zeros_like(center)
    mn_r[mn < 0] = -mn[mn < 0]
    mn[mn < 0] = 0

    mx = center + size / 2
    delta = mx - np.array(image.shape)
    m = mx > np.array(image.shape)
    mx_r = size * np.ones_like(center)
    mx_r[m] -= mx[m] - np.array(image.shape)[m]
    mx[m] = np.array(image.shape)[m]

    # Build the result.
    result = np.zeros((size, size))
    result[mn_r[0]:mx_r[0], mn_r[1]:mx_r[1]] = image[mn[0]:mx[0], mn[1]:mx[1]]

    return center, result


def trim_image(image, size):
    xmin = int(0.5 * (image.shape[0] - size))
    ymin = int(0.5 * (image.shape[0] - size))
    return image[xmin:xmin + size, ymin:ymin + size]


def _worker(scene, data):
    # Do the inference.
    psf = scene._infer_psf(data)
    new_scene = scene._infer_scene(data)
    return psf, new_scene


class Scene(object):
    """
    A `Scene` object describes and learns the "true" image from a lucky
    imaging data stream.

    """
    def __init__(self, imgglob, outdir="", psf_hw=13, size=None, sky=0.,
            kernel=None, psfreg=100., sceneL2=1. / 64.):
        # All the metadata.
        self.glob = imgglob
        self.outdir = os.path.abspath(outdir)
        self.sky = sky
        self.inferred_sky = 0

        self.img_number = 0
        self.pass_number = 0

        # Get the image list.
        entries = glob.glob(self.glob)
        self.image_list = [os.path.abspath(e) for e in sorted(entries)]
        assert len(self.image_list) > 0, \
                "There are no files matching '{0}'".format(imgglob)

        # Set the scene size.
        image = self.first_image
        assert size <= min(image.shape), \
                "The scene size must be <= to the data size."
        self.size = size

        # L2 norm weights.
        self.psfreg = psfreg
        self.sceneL2 = sceneL2

        # Initialize the PSF image as a delta function.
        self.psf_hw = psf_hw
        pd = 2 * psf_hw + 1
        self.psf = np.zeros((pd, pd))
        self.psf[psf_hw, psf_hw] = 1.

        if kernel is None:
            # Make the PSF convolution kernel here. There's a bit of black
            # MAGIC that could probably be fixed. The kernel is implicitly a
            # `sigma = 1. pix` Gaussian.
            self.kernel = np.exp(-0.5 * (np.arange(-5, 6)[:, None] ** 2
                + np.arange(-5, 6)[None, :] ** 2))
            self.kernel /= np.sum(self.kernel)
        else:
            self.kernel = kernel

        # Initialize the scene as a centered Gaussian.
        s = self.size + 2 * psf_hw
        x = np.linspace(-0.5 * s, 0.5 * s, s) ** 2
        r = np.sqrt(x[:, None] + x[None, :])
        self.scene = np.exp(-0.5 * r) / np.sqrt(2 * np.pi)

    def setup(self):
        # HACK
        self.size += 2 * self.psf_hw

        # Run lucky imaging. MAGIC: co-add the top 1 percent.
        images, ranks, scene = self.run_lucky(top_percent=1)
        self.scene = scene - np.median(scene)

        # HACK part 2.
        self.size -= 2 * self.psf_hw

        # Calculate the mask that we will use to unravel the scene.
        self.scene_mask = unravel_scene(len(self.scene), self.psf_hw)

        # And the PSF.
        self.psf_rows, self.psf_cols = \
                unravel_psf(len(self.scene), self.psf_hw)

    def run_lucky(self, do_coadd=True, top=None, top_percent=None):
        """
        Run traditional lucky imaging on a stream of data.

        ## Keyword Arguments

        * `do_coadd` (bool): Return the coadded image?
        * `top` (int): How many images should be coadded.

        """
        Ndata = len([f for f in self.image_list])

        data = np.empty((Ndata, self.size, self.size))

        self.coords = {}
        results = {}
        for n, fn in enumerate(self.image_list):
            image = load_image(fn)
            coords, result = centroid_image(image, self.scene, self.size)
            self.coords[fn] = coords

            data[n] = result

            results[fn] = (n, float(result[self.size / 2, self.size / 2]))

        # Sort by brightest centroided pixel.
        ranked = sorted(results, reverse=True, key=lambda k: results[k][-1])
        fns, ranks = [], []
        for k in ranked:
            fns.append(k)
            ranks.append(results[k][-1])

        if not do_coadd:
            return fns, ranks

        # Do the co-add.
        if top is None and top_percent is None:
            top = len(ranked)
        elif top_percent is not None:
            top = max(1, int(top_percent * 0.01 * len(ranked)))

        final = np.zeros((self.size, self.size))
        for i, k in enumerate(fns[:top]):
            final += data[results[k][0]] / float(top)

        return fns, ranks, final

    @property
    def first_image(self):
        """Get the data for the first image"""
        return load_image(self.image_list[0])

    def run_inference(self, basepath=None, npasses=5, current_pass=0,
            current_img=None, do_centroiding=True, subtract_median=False,
            use_nn=True):
        """
        Run the full inference on the dataset.

        ## Keyword Arguments

        * `npasses` (int): The number of passes to run.
        * `current_pass` (int): The pass number to start at. This is used
          for restarting.
        * `current_img` (int): The image number to start at on the first
          pass through the data. This is used for restarting.

        """
        if basepath is not None:
            self.basepath = basepath

        if current_img is None:
            current_img = self.img_number

        N = len([i for i in self.image_list])

        for self.pass_number in xrange(current_pass, npasses):
            for self.img_number, self.fn in enumerate(self.image_list):
                if self.img_number >= current_img:
                    image = load_image(self.fn)
                    coords, data = \
                            centroid_image(image, self.scene, self.size,
                                    coords=self.coords[self.fn])

                    data += self.sky - np.min(data)

                    # If it's the first pass, `alpha` should decay and we
                    # should use _non-negative_ optimization.
                    if self.pass_number == 0:
                        alpha = min(2. / (1 + self.img_number), 0.25)
                        nn = use_nn  # True
                    else:
                        alpha = 2. / N  # MAGIC: 2.
                        nn = False

                    # Do the inference.
                    self.old_scene = np.array(self.scene)
                    self.psf, self.this_scene = _worker(self, data)
                    self.scene = (1 - alpha) * self.scene \
                                            + alpha * self.this_scene
                    if nn:
                        self.scene[self.scene < 0] = 0.0

                    # WTF?!?
                    gc.collect()

                    # Subtract the median.
                    if subtract_median:
                        self.scene -= np.median(self.scene)

                    # Save the output.
                    self._save_state(data)

            # After one full pass through the data, make sure that the index
            # of the zeroth image is reset. We only want to start from this
            # image on the first pass through the data when we're restarting.
            current_img = 0

    @dfm_time
    def _infer_psf(self, data, useL2=False):
        """
        Take data and a current belief about the scene; infer the psf for
        this image given the scene.  This code infers a sky level
        simultaneously.  That might seem like a detail, but it matters.

        """
        # Sort out the dimensions.
        P = 2 * self.psf_hw + 1
        psf_size = P ** 2

        D = data.shape[0]
        data_size = D ** 2

        # Build scene matrix from kernel-convolved scene.
        kc_scene = convolve(self.kernel, self.scene, mode="same")

        scene_matrix = np.zeros((data_size + 1, psf_size + 1))

        # Unravel the scene.
        scene_matrix[:data_size, :psf_size] = \
                                    kc_scene.flatten()[self.scene_mask]

        # Add the sky.
        scene_matrix[:data.size, psf_size] = 1

        scene_matrix[data_size, :psf_size] = self.psfreg * 1.
        data_vector = np.append(data.flatten(), self.psfreg * np.ones(1))

        # Infer the new PSF.
        new_psf, rnorm = op.nnls(scene_matrix, data_vector)

        # Save the inferred sky level.
        self.inferred_sky = new_psf[-1]

        # Do the index gymnastics to get the correct inferred PSF.
        # NOTE: here, we're first dropping the sky and then reversing the
        # PSF object because of the way that the `convolve` function is
        # defined.
        return new_psf[:-1][::-1].reshape((P, P))

    @dfm_time
    def _infer_scene(self, data):
        """
        Take data and a current belief about the PSF; infer the scene for
        this image given the PSF.

        """
        # NOTE: the PSF is reversed here.
        vals = np.zeros((data.size, self.psf.size)) \
                + self.psf.flatten()[None, ::-1]
        vals = vals.flatten()

        # Append the identity for the L2 norm.
        vals = np.append(vals, self.sceneL2 + np.zeros(self.scene.size))

        psf_matrix = csr_matrix((vals, (self.psf_rows, self.psf_cols)),
                shape=(data.size + self.scene.size, self.scene.size))

        # Infer scene and return
        data_vector = np.append(data.flatten(), np.zeros(self.scene.size))
        results = lsqr(psf_matrix, data_vector)

        new_scene = results[0].reshape(self.scene.shape)

        return new_scene

    def _save_state(self, data):
        _id = "{0:d}-{1:08}".format(self.pass_number, self.img_number)
        outfn = os.path.join(self.outdir, _id + ".fits")

        hdus = [pyfits.PrimaryHDU(data),
                pyfits.ImageHDU(self.this_scene),
                pyfits.ImageHDU(self.scene),
                pyfits.ImageHDU(self.psf),
                pyfits.ImageHDU(self.kernel)]

        hdus[0].header.update("datafn", self.fn)
        hdus[0].header.update("size", self.size)
        hdus[0].header.update("pass", self.pass_number)
        hdus[0].header.update("image", self.img_number)
        hdus[0].header.update("sky", self.inferred_sky)
        hdus[1].header.update("status", "old")
        hdus[2].header.update("status", "new")

        pyfits.HDUList(hdus).writeto(outfn, clobber=True)

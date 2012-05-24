"""
This file is part of The Thresher.

"""

__all__ = ["Scene", "load_image", "centroid_image"]

import os
import logging
import gc

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.signal import fftconvolve as convolve
import scipy.optimize as op

import pyfits


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


def centroid_image(image, scene, size):
    """
    Centroid an image based on the current scene by projecting and
    convolving.

    """
    ip0, ip1 = np.sum(image, axis=0), np.sum(image, axis=1)
    sp0, sp1 = np.sum(scene, axis=0), np.sum(scene, axis=1)

    y0 = np.argmax(convolve(ip0, sp0, mode="valid"))
    x0 = np.argmax(convolve(ip1, sp1, mode="valid"))

    xmin = int(x0 + 0.5 * (scene.shape[0] - size))
    ymin = int(y0 + 0.5 * (scene.shape[1] - size))

    logging.info("Got image center {0}, {1}"
            .format(xmin + 0.5 * size, ymin + 0.5 * size))

    return image[xmin:xmin + size, ymin:ymin + size]


def trim_image(image, size):
    xmin = int(0.5 * (image.shape[0] - size))
    ymin = int(0.5 * (image.shape[0] - size))
    return image[xmin:xmin + size, ymin:ymin + size]


class Scene(object):
    """
    A `Scene` object describes and learns the "true" image from a lucky
    imaging data stream.

    """
    def __init__(self, basepath=".", outdir="", psf_hw=13, size=None, sky=0.,
            initial_scene=None, kernel=None, psfL2=0.25, sceneL2=1. / 64.):
        # All the metadata.
        self.basepath = os.path.abspath(basepath)
        self.outdir = os.path.abspath(outdir)
        self.psf_hw = psf_hw
        self.sky = sky

        self.img_number = 0
        self.pass_number = 0

        # Set the scene size.
        image = self.first_image
        if size is None:
            self.size = min(image.shape)
        else:
            assert size <= min(image.shape), \
                    "The scene size must be <= to the data size."
            self.size = size

        # L2 norm weights.
        self.psfL2 = psfL2
        self.sceneL2 = sceneL2

        # Initialize the PSF image as a delta function.
        pd = 2 * psf_hw + 1
        self.psf = np.zeros((pd, pd))
        self.psf[psf_hw, psf_hw] = 1.

        # If it's not provided, initialize the scene to something reasonable.
        if initial_scene is None:
            # Initialize the scene as a centered Gaussian.
            x = np.linspace(-0.5 * self.size, 0.5 * self.size, self.size) ** 2
            r = np.sqrt(x[:, None] + x[None, :])
            self.scene = np.exp(-0.5 * r) / np.sqrt(2 * np.pi)
            self.scene = convolve(self.scene, self.psf, mode="full")
        else:
            self.scene = np.array(initial_scene)

        if kernel is None:
            # Make the PSF convolution kernel here. There's a bit of black
            # MAGIC that could probably be fixed. The kernel is implicitly a
            # `sigma = 1. pix` Gaussian.
            self.kernel = np.exp(-0.5 * (np.arange(-5, 6)[:, None] ** 2
                + np.arange(-5, 6)[None, :] ** 2))
            self.kernel /= np.sum(self.kernel)
        else:
            self.kernel = kernel

        # Calculate the mask that we will use to unravel the scene.
        self.scene_mask = unravel_scene(len(self.scene), self.psf_hw)

        # And the PSF.
        self.psf_rows, self.psf_cols = \
                unravel_psf(len(self.scene), self.psf_hw)

    @property
    def image_list(self):
        entries = os.listdir(self.basepath)
        for e in sorted(entries):
            if os.path.splitext(e)[1] == ".fits":
                yield os.path.join(self.basepath, e)

    @property
    def first_image(self):
        """Get the data for the first image"""
        return load_image(self.image_list.next())

    def initialize_with_data(self):
        """
        Get a scene that can be used for initialization based on the _first_
        piece of data.

        """
        image = self.first_image
        data = trim_image(image, self.size) + self.sky
        self.scene = np.zeros_like(self.scene)
        self.scene[self.psf_hw:-self.psf_hw, self.psf_hw:-self.psf_hw] = \
                data

    def run_inference(self, basepath=None, npasses=5, current_pass=0,
            current_img=None, do_centroiding=True):
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

        for self.pass_number in xrange(current_pass, npasses):
            for self.img_number, self.fn in enumerate(self.image_list):
                if self.img_number >= current_img:
                    image = load_image(self.fn)
                    if do_centroiding:
                        data = centroid_image(image, self.scene, self.size)
                    else:
                        data = trim_image(image, self.size)

                    data += self.sky

                    # If it's the first pass, `alpha` should decay and we
                    # should use _non-negative_ optimization.
                    if self.pass_number == 0:
                        alpha = min(2. / (1 + self.img_number), 0.25)
                        nn = True
                    else:
                        # self.scene -= np.median(self.scene)  # Hackeroni?
                        alpha = 2. / 300.  # Hack-o-rama?
                        nn = False

                    # On the first pass on the first image, normalize so that
                    # the PSF sums to ~1.
                    if self.img_number == 0 and self.pass_number == 0:
                        self._infer_psf(data)

                        # Properly normalize the PSF.
                        norm = np.sum(self.psf)
                        self.psf /= norm

                        # Re-infer the scene.
                        self.scene = self._infer_scene(data)

                    # Do the inference.
                    self._inference_step(data, alpha, nn)

                    # Save the output.
                    self._save_state(data)

            # After one full pass through the data, make sure that the index
            # of the zeroth image is reset. We only want to start from this
            # image on the first pass through the data when we're restarting.
            current_img = 0

    def _inference_step(self, data, alpha, nn):
        """
        Concatenation of `_infer_psf()` and `_infer_scene()`.  Applies
        `alpha` times the newly inferred scene to `(1. - alpha)` times the
        old scene.

        ## Arguments

        * `data` (numpy.ndarray): The new image.
        * `alpha` (float): The weight of the new scene.
        * `nn` (bool): Should the update enforce non-negativity?

        """
        assert 0 < alpha <= 1
        self._infer_psf(data)
        self.old_scene = np.array(self.scene)
        self.this_scene = self._infer_scene(data)
        self.scene = (1 - alpha) * self.scene \
                                + alpha * self.this_scene

        # self.scene -= np.median(self.scene)  # Crazy hackishness!

        if nn:
            self.scene[self.scene < 0] = 0.0

        gc.collect()

    def _infer_psf(self, data):
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
        kc_scene = self.scene  # convolve(self.scene, self.kernel, mode="same")
        scene_matrix = np.zeros((data_size + psf_size, psf_size + 1))

        # Unravel the scene.
        scene_matrix[:data_size, :psf_size] = \
                                    kc_scene.flatten()[self.scene_mask]

        # Add the sky.
        scene_matrix[:data.size, psf_size] = 1

        # And the L2 regularization.
        scene_matrix[data_size:, :psf_size] = self.psfL2 * np.eye(psf_size)

        # Build the data vector.
        data_vector = np.append(data.flatten(), np.zeros(psf_size))

        # Infer the new PSF.
        new_psf, rnorm = op.nnls(scene_matrix, data_vector)

        # Do the index gymnastics to get the correct inferred PSF.
        # NOTE: here, we're first dropping the sky and then reversing the
        # PSF object because of the way that the `convolve` function is
        # defined.
        self.psf = new_psf[:-1][::-1].reshape((P, P))

    def _infer_scene(self, data):
        """
        Take data and a current belief about the PSF; infer the scene for
        this image given the PSF.

        """
        # The dimensions.
        P = 2 * self.psf_hw + 1
        psf_size = P ** 2

        D = data.shape[0]
        data_size = D ** 2

        S = self.scene.shape[0]
        scene_size = S ** 2

        kc_psf = self.psf  # convolve(self.psf, self.kernel, mode="same")

        # NOTE: the PSF is reversed here.
        vals = np.zeros((data_size, psf_size)) + kc_psf.flatten()[None, ::-1]
        vals = vals.flatten()

        # Append the identity for the L2 norm.
        vals = np.append(vals, np.ones(scene_size))

        psf_matrix = csr_matrix((vals, (self.psf_rows, self.psf_cols)),
                shape=(data_size + scene_size, scene_size))

        # Infer scene and return
        data_vector = np.append(data.flatten(), np.zeros(scene_size))
        results = lsqr(psf_matrix, data_vector)

        new_scene = results[0].reshape((S, S))
        # new_scene -= np.median(new_scene)

        # logging.info("Got scene {0}, Min: {1}, Median: {2}, Max: {3}"
        #         .format(newScene.shape, newScene.min(), np.median(newScene),
        #             newScene.max()))

        # gc.collect()

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
        hdus[1].header.update("status", "old")
        hdus[2].header.update("status", "new")

        pyfits.HDUList(hdus).writeto(outfn, clobber=True)
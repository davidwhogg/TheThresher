"""
This file is part of The Thresher.

"""

__all__ = ["Scene"]

import os
import gc
import glob

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.signal import fftconvolve as convolve
import scipy.optimize as op

import pyfits

import utils


def _worker(scene, data):
    # Do the inference.
    psf = scene.infer_psf(data)
    dlds = scene.get_dlds(data)
    return psf, dlds


class Scene(object):
    """
    A `Scene` object describes and learns the "true" image from a lucky
    imaging data stream.

    """
    def __init__(self, imgglob, outdir="", psf_hw=13, size=None, sky=0.,
            kernel=None, psfreg=100., sceneL2=1. / 64.):
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

    def setup(self):
        # Run lucky imaging. MAGIC: co-add the top 1 percent.
        images, ranks, self.coords, scene = self.run_tli(self.image_list,
                size=self.size + 2 * self.psf_hw, top_percent=1)
        self.image_list = images
        scene = scene[1]
        self.scene = scene - np.median(scene)

        # Calculate the mask that we will use to unravel the scene.
        self.scene_mask = utils.unravel_scene(len(self.scene), self.psf_hw)

        # And the PSF.
        self.psf_rows, self.psf_cols = \
                utils.unravel_psf(len(self.scene), self.psf_hw)

    def run_tli(self, image_list, size=None, top=None, top_percent=None):
        """
        Run traditional lucky imaging on a stream of data.

        ## Keyword Arguments

        * `top` (int): How many images should be coadded.

        """
        Ndata = len(image_list)

        # These will be instantiated the first time through the loop.
        stack = None
        scene = None

        # Calculate the centers and ranks of the images.
        centers = {}
        ranks = {}
        for n, fn in enumerate(image_list):
            img = utils.load_image(fn)
            if size is None:
                size = np.max(img.shape)
            if stack is None:
                # Allocate a stack for the centroided images.
                stack = np.empty((Ndata, size, size))
            if scene is None:
                x = np.linspace(-0.5 * size, 0.5 * size, size) ** 2
                r = np.sqrt(x[:, None] + x[None, :])
                scene = 0.5 * np.exp(-0.5 * r) / np.pi

            center, result, mask = utils.centroid_image(img, scene, size)
            centers[fn] = center
            stack[n] = result
            ranks[fn] = (n, float(result[size / 2, size / 2]))

        # Sort by brightest centroided pixel.
        ranked = sorted(ranks, reverse=True, key=lambda k: ranks[k][1])
        fns, values = [], []
        for i in range(len(ranked)):
            k = ranked[i]
            fns.append(k)
            ranked.append(ranks[k][-1])

        # Do the co-add.
        if top is None and top_percent is None:
            top = len(ranked)
        elif top_percent is not None:
            top = max(1, int(top_percent * 0.01 * len(ranked)))

        top = np.atleast_1d(top)
        final = np.zeros((len(top) + 1, size, size))
        for j, t in enumerate(np.atleast_1d(top)):
            for i, k in enumerate(fns[:t]):
                final[j] += stack[ranks[k][0]] / float(t)
        for i, k in enumerate(fns):
            final[-1] += stack[ranks[k][0]] / float(t)

        return fns, ranks, centers, final

    def do_update(self, fn, alpha, median=True, nn=False):
        image = utils.load_image(fn)
        coords, data, mask = utils.centroid_image(image, self.scene, self.size,
                        coords=self.coords.get(self.fn, None))

        # Piston the data for numerical stability.
        data += self.sky - np.min(data)

        # Do the inference.
        self.old_scene = np.array(self.scene)
        self.psf, self.dlds = _worker(self, data)
        self.scene = self.scene + alpha * self.dlds

        # WTF?!?
        gc.collect()

        # Apply some serious HACKS!
        if median:
            self.scene -= np.median(self.scene)
        if nn:
            self.scene[self.scene < 0] = 0.0
        return data

    def run_inference(self, basepath=None, npasses=5,
            subtract_median=False, use_nn=True, top=None):
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

        N = len([i for i in self.image_list])

        iml = self.image_list
        if top is not None:
            iml = iml[:int(top)]
        for self.pass_number in xrange(npasses):
            if self.pass_number > 0:
                np.random.shuffle(iml)
            for self.img_number, self.fn in enumerate(iml):
                # If it's the first pass, `alpha` should decay and we
                # should use _non-negative_ optimization.
                if self.pass_number == 0:
                    alpha = min(2. / (1 + self.img_number), 0.25)
                    nn = use_nn  # True
                else:
                    alpha = 2. / N  # MAGIC: 2.
                    nn = False

                data = self.do_update(self.fn, alpha, median=subtract_median,
                        nn=nn)
                self.save(data)

    def get_psf_matrix(self, L2=True):
        S = len(self.scene)
        P = self.psf_hw

        D = S - 2 * P
        data_size = D ** 2
        scene_size = S ** 2
        psf_size = (2 * P + 1) ** 2

        # NOTE: the PSF is reversed here.
        vals = np.zeros((data_size, psf_size)) \
                + self.psf.flatten()[None, ::-1]
        vals = vals.flatten()

        rows, cols = self.psf_rows, self.psf_cols
        shape = [data_size, scene_size]

        # Append the identity for the L2 norm.
        if L2:
            vals = np.append(vals, self.sceneL2 + np.zeros(self.scene.size))
            rows = np.append(rows, np.arange(data_size,
                data_size + scene_size))
            cols = np.append(cols, np.arange(scene_size))

            shape[0] += scene_size

        psf_matrix = csr_matrix((vals, (rows, cols)), shape=shape)

        return psf_matrix

    @utils.timer
    def infer_psf(self, data, useL2=False):
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

    @utils.timer
    def infer_scene(self, data):
        """
        Take data and a current belief about the PSF; infer the scene for
        this image given the PSF.

        """
        # Infer scene and return
        data_vector = np.append(data.flatten(), np.zeros(self.scene.size))
        results = lsqr(self.get_psf_matrix(L2=True), data_vector)

        new_scene = results[0].reshape(self.scene.shape)

        return new_scene

    @utils.timer
    def get_dlds(self, data):
        """
        Take data and a current belief about the PSF; compute the gradient of
        log-likelihood wrt scene.

        """
        psf_matrix = self.get_psf_matrix(L2=False)

        dlds = psf_matrix.transpose().dot(data.flatten() -
                psf_matrix.dot(self.scene.flatten()))
        dlds = dlds.reshape(self.scene.shape)

        return dlds

    def save(self, data):
        _id = "{0:d}-{1:08}".format(self.pass_number, self.img_number)
        outfn = os.path.join(self.outdir, _id + ".fits")

        hdus = [pyfits.PrimaryHDU(data),
                pyfits.ImageHDU(self.dlds),
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

"""
This file is part of The Thresher.

"""

__all__ = ["Scene"]

import os
import gc
import logging

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.signal import fftconvolve as convolve
import scipy.optimize as op

import pyfits

import utils


class Scene(object):
    """
    A `Scene` object describes and learns the "true" image from a lucky
    imaging data stream.

    ## Arguments

    * `initial` (numpy.ndarray): An initial guess at the scene. It needs to
      be square for now, unfortunately.
    * `img_list` (list): The list of images to thresh.

    ## Keyword Arguments

    * `outdir` (str): The path to the directory where the output files will
      be written. This defaults to `out` in the current working directory.
      The directory will be created if needed.
    * `centers` (numpy.ndarray): A list of the coordinates of the centers
      of the images. If this isn't provided, we'll just assume that the data
      are properly registered.
    * `psf_hw` (int): The half width of the PSF object.
      to the full size of the data.
    * `kernel` (numpy.ndarray): The small (diffraction-limited) PSF to use
      for the light deconvolution.
    * `psfreg` (float): The strength of the PSF "sum-to-one" regularization.
    * `sceneL2` (float): The strength of the L2 regularization to apply to
      the scene.

    """
    def __init__(self, initial, image_list, outdir="", centers=None,
            psf_hw=13, kernel=None, psfreg=100., sceneL2=0.0, dc=0.0):
        # Metadata.
        self.image_list = image_list
        self.outdir = os.path.abspath(outdir)
        self.psf_hw = psf_hw
        self.psfreg = psfreg
        self.sceneL2 = sceneL2
        self.dc = dc

        # Sort out the center vector and save it as a dictionary associated
        # with specific filenames.
        self.centers = centers
        if centers is not None:
            self.centers = dict([(image_list[i], centers[i])
                for i in range(len(image_list))])

        # Inference parameters.
        self.sky = 0
        self.scene = np.array(initial)

        # 'Sky'-subtract the initial scene.
        self.scene -= np.median(self.scene)

        # Deal with the masked pixels if there are any in the initial scene
        # by setting them to the 'sky' level.
        self.scene[np.isnan(self.scene)] = 0.0

        # Check the dimensions of the initial scene and set the size.
        shape = self.scene.shape
        assert shape[0] == shape[1], "The initial scene needs to be square."
        self.size = shape[0] - 2 * self.psf_hw

        # The 'kernel' used for 'light deconvolution'.
        if kernel is None:
            self.kernel = np.exp(-0.5 * (np.arange(-5, 6)[:, None] ** 2
                + np.arange(-5, 6)[None, :] ** 2))
            self.kernel /= np.sum(self.kernel)
        else:
            self.kernel = kernel

        # Index gymnastics.
        self.scene_mask = utils.unravel_scene(self.size + 2 * self.psf_hw,
                self.psf_hw)
        self.psf_rows, self.psf_cols = \
                utils.unravel_psf(self.size + 2 * self.psf_hw, self.psf_hw)

    def do_update(self, fn, alpha, median=True, nn=False):
        """
        Do a single stochastic gradient update using the image in a
        given file and learning rate.

        ## Arguments

        * `fn` (str): The filename of the image to be used.
        * `alpha` (float): The learning rate.

        ## Keyword Arguments

        * `median` (bool): Subtract the median of the scene?
        * `nn` (bool): Project onto the non-negative plane?

        ## Returns

        * `data` (numpy.ndarray): The centered and cropped data image used
          for this update.

        """
        image = utils.load_image(fn)

        # Center the data.
        if self.centers is None:
            data = utils.trim_image(image, self.size)
            mask = ~np.isnan(data)
        else:
            result = utils.centroid_image(image, self.size,
                scene=self.scene, coords=self.centers[fn])
            data = result[1]
            mask = ~np.isnan(data) * result[2]

        # Add the DC offset.
        data += self.dc

        # Do the inference.
        self.old_scene = np.array(self.scene)

        self.psf, self.sky = self.infer_psf(data, mask)
        print "sky:", self.sky
        self.dlds = self.get_dlds(data, mask)

        # self.old_scene = self.scene + alpha * self.dlds
        self.scene += alpha * self.dlds

        # WTF?!?
        gc.collect()

        # Apply some serious HACKS!
        if median:
            self.scene -= np.median(self.scene)
        if nn:
            self.scene[self.scene < 0] = 0.0

        return data

    def run_inference(self, npasses=5, median=False, nn=True, top=None,
            thin=1):
        """
        Thresh the data.

        ## Keyword Arguments

        * `npasses` (int): The number of times to run through the data.
        * `median` (bool): Subtract the median of the scene at each update.
        * `nn` (bool): Constrain the inferred scene to be non-negative.
        * `top` (int): Only consider the top few images.
        * `thin` (int): Only save the state every few images.

        """
        N = len([i for i in self.image_list])

        iml = self.image_list
        if top is not None:
            iml = iml[:int(top)]
        for pass_number in xrange(npasses):
            if pass_number > 0:
                np.random.shuffle(iml)
            for img_number, fn in enumerate(iml):
                # If it's the first pass, `alpha` should decay and we
                # should use _non-negative_ optimization.
                if pass_number == 0:
                    alpha = min(2. / (1 + img_number), 0.25)
                    use_nn = nn
                else:
                    alpha = 2. / N  # MAGIC: 2.
                    use_nn = False

                data = self.do_update(fn, alpha, median=median, nn=use_nn)

                # Save the current state of the scene.
                if img_number % thin == 0:
                    self.save(fn, pass_number, img_number, data)

    def get_psf_matrix(self, L2=True):
        """
        Get the unraveled matrix for the current PSF.

        ## Keyword Arguments

        * `L2` (bool): Should the rows for the L2 norm be included?

        ## Returns

        * `psf_matrix` (scipy.sparse.csr_matrix): The sparse, unraveled PSF
          matrix.

        """
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
    def infer_psf(self, data, mask):
        """
        Take data and a current belief about the scene; infer the PSF for
        this image given the scene. This code infers a sky level
        simultaneously. That might seem like a detail, but it matters. This
        method uses the sparse non-negative least-squares algorithm from
        scipy.

        ## Arguments

        * `data` (numpy.ndarray): The image data.
        * `mask` (numpy.ndarray): The inverse variance map for the data.

        ## Returns

        * `psf` (numpy.ndarray): The inferred 2D PSF image.

        """
        if np.any(data * mask < 0):
            logging.warn("This data violates the model... "
                + "it has negative pixels. Consider using --dc option.")
        # Sort out the dimensions.
        P = 2 * self.psf_hw + 1
        psf_size = P ** 2

        D = data.shape[0]
        data_size = D ** 2

        # Build scene matrix from kernel-convolved scene.
        # kc_scene = convolve(self.kernel, self.scene, mode="same")
        kc_scene = np.array(self.scene)

        scene_matrix = np.zeros((data_size + 1, psf_size + 1))

        # Unravel the scene.
        scene_matrix[:data_size, :psf_size] = \
                                    kc_scene.flatten()[self.scene_mask]

        # Add the sky.
        scene_matrix[:data.size, psf_size] = 1

        scene_matrix[data_size, :psf_size] = self.psfreg * 1.
        data_vector = np.append(data.flatten(), self.psfreg * np.ones(1))

        # Build the mask vector. The `sqrt` means that we're treating the
        # mask like an inverse variance map.
        mask_vector = np.sqrt(np.append(mask.flatten(), np.ones(1)))

        # Infer the new PSF.
        new_psf, rnorm = op.nnls(scene_matrix * mask_vector[:, None],
                data_vector * mask_vector)

        # Get the inferred sky level.
        sky = new_psf[-1]

        # Do the index gymnastics to get the correct inferred PSF.
        # NOTE: here, we're first dropping the sky and then reversing the
        # PSF object because of the way that the `convolve` function is
        # defined.
        return new_psf[:-1][::-1].reshape((P, P)), sky

    @utils.timer
    def infer_scene(self, data):
        """
        Take data and a current belief about the PSF; infer the scene for
        this image given the PSF.

        ## Arguments

        * `data` (numpy.ndarray): The data.

        ## Returns

        * `new_scene` (numpy.ndarray): The scene implied by this particular
          data alone.

        ## Note

        This method has been deprecated by `get_dlds` and the proper
        stochastic gradient update. I'm keeping this here for now just in
        case.

        """
        # Infer scene and return
        data_vector = np.append(data.flatten(), np.zeros(self.scene.size))
        results = lsqr(self.get_psf_matrix(L2=True), data_vector)

        new_scene = results[0].reshape(self.scene.shape)

        return new_scene

    @utils.timer
    def get_dlds(self, data, mask):
        """
        Take data and a current belief about the PSF; compute the gradient of
        log-likelihood with respect to the scene.

        ## Arguments

        * `data` (numpy.ndarray): The data.
        * `mask` (numpy.ndarray): The inverse variance mask for the data.

        ## Returns

        * `dlds` (numpy.ndarray): The gradient of the likelihood function
          with respect to the scene parameters.

        """
        psf_matrix = self.get_psf_matrix(L2=False)

        dlds = psf_matrix.transpose().dot((data.flatten() - self.sky -
                psf_matrix.dot(self.scene.flatten()))
                * mask.flatten())
        dlds = dlds.reshape(self.scene.shape)

        return dlds

    def save(self, fn, pass_number, img_number, data):
        _id = "{0:d}-{1:08}".format(pass_number, img_number)
        outfn = os.path.join(self.outdir, _id + ".fits")

        hdus = [pyfits.PrimaryHDU(data),
                pyfits.ImageHDU(self.dlds),
                pyfits.ImageHDU(self.scene),
                pyfits.ImageHDU(self.psf),
                pyfits.ImageHDU(self.kernel),
                pyfits.ImageHDU(self.old_scene),
                pyfits.ImageHDU(convolve(self.psf, convolve(self.kernel,
                    self.old_scene, "same"), "valid") - data + self.sky)]

        hdus[0].header.update("datafn", fn)
        hdus[0].header.update("size", self.size)
        hdus[0].header.update("pass", pass_number)
        hdus[0].header.update("image", img_number)
        hdus[0].header.update("sky", self.sky)
        hdus[0].header.update("dc", self.dc)
        hdus[1].header.update("status", "old")
        hdus[2].header.update("status", "new")

        pyfits.HDUList(hdus).writeto(outfn, clobber=True)

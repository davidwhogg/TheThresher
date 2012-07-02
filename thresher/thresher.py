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

    ## Arguments

    * `imgglob` (str): The filesystem glob that the data files will match
      (e.g. `/path/to/data_*.fits`).

    ## Keyword Arguments

    * `outdir` (str): The path to the directory where the output files will
      be written. This defaults to `out` in the current working directory.
      The directory will be created if needed.
    * `psf_hw` (int): The half width of the PSF object.
    * `size` (int): The size of one side of the (square) "scene" to infer.
      This currently needs to be provided but it should eventually default
      to the full size of the data.
    * `sky` (float): A constant piston to apply to every observation.
    * `kernel` (numpy.ndarray): The small (diffraction-limited) PSF to use
      for the light deconvolution.
    * `psfreg` (float): The strength of the PSF "sum-to-one" regularization.
    * `sceneL2` (float): The strength of the L2 regularization to apply to
      the scene.

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
        """
        Do the initial setup for a Thresher run. This is not needed for a
        TLI run (hence why it is not in `__init__`). It first calculates the
        initial guess for the scene (using TLI) and then figures out the
        index gymnastics needed for unravelling the scene and PSF.

        """
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

        ## Arguments

        * `image_list` (list): The list of filenames for the images which
          will be ranked and combined using TLI.

        ## Keyword Arguments

        * `size` (int): The size of the scene to be determined.
        * `top` (int): How many images should be coadded?
        * `top_percent` (float): An alternative notation for `top` instead
          specified by a percentage.

        ## Returns

        * `fns` (list): The filenames ordered from best to worst as ranked
          by TLI.
        * `ranks` (list): The value of the ranking scalar corresponding to
          the images listed in `fns`.
        * `centers` (list): The coordinates of the centers of each image as
          determined by centroiding.
        * `coadd` (numpy.ndarray): The resulting co-added image.

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

    def run_inference(self, npasses=5, median=False, nn=True, top=None,
            thin=10):
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
                if self.pass_number == 0:
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
    def infer_psf(self, data):
        """
        Take data and a current belief about the scene; infer the PSF for
        this image given the scene. This code infers a sky level
        simultaneously. That might seem like a detail, but it matters. This
        method uses the sparse non-negative least-squares algorithm from
        scipy.

        ## Arguments

        * `data` (numpy.ndarray): The image data.

        ## Returns

        * `psf` (numpy.ndarray): The inferred 2D PSF image.

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
    def get_dlds(self, data):
        """
        Take data and a current belief about the PSF; compute the gradient of
        log-likelihood with respect to the scene.

        ## Arguments

        * `data` (numpy.ndarray): The data.

        ## Returns

        * `dlds` (numpy.ndarray): The gradient of the likelihood function
          with respect to the scene parameters.

        """
        psf_matrix = self.get_psf_matrix(L2=False)

        dlds = psf_matrix.transpose().dot(data.flatten() -
                psf_matrix.dot(self.scene.flatten()))
        dlds = dlds.reshape(self.scene.shape)

        return dlds

    def save(self, fn, pass_number, img_number, data):
        _id = "{0:d}-{1:08}".format(pass_number, img_number)
        outfn = os.path.join(self.outdir, _id + ".fits")

        hdus = [pyfits.PrimaryHDU(data),
                pyfits.ImageHDU(self.dlds),
                pyfits.ImageHDU(self.scene),
                pyfits.ImageHDU(self.psf),
                pyfits.ImageHDU(self.kernel)]

        hdus[0].header.update("datafn", fn)
        hdus[0].header.update("size", self.size)
        hdus[0].header.update("pass", pass_number)
        hdus[0].header.update("image", img_number)
        hdus[0].header.update("sky", self.inferred_sky)
        hdus[1].header.update("status", "old")
        hdus[2].header.update("status", "new")

        pyfits.HDUList(hdus).writeto(outfn, clobber=True)

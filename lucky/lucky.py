"""
This file is part of the Lucky Imaging project.

Issues:
-------
- Initialization involves np.median() and np.clip(), as does the first pass
  (pindex == 1) of optimization.  Insane.
- There are hard-coded numbers everywhere, especially the stupid "300" and
  some L2 norms.
- sky fitting and display of results when sky has been fit needs to be
  audited -- should be able to add a sky offset and get IDENTICAL results out.
- region of image set by borderx, bordery; hard-coded! MAGIC NUMBERS abound.
- The infer functions ought to take weight vectors -- this would permit
  dropping data for cross-validation tests and also inclusion of an error
  model.
- I think it memory leaks at least a bit (`Image`s don't get deleted?).
    The leak is somewhere in all the convolving, etc. in the inference steps

notes:
------
- The NNLS implementation may require citation.
- Search code for "hack" and "magic" for issues.

"""

__all__ = ["Scene"]

import os
import logging
import sqlite3
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


class Scene(object):
    """
    A `Scene` object describes and learns the "true" image from a lucky
    imaging data stream.

    """
    def __init__(self, basepath, outdir, psf_hw=13, size=100, sky=1.):
        self.basepath = os.path.abspath(basepath)
        self.outdir = os.path.abspath(outdir)
        self.psf_hw = psf_hw
        self.size = size
        self.sky = sky

        # Initialize the PSF image as a delta function.
        pd = 2 * psf_hw + 1
        self.psf = np.zeros((pd, pd))
        self.psf[psf_hw, psf_hw] = 1.

        self.default_psf = np.zeros_like(self.psf)
        self.default_psf[psf_hw - 2:psf_hw + 2, psf_hw - 1:psf_hw + 2] = 1.

        # Initialize the scene as a centered Gaussian.
        x = np.linspace(-0.5 * size, 0.5 * size, size) ** 2
        r = np.sqrt(x[:, None] + x[None, :])
        self.scene = np.exp(-0.5 * r) / np.sqrt(2 * np.pi)
        self.scene = convolve(self.scene, self.default_psf, mode="full")

        # L2 norm weights.
        self.psfL2 = 0.25
        self.sceneL2 = 1. / 64.

        # Make the PSF convolution kernel here. There's a bit of black
        # MAGIC that could probably be fixed. The kernel is implicitly a
        # `sigma = 1. pix` Gaussian.
        self.kernel = np.exp(-0.5 * (np.arange(-3, 4)[:, None] ** 2
            + np.arange(-3, 4)[None, :] ** 2))
        self.kernel /= np.sum(self.kernel)
        Kx, Ky = self.kernel.shape
        self.tinykernel = np.zeros_like(self.kernel)
        self.tinykernel[(Kx - 1) / 2, (Ky - 1) / 2] = 1.

        # The `sqlite` database object that will save the current state.
        self.db = sqlite3.connect(os.path.join(self.outdir, ".state.db"))

    def _setup_db(self):
        c = self.db.cursor()
        c.execute("drop table if exists states")
        c.execute("""create table states
                (id text primary key, infile text, outfile text)""")
        self.db.commit()
        c.close()

    @property
    def image_list(self):
        entries = os.listdir(self.basepath)
        for e in sorted(entries):
            if os.path.splitext(e)[1] == ".fits":
                yield os.path.join(self.basepath, e)

    def run_inference(self, npasses=5, current_pass=0, current_img=0):
        """
        Run the full inference on the dataset.

        ## Keyword Arguments

        * `npasses` (int): The number of passes to run.
        * `current_pass` (int): The pass number to start at. This is used
          for restarting.
        * `current_img` (int): The image number to start at on the first
          pass through the data. This is used for restarting.

        """
        # If we're not restarting, wipe and setup the database.
        if current_pass == 0 and current_img == 0:
            self._setup_db()

        for self.pass_number in xrange(current_pass, npasses):
            for self.img_number, self.fn in enumerate(self.image_list):
                if self.img_number >= current_img:
                    image = load_image(self.fn)
                    data = centroid_image(image, self.scene, self.size)
                    data += self.sky

                    # If it's the first pass, `alpha` should decay and we
                    # should use _non-negative_ optimization.
                    if self.pass_number == 0:
                        alpha = min(2. / (1 + self.img_number), 0.25)
                        nn = True
                    else:
                        self.scene -= np.median(self.scene)  # Hackeroni?
                        alpha = 2. / 300.  # Hack-o-rama?
                        nn = False

                    self._inference_step(data, alpha, nn)
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
        self.scene = (1 - alpha) * self.scene \
                                + alpha * self._infer_scene(data)
        self.scene -= np.median(self.scene)  # Crazy hackishness!

        # Hogg: is this right?
        if nn:
            self.scene[self.scene < 0] = 0.0

    def _infer_psf(self, data):
        """
        Take data and a current belief about the scene; infer the psf for
        this image given the scene.  This code infers a sky level
        simultaneously.  That might seem like a detail, but it matters.
        There is also a Gaussian-like kernel hard-coded in here that is a
        problem.

        Note that the returned PSF (`psfShape` in the code) is *larger*
        than the number of pixels implied by the number of free parameters
        (`psfParameterShape`).  That is, the PSF it is padded out, because
        of the aforementioned kernel.

        Bug: There is a reversal (a `[::-1]`) in the code that is not
        fully understood at present.

        Bug: Shouldn't make the kernels at every call; these should be
        static or passed in.

        """
        Kx, Ky = self.kernel.shape

        # deal with all the size and shape setup
        Nx, Ny = self.scene.shape
        Px, Py = data.shape
        Mx, My = (Nx - Px + 1, Ny - Py + 1)
        Qx, Qy = (Mx - Kx + 1, My - Ky + 1)
        assert Qx > 0 and Qy > 0, \
                "The scene must be larger than the data ({0}, {1})" \
                .format(Qx, Qy)
        psfParameterShape = (Qx, Qy)
        psfParameterSize = Qx * Qy

        # build scene matrix from kernel-convolved scene
        kernelConvolvedScene = convolve(self.scene, self.kernel, mode="same")
        sceneMatrix = np.zeros((data.size + psfParameterSize,
                                psfParameterSize + 1))
        for k in range(psfParameterSize):
            dx, dy = index2xy(psfParameterShape, k)
            dx -= Qx / 2
            dy -= Qx / 2
            sceneMatrix[:data.size, k] = kernelConvolvedScene[
                    (Mx / 2 + dx):(Mx / 2 + dx + Px),
                    (My / 2 + dy): (My / 2 + dy + Py)].reshape(data.size)

        # sky fitting
        sceneMatrix[:data.size, psfParameterSize] = 1.

        # L2 regularization
        sceneMatrix[data.size:data.size + psfParameterSize, :psfParameterSize]\
                = self.psfL2 * np.identity(psfParameterSize)

        # infer PSF and return
        dataVector = np.append(data.reshape(data.size),
                np.zeros(psfParameterSize))
        newPsfParameter, rnorm = op.nnls(sceneMatrix, dataVector)
        logging.info("Dropping sky level {0} in _infer_psf."
                                    .format(newPsfParameter[psfParameterSize]))
        newPsfParameter = newPsfParameter[:psfParameterSize]  # drop sky
        newDeconvolvedPsf = convolve(
                newPsfParameter[::-1].reshape(psfParameterShape),
                self.tinykernel, mode="full")

        logging.info("Got PSF {0}. Min: {1}, Median: {2}, Max: {3}"
                .format(newDeconvolvedPsf.shape, newDeconvolvedPsf.min(),
                    np.median(newDeconvolvedPsf), newDeconvolvedPsf.max()))

        # Save the new PSF.
        self.psf = newDeconvolvedPsf

    def _infer_scene(self, data):
        """
        Take data and a current belief about the PSF; infer the scene for
        this image given the PSF.  This code infers a sky level
        simultaneously.  That might seem like a detail, but it matters.

        Bug: There is a reversal (a `[::-1]`) in the code that is not
        fully understood at present.  A good guess is that it has
        something to do with the `convolve()` operation.

        """
        Px, Py = data.shape
        Mx, My = self.psf.shape
        Nx, Ny = (Px + Mx - 1, Py + My - 1)
        sceneShape = (Nx, Ny)
        sceneSize = Nx * Ny

        # build psf matrix from psf
        psfX, psfY = index2xy(self.psf.shape, np.arange(self.psf.size))
        psfVector = self.psf.reshape(self.psf.size)[::-1]  # HACK
        vals = np.zeros(data.size * self.psf.size)
        rows = np.zeros_like(vals).astype(int)
        cols = np.zeros_like(vals).astype(int)
        for k in range(data.size):
            dx, dy = index2xy(data.shape, k)
            s = slice(k * self.psf.size, (k + 1) * self.psf.size)
            vals[s] = psfVector
            rows[s] = k
            cols[s] = xy2index(sceneShape, psfX + dx, psfY + dy)

        # add entries for old-scene-based regularization
        vals = np.append(vals, np.zeros(sceneSize) + self.sceneL2)
        rows = np.append(rows, np.arange(data.size, data.size + sceneSize))
        cols = np.append(cols, np.arange(sceneSize))

        psfMatrix = csr_matrix((vals, (rows, cols)),
                shape=(data.size + sceneSize, sceneSize))

        # infer scene and return
        dataVector = np.append(data.reshape(data.size), np.zeros(sceneSize))
        (newScene, istop, niters, r1norm, r2norm, anorm, acond,
            arnorm, xnorm, var) = lsqr(psfMatrix, dataVector)
        newScene = newScene.reshape(sceneShape)
        newScene -= np.median(newScene)

        logging.info("Got scene {0}, Min: {1}, Median: {2}, Max: {3}"
                .format(newScene.shape, newScene.min(), np.median(newScene),
                    newScene.max()))

        gc.collect()

        return newScene

    def _save_state(self, data):
        _id = "{0:d}-{1:08}".format(self.pass_number, self.img_number)
        outfn = os.path.join(self.outdir, _id + ".fits")

        hdus = [pyfits.PrimaryHDU(data), pyfits.ImageHDU(self.scene),
                pyfits.ImageHDU(self.psf)]
        hdus[0].header.update("datafn", self.fn)
        hdus[0].header.update("size", self.size)

        pyfits.HDUList(hdus).writeto(outfn, clobber=True)

        c = self.db.cursor()
        c.execute("insert into states values (?,?,?)", [_id, self.fn, outfn])
        self.db.commit()
        c.close()

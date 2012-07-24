#!/usr/bin/env python
"""
This file is part of The Thresher.

A set of unit tests.
"""

import os
import numpy as np

from scipy.sparse import csr_matrix

import thresher
import utils


class Tests(object):
    def setUp(self):
        self.Nx, self.Ny = 4, 7
        self.shape = (self.Nx, self.Ny)

        self.data = np.zeros((50, 50))

        self.scene = np.zeros((64, 64))
        self.scene[48, 48] = 1.

    def test_indexing_basic(self):
        for x in xrange(self.Nx):
            for y in xrange(self.Ny):
                i = utils.xy2index(self.shape, x, y)
                assert (x, y) == utils.index2xy(self.shape, i)
                assert i == utils.xy2index(self.shape,
                        *utils.index2xy(self.shape, i))

    def test_indexing_grid(self):
        xgrid = np.zeros(self.shape) + np.arange(self.Nx)[:, None]
        ygrid = np.zeros(self.shape) + np.arange(self.Ny)[None, :]

        i1 = utils.xy2index(self.shape, xgrid, ygrid).astype(int)
        i2 = np.arange(self.Nx * self.Ny).reshape(self.shape)
        assert np.all(i1 == i2)

        x1, y1 = utils.index2xy(self.shape, i1)
        assert np.all(x1 == xgrid) and np.all(y1 == ygrid)

    def test_unravel_scene(self):
        """
        Test to make sure that the scene unraveling yields the correct
        results.

        """
        S, P = 4, 1
        unraveled = utils.unravel_scene(S, P)

        # Calculate the brute force unraveled scene.
        brute = np.array([[ 0,  1,  2,  4,  5,  6,  8,  9, 10],
                          [ 1,  2,  3,  5,  6,  7,  9, 10, 11],
                          [ 4,  5,  6,  8,  9, 10, 12, 13, 14],
                          [ 5,  6,  7,  9, 10, 11, 13, 14, 15]], dtype=int)

        assert np.all(brute == unraveled)

    def test_unravel_psf(self):
        """
        Test to make sure that the PSF unraveling yields the correct
        results.

        """
        S, P = 4, 1
        rows, cols = utils.unravel_psf(S, P)

        # Calculate the brute force unraveled scene.
        b_cols = np.array([ 0,  1,  2,  4,  5,  6,  8,  9, 10,
                            1,  2,  3,  5,  6,  7,  9, 10, 11,
                            4,  5,  6,  8,  9, 10, 12, 13, 14,
                            5,  6,  7,  9, 10, 11, 13, 14, 15], dtype=int)
        b_rows = np.concatenate([k * np.ones((2 * P + 1) ** 2, dtype=int)
                            for k in range(4)])

        assert np.all(rows == b_rows) and np.all(cols == b_cols)

    def test_convolution(self):
        """
        Test that the matrix operation and convolution give the same result.

        """
        # Build a simple scene and PSF as delta functions.
        scene = np.zeros((7, 7))
        psf = np.zeros((3, 3))
        scene[3, 3] = 1
        psf[1, 1] = 1

        # Do the convolution.
        convolution = thresher.convolve(scene, psf, mode="valid")

        # Do the matrix operation.
        M = utils.unravel_scene(len(scene), 1)
        matrix = np.dot(scene.flatten()[M], psf.flatten()[::-1]) \
                .reshape(convolution.shape)

        # Check the results.
        np.testing.assert_allclose(convolution, matrix)

    def test_centroid(self):
        """
        Test that the centroiding operation works as expected.

        """
        # Start by building a sample test image.
        nx, ny = 150, 200
        X, Y = np.meshgrid(range(nx), range(ny))
        centers = np.array([[50.0, 45.0], [75.6, 150.1]])
        widths = [10, 4]
        amps = [5, 2]
        img = np.zeros((ny, nx))
        for i, p in enumerate(centers):
            R2 = (X - p[0]) ** 2 + (Y - p[1]) ** 2
            img += amps[i] * np.exp(- 0.5 * R2 / widths[i] ** 2) / widths[i]

        # Then try a centered scene.
        nsx, nsy = 45, 57
        SX, SY = np.meshgrid(range(nsx), range(nsy))
        scene = np.exp(-0.5 * ((SX - (nsx - 1) / 2) ** 2 \
                + (SY - (nsy - 1) / 2) ** 2))

        # Run the centroid.
        size = 24
        coords, data, mask = utils.centroid_image(img, size, scene=scene)

        # Make sure that it gets the right coordinates.
        truth = centers[np.argmax(amps)]
        assert np.all(coords[::-1] == truth)

    def test_psf_inference(self):
        """
        Make sure that the PSF inference gives the right thing in the limit
        of a delta function scene.

        """
        hw = 10
        dim = 2 * hw + 1
        sky = 4000.0

        # Build a delta function scene.
        initial_scene = np.zeros((dim + 2 * hw, dim + 2 * hw))
        initial_scene[2 * hw, 2 * hw] = 1.0

        # Data.
        data = np.zeros((dim, dim)) + sky
        data[hw, hw] += 100.0
        data[hw - 1, hw] += 50.0
        data[hw + 1, hw] += 50.0
        data[hw, hw - 1] += 50.0
        data[hw, hw + 1] += 50.0

        mask = np.ones_like(data)

        # Kernel.
        kernel = np.zeros((3, 3))
        kernel[1, 1] = 1.0

        # Do the inference.
        scene = thresher.Scene(initial_scene, [], psf_hw=hw, psfreg=0.0,
                kernel=kernel)
        psf, sky = scene.infer_psf(data, mask)

        import matplotlib.pyplot as pl

        pl.figure()

        pl.subplot(121)
        pl.imshow(data, interpolation="nearest", cmap="gray")

        pl.subplot(122)
        pl.imshow(psf, interpolation="nearest", cmap="gray")

        pl.savefig("test.png")

        print psf


if __name__ == "__main__":
    tests = Tests()
    tests.setUp()
    tests.test_psf_inference()

#!/usr/bin/env python
"""
A set of unit tests for the lucky imaging pipeline.

"""

import os
import numpy as np

import lucky

import matplotlib.pyplot as pl


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
                i = lucky.xy2index(self.shape, x, y)
                assert (x, y) == lucky.index2xy(self.shape, i)
                assert i == lucky.xy2index(self.shape,
                        *lucky.index2xy(self.shape, i))

    def test_indexing_grid(self):
        xgrid = np.zeros(self.shape) + np.arange(self.Nx)[:, None]
        ygrid = np.zeros(self.shape) + np.arange(self.Ny)[None, :]

        i1 = lucky.xy2index(self.shape, xgrid, ygrid).astype(int)
        i2 = np.arange(self.Nx * self.Ny).reshape(self.shape)
        assert np.all(i1 == i2)

        x1, y1 = lucky.index2xy(self.shape, i1)
        assert np.all(x1 == xgrid) and np.all(y1 == ygrid)

    def test_unravel_scene(self):
        """
        Test to make sure that the scene unraveling yields the correct
        results.

        """
        S, P = 4, 1
        scene = np.arange(S ** 2).reshape((S, S))
        unraveled = lucky.unravel_scene(scene, P)

        # Calculate the brute force unraveled scene.
        brute = np.array([[ 0,  1,  2,  4,  5,  6,  8,  9, 10],
                          [ 1,  2,  3,  5,  6,  7,  9, 10, 11],
                          [ 4,  5,  6,  8,  9, 10, 12, 13, 14],
                          [ 5,  6,  7,  9, 10, 11, 13, 14, 15]], dtype=int)

        assert np.all(brute == unraveled)

    def get_scene(self):
        """
        Get a scene object initialized in the test data directory.

        """
        bp = os.path.join(os.path.abspath(__file__), "..", "..", "test")
        scene = lucky.Scene(bp, psf_hw=5)
        scene.initialize_with_data()
        return scene

    def test_psf_infer_convolution(self):
        """
        Test to make sure that the convolution happening within the infer
        PSF step is doing what we think it is.

        NOTE: The PSF must be reversed in the matrix operation for things to
        work out properly.

        """
        scene = self.get_scene()
        kc_scene = lucky.convolve(scene.scene, scene.kernel, mode="same")

        # Generate a random PSF.
        psf_size = 2 * scene.psf_hw + 1
        psf = np.random.rand(psf_size ** 2).reshape((psf_size,) * 2)

        # Do the convolution.
        convolution = lucky.convolve(kc_scene, psf, mode="valid")

        # Do the matrix operation with the _reversed_ PSF.
        matrix = np.dot(kc_scene.flatten()[scene.scene_mask],
                psf.flatten()[::-1]).reshape(convolution.shape)

        # Check the results.
        np.testing.assert_allclose(convolution, matrix)

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
        convolution = lucky.convolve(scene, psf, mode="valid")

        # Do the matrix operation.
        unraveled = lucky.unravel_scene(scene, 1)
        matrix = np.dot(unraveled, psf.flatten()[::-1]) \
                .reshape(convolution.shape)

        # Check the results.
        np.testing.assert_allclose(convolution, matrix)

    def test_psf_norm(self):
        """Test to make sure that the PSF is properly normalized."""
        scene = self.get_scene()
        scene._infer_psf(scene.first_image)
        np.testing.assert_allclose(np.sum(scene.psf), 1.)

if __name__ == "__main__":
    tests = Tests()
    tests.setUp()
    tests.test_psf_infer_convolution()

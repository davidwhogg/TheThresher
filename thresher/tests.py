#!/usr/bin/env python
"""
This file is part of The Thresher.

A set of unit tests.
"""

import os
import numpy as np

from scipy.sparse import csr_matrix

import thresher


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
                i = thresher.xy2index(self.shape, x, y)
                assert (x, y) == thresher.index2xy(self.shape, i)
                assert i == thresher.xy2index(self.shape,
                        *thresher.index2xy(self.shape, i))

    def test_indexing_grid(self):
        xgrid = np.zeros(self.shape) + np.arange(self.Nx)[:, None]
        ygrid = np.zeros(self.shape) + np.arange(self.Ny)[None, :]

        i1 = thresher.xy2index(self.shape, xgrid, ygrid).astype(int)
        i2 = np.arange(self.Nx * self.Ny).reshape(self.shape)
        assert np.all(i1 == i2)

        x1, y1 = thresher.index2xy(self.shape, i1)
        assert np.all(x1 == xgrid) and np.all(y1 == ygrid)

    def test_unravel_scene(self):
        """
        Test to make sure that the scene unraveling yields the correct
        results.

        """
        S, P = 4, 1
        unraveled = thresher.unravel_scene(S, P)

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
        rows, cols = thresher.unravel_psf(S, P)

        # Calculate the brute force unraveled scene.
        b_cols = np.array([ 0,  1,  2,  4,  5,  6,  8,  9, 10,
                            1,  2,  3,  5,  6,  7,  9, 10, 11,
                            4,  5,  6,  8,  9, 10, 12, 13, 14,
                            5,  6,  7,  9, 10, 11, 13, 14, 15], dtype=int)
        b_cols = np.append(b_cols, np.arange(S ** 2))

        b_rows = np.concatenate([k * np.ones((2 * P + 1) ** 2, dtype=int)
                            for k in range(4)])
        b_rows = np.append(b_rows, np.arange(4, 4 + S ** 2))

        assert np.all(rows == b_rows) and np.all(cols == b_cols)

    def get_scene(self):
        """
        Get a scene object initialized in the test data directory.

        """
        bp = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                "..", "test",
                "*.fits")
        scene = thresher.Scene(bp, psf_hw=17)
        scene.initialize_with_data()
        return scene

    def test_psf_infer_convolution(self):
        """
        test to make sure that the convolution happening within the infer
        psf step is doing what we think it is.

        note: the psf must be reversed in the matrix operation for things to
        work out properly.

        """
        scene = self.get_scene()
        kc_scene = thresher.convolve(scene.scene, scene.kernel, mode="same")

        # generate a random psf.
        psf_size = 2 * scene.psf_hw + 1
        psf = np.random.rand(psf_size ** 2).reshape((psf_size,) * 2)

        # do the convolution.
        convolution = thresher.convolve(kc_scene, psf, mode="valid")

        # do the matrix operation with the _reversed_ psf.
        matrix = np.dot(kc_scene.flatten()[scene.scene_mask],
                psf.flatten()[::-1]).reshape(convolution.shape)

        # check the results.
        np.testing.assert_allclose(convolution, matrix)

    def test_scene_infer_convolution(self):
        """
        Test to make sure that the convolution happening within the infer
        scene step is doing what we think it is.

        """
        scene = self.get_scene()

        S = len(scene.scene)
        D = S - 2 * scene.psf_hw

        # Generate a random psf.
        psf_size = 2 * scene.psf_hw + 1
        psf = np.random.rand(psf_size ** 2).reshape((psf_size,) * 2)

        # Generate the PSF matrix. NOTE: the PSF is reversed here.
        vals = np.zeros((D ** 2, psf_size ** 2)) + psf.flatten()[None, ::-1]
        vals = vals.flatten()
        vals = np.append(vals, np.ones(S ** 2))

        psf_matrix = csr_matrix((vals, (scene.psf_rows, scene.psf_cols)),
                shape=(D ** 2 + S ** 2, S ** 2))

        # Do the convolution.
        convolution = thresher.convolve(scene.scene, psf, mode="valid")

        # Do the matrix operation.
        matrix = psf_matrix.dot(scene.scene.flatten())
        matrix = matrix[:D ** 2].reshape((D, D))

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
        convolution = thresher.convolve(scene, psf, mode="valid")

        # Do the matrix operation.
        M = thresher.unravel_scene(len(scene), 1)
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
        coords, data = thresher.centroid_image(img, scene, size)

        # Make sure that it gets the right coordinates.
        truth = centers[np.argmax(amps)]
        assert np.all(coords[::-1] == truth)

        # Make sure that the trimming worked.
        np.testing.assert_allclose(data,
                img[truth[1] - size / 2:truth[1] + size / 2,
                    truth[0] - size / 2:truth[0] + size / 2])


if __name__ == "__main__":
    tests = Tests()
    tests.setUp()
    tests.test_centroid()

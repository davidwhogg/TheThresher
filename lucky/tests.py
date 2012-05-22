#!/usr/bin/env python
"""
A set of unit tests for the lucky imaging pipeline.

"""

import os
import numpy as np

import lucky


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
        S, P = 5, 1
        scene = np.arange(S ** 2).reshape((S, S))
        unraveled = lucky.unravel_scene(scene, P)

        # Calculate the brute force unraveled scene.
        brute = np.array([[ 0,  1,  2,  5,  6,  7, 10, 11, 12],
                          [ 1,  2,  3,  6,  7,  8, 11, 12, 13],
                          [ 2,  3,  4,  7,  8,  9, 12, 13, 14],
                          [ 5,  6,  7, 10, 11, 12, 15, 16, 17],
                          [ 6,  7,  8, 11, 12, 13, 16, 17, 18],
                          [ 7,  8,  9, 12, 13, 14, 17, 18, 19],
                          [10, 11, 12, 15, 16, 17, 20, 21, 22],
                          [11, 12, 13, 16, 17, 18, 21, 22, 23],
                          [12, 13, 14, 17, 18, 19, 22, 23, 24]], dtype=int)

        assert np.all(brute == unraveled)

    def test_psf_norm(self):
        """Test to make sure that the PSF is properly normalized."""
        bp = os.path.join(os.path.abspath(__file__), "..", "..", "test_data")
        scene = lucky.Scene(bp, psf_hw=10)
        scene.initialize_with_data()
        scene._infer_psf(scene.first_image)
        np.assert_allclose(np.sum(scene.psf), 1.)

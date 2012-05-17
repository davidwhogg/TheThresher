#!/usr/bin/env python
"""
A set of unit tests for the lucky imaging pipeline.

"""

import os
import numpy as np

from lucky import xy2index, index2xy, Scene


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
                i = xy2index(self.shape, x, y)
                assert (x, y) == index2xy(self.shape, i)
                assert i == xy2index(self.shape, *index2xy(self.shape, i))

    def test_indexing_grid(self):
        xgrid = np.zeros(self.shape) + np.arange(self.Nx)[:, None]
        ygrid = np.zeros(self.shape) + np.arange(self.Ny)[None, :]

        i1 = xy2index(self.shape, xgrid, ygrid).astype(int)
        i2 = np.arange(self.Nx * self.Ny).reshape(self.shape)
        assert np.all(i1 == i2)

        x1, y1 = index2xy(self.shape, i1)
        assert np.all(x1 == xgrid) and np.all(y1 == ygrid)

    def test_psf_norm(self):
        """Test to make sure that the PSF is properly normalized."""
        bp = os.path.join(os.path.abspath(__file__), "..", "..", "test_data")
        scene = Scene(bp, psf_hw=10)
        scene.initialize_with_data()
        scene._infer_psf(scene.first_image)
        print np.sum(scene.psf)
        assert False

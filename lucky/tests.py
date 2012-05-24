#!/usr/bin/env python
"""
This file is part of The Thresher.

A set of unit tests.
"""

import os
import numpy as np

from lucky import xy2index, index2xy, save_scene, read_scene


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

    def test_io(self):
        fn = ".unit_test.fits"
        save_scene(self.scene, fn)
        loadedScene = read_scene(fn)
        try:
            os.remove(fn)
        except:
            pass
        assert np.sum(np.abs(self.scene - loadedScene)) < 1e-10

# FIXME: Broken unit tests.
# newPsf, newScene = inference_step(data, scene, 0.001, 0.001,
#         runUnitTest=True)
# # assert(np.all(newPsf >= 0)) # WHY DOES THIS FAIL FOR l_bfgs_b?
# assert(np.all(newPsf > -1.e-3))  # should be more stringent
# assert(np.all(newPsf < 1.e-3))  # should be more stringent
# assert(np.all(newScene == 0))
# return None

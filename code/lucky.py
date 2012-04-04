# This file is part of the Lucky Imaging project.

import numpy as np
import scipy.sparse as sp

def image2matrix(psfImage, sceneShape):
    '''
    # `image2matrix()`:

    Take a PSF-like kernel image and reform it so that it is a matrix
    that can be used to do convolution of a "true scene" image in a
    matrix operation, in which the true scene image is flattened to a
    vector and the PSF image is expanded into a large matrix.  See,
    eg: <http://adsabs.harvard.edu/abs/2011A%26A...531A...9H>.

    ### input:

    * `psfImage`: The (Mx, My)-shaped convolution kernel image (think
      of it as the PSF).
    * `sceneShape`: The (Nx, Ny) tuple shape of the image to be
      convolved (think of it as the true scene).

    ### output:

    * A sparse matrix that has shape (Mx*My, (Mx+Nx-1)*(My+Ny-1)) (or
      possibly the transpose of that).  Use as follows:

          psfMatrix = image2matrix(psfImage, sceneImage.shape)
          modelVector = np.dot(psfMatrix, sceneImage.reshape(Nx*Ny))
          modelImage = modelVector.reshape((Mx+Nx-1, My+Ny-1))
    '''
    return None

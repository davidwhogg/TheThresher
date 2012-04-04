# This file is part of the Lucky Imaging project.

# issues:
# -------
# - should use scipy.sparse.

import numpy as np

def xy2index(shape, x, y):
    '''
    # `xy2index()`:

    Go from x,y position in a 2-d numpy array to one-d index in the
    flattened 1-d array.
    '''
    return (x * shape[1] + y)

def index2xy(shape, i):
    '''
    # `index2xy()`:

    Inverse of `xy2index()`.
    '''
    return ((i / shape[1]), (i % shape[1]))

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

    * A sparse matrix that has shape ((Mx+Nx-1)*(My+Ny-1), Nx*Ny).
      Use as follows:

          psfMatrix = image2matrix(psfImage, sceneImage.shape)
          modelVector = np.dot(psfMatrix, sceneImage.reshape(Nx*Ny))
          modelImage = modelVector.reshape((Mx+Nx-1, My+Ny-1))

    ### issues:

    * There is probably a faster way to do all this.
    * Returns a numpy array; should return a scipy.sparse matrix.
    '''
    Mx, My = psfImage.shape
    Nx, Ny = sceneShape
    sceneIndex = np.arange(Nx * Ny)
    sceneX, sceneY = index2xy(sceneShape, sceneIndex)
    modelShape = (Mx + Nx - 1, My + Ny - 1)
    Px, Py = modelShape
    psfMatrix = np.zeros((Px * Py, Nx * Ny))
    for i in range(Mx * My):
        psfX, psfY = index2xy(psfImage.shape, i)
        modelIndex = xy2index(modelShape, psfX + sceneX, psfY + sceneY)
        psfMatrix[modelIndex, sceneIndex] = psfImage[psfX, psfY]
    return psfMatrix

def inference_step(data, psf, scene):
    '''
    # `inference_step()`:

    Take data and a current belief about the scene; infer the psf for
    this image given the scene, and infer a new scene given the
    inferred image and the inferred psf.

    ### input:

    * `data`: An individual image.
    * `psf`: A PSF image (used only for shape and size information).
    * `scene`: The current best guess for the scene that (after PSF
      convolution) generates the image.

    ### output:

    * `newPsf`:
    * `newScene`:

    '''
    dataVector = data.reshape(data.size)
    sceneMatrix = image2matrix(scene, psf.shape)
    newPsf, res, rank, s = np.linalg.lstsq(sceneMatrix, dataVector)
    newPsf = newPsf.reshape(psf.shape)
    psfMatrix = image2matrix(newPsf, scene.shape)
    newScene, res, rank, s = np.linalg.lstsq(psfMatrix, dataVector)
    return newPsf, newScene.reshape(scene.shape)

def unit_tests():
    '''
    # `unit_tests()`:

    Run a set of unit tests
    '''
    shape = (4, 7)
    Nx, Ny = shape
    for x in range(Nx):
        for y in range(Ny):
            i = xy2index(shape, x, y)
            assert((x, y) == index2xy(shape, i))
            assert(i == xy2index(shape, *index2xy(shape, i)))
    xgrid = np.zeros(shape) + np.arange(Nx)[:,None]
    ygrid = np.zeros(shape) + np.arange(Ny)[None,:]
    i1 = xy2index(shape, xgrid, ygrid).astype(int)
    i2 = np.arange(Nx * Ny).reshape(shape)
    print i1 - i2
    assert(np.all((i1 - i2) == 0))
    x1, y1 = index2xy(shape, i1)
    print x1 - xgrid
    print y1 - ygrid
    assert(np.all((x1 - xgrid) == 0))
    assert(np.all((y1 - ygrid) == 0))
    psf = np.zeros((5, 7))
    Mx, My = psf.shape
    psf[2,2] = 0.5
    psf[2,3] = 0.25
    psf[3,3] = 0.25
    psf[4,4] = 0.25
    scene = np.zeros((11,11))
    Nx, Ny = scene.shape
    scene[3,3] = 1.0
    scene[3,9] = 2.0
    scene[8,3] = 3.0
    scene[7,9] = 4.0
    psfMatrix = image2matrix(psf, scene.shape)
    modelVector = np.dot(psfMatrix, scene.reshape(Nx*Ny))
    modelImage1 = modelVector.reshape((Mx+Nx-1, My+Ny-1))
    sceneMatrix = image2matrix(scene, psf.shape)
    modelVector = np.dot(sceneMatrix, psf.reshape(Mx*My))
    modelImage2 = modelVector.reshape((Mx+Nx-1, My+Ny-1))
    print modelImage1 - modelImage2
    assert(np.all((modelImage1 - modelImage2) == 0))
    modelImage3 = np.zeros((Mx+Nx-1, My+Ny-1))
    for psfx in range(Mx):
        for psfy in range(My):
            modelImage3[psfx:psfx+Nx,psfy:psfy+Ny] += psf[psfx, psfy] * scene
    print modelImage1 - modelImage3
    assert(np.all((modelImage1 - modelImage3) == 0))
    data = modelImage3 + 0.01 * np.random.normal(size=modelImage3.shape)
    newPsf, newScene = inference_step(data, psf, scene)
    print psf, newPsf
    print 'all tests passed'
    return None

if __name__ == '__main__':
    unit_tests()

"""
This file is part of the Lucky Imaging project.

issues:
-------
- smoothness regularization for scene?
- L2 regularization for PSF
- L2 regularization for scene
- weak regularization sum(PSF) ~= 1

"""

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc
    rc('font',**{'family':'serif','serif':'Computer Modern Roman','size':12})
    rc('text', usetex=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.signal import correlate as convolve # Hey dstn CHECK THIS OUT
import scipy.optimize as op

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

def infer_psf(data, scene):
    '''
    # `infer_psf()`:

    Take data and a current belief about the scene; infer the psf for
    this image given the scene.  This code infers a sky level
    simultaneously.  That might seem like a detail, but it matters.
    There is also a Gaussian-like kernel hard-coded in here that is a
    problem.

    Note that the returned PSF (`psfShape` in the code) is *larger*
    than the number of pixels implied by the number of free parameters
    (`psfParameterShape`).  That is, the PSF it is padded out, because
    of the aforementioned kernel.

    ### input:

    * `data`: An individual image.
    * `scene`: The current best guess for the scene that (after PSF
      convolution) generates the image.

    ### output:

    * `psf`:
    '''
    # make kernel
    # magic numbers `-0.5` and `(3,-4)` in next line
    kernel = np.exp(-0.5 * (np.arange(-3,4)[:,None]**2 + np.arange(-3,4)[None,:]**2))
    Kx, Ky = kernel.shape

    # deal with all the size and shape setup
    Nx, Ny = scene.shape
    Px, Py = data.shape
    Mx, My = (Nx - Px + 1, Ny - Py + 1)
    psfShape = (Mx, My)
    psfSize = Mx * My
    Qx, Qy = (Mx - Kx + 1, My - Ky + 1)
    assert(Qx > 0)
    assert(Qy > 0)
    psfParameterShape = (Qx, Qy)
    psfParameterSize = Qx * Qy

    # build scene matrix from kernel-convolved scene
    kernelConvolvedScene = convolve(scene, kernel, mode="same")
    sceneMatrix = np.zeros((data.size, psfParameterSize))
    for k in range(psfParameterSize):
        dx, dy = index2xy(psfParameterShape, k)
        dx -= Qx / 2
        dy -= Qx / 2
        sceneMatrix[:,k] = kernelConvolvedScene[(Mx / 2 + dx): -(Mx / 2 - dx), (My / 2 + dy): -(My / 2 - dy)].reshape(data.size)

    # infer PSF and return
    dataVector = data.reshape(data.size)
    if True: # l_bfgs_b method
        def cost(psf):
            return (np.sum((dataVector - psf[0] - np.dot(sceneMatrix, psf[1:]))**2),
                    np.append(-2. * np.sum(dataVector - psf[0] - np.dot(sceneMatrix, psf[1:])), -2. * np.dot(sceneMatrix.T, (dataVector - psf[0] - np.dot(sceneMatrix, psf[1:])))))
        bounds = [(None, None)].append([(0., None) for p in range(psfParameterSize)])
        newPsfParameter, f, d = op.fmin_l_bfgs_b(cost, np.zeros(psfParameterSize + 1)) # HARDCORE OPTIONS: factr=0., pgtol=0.
        print d
        newPsf = convolve(newPsfParameter[1:].reshape(psfParameterShape), kernel, mode="full")
    if False: # levmar method
        def resid(lnpsf):
            return dataVector - lnpsf[0] - np.dot(sceneMatrix, np.exp(lnpsf[1:]))
        (newLnPsfParameter, cov_x, infodict, mesg, ier) = op.leastsq(resid, np.zeros(psfParameterSize + 1), full_output=True, xtol=0., ftol=0.)
        print ier, infodict['nfev']
        newPsf = convolve(np.exp(newLnPsfParameter[1:]).reshape(psfParameterShape), kernel, mode="full")
    print "got PSF", newPsf.shape
    return newPsf

def infer_scene(data, psf, l2norm):
    '''
    # `infer_scene()`:

    Take data and a current belief about the PSF; infer the scene for
    this image given the PSF.  This code infers a sky level
    simultaneously.  That might seem like a detail, but it matters.

    ### input:

    * `data`: An individual image.
    * `psf`: A PSF image (used only for shape and size information).
    * `l2norm`: Amplitude for the (required) L2-norm regularization.

    ### output:

    * `scene`
    '''
    # deal with all the size and shape setup
    Px, Py = data.shape
    Mx, My = psf.shape
    Nx, Ny = (Px + Mx - 1, Py + My - 1)
    sceneShape = (Nx, Ny)
    sceneSize = Nx * Ny

    # build psf matrix from psf
    psfX, psfY = index2xy(psf.shape, np.arange(psf.size))
    psfVector = psf.reshape(psf.size)
    vals = np.zeros(data.size * psf.size)
    rows = np.zeros_like(vals).astype(int)
    cols = np.zeros_like(vals).astype(int)
    for k in range(data.size):
        dx, dy = index2xy(data.shape, k)
        s = slice(k * psf.size, (k + 1) * psf.size)
        vals[s] = psfVector
        rows[s] = k
        cols[s] = xy2index(sceneShape, psfX + dx, psfY + dy)
    # add entries for sky estimation
    vals = np.append(vals, np.ones(data.size))
    rows = np.append(cols, np.arange(data.size))
    cols = np.append(cols, np.zeros(data.size) + sceneSize)
    # add entries for L2 norm regularization
    vals = np.append(vals, np.zeros(sceneSize) + l2norm)
    rows = np.append(rows, range(sceneSize))
    cols = np.append(cols, range(sceneSize))
    psfMatrix = csr_matrix((vals, (rows, cols)), shape=(data.size + sceneSize, sceneSize + 1))
    print 'constructed psfMatrix'

    # infer scene and return
    dataVector = np.append(data.reshape(data.size), np.zeros(sceneSize))
    skyvec = np.zeros(data.size + sceneSize)
    if True: # use sparse `lsqr`
        (newScene, istop, niters, r1norm, r2norm, anorm, acond,
         arnorm, xnorm, var) = lsqr(psfMatrix, dataVector)
    if False: # use Lev-Mar `leastsq`
        def sresid(scenepars):
            skyvec[0 : data.size] = scenepars[0]
            return dataVector - skyvec - psfMatrix * scenepars[1:]
        (newScene, cov_x, infodict, mesg, ier) = op.leastsq(sresid, np.zeros(sceneSize + 1), full_output=True)
        print ier, infodict['nfev']
    print "got scene", newScene.shape
    return newScene[:sceneSize].reshape(sceneShape)

def inference_step(data, scene, l2norm, plot=None):
    '''
    # `inference_step()`:

    Concatenation of `infer_psf()` and `infer_scene()`.

    ### inputs:

    * `data`: Image data
    * `scene`: First guess at scene, must be (substantially) larger
      than data image.
    * `l2norm`: Amplitude of L2 regularization; units TBA.
    * `plot`: If not `None`, make a standard plot with this name.

    ### outputs:

    * `newPsf`: inferred PSF.
    * `newScene`: inferred scene.
    '''
    newPsf = infer_psf(data, scene)
    newScene = infer_scene(data, newPsf, l2norm)
    if plot is not None:
        plot_inference_step(data, scene, newPsf, newScene, plot)
    return newPsf, newScene

def plot_inference_step(data, scene, newPsf, newScene, filename):
    '''
    # `plot_inference_step()`:

    Make plots for `inference_step()`.
    '''
    fig = plt.figure(figsize=(10,10))
    plt.clf()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    my_plot = lambda ax, im: ax.imshow(im, cmap="gray", interpolation="nearest")
    my_plot(ax1, data)
    ax1.set_title("Data")
    my_plot(ax2, scene)
    ax2.set_title("Previous Scene")
    my_plot(ax3, newPsf)
    ax3.set_title("Inferred PSF")
    my_plot(ax4, newScene)
    ax4.set_title("Inferred Scene")
    plt.savefig(filename)
    return None

def unit_tests():
    '''
    # `unit_tests()`:

    Run a set of unit tests.
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
    data = np.zeros((64, 64))
    scene = np.zeros((96, 96))
    scene[48, 48] = 1.
    psf = np.zeros((33, 33))
    newPsf, newScene = inference_step(data, scene, 0.001)
    print newPsf
    assert(np.all(newPsf == 0))
    assert(np.all(newScene == 0))
    print 'unit_tests(): all tests passed'
    return None

def functional_tests():
    '''
    # `functional_tests()`:

    Run a set of functional tests.
    '''
    truescene = np.zeros((48, 48))
    truescene[22, 26] = 1.
    truepsf  = np.exp(-0.5   * (((np.arange(15)-8)[:,None])**2 + ((np.arange(15)-4)[None,:])**2))
    truepsf += np.exp(-0.125 * (((np.arange(15)-6)[:,None])**2 + ((np.arange(15)-6)[None,:])**2))
    truedata = convolve(truescene, truepsf, mode="valid")
    data = truedata + 0.01 * np.random.normal(size=(truedata.shape))
    newPsf, newScene = inference_step(data, truescene, 0.001, plot="functional.png")
    print 'functional_tests(): all tests passed'
    return None

if __name__ == '__main__':
    unit_tests()
    functional_tests()
    import os
    from data import Image

    images = Image.get_all()

    hw = 7
    psf = np.zeros((2*hw+1, 2*hw+1))

    scene = images[0][hw:-hw, hw:-hw]

    fig = plt.figure(figsize=(10,10))

    bp = os.path.join(Image._bp, "img")
    try:
        os.makedirs(bp)
    except os.error:
        pass

    for count, img in enumerate(images):
        data = img.image
        psf, newScene = inference_step(data, psf, scene)

        plt.clf()

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        my_plot = lambda ax, im: ax.imshow(im, cmap="gray", interpolation="nearest")

        my_plot(ax1, data)
        ax1.set_title("Data")

        my_plot(ax2, scene)
        ax2.set_title("Previous Scene")

        my_plot(ax3, psf)
        ax3.set_title("Inferred PSF")

        my_plot(ax4, newScene)
        ax4.set_title("Inferred Scene")

        plt.savefig(os.path.join(bp, "%04d.png"%count))

        ndata = 1 + count
        scene = ((ndata - 1.) / ndata) * scene + (1. / ndata) * newScene


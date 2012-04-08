"""
This file is part of the Lucky Imaging project.

issues:
-------
- When I try passing jacobian to lev-mar, it doesn't work; I think this is because of the zeroes / infinities that come in near zero flux, but I don't know how to transform the problem to remove these.  And or I could be wrong.  -Hogg
- Needs to save the PSF and scene inferred from each image.
- l_bfgs_b non-negative optimization is FAILING (derivative wrong?)
- Super slow on PSF estimation!
- I think it memory leaks at least a bit (`Image`s don't get deleted?).

notes:
------
- The L-BFGS-B implementation we are using comes with a citation requirement; see `l_bfgs_b` documentation.

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
from scipy.signal import convolve
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

def infer_psf(data, scene, l2norm, runUnitTest=False):
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

    Bug: There is a reversal (a `[::-1]`) in the code that is not
    fully understood at present.

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
    sceneMatrix = np.zeros((data.size + psfParameterSize, psfParameterSize + 1))
    for k in range(psfParameterSize):
        dx, dy = index2xy(psfParameterShape, k)
        dx -= Qx / 2
        dy -= Qx / 2
        sceneMatrix[:data.size, k] = kernelConvolvedScene[(Mx / 2 + dx): -(Mx / 2 - dx), (My / 2 + dy): -(My / 2 - dy)].reshape(data.size)

    # sky fitting
    sceneMatrix[:data.size, psfParameterSize] = 1.

    # L2 regularization
    sceneMatrix[data.size:(data.size+psfParameterSize), :psfParameterSize] = l2norm * np.identity(psfParameterSize)

    # infer PSF and return
    dataVector = np.append(data.reshape(data.size), np.zeros(psfParameterSize))
    if True: # nnls method
        newPsfParameter, rnorm = op.nnls(sceneMatrix, dataVector)
        newPsfParameter = newPsfParameter[:psfParameterSize] # drop sky
    if False: # l_bfgs_b method -- DOES NOT SEEM TO RESPECT BOUNDS!
        def cost(psf):
            return np.sum((dataVector - np.dot(sceneMatrix, psf))**2)
        def deriv(psf):
            return -2. * np.dot(sceneMatrix.T, (dataVector - np.dot(sceneMatrix, psf)))
        # note MAGIC NUMBERS 10000 because I am afraid of using `None`
        bounds = [(0., 10000.) for p in range(psfParameterSize)]
        bounds.append((-10000., 10000.))
        # in the next line, should have `fprime` argument NOT `approx_grad` argument
        newPsfParameter, f, d = op.fmin_l_bfgs_b(cost, np.ones(psfParameterSize + 1), fprime=deriv, approx_grad=False, epsilon=1.e-6) # HARDCORE OPTIONS: factr=0., pgtol=0.
        newPsfParameter = newPsfParameter[:psfParameterSize] # drop sky
    if False: # levmar method
        def resid(lnpsf):
            foo = np.exp(lnpsf) # exp psf amplitudes
            foo[-1] = lnpsf[-1] # deal with sky
            return np.dot(sceneMatrix, foo) - dataVector
        def jacobian(lnpsf):
            foo = np.exp(lnpsf) # exp psf amplitudes
            foo[-1] = 1. # deal with sky
            return sceneMatrix * foo[None,:]
        def unit_test_jacobian(lnpsf):
            delta = 1.e-4 # magic number
            j0 = jacobian(lnpsf)
            r0 = resid(lnpsf)
            for k,p in enumerate(lnpsf):
                l1 = 1. * lnpsf
                l1[k] += delta
                badness = (resid(l1) - r0) / delta - j0[:,k]
                print k, badness
                assert(np.all(badness**2 <= 1e-6 * j0[:,k]**2))
        first_guess = np.zeros(psfParameterSize + 1)
        if runUnitTest:
            unit_test_jacobian(first_guess)
            unit_test_jacobian(np.random.normal(size=first_guess.size))
        # should add option `Dfun=jacobian` to the next line for speed, but if fails if I do. - Hogg
        (newLnPsfParameter, cov_x, infodict, mesg, ier) = op.leastsq(resid, first_guess, full_output=True, xtol=1.e-5, ftol=1e-5) # HARDCORE OPTIONS: xtol=0., ftol=0.
        if runUnitTest:
            unit_test_jacobian(newLnPsfParameter)
        newPsfParameter = np.exp(newLnPsfParameter[:psfParameterSize]) # drop sky
    newPsf = convolve(newPsfParameter[::-1].reshape(psfParameterShape), kernel, mode="full")
    print "got PSF", newPsf.shape, np.min(newPsf), np.max(newPsf)
    return newPsf

def infer_scene(data, psf, l2norm):
    '''
    # `infer_scene()`:

    Take data and a current belief about the PSF; infer the scene for
    this image given the PSF.  This code infers a sky level
    simultaneously.  That might seem like a detail, but it matters.

    Bug: There is a reversal (a `[::-1]`) in the code that is not
    fully understood at present.  A good guess is that it has
    something to do with the `convolve()` operation.

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
    psfVector = psf.reshape(psf.size)[::-1] # HACK
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
    vals = np.append(vals, np.zeros(data.size))
    rows = np.append(rows, np.arange(data.size).astype(int))
    cols = np.append(cols, np.zeros(data.size).astype(int) + sceneSize)
    # add entries for L2 norm regularization
    vals = np.append(vals, np.zeros(sceneSize) + l2norm)
    rows = np.append(rows, np.arange(data.size, data.size + sceneSize))
    cols = np.append(cols, np.arange(sceneSize))
    psfMatrix = csr_matrix((vals, (rows, cols)), shape=(data.size + sceneSize, sceneSize + 1))
    print 'constructed psfMatrix:', min(rows), max(rows), data.size, sceneSize, min(cols), max(cols), sceneSize

    # infer scene and return
    dataVector = np.append(data.reshape(data.size), np.zeros(sceneSize))
    skyVector = np.zeros(data.size + sceneSize)
    (newScene, istop, niters, r1norm, r2norm, anorm, acond,
     arnorm, xnorm, var) = lsqr(psfMatrix, dataVector)
    newScene = newScene[:sceneSize].reshape(sceneShape)
    print "got scene", newScene.shape, np.min(newScene), np.max(newScene)
    return newScene

def inference_step(data, scene, l2norm, plot=None, runUnitTest=False):
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
    newPsf = infer_psf(data, scene, l2norm, runUnitTest=runUnitTest)
    newScene = infer_scene(data, newPsf, l2norm)
    if plot is not None:
        plot_inference_step(data, scene, newPsf, newScene, plot)
    return newPsf, newScene

def plot_inference_step(data, scene, newPsf, newScene, filename):
    '''
    # `plot_inference_step()`:

    Make plots for `inference_step()`.
    '''
    fig = plt.figure(figsize=(12,8))
    plt.clf()
    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)
    my_plot = lambda ax, im: ax.imshow(im, cmap="gray", interpolation="nearest")
    hw1 = (newPsf.shape[0] - 1) / 2
    hw2 = (data.shape[0] - newPsf.shape[0] - 1) / 2
    my_plot(ax1, data)
    ax1.set_title("data")
    my_plot(ax2, convolve(scene, newPsf, mode="valid"))
    ax2.set_title(r"[inferred PSF] $\ast$ [previous scene]")
    my_plot(ax3, convolve(newScene, newPsf, mode="valid"))
    ax3.set_title(r"[inferred PSF] $\ast$ [inferred scene]")
    bigPsf = np.zeros_like(data)
    bigPsf[hw2+1:-hw2,hw2+1:-hw2] = newPsf
    my_plot(ax4, bigPsf)
    ax4.set_title("inferred PSF (padded)")
    my_plot(ax5, scene[hw1:-hw1,hw1:-hw1])
    ax5.set_title("previous scene (cropped)")
    my_plot(ax6, newScene[hw1:-hw1,hw1:-hw1])
    ax6.set_title("inferred scene (cropped)")
    hogg_savefig(filename)
    return None

def hogg_savefig(fn):
    '''
    # `hogg_savefig()`:

    Hogg likes a verbose `savefig()`!
    '''
    print "writing %s" % fn
    return plt.savefig(fn)

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
    data = np.zeros((50, 50))
    scene = np.zeros((64, 64))
    scene[48, 48] = 1.
    newPsf, newScene = inference_step(data, scene, 0.001, runUnitTest=True)
    print "psf:", newPsf.shape, np.min(newPsf), np.max(newPsf)
    print "scene:", newScene.shape, np.min(newScene), np.max(newScene)
    # assert(np.all(newPsf >= 0)) # WHY DOES THIS FAIL FOR l_bfgs_b?
    assert(np.all(newPsf > -1.e-3)) # should be more stringent
    assert(np.all(newPsf <  1.e-3)) # should be more stringent
    assert(np.all(newScene == 0))
    print 'unit_tests(): all tests passed'
    return None

def functional_tests():
    '''
    # `functional_tests()`:

    Run a set of functional tests.
    '''
    truescene = np.zeros((48, 48))
    truescene[23, 26] = 1.
    truepsf  = 1.  * np.exp(-0.5   * (((np.arange(17)-8)[:,None])**2 + ((np.arange(17)-5)[None,:])**2))
    truepsf += 1.  * np.exp(-0.125 * (((np.arange(17)-6)[:,None])**2 + ((np.arange(17)-6)[None,:])**2))
    truepsf += 2.  * np.exp(-0.5   * (((np.arange(17)-10)[:,None])**2 + ((np.arange(17)-10)[None,:])**2))
    truedata = convolve(truescene, truepsf, mode="valid")
    data = truedata + 0.03 * np.random.normal(size=(truedata.shape))
    newPsf, newScene = inference_step(data, truescene, 1., plot="functional01.png")
    funkyscene = np.zeros((48, 48))
    funkyscene[15:32,15:32] = truepsf
    newPsf, newScene = inference_step(data, funkyscene, 1., plot="functional02.png")
    print 'functional_tests(): all tests run (look at images to assess)'
    return None

if __name__ == '__main__':
    unit_tests()
    ## functional_tests()
    import os
    from data import Image
    images = Image.get_all()
    hw = 13
    psf = np.zeros((2*hw+1, 2*hw+1))
    psf[hw,hw] = 1.
    scene = convolve(psf, images[0].image, mode="full")
    try:
        os.makedirs("img")
    except os.error:
        pass
    for count, img in enumerate(images[1:]):
        print "starting work on img", count
        data = img.image
        psf, newScene = inference_step(data, scene, (1. / 32.), plot="img/%04d.png"%count)
        ndata = 2 + count
        scene = ((ndata - 1.) / ndata) * scene + (1. / ndata) * newScene

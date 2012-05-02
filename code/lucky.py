"""
This file is part of the Lucky Imaging project.

Issues:
-------
- The initialization and optimization is handled in the __main__ code;
  that's bad.
- Initialization involves np.median() and np.clip(), as does the first pass
  (pindex == 1) of optimization.  Insane.
- There are hard-coded numbers everywhere, especially the stupid "300" and
  some L2 norms.
- centroiding is a hack-orama; replace with a function that takes bigdata,
  scene and does one-d convolves and returns xc, yc
- sky fitting and display of results when sky has been fit needs to be
  audited -- should be able to add a sky offset and get IDENTICAL results out.
- region of image set by borderx, bordery; hard-coded! MAGIC NUMBERS abound.
- The infer functions ought to take weight vectors -- this would permit
  dropping data for cross-validation tests and also inclusion of an error
  model.
- I think it memory leaks at least a bit (`Image`s don't get deleted?).

notes:
------
- The NNLS implementation may require citation.
- Search code for "hack" and "magic" for issues.

"""

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import rc
    rc('font', family="serif", serif="Computer Modern Roman", size=12)
    rc('text', usetex=True)

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import lsqr
from scipy.signal import convolve
import scipy.optimize as op
import pyfits


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

    Bug: Shouldn't make the kernels at every call; these should be
    static or passed in.

    ### input:

    * `data`: An individual image.
    * `scene`: The current best guess for the scene that (after PSF
      convolution) generates the image.

    ### output:

    * `psf`:
    '''
    # make kernel
    # magic numbers `-0.5` and `(3,-4)` in next line; implicitly
    # sigma = 1. pix Gaussian
    kernel = np.exp(-0.5 * (np.arange(-3, 4)[:, None] ** 2
        + np.arange(-3, 4)[None, :] ** 2))
    kernel /= np.sum(kernel)
    Kx, Ky = kernel.shape
    tinykernel = np.zeros_like(kernel)
    tinykernel[(Kx - 1) / 2, (Ky - 1) / 2] = 1.

    # deal with all the size and shape setup
    Nx, Ny = scene.shape
    Px, Py = data.shape
    Mx, My = (Nx - Px + 1, Ny - Py + 1)
    # psfShape = (Mx, My)
    # psfSize = Mx * My
    Qx, Qy = (Mx - Kx + 1, My - Ky + 1)
    assert(Qx > 0)
    assert(Qy > 0)
    psfParameterShape = (Qx, Qy)
    psfParameterSize = Qx * Qy

    # build scene matrix from kernel-convolved scene
    kernelConvolvedScene = convolve(scene, kernel, mode="same")
    sceneMatrix = np.zeros((data.size + psfParameterSize,
                            psfParameterSize + 1))
    for k in range(psfParameterSize):
        dx, dy = index2xy(psfParameterShape, k)
        dx -= Qx / 2
        dy -= Qx / 2
        sceneMatrix[:data.size, k] = kernelConvolvedScene[
                (Mx / 2 + dx):(Mx / 2 + dx + Px),
                (My / 2 + dy): (My / 2 + dy + Py)].reshape(data.size)

    # sky fitting
    sceneMatrix[:data.size, psfParameterSize] = 1.

    # L2 regularization
    sceneMatrix[data.size:(data.size + psfParameterSize), :psfParameterSize] \
            = l2norm * np.identity(psfParameterSize)

    # infer PSF and return
    dataVector = np.append(data.reshape(data.size), np.zeros(psfParameterSize))
    newPsfParameter, rnorm = op.nnls(sceneMatrix, dataVector)
    print "infer_psf(): dropping sky level", newPsfParameter[psfParameterSize]
    newPsfParameter = newPsfParameter[:psfParameterSize]  # drop sky
    newPsf = convolve(newPsfParameter[::-1].reshape(psfParameterShape),
            kernel, mode="full")
    newDeconvolvedPsf = convolve(
            newPsfParameter[::-1].reshape(psfParameterShape),
            tinykernel, mode="full")
    print "infer_psf(): got PSF", \
            newPsf.shape, np.min(newPsf), np.median(newPsf), np.max(newPsf)
    return newPsf, newDeconvolvedPsf


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
    * `l2norm`: Amplitude for the (required) L2 regularization.

    ### output:

    * `scene`
    '''
    # deal with all the size and shape setup
    assert(l2norm > 0.)
    Px, Py = data.shape
    Mx, My = psf.shape
    Nx, Ny = (Px + Mx - 1, Py + My - 1)
    sceneShape = (Nx, Ny)
    sceneSize = Nx * Ny

    # build psf matrix from psf
    psfX, psfY = index2xy(psf.shape, np.arange(psf.size))
    psfVector = psf.reshape(psf.size)[::-1]  # HACK
    vals = np.zeros(data.size * psf.size)
    rows = np.zeros_like(vals).astype(int)
    cols = np.zeros_like(vals).astype(int)
    for k in range(data.size):
        dx, dy = index2xy(data.shape, k)
        s = slice(k * psf.size, (k + 1) * psf.size)
        vals[s] = psfVector
        rows[s] = k
        cols[s] = xy2index(sceneShape, psfX + dx, psfY + dy)

    # add entries for old-scene-based regularization
    vals = np.append(vals, np.zeros(sceneSize) + l2norm)
    rows = np.append(rows, np.arange(data.size, data.size + sceneSize))
    cols = np.append(cols, np.arange(sceneSize))

    psfMatrix = csr_matrix((vals, (rows, cols)),
            shape=(data.size + sceneSize, sceneSize))
    print 'infer_scene(): constructed psfMatrix:', min(rows), max(rows), \
            data.size, sceneSize, min(cols), max(cols), sceneSize

    # infer scene and return
    dataVector = np.append(data.reshape(data.size), np.zeros(sceneSize))
    (newScene, istop, niters, r1norm, r2norm, anorm, acond,
     arnorm, xnorm, var) = lsqr(psfMatrix, dataVector)
    newScene = newScene.reshape(sceneShape)
    newScene -= np.median(newScene)
    print "infer_scene(): got scene", newScene.shape, np.min(newScene), \
            np.median(newScene), np.max(newScene)
    return newScene


def inference_step(data, oldScene, alpha, psfL2norm, sceneL2norm, nonNegative,
                   reconvolve=None, plot=None, splot=None, runUnitTest=False):
    '''
    # `inference_step()`:

    Concatenation of `infer_psf()` and `infer_scene()`.  Applies
    `alpha` times the newly inferred scene to `(1. - alpha)` times the
    old scene.  Possibly also regularizes with L2norm and
    non-negativity.

    ### inputs:

    * `data`: Image data
    * `oldScene`: First guess at scene, or scene from previous
      iteration; must be (substantially) larger than data image.
    * `alpha`: fraction of a full step to take; should be something
      like `1./nIteration`.
    * `psfl2norm`: Amplitude of L2 regularization for PSF; units TBA.
    * `nonNegative`: If `True`, apply non-negative clip.  Harsh!.
    * `reconvolve`: Don't ever use this.
    * `plot`: If not `None`, make a standard plot with this name.
    * `splot`: If not `None`, make a hard-stretch plot with this name.
    * `runUnitTest`: If `True`, pass forward unit test requests to
      sub-functions.

    ### outputs:

    * `newPsf`: inferred PSF.
    * `newScene`: updated scene.
    '''
    assert(alpha > 0.)
    assert(alpha <= 1.)
    foo, newPsf = infer_psf(data, oldScene, psfL2norm, runUnitTest=runUnitTest)
    thisScene = infer_scene(data, newPsf, sceneL2norm)
    print "inference_step(): updating with", alpha, psfL2norm, nonNegative
    if reconvolve is not None:
        thisScene = convolve(thisScene, reconvolve, mode="same")
    newScene = (1. - alpha) * oldScene + alpha * thisScene
    newScene -= np.median(newScene)  # brutal hack
    if nonNegative:
        # this is ugly but apparently okay
        newScene = np.clip(newScene, 0.0, np.Inf)
        print 'inference_step(): clipped scene to non-negative'
    print 'inference_step(): new scene:', np.min(newScene), \
            np.median(newScene), np.max(newScene)
    if plot is not None:
        plot_inference_step(data, thisScene, newPsf, newScene, plot)
    if splot is not None:
        plot_inference_step(data, thisScene, newPsf, newScene, splot,
                stretch=3.)
    return newPsf, newScene


def plot_inference_step(data, thisScene, thisPsf, newScene, filename,
        stretch=None):
    '''
    # `plot_inference_step()`:

    Make plots for `inference_step()`.
    '''
    fig = plt.figure(figsize=(12, 8))
    plt.clf()
    ax1 = fig.add_subplot(231)

    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    def hogg_plot_image_histeq(ax, im):
        shape = im.shape
        size = im.size
        foo = np.zeros(size)
        foo[np.argsort(im.reshape(size))] = np.arange(size)
        return ax.imshow(foo.reshape(shape), cmap="gray",
                interpolation="nearest")

    def hogg_plot_image(ax, im, stretch):
        if stretch is None:
            a = np.median(im)
            b = np.max(im)
            vmin = a - b
            vmax = a + b
        else:
            a = np.median(im)
            b = np.sort(im.reshape(im.size))[0.95 * im.size]
            vmin = a - 3. * b / stretch
            vmax = a + 3. * b / stretch
        return ax.imshow(im, cmap="gray", interpolation="nearest", vmin=vmin,
                vmax=vmax)

    hw1 = (thisPsf.shape[0] - 1) / 2
    hw2 = (data.shape[0] - thisPsf.shape[0] - 1) / 2
    hogg_plot_image(ax1, data, stretch)
    ax1.set_title("data")
    hogg_plot_image(ax2, convolve(scene, thisPsf, mode="valid"), stretch)
    ax2.set_title(r"[inferred PSF] $\ast$ [previous scene]")
    hogg_plot_image(ax3, convolve(newScene, thisPsf, mode="valid"), stretch)
    ax3.set_title(r"[inferred PSF] $\ast$ [inferred scene]")
    bigPsf = np.zeros_like(data)
    bigPsf[hw2:hw2 + thisPsf.shape[0], hw2:hw2 + thisPsf.shape[1]] = thisPsf
    hogg_plot_image(ax4, bigPsf, stretch)
    ax4.set_title("inferred PSF (deconvolved)")
    hogg_plot_image(ax5, thisScene[hw1:-hw1, hw1:-hw1], stretch)
    ax5.set_title("inferred scene (cropped)")
    hogg_plot_image(ax6, newScene[hw1:-hw1, hw1:-hw1], stretch)
    ax6.set_title("updated scene (cropped)")
    hogg_savefig(filename)
    return None


def hogg_savefig(fn):
    '''
    # `hogg_savefig()`:

    Hogg likes a verbose `savefig()`!
    '''
    print "hogg_savefig(): writing %s" % fn
    return plt.savefig(fn)


def save_scene(image, fn, clobber=True):
    '''
    # `save_scene()`:

    Given an image (2D `numpy.ndarray`) and a file name, write a fits file.

    Optionally, clobber the file if it already exists. NOTE: this is the
    _default_ behavior here but not in `pyfits`.

    '''
    if not fn[-5:].lower() == ".fits":
        fn += ".fits"
    hdu = pyfits.PrimaryHDU(image)
    hdu.writeto(fn, clobber=clobber)
    print "save_scene(): wrote %s" % fn
    return None


def read_scene(fn):
    '''
    # `read_scene(fn)`:

    Read a scene image from a FITS file and return as a `numpy.ndarray`.

    '''
    f = pyfits.open(fn)
    data = np.array(f[0].data, dtype=float)
    print "read_scene(): read %s" % fn
    return data


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
    xgrid = np.zeros(shape) + np.arange(Nx)[:, None]
    ygrid = np.zeros(shape) + np.arange(Ny)[None, :]
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

    # Unit tests for save/read scene functions.
    fn = ".unit_test.fits"
    save_scene(scene, fn)
    loadedScene = read_scene(fn)
    try:
        os.remove(fn)
    except:
        pass
    assert np.sum(np.abs(scene - loadedScene)) < 1e-10

    newPsf, newScene = inference_step(data, scene, 0.001, 0.001,
            runUnitTest=True)
    print "psf:", newPsf.shape, np.min(newPsf), np.max(newPsf)
    print "scene:", newScene.shape, np.min(newScene), np.max(newScene)
    # assert(np.all(newPsf >= 0)) # WHY DOES THIS FAIL FOR l_bfgs_b?
    assert(np.all(newPsf > -1.e-3))  # should be more stringent
    assert(np.all(newPsf < 1.e-3))  # should be more stringent
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
    truepsf = 1. * np.exp(-0.5 * (((np.arange(17) - 8)[:, None]) ** 2 \
            + ((np.arange(17) - 5)[None, :]) ** 2))
    truepsf += 1. * np.exp(-0.125 * (((np.arange(17) - 6)[:, None]) ** 2 \
            + ((np.arange(17) - 6)[None, :]) ** 2))
    truepsf += 2. * np.exp(-0.5 * (((np.arange(17) - 10)[:, None]) ** 2 \
            + ((np.arange(17) - 10)[None, :]) ** 2))
    truedata = convolve(truescene, truepsf, mode="valid")
    data = truedata + 0.03 * np.random.normal(size=(truedata.shape))
    newPsf, newScene = inference_step(data, truescene, 1., 1.,
            plot="functional01.png")
    funkyscene = np.zeros((48, 48))
    funkyscene[15:32, 15:32] = truepsf
    newPsf, newScene = inference_step(data, funkyscene, 1., 1.,
            plot="functional02.png")
    print 'functional_tests(): all tests run (look at images to assess)'
    return None

if __name__ == '__main__':
    import sys
    import os
    from data import get_image_list, get_image

    if "--test" in sys.argv:
        unit_tests()
        ## functional_tests()
        sys.exit(0)

    # Default data path to the Mars dataset.
    bp = os.getenv("LUCKY_DATA",
            "/data2/dfm/lucky/bpl1m001-en07-20120304/unspooled")
    img_dir = "mars"
    center = True
    binary = False
    trinary = False
    if "--binary" in sys.argv:
        bp = os.getenv("BINARY_DATA", "/data2/dfm/lucky/binary")
        img_dir = "binary"
        center = False
        binary = True
    if "--binary_short" in sys.argv:
        bp = "/data2/dfm/lucky/binary_short"
        img_dir = "binary_short"
        center = False
        binary = True
    if "--triple" in sys.argv:
        bp = os.getenv("TRIPLE_DATA", "/data2/dfm/lucky/triple")
        img_dir = "triple"
        center = False
        trinary = True

    try:
        os.makedirs(img_dir)
    except os.error:
        pass

    hw = 13
    psf = np.zeros((2 * hw + 1, 2 * hw + 1))
    psf[hw, hw] = 1.
    defaultpsf = psf
    defaultpsf[hw, hw] = 1.
    defaultpsf[hw - 1, hw] = 1.
    defaultpsf[hw + 1, hw] = 1.
    defaultpsf[hw, hw - 1] = 1.
    defaultpsf[hw, hw + 1] = 1.
    size = 100
    sky = 1.
    if trinary:
        size = 64
        sky = 7.
    # do the full inference
    for pindex in (1, 2, 3, 4, 5):
        savefn = "pass%1d.fits" % pindex
        if os.path.exists(savefn):
            scene = read_scene(savefn)
        else:
            for count, fn in enumerate(get_image_list(bp)):
                bigdata = get_image(fn, center=center)
                print "__main__: bigdata median", np.median(bigdata)
                # must be square or else something is f**king up
                assert(bigdata.shape[0] == bigdata.shape[1])
                if count == 0:
                    # initialization is insane here; this could be improved
                    # NOTE MAGIC NUMBERS
                    borderx = (bigdata.shape[0] - size) / 2
                    bordery = borderx
                    if binary:
                        borderx, bordery = 42, 65  # hard coded MAGIC NUMBERS
                    if trinary:
                        # hard coded MAGIC NUMBERS
                        borderx, bordery = 74 + 18, 66 + 18
                    data = bigdata[borderx:borderx + size,
                                   bordery:bordery + size]
                    dataShape = data.shape
                    # initialize scene -- should be its own function
                    if pindex == 1:
                        scene = convolve(data, defaultpsf, mode="full")
                        scene -= np.median(scene)
                        scene = np.clip(scene, 0.0, np.Inf)
                    foo = convolve(bigdata, scene, mode="valid")
                    mi = np.argmax(foo)
                    x0, y0 = mi / foo.shape[1], mi % foo.shape[1]
                else:
                    # if this difference isn't large, the centroiding is
                    # useless
                    assert((bigdata.shape[0] - scene.shape[0]) > 20)
                    smoothscene = convolve(scene, defaultpsf, mode="same")
                    mi = np.argmax(convolve(bigdata, smoothscene,
                        mode="valid"))
                    xc, yc = (mi / foo.shape[1]) - x0, (mi % foo.shape[1]) - y0
                    print "__main__: got centroid shift", (xc, yc)
                    data = bigdata[borderx + xc:borderx + xc + size,
                                   bordery + yc:bordery + yc + size]
                # if this isn't true then some edges got hit
                assert(data.shape == dataShape)
                if pindex == 1:
                    alpha = 2. / (1. + float(count))
                    nn = True
                else:
                    alpha = 2. / 300.  # HACK-O-RAMA
                    nn = False
                    scene -= np.median(scene)  # hack
                if alpha > 0.25:
                    alpha = 0.25
                data += sky  # hack
                plot = None
                splot = None
                if (count % 10) == 0:
                    plot = os.path.join(img_dir, "pass%1d_%04d.png"
                            % (pindex, count))
                    splot = os.path.join(img_dir, "pass%1d_%04ds.png"
                            % (pindex, count))
                psf, scene = inference_step(data, scene, alpha,
                                            1. / 4., 1. / 64., nn,
                                            plot=plot, splot=splot)
                print bigdata.shape, data.shape, psf.shape, scene.shape
                del bigdata
            save_scene(scene, savefn)

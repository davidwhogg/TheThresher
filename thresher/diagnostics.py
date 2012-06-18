import numpy as np
import scipy.optimize as op


X, Y = np.meshgrid(range(-1, 2), range(-1, 2))
X, Y = X.flatten(), Y.flatten()
A = np.vstack([np.ones(9), X, Y, X * X, X * Y, Y * Y]).T
ATAinvA = np.dot(np.linalg.inv(np.dot(A.T, A)), A.T)


def centroid_source(img, x0, y0):
    """
    Find the centroid of a source by doing a 2nd order fit.

    ## Arguments

    * `img` (numpy.ndarray): The data.
    * `x0` (int): The brightest pixel in the zeroth dimension of `img`.
    * `y0` (int): The brightest pixel in the first dimension of `img`.

    ## Returns

    * `xc` (float): The center of the source in the zeroth dimension of `img`.
    * `yc` (float): The same in the first dimension.

    """
    x0, y0 = int(x0), int(y0)
    patch = img[x0 - 1:x0 + 2, y0 - 1:y0 + 2]
    a, b, c, d, e, f = np.dot(ATAinvA, patch.flatten())
    yc = (2 * c * d + e * b) / (e * e - 4 * f * d)
    xc = (2 * b * f + e * c) / (e * e - 4 * f * d)
    assert np.abs(xc) < 1.5 and np.abs(yc) < 1.5, \
            "(x0, y0) = ({0}, {1}) and (dx, dy) = ({2}, {3})" \
            .format(x0, y0, xc, yc)
    return x0 + xc, y0 + yc


def find_sources(img, K, delta=3, padding=0):
    """
    Detect the K brightest sources in an image using brightest pixel and then
    refined using `centroid_source`.

    ## Arguments

    * `img` (numpy.ndarray): The image.
    * `K` (int): The number of sources to find.

    ## Keyword Arguments

    * `delta` (float): The minimum distance between sources (in pixels).
    * `padding` (float): The number of pixels around the outer edge of the
      image to ignore.

    ## Returns

    * `coords` (numpy.ndarray): List of source positions with two columns for
      the zeroth and first dimensions of `img` respectively.

    """
    # Account for the size of centroiding patch.
    padding += 1

    # Find the brightest pixels.
    inds = np.argsort(img.flatten())[::-1]
    coords = np.array([(-delta, -delta)])

    # Loop over the brightest pixels until we find `K` acceptable sources.
    for i in inds[1:]:
        tmp = np.array(np.unravel_index(i, img.shape))
        if np.all(np.sum((coords - tmp) ** 2, axis=1) > delta ** 2) and \
                padding <= tmp[0] < img.shape[0] - padding and \
                padding <= tmp[1] < img.shape[1] - padding:
            # Refine the centroiding.
            try:
                res = centroid_source(img, *tmp)
            except:
                pass
            else:
                coords = np.concatenate([coords, [res]], axis=0)
        if len(coords) >= K + 1:
            break

    return coords[1:]


def _generate(shape, coords, p):
    result = np.zeros(shape)
    x, y = np.arange(shape[0]), np.arange(shape[1])
    w2 = p[0] ** 2
    norm = 1.0 / (2 * np.pi * w2)
    gamm = -0.5 / w2

    for i, c in enumerate(coords):
        r2 = ((x - c[0]) ** 2)[:, None] + ((y - c[1]) ** 2)[None, :]
        result += p[i + 1] * np.exp(gamm * r2) * norm

    return result


def _synthesize_patch(shape, coords, flux, sigma2):
    x, y = np.arange(shape[0]), np.arange(shape[1])
    r2 = ((x - coords[0]) ** 2)[:, None] + ((y - coords[1]) ** 2)[None, :]
    result = flux * np.exp(-0.5 * r2 / sigma2) / (2 * np.pi * sigma2)
    return result


def _chi(p, coords, img):
    sigma2 = 1.0  # p[0] ** 2
    fluxes = p[1:]

    patch_size = 3
    shape = [2 * patch_size] * 2
    npatch = np.prod(shape)

    chi = np.zeros(len(coords) * npatch)

    for i, (xc, yc) in enumerate(coords):
        xmn = int(xc - patch_size)
        xmx = xmn + 2 * patch_size
        ymn = int(yc - patch_size)
        ymx = ymn + 2 * patch_size
        data = img[xmn:xmx, ymn:ymx]
        patch = _synthesize_patch(shape, (xc - xmn, yc - ymn), fluxes[i],
                sigma2)
        chi[i * npatch:(i + 1) * npatch] = (data - patch).flatten()

    return chi


def measure_sources(img, K, w=3., padding=None):
    """
    Find and measure the flux of the K brightest sources in an image.

    ## Arguments

    * `img` (numpy.ndarray): The image.
    * `K` (int): The number of sources to find.

    ## Keyword Arguments

    * `w` (float): Initial guess for the width of sources (in pixels).

    """
    if padding is None:
        padding = 10
    # MAGIC: exclude sources within `3 * w` pixels of known ones.
    coords = find_sources(img, K, 3 * w)

    # Fit for the fluxes.
    p0 = 100 + 400 * np.ones(K + 1)
    p0[0] = w

    p1 = op.leastsq(_chi, p0, args=(coords, img))
    p1 = p1[0]

    return p1[0] ** 2, np.hstack((coords, np.atleast_2d(p1[1:]).T))


if __name__ == "__main__":
    import matplotlib.pyplot as pl

    # Test centroid.
    x0, y0 = 3.1, 6.2
    img = _generate((10, 10), [[x0, y0]], [1.0, 1.0])
    xc, yc = centroid_source(img, np.floor(x0), np.floor(y0))
    print("Error in centroid: {0}".format((np.abs(xc - x0), np.abs(yc - y0))))
    pl.imshow(img, interpolation="nearest", cmap="gray")
    pl.plot(yc, xc, "+b")
    pl.plot(y0, x0, "+r")
    pl.savefig("centroid.png")

    # Test full pipeline.
    # Generate a test catalog.
    nx, ny = 200, 300
    K0 = 20
    w0 = 1
    truth = np.random.rand(K0 * 3).reshape((K0, 3))
    truth[:, 0] *= nx
    truth[:, 1] *= ny
    truth[:, 2] *= 400
    truth[:, 2] += 100

    # Generate the image.
    img = _generate((nx, ny), truth, np.append(w0, truth[:, 2]))

    # Add noise.
    img += 1 * np.random.randn(nx * ny).reshape(img.shape)

    # Measure the sources.
    K = 3
    w, coords = measure_sources(img, K, w=w0)

    print "Truth:"
    print truth[np.argsort(truth[:, 2])[::-1][:K]]
    print "Detections:"
    print coords
    print "Source size:",
    print w0, w

    pl.figure()
    pl.subplot(121)
    pl.imshow(img, interpolation="nearest", cmap="gray")

    pl.subplot(122)
    pl.imshow(_generate(img.shape, coords, np.append(w, coords[:, 2])),
        interpolation="nearest", cmap="gray")

    pl.savefig("detect.pdf")

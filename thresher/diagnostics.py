import numpy as np
import scipy.optimize as op
from scipy.ndimage.filters import gaussian_filter


def find_sources(img, K, delta=3):
    """
    Detect the K brightest sources in an image.

    ## Arguments

    * `img` (numpy.ndarray): The image.
    * `K` (int): The number of sources to find.

    ## Keyword Arguments

    * `delta` (float): The minimum distance between sources (in pixels).

    """
    inds = np.argsort(img.flatten())[::-1]
    coords = np.array([np.unravel_index(inds[0], img.shape)])

    for i in inds[1:]:
        tmp = np.array(np.unravel_index(i, img.shape))
        if not np.any(np.sum(np.abs(coords - tmp), axis=1) < delta):

            coords = np.concatenate([coords, [tmp]], axis=0)
        if len(coords) >= K:
            break

    return coords


def _generate_image(shape, coords, p):
    # Build the delta function map.
    delta_map = np.zeros(shape)
    delta_map[coords[:, 0].astype(int), coords[:, 1].astype(int)] = p[1:]

    return gaussian_filter(delta_map, p[0])


def measure_sources(img, K, w=1.):
    """
    Find and measure the flux of the K brightest sources in an image.

    ## Arguments

    * `img` (numpy.ndarray): The image.
    * `K` (int): The number of sources to find.

    ## Keyword Arguments

    * `w` (float): Initial guess for the width of sources (in pixels).

    """
    # MAGIC: exclude sources within `6 * w` pixels of known ones.
    coords = find_sources(img, K, 3 * w)

    # Fit for the fluxes.
    p0 = np.ones(K + 1)
    p0[0] = w
    bounds = [(1, None)] + [(0, None) for i in range(K)]

    chi2 = lambda p: np.sum((img - _generate_image(img.shape, coords, p)) ** 2)
    p1 = op.fmin_l_bfgs_b(chi2, p0, approx_grad=True, bounds=bounds)
    p1 = p1[0]

    return p1[0], np.hstack((coords, np.atleast_2d(p1[1:]).T))


if __name__ == "__main__":
    import matplotlib.pyplot as pl

    nx, ny = 200, 300
    K0 = 10
    w0 = 3
    truth = np.random.rand(K0 * 3).reshape((K0, 3))
    truth[:, 0] *= nx
    truth[:, 1] *= ny
    truth[:, 2] *= 500
    img = _generate_image((nx, ny), truth, np.append(w0, truth[:, 2]))
    img += 1 * np.random.randn(nx * ny).reshape(img.shape)

    w, coords = measure_sources(img, 10, w=w0)
    print truth[np.argsort(truth[:, 2])[::-1][:10]]
    print coords
    print w0, w

    pl.figure()
    pl.subplot(121)
    pl.imshow(img, interpolation="nearest", cmap="gray")

    pl.subplot(122)
    pl.imshow(_generate_image(img.shape, truth, np.append(w0, truth[:, 2])) -
            _generate_image(img.shape, coords, np.append(w, coords[:, 2])),
        interpolation="nearest", cmap="gray")

    pl.savefig("detect.pdf")

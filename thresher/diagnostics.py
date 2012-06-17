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
    coords = np.array([(-delta, -delta)])

    for i in inds[1:]:
        tmp = np.array(np.unravel_index(i, img.shape))
        if not np.any(np.sum(np.abs(coords - tmp), axis=1) < delta):
            rng = [max(0, tmp[0] - delta),
                   min(img.shape[0], tmp[0] + delta),
                   max(0, tmp[1] - delta),
                   min(img.shape[1], tmp[1] + delta)]
            smimg = img[rng[0]:rng[1], rng[2]:rng[3]]
            Y, X = np.meshgrid(range(rng[2], rng[3]), range(rng[0], rng[1]))
            norm = np.sum(smimg)
            x0, y0 = np.sum(X * smimg) / norm, np.sum(Y * smimg) / norm
            coords = np.concatenate([coords, [[x0, y0]]], axis=0)
        if len(coords) >= K + 1:
            break

    return coords[1:]


def _generate_exact(shape, coords, p):
    result = np.zeros(shape)
    x, y = np.arange(shape[0]), np.arange(shape[1])
    w2 = p[0]
    norm = np.sqrt(2 * np.pi * w2)
    gamm = -0.5 / w2
    for i, c in enumerate(coords):
        r2 = ((x - c[0]) ** 2)[:, None] + ((y - c[1]) ** 2)[None, :]
        result += p[i + 1] * np.exp(gamm * r2) / norm

    return result


def _generate_image(shape, coords, p):
    # Build the delta function map.
    delta_map = np.zeros(shape)
    delta_map[coords[:, 0].astype(int), coords[:, 1].astype(int)] = p[1:]

    return gaussian_filter(delta_map, p[0])


def measure_sources(img, K, w=3.):
    """
    Find and measure the flux of the K brightest sources in an image.

    ## Arguments

    * `img` (numpy.ndarray): The image.
    * `K` (int): The number of sources to find.

    ## Keyword Arguments

    * `w` (float): Initial guess for the width of sources (in pixels).

    """
    # MAGIC: exclude sources within `3 * w` pixels of known ones.
    coords = find_sources(img, K, 3 * w)

    # Fit for the fluxes.
    p0 = np.ones(K + 1)
    p0[0] = w
    bounds = [(1, None)] + [(0, None) for i in range(K)]

    chi2 = lambda p: np.sum((img - _generate_exact(img.shape, coords, p)) ** 2)
    p1 = op.fmin_l_bfgs_b(chi2, p0, approx_grad=True, bounds=bounds)
    p1 = p1[0]

    return p1[0], np.hstack((coords, np.atleast_2d(p1[1:]).T))


if __name__ == "__main__":
    import matplotlib.pyplot as pl

    nx, ny = 200, 300
    K0 = 10
    w0 = 4
    truth = np.random.rand(K0 * 3).reshape((K0, 3))
    truth[:, 0] *= nx
    truth[:, 1] *= ny
    truth[:, 2] *= 500
    img = _generate_exact((nx, ny), truth, np.append(w0, truth[:, 2]))
    img += 1 * np.random.randn(nx * ny).reshape(img.shape)

    w, coords = measure_sources(img, 10, w=w0)
    print truth[np.argsort(truth[:, 2])[::-1][:10]]
    print coords
    print w0, w

    pl.figure()
    pl.subplot(121)
    pl.imshow(img, interpolation="nearest", cmap="gray")

    pl.subplot(122)
    pl.imshow(_generate_exact(img.shape, coords, np.append(w, coords[:, 2])),
        interpolation="nearest", cmap="gray")

    pl.savefig("detect.pdf")

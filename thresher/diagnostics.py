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
    assert np.abs(xc) < 1 and np.abs(yc) < 1, \
            "(x0, y0) = ({0}, {1}) and (dx, dy) = ({2}, {3}) {4}" \
            .format(x0, y0, xc, yc, patch)
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
            except Exception:
                pass
            else:
                coords = np.concatenate([coords, [res]], axis=0)
        if len(coords) >= K + 1:
            break

    return coords[1:]


def _generate(shape, coords, p):
    result = np.zeros(shape)
    x, y = np.arange(shape[0]), np.arange(shape[1])
    norm = 1.0 / (2 * np.pi)
    gammas = [1.0 / p[0] ** 2, 1.0 / (p[0] ** 2 + p[1] ** 2)]
    strehl = logistic(p[2])
    # strehl = 0.5
    amps = [strehl, 1 - strehl]

    for i, c in enumerate(coords):
        r2 = ((x - c[0]) ** 2)[:, None] + ((y - c[1]) ** 2)[None, :]
        for j in range(2):
            result += amps[j] * p[i + 3] * np.exp(-0.5 * r2 * gammas[j]) \
                    * norm * gammas[j]

    return result


def _synthesize_patch(shape, coords, flux, sigma2):
    x, y = np.arange(shape[0]), np.arange(shape[1])
    r2 = ((x - coords[0]) ** 2)[:, None] + ((y - coords[1]) ** 2)[None, :]
    result = flux * np.exp(-0.5 * r2 / sigma2) / (2 * np.pi * sigma2)
    return result


def _get_patch(img, xc, yc, patch_size):
    xmn = int(xc - patch_size)
    xmx = xmn + 2 * patch_size
    ymn = int(yc - patch_size)
    ymx = ymn + 2 * patch_size
    data = img[xmn:xmx, ymn:ymx]
    return xmn, ymn, data


def _chi(p, coords, img):
    print p[:3]
    model = _generate(img.shape, coords, p)
    return (img - model).flatten()


def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))


def inv_logistic(x):
    return np.log(x / (1 - x))


def measure_sources(img, coords, w1=3., w2=5., strehl=0.5, padding=None):
    """
    Find and measure the flux of the K brightest sources in an image.

    ## Arguments

    * `img` (numpy.ndarray): The image.
    * `coords` (list):

    ## Keyword Arguments

    * `w` (float): Initial guess for the width of sources (in pixels).

    """
    if padding is None:
        padding = 10

    # Fit for the fluxes.
    p0 = np.mean(img) * np.ones(len(coords) + 3)
    p0[0] = w1
    p0[1] = w2
    p0[2] = inv_logistic(strehl)

    p1 = op.leastsq(_chi, p0, args=(coords, img), factor=0.1)
    p1 = p1[0]

    fluxes = p1[3:]
    inds = np.argsort(fluxes)[::-1]
    coords = np.array(coords)[inds]
    fluxes = fluxes[inds]

    return np.abs(p1[0]), np.abs(p1[1]), logistic(p1[2]), coords, fluxes


def robust_statistics(x, nsig=2.5):
    # Sigma-clipping.
    fs = np.array(x)
    mu = np.median(fs)
    std = np.max(fs) - np.min(fs)
    for i in range(100):
        inrange = (fs - mu > -nsig * std) * (fs - mu < nsig * std)
        mu = np.median(fs[inrange])
        newstd = np.sqrt(np.mean((fs[inrange] - mu) ** 2))
        if newstd - std == 0:
            print i
            break
        std = newstd

    return mu, newstd


def estimate_noise(img, w, N=1000, padding=None, full_output=False):
    if padding is None:
        padding = 10
    sigma2 = w ** 2

    patch_size = int(5 * np.abs(w))
    shape = [2 * patch_size] * 2

    fs = []
    patch = _synthesize_patch(shape, [0.5 * patch_size] * 2, 1.0, sigma2)
    while 1:
        xc, yc = padding + \
                np.random.rand(2) * (np.array(img.shape) - 2 * padding)
        xmn, ymn, data = _get_patch(img, xc, yc, patch_size)
        chi = lambda p: (data - float(p) * patch).flatten()
        try:
            p1 = op.leastsq(chi, [np.mean(data)])
        except:
            pass
        else:
            fs.append(float(p1[0]))
        if len(fs) >= N:
            break

    fs0 = np.array(fs)

    mu, noise = robust_statistics(fs, nsig=2.5)

    if full_output:
        return noise, mu, fs0

    return noise


def do_diagnostics(img, prefix, sources=None, nsources=10):
    # MAGIC: risky 2.5 sigma for sigma clipping.
    sky, sky_noise = robust_statistics(img.flatten(), nsig=2.5)
    img -= sky
    img /= sky_noise

    # Find and photometer the sources.
    w0 = 3
    if sources is None:
        coords = find_sources(img, nsources, 3 * w0)
    else:
        coords = np.array([centroid_source(img, *c) for c in sources])

    w1, w2, strehl, coords, fluxes = measure_sources(img, coords, w0)

    # Synthesize a PSF patch.
    # patch_size = int(5 * np.abs(w0))
    # shape = [2 * patch_size] * 2
    # patch = _synthesize_patch(shape, [0.5 * patch_size] * 2, 1.0, w ** 2)
    # print coords
    # print np.sum(patch), np.sum(patch ** 2)

    # patch_noise = sky_noise / np.sqrt(np.sum(patch ** 2))

    # print("S/N : {0}".format(prefix))
    # print(coords[:, 2] / patch_noise)

    delta = 1.0 / w1 ** 2
    vrange = [-0.001 * delta, 0.01 * delta]

    pl.figure(figsize=(10, 10))
    pl.subplot(221)
    pl.imshow(img / fluxes[0], interpolation="nearest", cmap="gray",
            vmin=vrange[0], vmax=vrange[1])
    pl.plot(coords[:, 1], coords[:, 0], "+r")
    [pl.text(coords[i, 1], coords[i, 0], str(i)) for i in range(len(coords))]
    pl.xlim(0, 79)
    pl.ylim(0, 79)
    pl.title(r"{0} $\sigma = {1:.2f}$".format(prefix, w1))

    pl.subplot(222)
    model = _generate(img.shape, coords,
            np.concatenate([[w1, w2, inv_logistic(strehl)], fluxes]))
    pl.imshow(model / fluxes[0], interpolation="nearest", cmap="gray",
            vmin=vrange[0], vmax=vrange[1])
    pl.plot(coords[:, 1], coords[:, 0], "+r")
    [pl.text(coords[i, 1], coords[i, 0], str(i)) for i in range(len(coords))]
    pl.xlim(0, 79)
    pl.ylim(0, 79)
    pl.title("Model")

    pl.subplot(223)
    chi = _chi(np.concatenate([[w1, w2, inv_logistic(strehl)], fluxes]),
            coords, img).reshape(img.shape)
    pl.imshow(chi, interpolation="nearest", cmap="gray",
            vmin=-5, vmax=5)
    pl.plot(coords[:, 1], coords[:, 0], "+r")
    [pl.text(coords[i, 1], coords[i, 0], str(i)) for i in range(len(coords))]
    pl.xlim(0, 79)
    pl.ylim(0, 79)
    pl.title("Chi")

    pl.savefig("{0}_img.pdf".format(prefix))


if __name__ == "__main__":
    import matplotlib.pyplot as pl
    import pyfits

    hdus = pyfits.open("test/thresh.fits")
    img = np.array(hdus[2].data, dtype=float)
    hdus.close()
    zero = img.shape[0] / 2
    img = img[zero - 40:zero + 40, zero - 40:zero + 40]

    coords = np.array([[40, 40], [33, 44], [29, 19]])

    do_diagnostics(img, "thresh", sources=coords)

    assert 0

    img -= np.median(img)
    thresh_pxl = img.flatten()
    w, coords = measure_sources(img, 5, w=3.0)

    noise_thresh, mu_thresh, fs_thresh = estimate_noise(img, w, full_output=True)
    print coords[0, 2] / noise_thresh

    img /= coords[0, 2]

    mu = np.median(img)
    delta = 1.0 / w ** 2

    pl.figure(figsize=(10, 5))
    pl.subplot(121)
    pl.imshow(img, interpolation="nearest", cmap="gray",
            vmin=mu - 0.2 * delta, vmax=delta)
    pl.plot(coords[:2, 1], coords[:2, 0], "+r")
    pl.xlim(0, 79)
    pl.ylim(0, 79)
    pl.title(r"Thresher --- $\sigma = {0:.2f}$".format(w))

    hdus = pyfits.open("test/tli.fits")
    img = np.array(hdus[2].data, dtype=float)
    hdus.close()
    img -= np.median(img)
    tli_pxl = img.flatten()
    w, coords = measure_sources(img, 5, w=3.0)

    noise_tli, mu_tli, fs_tli = estimate_noise(img, w, full_output=True)
    print coords[0, 2] / noise_tli

    img /= coords[0, 2]

    pl.subplot(122)
    pl.imshow(img, interpolation="nearest", cmap="gray",
            vmin=mu - 0.2 * delta, vmax=delta)
    pl.plot(coords[:2, 1], coords[:2, 0], "+r")
    pl.xlim(0, 79)
    pl.ylim(0, 79)
    pl.title(r"TLI --- $\sigma = {0:.2f}$".format(w))

    pl.savefig("diagnostics.pdf")

    # Pixel histograms.
    quantile = lambda qs, x: [x[np.argsort(x)][int(q * len(x))]
            for q in np.atleast_1d(qs)]
    pl.clf()
    rng = quantile([0., 0.95], thresh_pxl)
    pl.hist(thresh_pxl, 200, range=rng, normed=True, histtype="step")
    thresh_sky, thresh_sky_noise = robust_statistics(thresh_pxl)
    x = np.linspace(rng[0], rng[1], 5000)
    pl.plot(x, np.exp(-0.5 * (x - thresh_sky) ** 2 / thresh_sky_noise ** 2) / np.sqrt(2 * np.pi * thresh_sky_noise ** 2))
    pl.xlim(rng)
    pl.title("Thresher Pixels")
    pl.savefig("thresher_pxls.pdf")

    pl.clf()
    rng = quantile([0., 0.95], tli_pxl)
    pl.hist(tli_pxl, 500, range=rng)
    pl.xlim(rng)
    pl.title("TLI Pixels")
    pl.savefig("tli_pxls.pdf")

    # Plot histograms.
    pl.clf()
    rng = 5 * noise_thresh * np.array([-1, 1])
    n, b = np.histogram(fs_thresh, 100, normed=True, range=rng)
    b = 0.5 * (b[1:] + b[:-1])
    pl.plot(b, n, label="Thresher")
    x = np.linspace(rng[0], rng[1], 5000)
    pl.plot(x, np.exp(-0.5 * (x - mu_thresh) ** 2 / noise_thresh ** 2) / np.sqrt(2 * np.pi * noise_thresh ** 2))
    pl.title("Thresher")
    pl.savefig("thresh_hist.pdf")

    pl.clf()
    rng = 5 * noise_tli * np.array([-1, 1])
    n, b = np.histogram(fs_tli, 100, normed=True, range=rng)
    b = 0.5 * (b[1:] + b[:-1])
    pl.plot(b, n, label="TLI")
    x = np.linspace(rng[0], rng[1], 5000)
    pl.plot(x, np.exp(-0.5 * (x - mu_tli) ** 2 / noise_tli ** 2) / np.sqrt(2 * np.pi * noise_tli ** 2))
    pl.title("TLI")
    pl.savefig("tli_hist.pdf")

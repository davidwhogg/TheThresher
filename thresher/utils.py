__all__ = ["s2n", "resolution"]

import numpy as np


def _measure_counts(img, pos, diam):
    """
    Measure the counts within an annulus at a particular position.

    """
    ny, nx = img.shape
    pos_x, pos_y = np.meshgrid(np.arange(nx), np.arange(ny))
    r = np.sqrt((pos_x - pos[0]) ** 2 + (pos_y - pos[1]) ** 2)
    m = r < diam

    # Hack: I'm weighting the counts by how many pixels are being included.
    # My intuition is that this should roughly deal with edge effects.
    return np.sum(img[pos_y[m], pos_x[m]]) / np.sum(m)


def s2n(img, pos, diam=5, nnoise=100):
    """
    Approximate the S/N for an image.

    ## Arguments

    * `img` (numpy.ndarray): The 2D image to estimate the S/N for.
    * `pos` (tuple): The coordinates of the "signal".

    ## Keyword Arguments

    * `diam` (float): The diameter of the annulus within which the number of
      counts will be measured.
    * `nnoise` (int): The number of "noise" estimates to make.

    """
    ny, nx = img.shape
    median = np.median(img)

    # Get the counts at the source as an approximation of the "signal".
    s_counts = _measure_counts(img, pos, diam) - median

    # Estimate the noise.
    ncounts = np.zeros(nnoise)
    for i in range(nnoise):
        r = 0
        while r < diam:
            p0 = (np.random.randint(nx), np.random.randint(ny))
            r = np.sqrt((pos[0] - p0[0]) ** 2 + (pos[1] - p0[1]) ** 2)
        ncounts[i] = _measure_counts(img, p0, diam) - median

    return s_counts / np.sqrt(np.mean(ncounts ** 2))


def resolution():
    pass


if __name__ == "__main__":
    import matplotlib.pyplot as pl

    nx, ny = 50, 100
    pos = (20.3, 75.1)

    pos_x, pos_y = np.meshgrid(np.arange(nx), np.arange(ny))
    r = np.sqrt((pos_x - pos[0]) ** 2 + (pos_y - pos[1]) ** 2)

    w = 3
    s = 1000.
    img = s * np.exp(-0.5 * r ** 2 / w ** 2) / (2 * np.pi) / w ** 2

    # Add noise.
    n = 1.
    img += n * np.random.randn(img.size).reshape(img.shape)

    print s2n(img, pos)

    pl.imshow(img, interpolation="nearest")
    pl.savefig("sn.png")

"""
Data interface for the lucky imaging project.

Note: the filenames are currently _hard-coded_ for this particular project.

"""

__all__ = ["get_image_list", "get_image"]

import os

import numpy as np
import pyfits


def get_image_list(bp):
    entries = os.listdir(bp)
    r = []

    for e in sorted(entries):
        if os.path.splitext(e)[1] == ".fits":
            r.append(os.path.join(bp, e))

    return r


def get_image(fn, center=False):
    print ".image(): reading %s" % fn
    f = pyfits.open(fn)
    data = np.array(f[0].data, dtype=float)
    f.close()

    hw = 128
    if center:
        # Centroid and take an image section.
        s = np.sum(data)
        xc = np.argmin(np.abs(0.5 - np.cumsum(np.sum(data, axis=1)) / s))
        yc = np.argmin(np.abs(0.5 - np.cumsum(np.sum(data, axis=0)) / s))
    else:
        xc = data.shape[0] / 2
        yc = data.shape[1] / 2
    img = data[xc - hw:xc + hw, yc - hw:yc + hw]

    # re-scale flux values for no reason except to make the L2
    # norm more interpretable.
    rms = np.sqrt(np.mean((img - np.mean(img)) ** 2))
    img /= rms

    return img

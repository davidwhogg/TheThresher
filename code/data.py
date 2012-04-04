"""
Data interface for the lucky imaging project.

Note: the filenames are currently _hard-coded_ for this particular project.

"""

__all__ = ["Image"]

import os

import numpy as np

import pyfits

_bp = os.environ.get("LUCKY_DATA",
                "/data2/dfm/mars/bpl1m001-en07-20120304/unspooled")

class Image(object):
    """
    A simple wrapper around the FITS file for the Mars lucky imaging dataset.

    ## Arguments

    * `_id` (int): The

    """
    def __init__(self, _id):
        fn = "bpl1m001-en07-20120304-%04d-e00.fits"%_id
        full_path = os.path.join(_bp, fn)
        f = pyfits.open(full_path)

        # Grab the image and header from the FITS file.
        w = 64
        self.image = np.array(f[0].data[w:-w:2, w:-w:2], dtype=float)
        self.info = {}
        for k in f[0].header.keys():
            self.info[k] = f[0].header[k]

        f.close()

    def __getitem__(self, s):
        return self.image[s]

if __name__ == "__main__":
    import matplotlib.pyplot as pl

    i = Image(115)
    pl.imshow(i.image)

    pl.savefig("115.png")


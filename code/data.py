"""
Data interface for the lucky imaging project.

Note: the filenames are currently _hard-coded_ for this particular project.

"""

__all__ = ["Image"]

import os

import numpy as np

import pyfits

class Image(object):
    """
    A simple wrapper around the FITS file for the Mars lucky imaging dataset.

    ## Arguments

    * `_id` (int): The

    """

    _bp = os.environ.get("LUCKY_DATA",
                "/data2/dfm/mars/bpl1m001-en07-20120304/unspooled")
    _fn_format = "bpl1m001-en07-20120304-{0}-e00.fits"

    def __init__(self, _id):
        self.fn = self._fn_format.format("%04d"%_id)
        self.path = os.path.join(self._bp, self.fn)
        self._image = None

    @classmethod
    def get_all(cls):
        pass

    @property
    def image(self):
        """Lazily load the image file only when needed."""
        if self._image is not None:
            return self._image

        f = pyfits.open(self.path)

        # Grab the image and header from the FITS file.
        w = 64
        data = np.array(f[0].data, dtype=float)
        self.info = {}
        # The following is commented out because it doesn't work for Hogg on broiler.
        #for k in f[0].header.keys():
        #    self.info[k] = f[0].header[k]
        f.close()

        # Centroid and take an image section.
        s = np.sum(data)
        xc = np.argmin(np.abs(0.5 - np.cumsum(np.sum(data, axis=1)) / s))
        yc = np.argmin(np.abs(0.5 - np.cumsum(np.sum(data, axis=0)) / s))
        self.info['xc'] = xc
        self.info['yc'] = yc
        hw = 50
        self._image = data[xc-hw : xc+hw, yc-hw : yc+hw]

        return self._image

    def __getitem__(self, s):
        return self.image[s]

if __name__ == "__main__":
    import matplotlib.pyplot as pl

    i = Image(115)
    pl.imshow(i.image)

    pl.savefig("115.png")


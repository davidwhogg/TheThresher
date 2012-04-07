"""
Data interface for the lucky imaging project.

Note: the filenames are currently _hard-coded_ for this particular project.

"""

__all__ = ["Image"]

import os
import re

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

    def __init__(self, _id=None, fn=None):
        assert _id is not None or fn is not None
        if fn is None:
            fn = self._fn_format.format("%04d"%_id)
        self.path = os.path.join(self._bp, fn)
        self._image = None

    @classmethod
    def get_all(cls):
        """
        Get a list of `Image` objects for all of the FITS files in the base
        directory.

        """
        entries = os.listdir(cls._bp)
        result = []
        for e in entries:
            if os.path.splitext(e)[1] == ".fits":
                result.append(cls(fn=e))
        return result

    @property
    def image(self):
        """Lazily load the image file only when needed."""
        if self._image is not None:
            return self._image

        f = pyfits.open(self.path)

        # Grab the image and header from the FITS file.
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

        # re-scale flux values for no reason except to make the L2
        # norm more interpretable.
        rms = np.sqrt(np.mean((self._image - np.mean(self._image))**2))
        self.info['rms'] = rms
        self._image /= rms

        return self._image

    def __getitem__(self, s):
        return self.image[s]

if __name__ == "__main__":
    print Image.get_all()


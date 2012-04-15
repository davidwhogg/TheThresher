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
    def __init__(self, _id=None, fn=None,
            bp="/data2/dfm/lucky/bpl1m001-en07-20120304/unspooled",
            center=False):
        self._bp    = bp
        self.path   = os.path.join(self._bp, fn)
        self._image = None
        self._center = center

    def _clear(self):
        del self._image

    @classmethod
    def get_all(cls, bp="/data2/dfm/lucky/bpl1m001-en07-20120304/unspooled",
            center=False):
        """
        Get a list of `Image` objects for all of the FITS files in the base
        directory.

        """
        entries = os.listdir(bp)
        for e in entries:
            if os.path.splitext(e)[1] == ".fits":
                o = cls(fn=e, bp=bp, center=center)
                yield o
                o._clear()
                del o

    @property
    def image(self):
        """Lazily load the image file only when needed."""
        if self._image is not None:
            return self._image

        f = pyfits.open(self.path)

        # Grab the image and header from the FITS file.
        data = np.array(f[0].data, dtype=float)
        self.info = {}
        f.close()

        hw = 80
        if self._center:
            # Centroid and take an image section.
            s = np.sum(data)
            xc = np.argmin(np.abs(0.5 - np.cumsum(np.sum(data, axis=1)) / s))
            yc = np.argmin(np.abs(0.5 - np.cumsum(np.sum(data, axis=0)) / s))
        else:
            xc = data.shape[0] / 2 - 42 # MAGIC
            yc = data.shape[1] / 2 - 25 # MAGIC
        self.info['xc'] = xc
        self.info['yc'] = yc
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


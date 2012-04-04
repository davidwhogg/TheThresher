"""
Data interface for the lucky imaging project.

Note: the filenames are currently _hard-coded_ for this particular project.

"""

__all__ = ["Image"]

import os

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
        self.fn = self.fn_format.format("%04d"%_id)
        self.path = os.path.join(self._bp, self.fn)
        self._image = None

    @classmethod
    def get_all(cls):


    @property
    def image(self):
        """Lazily load the image file only when needed."""
        if self._image is not None:
            return self._image

        f = pyfits.open(self.path)

        # Grab the image and header from the FITS file.
        self._image = f[0].data
        self.info = {}
        for k in f[0].header.keys():
            self.info[k] = f[0].header[k]

        f.close()

        return self._image

    def __getitem__(self, s):
        return self.image[s]

if __name__ == "__main__":
    import matplotlib.pyplot as pl

    i = Image(115)
    pl.imshow(i.image)

    pl.savefig("115.png")


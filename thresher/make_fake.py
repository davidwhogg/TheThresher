"""
This file is part of The Thresher.
"""

import numpy as np
import pyfits as pyf

def make_image(shape, positions, fluxes, sky, sky_sigma, psf_sigma):
    """
    Make one image.
    """
    img = np.zeros(shape)
    img += sky
    img += sky_sigma * np.random.normal(size=shape)
    nx, ny = shape
    xg, yg = np.meshgrid(range(nx), range(ny))
    for pos, f in zip(positions, fluxes):
        kernel = np.exp(-0.5 * ((xg - pos[0]) ** 2 + (yg - pos[1]) ** 2))
        img += f * kernel / (2. * np.pi * psf_sigma ** 2)
    return img

def make_constant_psf_imaging_run(N, prefix):
    """
    Make a set of images with identical PSF and sources.
    Save in a set of FITS files.
    Lots of stuff hard-coded.
    """
    shape = (128, 128)
    positions = np.array([[13.1, 45.5],
                          [65.3, 61.2],
                          [51.5, 73.7]])
    fluxes = np.array([10., 100., 50.])
    sky_sigma = 1.0
    psf_sigma = 1.5
    for n in range(N):
        sky = 10. * np.random.uniform()
        offset = 20. * np.random.normal(size=(1,2))
        ps = positions + offset
        fn = "%s_%05d.fits" % (prefix, n)
        img = make_image(shape, ps, fluxes, sky, sky_sigma, psf_sigma)
        pyf.PrimaryHDU(img).writeto(fn, clobber=True)
    pass

if __name__ == "__main__":
    make_constant_psf_imaging_run(300, 'fake')

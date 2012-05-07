__all__ = ["plot_inference_step", "hogg_savefig"]

import logging

import numpy as np
from scipy.signal import convolve
import matplotlib.pyplot as plt


def hogg_plot_image_histeq(ax, im):
    shape = im.shape
    size = im.size
    foo = np.zeros(size)
    foo[np.argsort(im.reshape(size))] = np.arange(size)
    return ax.imshow(foo.reshape(shape), cmap="gray",
            interpolation="nearest")


def hogg_plot_image(ax, im, stretch):
    if stretch is None:
        a = np.median(im)
        b = np.max(im)
        vmin = a - b
        vmax = a + b
    else:
        a = np.median(im)
        b = np.sort(im.reshape(im.size))[0.95 * im.size]
        vmin = a - 3. * b / stretch
        vmax = a + 3. * b / stretch
    return ax.imshow(im, cmap="gray", interpolation="nearest", vmin=vmin,
            vmax=vmax)


def plot_inference_step(fig, data, thisScene, thisPsf, newScene, filename,
        stretch=None):
    """
    Make plots for `inference_step()`.

    """
    fig.clf()
    ax1 = fig.add_subplot(231)

    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    hw1 = (thisPsf.shape[0] - 1) / 2
    hw2 = (data.shape[0] - thisPsf.shape[0] - 1) / 2
    hogg_plot_image(ax1, data, stretch)
    ax1.set_title("data")

    # NOTE: Hogg, you had `scene` instead of `thisScene` in the line below
    # which was resolving to the _global_ scene object. I think that
    # `thisScene` is what we actually want though...
    hogg_plot_image(ax2, convolve(thisScene, thisPsf, mode="valid"), stretch)
    ax2.set_title(r"[inferred PSF] $\ast$ [previous scene]")
    hogg_plot_image(ax3, convolve(newScene, thisPsf, mode="valid"), stretch)
    ax3.set_title(r"[inferred PSF] $\ast$ [inferred scene]")
    bigPsf = np.zeros_like(data)
    bigPsf[hw2:hw2 + thisPsf.shape[0], hw2:hw2 + thisPsf.shape[1]] = thisPsf
    hogg_plot_image(ax4, bigPsf, stretch)
    ax4.set_title("inferred PSF (deconvolved)")
    hogg_plot_image(ax5, thisScene[hw1:-hw1, hw1:-hw1], stretch)
    ax5.set_title("inferred scene (cropped)")
    hogg_plot_image(ax6, newScene[hw1:-hw1, hw1:-hw1], stretch)
    ax6.set_title("updated scene (cropped)")
    hogg_savefig(filename)
    return None


def hogg_savefig(fn):
    """
    Hogg likes a verbose `savefig()`!

    """
    logging.info("hogg_savefig(): writing {0}".format(fn))
    return plt.savefig(fn)

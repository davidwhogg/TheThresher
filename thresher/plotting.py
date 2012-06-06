"""
This file is part of The Thresher

"""

__all__ = ["plot_inference_step"]

import itertools

import numpy as np
from scipy.signal import convolve
from scipy.special import erf


def plot_image(ax, img, size=None, vrange=None):
    """
    A wrapper to nicely plot an image the way that Hogg likes it.

    ## Arguments

    * `ax` (matplotlib.Axes): The axes to plot into.
    * `img` (numpy.ndarray): The image.

    ## Keyword Arguments

    * `size` (int): The size to crop/pad the image to.
    * `vrange` (tuple): The image stretch range.

    """
    if vrange is None:
        a, b = np.median(img), np.max(img)
        vmin, vmax = a - 0.2 * b, a + b
    else:
        vmin, vmax = vrange

    if size is None:
        size = np.mean(img.shape)

    # Invert the image.
    vmin, vmax = -vmax, -vmin

    ax.imshow(-img, cmap="gray", interpolation="nearest",
            vmin=vmin, vmax=vmax)

    # Crop/pad to the right size.
    xmin, xmax = (img.shape[0] - size) / 2, (img.shape[0] + size) / 2 - 1
    ymin, ymax = (img.shape[1] - size) / 2, (img.shape[1] + size) / 2 - 1
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def estimate_sigma(scene):
    # Let's estimate the variance of new_scene. Hack anyone?
    nsigma = np.arange(1, 5)
    fracs = 1 - 0.5 * (1 - erf(nsigma / np.sqrt(2)))
    median = np.median(scene)

    scene_sort = np.sort(scene.flatten())
    mask = np.array(fracs * scene.size, dtype=int)
    quantiles = scene_sort[mask]

    sigmas = (quantiles - median) / nsigma
    ind = np.arange(len(sigmas))[sigmas > 0][1]
    sigma = sigmas[ind]

    return sigma


def plot_inference_step(fig, data, this_scene, new_scene, dpsf, kernel,
        meta=[], sky=0.0):
    """
    Plot the images produced by a single update step.

    NOTE: The stretch is the _same_ in most of the plots.

    ## Arguments

    * `fig` (matplotlib.Figure): The figure to clear and plot into.
    * `data` (numpy.ndarray): The data image.
    * `this_scene` (numpy.ndarray): The scene implied by _this datapoint
      only_.
    * `new_scene` (numpy.ndarray): The updated scene.
    * `dpsf` (numpy.ndarray): The deconvolved PSF image.
    * `kernel` (numpy.ndarray): The user-defined kernel.

    """
    fig.clf()

    # Build the subplots.
    rows, cols = 3, 4
    axes = []

    for ri, ci in itertools.product(range(rows), range(cols)):
        axes.append(fig.add_subplot(rows, cols, ri * cols + ci + 1))
        axes[-1].set_xticklabels([])
        axes[-1].set_yticklabels([])

    # Calculate stretch.
    sigma = estimate_sigma(new_scene)
    scene_range = np.array([-2.5, 5]) * sigma

    # Set up which data will go in which panel.
    psf = convolve(dpsf, kernel, mode="same")
    norm = np.sum(dpsf)
    predicted = convolve(this_scene, psf, mode="valid") + sky
    delta = data - predicted
    panels = [[("PSF", psf),
        ("Data", data, np.median(data) + scene_range * norm),
        ("This Scene", this_scene, np.median(this_scene) + scene_range),
        ("This Scene", this_scene, np.median(this_scene) + 0.1 * scene_range)],
        [("dPSF", dpsf),
        (r"Predicted Data", predicted, np.median(data) + scene_range * norm),
        ("New Scene", new_scene, np.median(new_scene) + scene_range),
        ("New Scene", new_scene, np.median(new_scene) + 0.1 * scene_range)],
        [("", None),
        (r"Data - Predicted", delta, np.median(delta) + scene_range),
        ("Update", new_scene - this_scene),
        ("annotations", None)]]

    # Do the plotting.
    size = data.shape[0]  # The size to pad/crop to.
    for i, (ri, ci) in enumerate(itertools.product(range(rows), range(cols))):
        ax = axes[i]
        panel = panels[ri][ci]
        title, content = panel[0], panel[1]
        if len(panel) > 2:
            vrange = panel[2]
        else:
            vrange = None

        if content is not None:
            plot_image(ax, content, size=size, vrange=vrange)
            ax.set_title(title)
        elif title == "annotations":
            # Put some stats in this axis.
            line_height = 0.13
            txt = meta
            txt.append("Sky: {0:0.4f}".format(sky))
            txt.append(r"$\sum \mathrm{{dPSF}} = {0:.4f}$"
                    .format(norm))
            txt.append(r"$\sum \mathrm{{PSF}} = {0:.4f}$"
                    .format(np.sum(psf)))
            txt.append("median(Data) = {0:.4f}".format(np.median(data)))
            txt.append("median(This Scene) = {0:.4f}"
                    .format(np.median(this_scene)))
            txt.append("median(New Scene) = {0:.4f}"
                    .format(np.median(new_scene)))

            for i in range(len(txt)):
                ax.text(0, 1 - i * line_height, txt[i], ha="left", va="top",
                        transform=ax.transAxes)

            ax.set_axis_off()
        else:
            ax.set_axis_off()

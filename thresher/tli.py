__all__ = ["run_tli"]


import numpy as np
from scipy.signal import fftconvolve as convolve

import utils


def pad_image_and_weight(image, weight, final_shape, offset=None):
    final_image = np.zeros(final_shape, dtype=float)
    final_weight = np.zeros(final_shape, dtype=bool)

    shape = image.shape
    rng = (0.5 * (np.atleast_1d(final_shape) - np.atleast_1d(shape))) \
            .astype(int)
    if offset is not None:
        rng -= offset

    final_image[rng[0]:rng[0] + shape[0], rng[1]:rng[1] + shape[1]] = \
            image
    final_weight[rng[0]:rng[0] + shape[0], rng[1]:rng[1] + shape[1]] = \
            weight.astype(bool)

    return final_image, final_weight


def run_tli(image_list, top=None, top_percent=None, shift=True):
    """
    Run traditional lucky imaging on a stream of data.

    ## Arguments

    * `image_list` (list): The list of filenames for the images which
        will be ranked and combined using TLI.

    ## Keyword Arguments

    * `top` (int or list): How many images should be co-added? This can be
      a list of `int`s so that multiple sets can be co-added simultaneously.
    * `top_percent` (float): An alternative notation for `top` instead
        specified by a percentage.
    * `shift` (bool): Should the images be shifted before co-adding? This
      defaults to `True`.

    ## Returns

    * `fns` (list): The filenames ordered from best to worst as ranked
        by TLI.
    * `ranks` (list): The value of the ranking scalar corresponding to
        the images listed in `fns`.
    * `centers` (list): The coordinates of the centers of each image as
        determined by centroiding.
    * `coadd` (numpy.ndarray): The resulting co-added images. The shape will
      be `(len(top) + 1, N, M)` where the `-1` entry is the full co-add and
      `N` and `M` are decided based on the size of result needed based on the
      offsets.

    """
    # Build the scene to convolve with. It's a small, pixelated Gaussian.
    dim = 10
    x = np.linspace(-0.5 * dim, 0.5 * dim, dim) ** 2
    r = np.sqrt(x[:, None] + x[None, :])
    scene = 0.5 * np.exp(-0.5 * r) / np.pi
    s_dim = (np.array(scene.shape) - 1) / 2

    # Initialized the first time through the data.
    final_shape = None

    # Calculate the centers and ranks of the images.
    centers = {}
    offsets = {}
    ranks = {}
    images = {}
    weights = {}
    for n, fn in enumerate(image_list):
        img = utils.load_image(fn)

        # Mask the pixels that are set to NaN in the image.
        weight = ~np.isnan(img)

        # This is a sky subtraction hack.
        img -= np.median(img[weight])

        # Discard the image if no pixels are included.
        if np.sum(weight):
            # Set those same pixels to the median value. This is a hack to
            # make the centroiding work.
            img[~weight] = 0

            # Do the centroiding and find the rank.
            convolved = convolve(img, scene, mode="valid")
            ind_max = convolved.argmax()
            center = np.unravel_index(ind_max, convolved.shape)
            rank = convolved.flat[ind_max]

            # Because of the "valid" in the convolve, we need to offset
            # based on the size of the "scene".
            center = np.array(center) + s_dim
            offset = (center - 0.5 * np.array(img.shape)).astype(int)

            # Keep track of how the largest offset affects the final shape
            # of the image.
            shape = np.array(img.shape)
            if final_shape is None:
                final_shape = shape
            if shift:
                final_shape = np.max(np.vstack(
                    [final_shape, shape + 2 * np.abs(offset)]), axis=0)

            # Save the image, weight and metadata.
            centers[fn] = center
            offsets[fn] = offset
            images[fn] = img
            weights[fn] = weight
            ranks[fn] = (n, rank)

    # Sort by brightest centroided pixel.
    ranked = sorted(ranks, reverse=True, key=lambda k: ranks[k][1])
    ordered_fns, ordered_ranks, ordered_centers = [], [], []
    for k in ranked:
        ordered_fns.append(k)
        ordered_ranks.append(ranks[k][-1])
        ordered_centers.append(list(centers[k]))

    ordered_ranks = np.array(ordered_ranks)
    ordered_centers = np.array(ordered_centers)

    # Pad the images to the right size.
    for k in ordered_fns:
        if shift:
            offset = offsets[k]
        else:
            offset = np.zeros(2)

        images[k], weights[k] = \
                pad_image_and_weight(images[k], weights[k], final_shape,
                        offset=offset)

    # Figure out the number of images that should be co-added.
    if top is None and top_percent is None:
        top = len(ordered_fns)
    elif top_percent is not None:
        top = [max(1, int(top_percent * 0.01 * len(ranked))), len(ordered_fns)]
    else:
        top = np.append(np.atleast_1d(top), len(ordered_fns))
    top = np.atleast_1d(top)

    # Allocate the memory for the final image.
    final_image = np.zeros([len(top)] + list(final_shape))
    final_weight = np.zeros([len(top)] + list(final_shape))

    # Do the co-add.
    for j, t in enumerate(top):
        for i, k in enumerate(ordered_fns[:t]):
            final_image[j] += images[k] * weights[k].astype(float)
            final_weight[j] += weights[k].astype(float)

    m = final_weight > 0
    final_image[m] /= final_weight[m]
    final_image[~m] = np.nan

    return ordered_fns, ordered_ranks, ordered_centers, final_image

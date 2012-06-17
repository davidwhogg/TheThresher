import numpy as np
import matplotlib.pyplot as pl
from scipy.sparse import csr_matrix

import thresher


S, P = 10, 2
D = S - 2 * P


def plot_psf_transform(ax1, ax2):
    px, py = np.meshgrid(range(2 * P + 1), range(2 * P + 1))
    r2 = (px - P) ** 2 + (py - P) ** 2
    psf = np.exp(-0.5 * r2)

    rows, cols = thresher.unravel_psf(S, P)
    rows = rows[:-S ** 2]
    cols = cols[:-S ** 2]
    vals = np.zeros((D ** 2, (2 * P + 1) ** 2)) + psf.flatten()[None, ::-1]
    vals = vals.flatten()

    psf_matrix = csr_matrix((vals, (rows, cols)), shape=(D ** 2, S ** 2))

    ax1.imshow(psf, interpolation="nearest", cmap="gray")
    ax2.imshow(psf_matrix.todense(), interpolation="nearest", cmap="gray")

    ax1.annotate("",
            xytext=(2 * P + 3, P), xycoords="data",
            xy=(S ** 2 / 2 + P + 0.5, P), textcoords="data",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    ax1.set_xlim([-S ** 2 / 2 + P + 0.5, S ** 2 / 2 + P + 0.5])
    ax1.set_ylim([-D ** 2 / 2 + P + 0.5, D ** 2 / 2 + P + 0.5])

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.set_frame_on(False)
    ax1.set_title(r"$\Psi_n$")

    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_frame_on(False)
    ax2.set_title(r"$P_n$")


def plot_scene_transform(ax1, ax2):
    x, y = np.meshgrid(range(S), range(S))
    r2 = (x - 0.5 * S + 0.5) ** 2 + (y - 0.5 * S + 0.5) ** 2
    scene = np.exp(-0.5 * r2)

    m = thresher.unravel_scene(S, P)
    scene_matrix = scene.flatten()[m]

    ax1.imshow(scene, interpolation="nearest", cmap="gray")
    ax2.imshow(scene_matrix, interpolation="nearest", cmap="gray")

    ax1.annotate("",
            xytext=(S + 3, 0.5 * S - 0.5), xycoords="data",
            xy=(S ** 2 / 2 + P + 0.5, 0.5 * S - 0.5), textcoords="data",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))

    ax1.set_xlim([-S ** 2 / 2 + 0.5 * S, S ** 2 / 2 + 0.5 * S])
    ax1.set_ylim([-D ** 2 / 2 + 0.5 * S, D ** 2 / 2 + 0.5 * S])

    dp = 2 * P + 1
    ax2.set_xlim([-S ** 2 / 2 + 0.5 * dp ** 2, S ** 2 / 2 + 0.5 * dp ** 2])

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)
    ax1.set_frame_on(False)
    ax1.set_title(r"$\Sigma$")

    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.set_frame_on(False)
    ax2.set_title(r"$S$")


def plot_transforms():
    pl.figure(figsize=(12, 5))

    psf_ax1 = pl.subplot(221)
    psf_ax2 = pl.subplot(222)
    plot_psf_transform(psf_ax1, psf_ax2)

    sc_ax1 = pl.subplot(223)
    sc_ax2 = pl.subplot(224)
    plot_scene_transform(sc_ax1, sc_ax2)

    pl.savefig("index_gymnastics.pdf")


if __name__ == "__main__":
    plot_transforms()

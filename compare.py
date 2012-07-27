import matplotlib.pyplot as pl
import numpy as np
import pyfits
import sys
import thresher

threshfn = sys.argv[2]
tlifn = sys.argv[3]
tlihdus = [int(i) for i in sys.argv[4:]]

f = pyfits.open(threshfn)
thresh_data = np.array(f[0].data, dtype=float)
dim = f[0].header["size"]
f.close()

thresh_data = thresher.utils.trim_image(thresh_data, dim)

f = pyfits.open(tlifn)
total = len(f[-1].data["filename"])
tli_data = [np.array(f[i].data, dtype=float) for i in tlihdus]
numbers = [f[i].header.get("number", "All %d" % total) for i in tlihdus]
f.close()

if len(tlihdus) == 1:
    fig = pl.figure(figsize=(6, 3))
    fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.85,
            wspace=0.05, hspace=0.1)
else:
    fig = pl.figure(figsize=(2.5 * len(tlihdus), 5))
    fig.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.9,
            wspace=0.05, hspace=0.1)

asinh = lambda img, mu, sigma, f: f * np.arcsinh((img - mu) / sigma) + 0.2

ax = fig.add_subplot(2, len(tlihdus), int(len(tlihdus) * 2))
sigma = thresher.plotting.estimate_sigma(thresh_data)
thresher.plotting.plot_image(ax, asinh(thresh_data, np.median(thresh_data),
    1 * sigma, 0.2), vrange=[0, 1])
ax.set_xticklabels([])
ax.set_yticklabels([])

for i, data in enumerate(tli_data):
    data = thresher.utils.trim_image(data, dim)
    ax = fig.add_subplot(2, len(tlihdus), i + 1)
    thresher.plotting.plot_image(ax, asinh(data, np.median(thresh_data),
        1 * sigma, 0.2), vrange=[0, 1])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(numbers[i])

pl.savefig(sys.argv[1])

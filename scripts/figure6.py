import sys
from pathlib import Path
from context import phantom

import imageio

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm.auto import tqdm

from phantom.utils import get_psnr

from phantom.mvtomo.algorithm_single import CGNE as CGNEs
from phantom.mvtomo.algorithms_mv import CGNE as CGNEm
from phantom.mvtomo.modifier import translate_operator


from phantom.plt_utils import (
    TEXTWIDTH_FULL,
    set_params,
    set_size,
    add_border_to_axis,
    remove_axis,
)

BACT_FOLDER = phantom.BACT_FOLDER
SCRIPT_DIR = Path(sys.argv[0]).resolve().parent


def make_rec(n_iters):
    y = imageio.volread(BACT_FOLDER / "y_full.tiff")[:, ::5, :]
    y_roi = imageio.volread(BACT_FOLDER / "y_roi.tiff")
    tx = np.loadtxt(BACT_FOLDER / "tx.txt")
    ty = np.loadtxt(BACT_FOLDER / "ty.txt")
    angles = (np.loadtxt(BACT_FOLDER / "angles.txt")[::5]).astype("float32")
    angles_roi = np.loadtxt(BACT_FOLDER / "angles_roi.txt").astype("float32")
    kwargs_mv = {
        "y": [y, y_roi],
        "angles": [angles, angles_roi],
        "nn_step": 2,
    }

    H, _, W = y.shape

    def x0():
        return np.zeros((H, W, W), dtype="float32")

    rec_single = CGNEs(x0(), y, angles, nn_step=2)
    rec_roi = CGNEs(x0(), y_roi, angles_roi, nn_step=2)
    rec_roi.A = translate_operator(rec_roi.A, tx, ty)

    rec_cgne_mv = CGNEm(x0(), **kwargs_mv)
    rec_cgne_mv.A[1] = translate_operator(rec_cgne_mv.A[1], tx, ty)

    vol_sparse = rec_single(n_iters)
    vol_roi = rec_roi(n_iters)
    vol_mv = rec_cgne_mv(n_iters)
    imageio.volwrite(BACT_FOLDER / f"cgne_sparse_{n_iters}.tiff", vol_sparse)
    imageio.volwrite(BACT_FOLDER / f"cgne_roi_{n_iters}.tiff", vol_roi)
    imageio.volwrite(BACT_FOLDER / f"cgne_mv_{n_iters}.tiff", vol_mv)


def load_rec(n_iter=40):
    fn_single = BACT_FOLDER / f"cgne_sparse_{n_iter}.tiff"
    fn_roi = BACT_FOLDER / f"cgne_roi_{n_iter}.tiff"
    fn_mv = BACT_FOLDER / f"cgne_mv_{n_iter}.tiff"

    if fn_single.exists() and fn_roi.exists() and fn_mv.exists():
        pass
    else:
        print(f"Reconstructions for {n_iter} iterations not found — generating them.")
        make_rec(n_iter)
    return imageio.volread(fn_single), imageio.volread(fn_roi), imageio.volread(fn_mv)


def make_figure():
    vol_sparse, vol_roi, vol_mv = load_rec()
    xi = 332
    zi = 390

    volumes = [vol_roi, vol_sparse, vol_mv]
    names = ["Interior", "Sparse", "Combined"]
    nx, ny, nz = vol_sparse.shape
    padxz = 50
    slices = [vol[zi, :, :][padxz:-padxz, padxz:-padxz] for vol in volumes]

    clim = np.percentile(slices[-1][slices[-1] > 0.001], (10, 90))
    cmap = "gray_r"

    f, ax = plt.subplots(ncols=3, figsize=set_size(TEXTWIDTH_FULL, n=1))

    for i, img in enumerate(slices):
        ax[i].imshow(img, clim=clim, cmap=cmap)

    for i, name in enumerate(names):
        ax[i].set_title(names[i])

    for axis in ax.ravel():
        remove_axis(axis)
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()
    f.savefig(SCRIPT_DIR / "example_bacteria.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    make_figure()

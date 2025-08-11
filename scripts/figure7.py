import sys
from pathlib import Path
from context import phantom

import imageio

import matplotlib.pyplot as plt
from matplotlib import patches
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

BCELL_FOLDER = phantom.BCELL_FOLDER
SCRIPT_DIR = Path(sys.argv[0]).resolve().parent


def x0(y):
    H, _, W = y.shape
    return np.zeros((H, W, W), dtype="float32")


def make_full(n_iters):
    y = imageio.volread(BCELL_FOLDER / "y_full.tiff")
    angles = (np.loadtxt(BCELL_FOLDER / "angles.txt")).astype("float32")
    rec_single = CGNEs(x0(y), y, angles, nn_step=2)
    vol_mv = rec_single(n_iters)
    imageio.volwrite(BCELL_FOLDER / f"cgne_full_{n_iters}.tiff", vol_mv)


def make_sparse(n_iters):
    y_sparse = imageio.volread(BCELL_FOLDER / "y_full.tiff")[:, ::5, :]
    angles = (np.loadtxt(BCELL_FOLDER / "angles.txt")[::5]).astype("float32")
    rec_single = CGNEs(x0(y_sparse), y_sparse, angles, nn_step=2)
    vol_mv = rec_single(n_iters)
    imageio.volwrite(BCELL_FOLDER / f"cgne_sparse_{n_iters}.tiff", vol_mv)


def make_roi(n_iters):
    y_roi = imageio.volread(BCELL_FOLDER / "y_roi.tiff")
    pass


def make_mv(n_iters):
    y_roi = imageio.volread(BCELL_FOLDER / "y_roi.tiff")
    pass


def make_rec(n_iters):
    tx = np.loadtxt(BCELL_FOLDER / "tx.txt")
    ty = np.loadtxt(BCELL_FOLDER / "ty.txt")
    angles_sparse = (np.loadtxt(BCELL_FOLDER / "angles.txt")[::5]).astype("float32")
    angles_roi = np.loadtxt(BCELL_FOLDER / "angles_roi.txt").astype("float32")
    kwargs_mv = {
        "y": [y, y_roi],
        "angles": [angles, angles_roi],
        "nn_step": 2,
    }

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
    imageio.volwrite(BCELL_FOLDER / f"cgne_sparse_{n_iters}.tiff", vol_sparse)
    imageio.volwrite(BCELL_FOLDER / f"cgne_roi_{n_iters}.tiff", vol_roi)
    imageio.volwrite(BCELL_FOLDER / f"cgne_mv_{n_iters}.tiff", vol_mv)


def load_rec(tag, n_iter=40):
    dispatch = {
        "full": make_full,
        "sparse": make_sparse,
        "roi": make_roi,
        "mv": make_mv,
    }

    filename = BCELL_FOLDER / f"cgne_{tag}_{n_iter}.tiff"
    if filename.exists():
        pass
    else:
        print(f"Reconstructions for {n_iter} iterations not found — generating them.")
        dispatch[tag](n_iter)
    return imageio.volread(filename)


def geteroi(vol, p, R):
    slices = [slice(int(x - R), (x + R)) for x in p]
    return vol[tuple(slices)]


def make_figure():
    vol_sparse, vol_roi, vol_mv = load_rec(5)

    roi2_low = geteroi(lac_low, p, r)
    roi2_sparse = geteroi(lac_low_sparse, p, r)
    roi2_comb = geteroi(lac_comb_sparse, p, r)

    # crop_bottom_roi = 30

    # # xi = rec_low.shape[0]//2
    # # xi = 250
    # xi = 332
    # cmap = "gray_r"
    # crop = 300
    # nx, ny, nz = rec_low.shape
    # clim = np.percentile(rec_low[rec_low > 0.001], (2, 87))
    # hr = rec_low.shape[0] - crop, rec_low.shape[2]
    # volumes = [lac_int, lac_low, lac_comb]
    # names = ["Interior", "Full View", "Combined"]

    # f = plt.figure(
    #     figsize=set_size(TEXTWIDTH_FULL, TEXTWIDTH_FULL), layout="constrained"
    # )

    # outer_grid = f.add_gridspec(3, 1, wspace=0.05, hspace=0, height_ratios=[1, 1, 1])
    # up_grid = outer_grid[0].subgridspec(1, 3, wspace=0, hspace=0)
    # down_grid = outer_grid[1].subgridspec(1, 4, wspace=0, hspace=0.01)
    # dm_grid = outer_grid[2].subgridspec(1, 4, wspace=0, hspace=0.01)

    # ax = up_grid.subplots()
    # for i, vol in enumerate(volumes):
    #     ax[i].imshow(vol[:-crop, xi, :], clim=clim, cmap=cmap)
    #     ax[i].set_title(names[i])

    # # first ROI
    # H = roi_L[0] - crop_bottom_roi
    # W = roi_L[2]
    # rect = patches.Rectangle(
    #     (roi0[2], roi0[0]), W, H, linewidth=2, edgecolor="C1", facecolor="none"
    # )
    # ax[0].add_patch(rect)

    # # second ROI
    # H = 128
    # W = 128
    # rect = patches.Rectangle(
    #     (dm_pos[2] - 64, dm_pos[0] - 64),
    #     W,
    #     H,
    #     linewidth=2,
    #     edgecolor="C0",
    #     facecolor="none",
    # )
    # ax[0].add_patch(rect)

    # zi = 62
    # ax2 = down_grid.subplots()
    # volumes = [roi_lac_low_AD, roi_lac_sparse_AD, roi_lac_comb_AD]
    # names = ["Low", "Sparse", "Comb."]
    # for i, vol in enumerate(volumes):
    #     ax2[i].imshow(volumes[i][:-crop_bottom_roi, :, zi], clim=clim, cmap=cmap)
    #     ax2[i].set_title(names[i])
    # ax2[3].imshow(imageio.v3.imread("snakes_comb.png"))

    # zi = 64
    # ax3 = dm_grid.subplots()
    # volumes = [roi2_low_AD, roi2_sparse_AD, roi2_comb_AD]
    # names = ["Low", "Sparse", "Comb."]
    # for i, vol in enumerate(volumes):
    #     ax3[i].imshow(volumes[i][:, :, zi], clim=clim, cmap=cmap)
    #     # ax3[i].set_title(names[i])
    # ax3[3].imshow(imageio.v3.imread("dm_comb.png"))

    # for axis in ax.ravel():
    #     remove_axis(axis, despine=False)
    # for axis in ax2.ravel():
    #     remove_axis(axis, despine=False)
    # for axis in ax3.ravel():
    #     remove_axis(axis, despine=False)

    # labels = dict(labelcols)
    # labels.pop("Nucleus")
    # labels["Membrane\nvesicle"] = "C0"
    # anchor_x, anchor_y = 0.2, -0.02  # Adjust as needed
    # x_offset = 0  # Start from anchor_x
    # char_width = 0.012  # Approximate width per character
    # extra_padding = 0.03  # Additional space between elements
    # for tag, color in labels.items():
    #     # Draw color box
    #     box = patches.FancyBboxPatch(
    #         (anchor_x + x_offset, anchor_y),
    #         0.03,
    #         0.015,  # Box size
    #         boxstyle="round,pad=0.01",
    #         facecolor=color,
    #         edgecolor="none",
    #         transform=f.transFigure,
    #         clip_on=False,
    #     )
    #     f.patches.append(box)
    #     x_offset += 0.05  # Space between box and text

    #     # Add text next to the box
    #     f.text(
    #         anchor_x + x_offset,
    #         anchor_y + 0.005,
    #         tag,
    #         fontsize=10,
    #         va="center",
    #         ha="left",
    #     )
    #     x_offset += len(tag) * char_width + extra_padding  # Adjust spacing dynamically

    # f.savefig("example_bcell_viz.pdf", bbox_inches="tight", dpi=600)


if __name__ == "__main__":
    make_figure()

import sys
from pathlib import Path
from context import phantom

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from tqdm.auto import tqdm

from phantom import InteriorPhantomPSF

from phantom.utils import get_psnr


from phantom.plt_utils import (
    TEXTWIDTH_FULL,
    set_params,
    set_size,
    add_border_to_axis,
    remove_axis,
)
from figure3 import janelia_roi, janelia_circroi

JANELIA_FOLDER = phantom.JANELIA_FOLDER
SCRIPT_DIR = Path(sys.argv[0]).resolve().parent


def get_reconstructions(cell: InteriorPhantomPSF):

    # dose per image
    dpi = 2.294312200191294
    intensity = 1000
    n_images = 499
    list_full = [11, 23, 47, 97, 199, 307, 389]

    ref_ref35 = cell.oracle_rec_poisson(
        "35", n_images, noneg=True, intensity=intensity, seed=1
    )
    # ref_ref60 = cell.oracle_rec_poisson('60',n_images, noneg=True,intensity = intensity,seed = 1)

    recdict = dict()
    recdict_deconv_g = dict()

    for n_full in tqdm(list_full):
        # leftover dose from removed full FOV scans
        dose_left = intensity * (n_images - n_full)
        I_interior = int(dpi * dose_left / n_images)
        print(f"nfull {n_full} intensity {intensity} dose {dose_left} I {I_interior} ")
        kwargs_raw = {
            "noneg": True,
            "seed": 1,
            "intensity_full": intensity,
            "intensity_roi": I_interior,
        }

        recdict[n_full] = cell.oracle_rec_mv_poisson(
            "60", "35", n_full, n_images, **kwargs_raw
        )
        recdict_deconv_g[n_full] = cell.oracle_rec_mv_poisson_deco_gauss(
            "60", "35", n_full, n_images, **kwargs_raw
        )
    return ref_ref35, recdict, recdict_deconv_g


def roi_slice(image, L):
    midpoint = [l // 2 for l in image.shape]
    R = int(L // 2)
    slice0 = slice(midpoint[0] - R, midpoint[0] + R)
    slice1 = slice(midpoint[1] - R, midpoint[1] + R)
    return image[slice0, slice1]


def make_figure():
    print("Loading phantom images...")
    phantom, pixel_size = janelia_roi()
    phantom_crop = janelia_circroi(phantom, pixel_size)
    roi_width = int(6 / pixel_size)
    cell = InteriorPhantomPSF(phantom_crop, roi_width, pixel_size)

    ref_ref35, recdict, recdict_deconv_g = get_reconstructions(cell)

    def metric_psnr(x, y):
        return get_psnr(x, y, mask=cell.roi_mask(roi_width))

    psnr_ref = metric_psnr(ref_ref35, cell.phantom_pix)
    psnr_roi = [metric_psnr(rec, cell.phantom_pix) for n, rec in tqdm(recdict.items())]
    psnr_full = [get_psnr(rec, cell.phantom_pix) for n, rec in tqdm(recdict.items())]
    psnr_roi_deco_g = [
        metric_psnr(rec, cell.phantom_pix) for n, rec in tqdm(recdict_deconv_g.items())
    ]

    slice_nr = cell.phantom_pix.shape[0] // 2
    disp_width = roi_width / 2
    ref_slice = roi_slice(cell.phantom_pix[slice_nr], disp_width)
    clim = np.percentile(ref_slice, (5, 95))
    cmap = "gray_r"
    cmap = "viridis"
    slice_nr

    set_params()
    fig = plt.figure(layout="constrained", figsize=set_size(TEXTWIDTH_FULL, n=1))
    gs = GridSpec(2, 6, figure=fig)

    ax_psnr = fig.add_subplot(gs[1, 0:2])
    ax_full = fig.add_subplot(gs[1, 2:4])
    ax_ref = fig.add_subplot(gs[1, 4:])

    ax_rec1 = fig.add_subplot(gs[0, 0:2])
    ax_rec2 = fig.add_subplot(gs[0, 2:4])
    ax_rec3 = fig.add_subplot(gs[0, 4:])

    slice_nr = 48

    ax_psnr.plot(recdict.keys(), psnr_full, label="Full Sample")
    ax_psnr.plot(recdict.keys(), psnr_roi, label="ROI")
    ax_psnr.plot(recdict.keys(), psnr_roi_deco_g, label="ROI deconv")

    clim = np.percentile(ref_slice, (5, 95))

    dict_key = 23

    big_slice = recdict[dict_key][slice_nr]
    ax_full.imshow(big_slice, clim=clim, cmap=cmap)
    ax_ref.imshow(
        roi_slice(cell.phantom_pix[slice_nr], disp_width), clim=clim, cmap=cmap
    )
    ax_rec1.imshow(
        roi_slice(recdict[dict_key][slice_nr], disp_width), clim=clim, cmap=cmap
    )
    ax_rec2.imshow(
        roi_slice(recdict_deconv_g[dict_key][slice_nr], disp_width),
        clim=clim,
        cmap=cmap,
    )
    ax_rec3.imshow(roi_slice(ref_ref35[slice_nr], disp_width), clim=clim, cmap=cmap)

    ax_psnr.set_xlabel(r"$N_\text{full}$")
    ax_psnr.set_ylabel("PSNR")
    ax_psnr.axhline(psnr_ref, color="C3")

    ax_psnr.scatter(
        dict_key, psnr_roi[1], marker="o", facecolors="none", edgecolors="C1"
    )
    ax_psnr.scatter(
        dict_key, psnr_roi_deco_g[1], marker="o", facecolors="none", edgecolors="C2"
    )

    ax_psnr.legend()

    ax_full.set_xlabel(r"Full Slice, $N_\text{full} = 23$ ")
    ax_ref.set_xlabel("Phantom")

    ax_rec1.set_xlabel(r"$N_\text{full} = 23$")
    ax_rec2.set_xlabel(r"$N_\text{full} = 23$, deconv")
    ax_rec3.set_xlabel("Reference 35 nm")

    add_border_to_axis(ax_rec1, pad=2, lw=2, color="C1")
    add_border_to_axis(ax_rec2, pad=2, lw=2, color="C2")
    add_border_to_axis(ax_rec3, pad=2, lw=2, color="C3")

    for axis in [ax_rec1, ax_rec2, ax_rec3]:
        remove_axis(axis, despine=False)
    for axis in [
        ax_full,
        ax_ref,
    ]:
        remove_axis(axis)

    plt.show()
    fig.savefig(SCRIPT_DIR / "dose_opt_psf.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    make_figure()

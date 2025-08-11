import sys
from pathlib import Path
from scipy import ndimage as ndi
import numpy as np
from tqdm.auto import tqdm

from context import phantom

from phantom.plt_utils import (
    TEXTWIDTH_FULL,
    set_params,
    set_size,
    add_border_to_axis,
    remove_axis,
)
import matplotlib.pyplot as plt
import pandas as pd


from phantom.read_write_mrc import read_mrc
from phantom import InteriorPhantom

from phantom.utils import get_psnr
from janelia_phantom import load_janelia_phantom

SCRIPT_DIR = Path(sys.argv[0]).resolve().parent
EXPORT = SCRIPT_DIR / "figures"
TABLEPATH = SCRIPT_DIR / "tables" / "interior_nullspace.csv"


def getmetrics(cell: InteriorPhantom, n_roi, n_full, roi_width, roi_width_full, n_null):
    angles_full = np.linspace(0, np.pi, n_full, endpoint=False)
    angles_roi = np.linspace(0, np.pi, n_roi, endpoint=False)

    cell.make_full(angles_full)
    cell.make_roi(angles_roi, roi_width=roi_width)

    rec_mv = cell.mv_rec_oracle()
    null_mv = cell.nullspace(mode="combined", n_iter=n_null)

    cell.make_roi(angles_roi, roi_width=roi_width_full)
    rec_full = cell.mv_rec_oracle()
    null_full = cell.nullspace(mode="combined", n_iter=n_null)

    mask = cell.roi_mask(roi_width)
    L2_x0 = np.sum(cell.phantom_pix**2)
    L2_x0_roi = np.sum(mask * cell.phantom_pix**2)

    L2_full = np.sum(null_full**2)
    L2_full_roi = np.sum(mask * null_full**2)

    L2_mv = np.sum(null_mv**2)
    L2_mv_roi = np.sum(mask * null_mv**2)

    resdict = {
        "n_roi": n_roi,
        "n_full": n_full,
        "psnr_full": get_psnr(rec_full, cell.phantom_oracle),
        "psnr_mv": get_psnr(rec_mv, cell.phantom_oracle),
        "psnr_full_roi": get_psnr(rec_full, cell.phantom_oracle, mask),
        "psnr_mv_roi": get_psnr(rec_mv, cell.phantom_oracle, mask),
        "xnull_full": L2_full / L2_x0,
        "xnull_full_roi": L2_full_roi / L2_x0_roi,
        "xnull_mv": L2_mv / L2_x0,
        "xnull_mv_roi": L2_mv_roi / L2_x0_roi,
    }
    return resdict


def get_phantom_cell():
    original_data, original_pixel_size = load_janelia_phantom(40)
    downscale = 4
    pixel_size = original_pixel_size * downscale
    scaled = ndi.zoom(original_data, 1 / downscale, order=1)
    H = scaled.shape[1]
    phantom = np.transpose(scaled[:, int(1.5 / 5 * H) : -int(1 / 5 * H), :], (1, 0, 2))
    FWHM = 60 / 40
    roi_width = int(10 / pixel_size)  # 10 UM wide ROI
    phantom = np.ascontiguousarray(phantom)
    return InteriorPhantom(phantom, roi_width, pixel_size, FWHM)


def generate_table():

    cell = get_phantom_cell()
    detector_width = np.linalg.norm(np.array(cell.phantom_oracle.shape[1:]))
    roi_width = int(10 / cell.pixel_size)  # 10 UM wide ROI
    pad = (detector_width - roi_width) // 2
    roi_width_full = int(roi_width + 2 * pad)

    cosnt_opts = {
        "n_null": 200,
        "roi_width": roi_width,
        "roi_width_full": roi_width_full,
    }

    n_angles = 273
    a_range = (2, n_angles)
    func = lambda x: x**2
    n = np.arange(13)
    xx = (a_range[0] + (a_range[1] - a_range[0]) * func(n) / func(n[-1])).astype(int)
    rows = []
    for n_full in tqdm(xx):
        rows.append(getmetrics(cell, n_angles, n_full, **cosnt_opts))

    df = pd.DataFrame(rows)
    df.to_csv(TABLEPATH, index=False)


def plot_figure():
    set_params()
    df = pd.read_csv(TABLEPATH)
    # EXAMPLE PARAMETERS
    n_rec = 100
    n_null = 200
    n_angles = 273
    n_full = 7
    cell = get_phantom_cell()
    roi_width = int(10 / cell.pixel_size)

    angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    angles_full = np.linspace(0, np.pi, n_full, endpoint=False)

    cell.make_full(angles_full)
    cell.make_roi(angles, roi_width=roi_width)

    rec_roi = cell.roi_rec(n_iter=n_rec)
    rec_full = cell.full_rec(n_iter=n_rec)
    rec_mv = cell.mv_rec(n_iter=n_rec)

    null_f = cell.nullspace(mode="full", n_iter=n_null)
    null_roi = cell.nullspace(mode="interior", n_iter=n_null)
    null_mv = cell.nullspace(mode="combined", n_iter=n_null)

    cmap = "gray_r"
    index = cell.phantom_pix.shape[0] // 2

    clim1 = np.percentile(cell.phantom_pix[index], (1, 99))
    clim2 = np.percentile(null_f[index], (1, 99))

    fig = plt.figure(
        layout="constrained", figsize=set_size(TEXTWIDTH_FULL, TEXTWIDTH_FULL)
    )
    subfigs = fig.subfigures(1, 2, wspace=0.01, width_ratios=[8, 5])
    axsLeft = subfigs[0].subplots(3, 2)
    axR1 = subfigs[1].add_subplot(311)
    axR2 = subfigs[1].add_subplot(312, sharex=axR1)
    axR0 = subfigs[1].add_subplot(313)

    axsLeft[0, 0].imshow(rec_full[index], cmap=cmap, clim=clim1)
    axsLeft[1, 0].imshow(rec_roi[index], cmap=cmap, clim=clim1)
    axsLeft[2, 0].imshow(rec_mv[index], cmap=cmap, clim=clim1)
    axsLeft[0, 1].imshow(null_f[index], cmap=cmap, clim=clim2)
    axsLeft[1, 1].imshow(null_roi[index], cmap=cmap, clim=clim2)
    axsLeft[2, 1].imshow(null_mv[index], cmap=cmap, clim=clim2)

    add_border_to_axis(axsLeft[0, 0], color="C0", pad=-2, linewidth=3)
    pos = [s // 2 for s in rec_full[index].shape]
    roi_circle = plt.Circle(
        pos[::-1], roi_width / 2, color="C1", fill=False, linewidth=3
    )
    axsLeft[0, 0].add_patch(roi_circle)
    for ax in axsLeft.ravel():
        remove_axis(ax)

    def line(vol):
        thickness = 5
        index_line = vol.shape[1] // 2
        return np.mean(
            vol[index, index_line - thickness : index_line + thickness, :], 0
        )

    index_line = rec_full.shape[1] // 2
    axsLeft[2, 0].axhline(index_line, color="C3")

    axR0.plot(line(cell.phantom_pix) / cell.pixel_size, label="Phantom")
    axR0.plot(line(rec_full) / cell.pixel_size, label="Sparse")
    axR0.plot(line(rec_roi) / cell.pixel_size, label="Interior")
    axR0.plot(line(rec_mv) / cell.pixel_size, label="Combined")
    axR0.legend(prop={"size": 8})

    axR1.plot(df["n_full"], df["psnr_mv"], "C0", label=r"Full Volume")
    axR1.plot(df["n_full"], df["psnr_mv_roi"], "C1", label=r"Interior ROI")
    axR2.plot(df["n_full"], 100 * df["xnull_mv"], "C0", label=r"Full Volume")
    axR2.plot(df["n_full"], 100 * df["xnull_mv_roi"], "C1", label=r"Interior ROI")
    axR2.legend()

    label_size = 12
    axsLeft[0, 0].set_ylabel("Sparse", fontsize=label_size)
    axsLeft[1, 0].set_ylabel("Interior", fontsize=label_size)
    axsLeft[2, 0].set_ylabel("Combined", fontsize=label_size)

    axsLeft[0, 0].set_title("Reconstruction", fontsize=label_size)
    axsLeft[0, 1].set_title("Null Space", fontsize=label_size)

    axR0.set_ylabel(r"LAC [\si{\per\um}]")

    axR2.set_xlabel("Number of full view projections")
    axR1.set_ylabel("PSNR")
    axR2.set_ylabel(r"$ | \bm{x}_\text{null} | / | \bm{x}_0 | $ [\%]")

    axR1.yaxis.get_major_locator().set_params(integer=True)
    plt.show()
    fig.savefig(EXPORT / "nullspace_combined.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    generate_table()
    plot_figure()

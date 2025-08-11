import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from external.fourier_ring_corr import FSC

from functools import reduce
from scipy.signal.windows import hamming, hann
from scipy import ndimage as ndi

from context import phantom

from phantom import InteriorPhantomPSF
import pandas as pd

from phantom.plt_utils import (
    TEXTWIDTH_FULL,
    set_params,
    set_size,
    add_border_to_axis,
    remove_axis,
)
from figure3 import janelia_roi, janelia_circroi

SCRIPT_DIR = Path(sys.argv[0]).resolve().parent
EXPORT = SCRIPT_DIR / "figures"
TABLEPATH = SCRIPT_DIR / "tables" / "dose_fractionation.csv"

ANGLE_LIST = [51, 101, 201, 499, 881]
TOTAL_DOSES = [300 * 201, 3000 * 201, 30000 * 201]


def window(shape, func, **kwargs):
    vs = [func(l, **kwargs) for l in shape]
    return reduce(np.multiply, np.ix_(*vs))


def middle_frc(volume, volume_ref, L, disp=0, windowfunc=None):
    midpoint = [l // 2 for l in volume.shape]
    R = int(L // 2)
    slice1 = slice(midpoint[1] - R, midpoint[1] + R)
    slice2 = slice(midpoint[2] - R, midpoint[2] + R)

    image = volume[midpoint[0], slice1, slice2]
    image_ref = volume_ref[midpoint[0], slice1, slice2]

    W = np.ones(image.shape)
    if windowfunc is not None:
        W = window(W.shape, windowfunc)

    x, y = FSC(W * image, W * image_ref, disp=disp)
    x = x[: R + 1]
    y = y[: R + 1]

    return x, y


def make_data():
    print("Loading phantom ...")
    phantom, pixel_size = janelia_roi()
    phantom_crop = janelia_circroi(phantom, pixel_size)
    roi_width = int(6 / pixel_size)
    cell = InteriorPhantomPSF(phantom_crop, roi_width, pixel_size)

    print("Loading phantom ...")
    data = dict()
    for dose in tqdm(TOTAL_DOSES):
        for na in tqdm(ANGLE_LIST, leave=False):
            I = int(dose / na)
            rec_kwargs = {
                "n_angles": na,
                "intensity": I,
                "noneg": True,
                "seed": 1,
            }
            rec = cell.oracle_rec_poisson("35m", **rec_kwargs)
            x, y = middle_frc(rec, cell.phantom_pix, L=roi_width / np.sqrt(2))
            data[f"{dose}_{na}_x"] = x
            data[f"{dose}_{na}_y"] = y

    df = pd.DataFrame(data)
    df.to_csv(TABLEPATH, index=False)
    return data


def make_figure():
    set_params()

    print("Loading data ...")
    data = pd.read_csv(TABLEPATH)

    f, axes = plt.subplots(
        ncols=len(TOTAL_DOSES),
        sharey=True,
        sharex=True,
        figsize=set_size(TEXTWIDTH_FULL, n=2),
    )
    for ax, dose in zip(axes, TOTAL_DOSES):
        for j, na in enumerate(ANGLE_LIST):
            I = int(dose / na)

            x = data[f"{dose}_{na}_x"]
            y = data[f"{dose}_{na}_y"]
            ys = ndi.gaussian_filter1d(y, sigma=3, mode="nearest")

            color = f"C{j}"

            ax.plot(x, ys, label=rf"$N_a$ = {na}", color=color)
            ax.plot(x, y, "-.", color=color, alpha=0.5)

            ax.set_ylim(-0.1, 1.1)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Nyquist Frequency")

    axes[0].legend(prop={"size": 8})
    axes[0].set_ylabel("FRC")

    axes[0].set_title(r"$\sum I_0 = 300 \times 201$")
    axes[1].set_title(r"$10 \times (\sum I_0)$")
    axes[2].set_title(r"$100 \times (\sum I_0)$")

    padh = 0.10
    badb = 0.2
    f.subplots_adjust(top=1 - 0.1, bottom=badb, left=padh, right=1 - padh)
    f.savefig(EXPORT / "limited_angle_check.pdf", format="pdf")


if __name__ == "__main__":
    # make_data()
    make_figure()

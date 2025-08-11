import sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
from skimage.metrics import peak_signal_noise_ratio
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from external.fourier_ring_corr import FSC

from context import phantom
from phantom.utils import crop_to_nonzero_of, profile_func
from phantom.read_write_mrc import read_mrc

from phantom import InteriorPhantomPSF

from phantom.plt_utils import set_params, remove_axis, TEXTWIDTH_FULL, set_size
from janelia_phantom import load_janelia_phantom

SCRIPT_DIR = Path(sys.argv[0]).resolve().parent
EXPORT = SCRIPT_DIR / "figures"
TABLEPATH1 = SCRIPT_DIR / "tables" / "XM2_janelia_noiseless.csv"
TABLEPATH2 = SCRIPT_DIR / "tables" / "XM2_janelia_noiseless_frc.csv"


def janelia_roi(downscale=1):
    original_data, original_pixel_size = load_janelia_phantom(20)

    pixel_size = original_pixel_size * downscale
    scaled = ndi.zoom(original_data, 1 / downscale, order=1)
    H = scaled.shape[1]

    phantom_vol = np.transpose(
        scaled[:, int(2 / 5 * H) : -int(1 / 5 * H), :], (1, 0, 2)
    )
    phantom_vol[phantom_vol < 0] = 0
    return phantom_vol, pixel_size


def janelia_circroi(phantom_vol, pixel_size, downscale=1):
    roi_rad_outer = int(8 / pixel_size)  # 8 UM wide ROI

    center_roi = [470 // downscale, 500 // downscale]
    shape = phantom_vol.shape
    X, Y, Z = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
    )

    R2 = (Y - center_roi[0]) ** 2 + (Z - center_roi[1]) ** 2

    mask = R2 < roi_rad_outer**2
    phantom_crop = crop_to_nonzero_of(phantom_vol * mask, mask)
    phantom_crop = np.ascontiguousarray(phantom_crop)
    return phantom_crop


def make_table1():
    print("Loading phantom images...")
    phantom, pixel_size = janelia_roi()
    phantom_crop = janelia_circroi(phantom, pixel_size)
    roi_width = int(6 / pixel_size)
    cell = InteriorPhantomPSF(phantom_crop, roi_width, pixel_size)

    single_kwargs = {
        "n_angles": 499,
        "intensity": 10000,
        "noneg": True,
        "seed": 1,
    }
    mv_kwargs = {
        "n_angles_full": 499,
        "n_angles_roi": 499,
        "intensity_full": 10000,
        "intensity_roi": 10000,
        "noneg": True,
        "seed": 1,
    }

    print("Reconstructing...")
    rec_35m_poisson = cell.oracle_rec_poisson("35m", **single_kwargs)
    rec_35_poisson = cell.oracle_rec_poisson("35", **single_kwargs)
    rec_60_poisson = cell.oracle_rec_poisson("60", **single_kwargs)
    rec_interior = cell.oracle_rec_interior_poisson("35", **single_kwargs)
    rec_mv = cell.oracle_rec_mv_poisson("60", "35", **mv_kwargs)

    coeff = 2 * pixel_size
    rad_dist = cell.center_distance()
    mask_cell = ndi.binary_fill_holes(ndi.minimum_filter(cell.phantom_pix > 0, size=3))
    data_range = np.percentile(cell.phantom_pix[mask_cell], 99)

    def calc_psnr(x, y):
        return peak_signal_noise_ratio(x, y, data_range=data_range)

    common_kwargs = {
        "reference": cell.phantom_pix,
        "edt": rad_dist,
        "mask": mask_cell,
        "func": calc_psnr,
        "sampling": 10,
        "r_max": 400,
    }

    print("Measuring profiles...")
    x, y_35m = profile_func(volume=rec_35m_poisson, **common_kwargs)
    x, y_35 = profile_func(volume=rec_35_poisson, **common_kwargs)
    x, y_60 = profile_func(volume=rec_60_poisson, **common_kwargs)
    x, y_int = profile_func(volume=rec_interior, **common_kwargs)
    x, y_mv = profile_func(volume=rec_mv, **common_kwargs)

    resdict = {
        "roi": coeff * x,
        "35m": y_35m,
        "35": y_35,
        "60": y_60,
        "int": y_int,
        "mv": y_mv,
    }

    df = pd.DataFrame(resdict)
    df.to_csv(TABLEPATH1, index=False)


def middle_frc(volume, volume_ref, L=None, disp=0, smooth=True, n_slices=3):
    midpoint = [l // 2 for l in volume.shape]
    R = int(L // 2)
    slice1 = slice(midpoint[1] - R, midpoint[1] + R)
    slice2 = slice(midpoint[2] - R, midpoint[2] + R)

    y_list = []
    indecies = np.round(
        np.linspace(0, volume.shape[0] - 1, n_slices + 2, endpoint=True)
    ).astype(int)
    for index in tqdm(indecies[1:-1], leave=False):
        image = volume[midpoint[0], slice1, slice2]
        image_ref = volume_ref[midpoint[0], slice1, slice2]

        x, y = FSC(image, image_ref, disp=disp)
        y_list.append(y)

    y = np.mean(y_list, 0)
    x = x[: R + 1]
    y = y[: R + 1]

    return x, y


def make_table2():
    print("Loading phantom images...")
    phantom, pixel_size = janelia_roi()
    phantom_crop = janelia_circroi(phantom, pixel_size)

    roi_width = int(6 / pixel_size)
    cell = InteriorPhantomPSF(phantom_crop, roi_width, pixel_size)

    single_kwargs = {"n_angles": 499, "intensity": 10000, "noneg": True, "seed": 1}
    mv_kwargs = {
        "n_angles_full": 499,
        "n_angles_roi": 499,
        "intensity_full": 10000,
        "intensity_roi": 10000,
        "noneg": True,
        "seed": 1,
    }

    print("Reconstructing...")
    rec_35_poisson = cell.oracle_rec_poisson("35", **single_kwargs)
    rec_60_poisson = cell.oracle_rec_poisson("60", **single_kwargs)
    rec_interior = cell.oracle_rec_interior_poisson("35", **single_kwargs)
    rec_mv = cell.oracle_rec_mv_poisson("60", "35", **mv_kwargs)

    print("Measuring FRC...")
    x, frc_60 = middle_frc(rec_60_poisson, cell.phantom_pix, smooth=False, L=roi_width)
    _, frc_35 = middle_frc(rec_35_poisson, cell.phantom_pix, smooth=False, L=roi_width)
    _, frc_int = middle_frc(rec_interior, cell.phantom_pix, smooth=False, L=roi_width)
    _, frc_mv = middle_frc(rec_mv, cell.phantom_pix, smooth=False, L=roi_width)

    resdict = {
        "freq": x,
        "frc_60": frc_60,
        "frc_35": frc_35,
        "frc_int": frc_int,
        "frc_mv": frc_mv,
        "frc_60_s": ndi.gaussian_filter1d(frc_60, sigma=3, mode="nearest"),
        "frc_35_s": ndi.gaussian_filter1d(frc_35, sigma=3, mode="nearest"),
        "frc_int_s": ndi.gaussian_filter1d(frc_int, sigma=3, mode="nearest"),
        "frc_mv_s": ndi.gaussian_filter1d(frc_mv, sigma=3, mode="nearest"),
    }

    df = pd.DataFrame(resdict)
    df.to_csv(TABLEPATH2, index=False)


def make_figure():
    df1 = pd.read_csv(TABLEPATH1)
    df2 = pd.read_csv(TABLEPATH2)

    set_params()
    fig, (ax1, ax2) = plt.subplots(
        ncols=2, figsize=set_size(TEXTWIDTH_FULL, n=2), layout="tight"
    )

    labels = {
        # '60':r'Full \qty{60}{nm}',
        "60": r"Full 60 nm",
        "35": r"Full 35 nm",
        "int": r"Interior",
        "mv": r"Combined",
    }
    colors = {
        "60": "C0",
        "35": "C1",
        "int": "C2",
        "mv": "C3",
    }

    for key in ["60", "35", "int", "mv"]:
        ax1.plot(df1["roi"], df1[key] - df1["35m"], label=labels[key])

        frckey = f"frc_{key}"
        frckey_s = f"frc_{key}_s"

        ax2.plot(df2["freq"], df2[frckey], alpha=0.5, color=colors[key])
        ax2.plot(df2["freq"], df2[frckey_s], color=colors[key], label=labels[key])

    ax1.set_ylim(-6.5, 0.5)
    ax1.set_xlim(0, 16)
    ax1.set_ylabel(r"Relative PSNR [dB]")
    ax1.set_xlabel(r"ROI width [\si{\um}]")
    ax2.set_ylabel(r"FRC")
    ax2.set_xlabel(r"Nyquist Frequency")

    ax1.legend(loc=4)
    ax2.legend()

    plt.show()
    fig.savefig(EXPORT / "xm2_janelia_noiseless.pdf", format="pdf", bbox_inches="tight")


if __name__ == "__main__":
    make_table1()
    make_table2()

    make_figure()

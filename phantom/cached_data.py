import numpy as np
from ncxt_psftomo import sA_psf
import imageio

from .psf import effective_psf, mzp35, mzp60

import phantom

BASE = phantom.WD_FOLDER

MZPFUNC = {
    "35": mzp35,
    "60": mzp60,
}


def cached_psf_R(OZW, L, R, pix_nm, BW):
    if OZW == "bl":
        psf = np.zeros((L, 3, 3))
        psf[:, 1, 1] = 1
        return psf

    if OZW == "35m":
        psf = cached_psf_R("35", L, R, pix_nm, BW)
        psf_mid = 1.0 * psf[L // 2, :, :]
        for i in range(L):
            psf[i, :, :] = psf_mid
        return psf

    filename = BASE / f"theoretical_psf_{OZW}_{L}_r{R}_{pix_nm}_{BW}.npy"
    if filename.exists():
        print("Loading cached PSF")
        psf = np.load(filename)
        return np.nan_to_num(psf, nan=1 / np.prod(psf.shape[1:]))

    pdf = effective_psf(MZPFUNC[OZW], L=L, pix_nm=pix_nm, bandwidth=BW, R=R)
    np.save(filename, pdf)
    return pdf


def cached_psf(OZW, L, pix_nm, BW):
    if OZW == "bl":
        psf = np.zeros((L, 3, 3))
        psf[:, 1, 1] = 1
        return psf

    if OZW == "35m":
        psf = cached_psf("35", L, pix_nm, BW)
        psf_mid = 1.0 * psf[L // 2, :, :]
        for i in range(L):
            psf[i, :, :] = psf_mid
        return psf

    filename = BASE / f"theoretical_psf_{OZW}_{L}_{pix_nm}_{BW}.npy"
    if filename.exists():
        print("Loading cached PSF")
        psf = np.load(filename)
        return np.nan_to_num(psf, nan=1 / np.prod(psf.shape[1:]))

    pdf = effective_psf(MZPFUNC[OZW], L=L, pix_nm=pix_nm, bandwidth=BW)
    np.save(filename, pdf)
    return pdf


def cached_proj(phantom, angles, OZW, pix_nm, BW):
    shapestr = "x".join([str(x) for x in phantom.shape])
    filename = BASE / f"psf_proj_{shapestr}_{len(angles)}_{OZW}_{pix_nm}_{BW}.tiff"

    if filename.exists():
        print("Loading cached Projections")
        return imageio.volread(filename)

    phantom_psf = np.transpose(phantom, (2, 1, 0))

    full_width = int(np.ceil(np.linalg.norm(phantom_psf.shape[:2])))
    print("full_width", full_width)
    psf_len = int(2 ** np.ceil(np.log2(full_width)))
    print("psf_len", psf_len)
    psf = cached_psf(OZW, psf_len, pix_nm, BW)
    projections = np.zeros((len(angles), full_width, phantom_psf.shape[2]))
    print("Calculating forward projections for ", OZW)
    sA_psf(phantom_psf, projections, psf, angles)

    projections = np.transpose(projections, (2, 0, 1))
    imageio.volsave(filename, projections.astype("float32"))
    return projections.astype("float32")


def cached_volfunc(volfunc, name, make_if_missing=True):
    filename = BASE / (name + ".tiff")
    # print('Looking for',name)

    if filename.exists():
        print("Loading cached volfunc", filename)
        return imageio.volread(filename)
    if not make_if_missing:
        return None

    print("Making", name)
    volume = volfunc()
    print(f"vol is of shape", volume.shape)

    imageio.volsave(filename, volume.astype("float32"))
    return volume.astype("float32")

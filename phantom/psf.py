import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm
from scipy.special import jv


def Lommel(u, v, n, tol=1e-12, max_iters=100):
    U = u * 0
    for m in range(max_iters):
        next_el = (-1) ** m * (u / v) ** (n + 2 * m) * jv(n + 2 * m, v)
        norm_next = np.linalg.norm(next_el)
        norm_current = np.linalg.norm(u)
        if norm_next / norm_current < tol:
            return U
        U += next_el
    return U


def PSF_uv_vec(u, v):
    sel_uv = np.abs(u) + np.abs(v) == 0
    sel_u = (u == 0) ^ sel_uv
    sel_v = (v == 0) ^ sel_uv
    sel = ~(sel_u + sel_v + sel_uv)

    h = np.zeros(u.shape)

    h[sel] = (2 / u[sel]) ** 2 * (
        Lommel(u[sel], v[sel], 1) ** 2 + Lommel(u[sel], v[sel], 2) ** 2
    )

    h[sel_u] = (2 * jv(1, v[sel_u]) / v[sel_u]) ** 2
    h[sel_v] = (np.sin(u[sel_v] / 4) / (u[sel_v] / 4)) ** 2
    h[sel_uv] = 1
    return h


def PSF_rz_vec(r, z, lamb, NA):
    u = 2 * np.pi / lamb * NA**2 * z
    v = 2 * np.pi / lamb * NA * r
    return PSF_uv_vec(u, v)


PREFIX = {
    "Qm": 30,
    "Rm": 27,
    "Ym": 24,
    "Zm": 21,
    "Em": 18,
    "Pm": 15,
    "Tm": 12,
    "Gm": 9,
    "Mm": 6,
    "km": 3,
    "hm": 2,
    "dam": 1,
    "m": 0,
    "dm": -1,
    "cm": -2,
    "mm": -3,
    "um": -6,
    "nm": -9,
    "pm": -12,
    "fm": -15,
    "am": -18,
    "zm": -21,
    "ym": -24,
    "rm": -27,
    "qm": -30,
}


def unit(unit):
    return 10 ** PREFIX[unit]


def psf_volume(resolution, depth_of_field, R, L, df):
    NA = resolution / (0.610 * depth_of_field)
    wavelength = depth_of_field * NA**2

    r = np.linspace(-R, R, 2 * R + 1)
    z = np.linspace(0, L, L) - L / 2 + df
    zz, rr1, rr2 = np.meshgrid(z, r, r, indexing="ij")
    rr = np.sqrt(rr1**2 + rr2**2)
    h = PSF_rz_vec(rr, zz, wavelength, NA)
    for h_slice in h:
        h_slice /= np.sum(h_slice)

    return h


E = 550
H = 4.135667696e-15  # eVs
C = 299792458  # m/s
WL = H * C / E


def mzp60(energy, energy_ref=517):
    f0 = 1916 * unit("um")
    ozw = 60 * unit("nm")
    NA = WL / (2 * ozw)

    wl = H * C / energy
    resolution = 0.61 * wl / NA
    dof = wl / NA**2

    f_x = energy / E * f0
    f_ref = energy_ref / E * f0
    deltaf = f_x - f_ref
    return resolution, dof, deltaf


def mzp35(energy, energy_ref=517):
    f0 = 931 * unit("um")
    ozw = 35 * unit("nm")
    NA = WL / (2 * ozw)

    wl = H * C / energy
    resolution = 0.61 * wl / NA
    dof = wl / NA**2

    f_x = energy / E * f0
    f_ref = energy_ref / E * f0
    deltaf = f_x - f_ref
    return resolution, dof, deltaf


def effective_psf(mzp_x, L=512, pix_nm=40, n_sampling=21, bandwidth=300, R=5):
    sigma = 517 / bandwidth / 2.355
    xx = np.linspace(517 - 3 * sigma, 517 + 3 * sigma, n_sampling)
    yy = norm.pdf(xx, loc=517, scale=sigma)

    pix = pix_nm * unit("nm")
    psf_list = []

    for energy in tqdm(xx):
        res, dof, f = mzp_x(energy)
        psf_list.append(psf_volume(res / pix, dof / pix, R, L, f / pix))

    psf_eff = np.sum(
        [weight / np.sum(yy) * psf for weight, psf in zip(yy, psf_list)], 0
    )

    return psf_eff

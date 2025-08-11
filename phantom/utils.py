import numpy as np
from scipy import ndimage as ndi 

from skimage.metrics import peak_signal_noise_ratio

def add_noise(x, I, ff = 1,seed=None):
    I = ff*I
    I_im = I*np.exp(-x)
    np.random.seed(seed)
    I_im = np.random.poisson(I_im)
    I_im[I_im<1]=1
    return -np.log(1.0*I_im/I)

def get_psnr(x, ref, mask=None):
    data_range = np.percentile(ref,99)
    if mask is None:
        return peak_signal_noise_ratio(x, ref, data_range=data_range)
    return peak_signal_noise_ratio(x[mask>0], ref[mask>0], data_range=data_range)


def x_translation(center_relative, angles):
    offset_x = center_relative[1] * np.cos(angles) + center_relative[0] * np.sin(angles)
    return offset_x

def getCropLim(image, pad=64, th=0.01):

    num_el_0 = ndi.maximum_filter(np.sum(np.sum(image, 1), 1), 2 * pad + 1)
    num_el_1 = ndi.maximum_filter(np.sum(np.sum(image, 0), 1), 2 * pad + 1)
    num_el_2 = ndi.maximum_filter(np.sum(np.sum(image, 0), 0), 2 * pad + 1)

    prof_list = [num_el_0, num_el_1, num_el_2]
    count_lim = [th * np.max(x) for x in prof_list]

    clims = [
        (np.min(np.where(p > c)), np.max(np.where(p > c))+1)
        for p, c in zip(prof_list, count_lim)
    ]

    return clims


def crop_to_nonzero_of(img, label, pad=1):
    clim = getCropLim(label, pad=pad)
    return img[
        clim[0][0] : clim[0][1], clim[1][0] : clim[1][1], clim[2][0] : clim[2][1]
    ]

def profile_func(
    volume, reference, edt, mask, func, n_bins=20, sampling=100, r_max=None, unsafe=False
):
    r_max = np.max(edt) if r_max == None else r_max
    n_bins = int(min(r_max, n_bins))

    x = edt[mask > 0][::sampling]
    y = volume[mask > 0][::sampling]
    y_ref = reference[mask > 0][::sampling]

    edges = np.linspace(0, r_max, n_bins + 1)
    dx = edges[1] - edges[0]
    r_ind = np.round((np.ravel(x) - 0.5 * dx) / dx).astype(int)

    if unsafe:
        binned_x = np.zeros((n_bins,))
        binned_y = np.zeros((n_bins,))
        for i in range(n_bins):
            binned_y[i] = func(np.ravel(y)[r_ind == i],np.ravel(y_ref)[r_ind == i])
            binned_x[i] = np.mean(np.ravel(x)[r_ind == i])
    else:
        binned_x = []
        binned_y = []
        for i in range(n_bins):
            if np.sum(r_ind == i):
                y_val = func( np.ravel(y)[r_ind == i],np.ravel(y_ref)[r_ind == i])
                binned_y.append(y_val)
                binned_x.append(np.mean(np.ravel(x)[r_ind == i]))
        binned_x = np.array(binned_x)
        binned_y = np.array(binned_y)

    return binned_x, binned_y     
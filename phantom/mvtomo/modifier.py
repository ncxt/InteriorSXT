import torch
import torch.nn.functional as F
import tomosipo as ts

import numpy as np
from tomosipo.Operator import _to_link
from scipy import ndimage as ndi
import warnings
from tomosipo.Operator import Operator, BackprojectionOperator
from tomosipo.Operator import _to_link, direct_bp, direct_fp, Data


def gaussian_kernel1d(size: int, sigma: float):
    """Generate a 1D Gaussian kernel."""
    coords = torch.arange(size) - size // 2
    kernel = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= kernel.sum()  # Normalize
    return kernel


def apply_nearest_neighbor_padding(tensor: torch.Tensor, kernel_size: int, axis: int):
    """Apply nearest neighbor padding along the specified axis."""
    pad = kernel_size // 2
    padding = [0, 0, 0, 0, 0, 0]

    if axis == 0:  # Depth axis
        padding[0] = pad
        padding[1] = pad
    elif axis == 0:  # Depth axis
        padding[2] = pad
        padding[3] = pad
    elif axis == 2:  # Width axis
        padding[4] = pad
        padding[5] = pad

    return F.pad(tensor, padding, mode="replicate")


def gaussian_blur_sep_nn(tensor: torch.Tensor, kernel_size=5, sigma=1.0):
    """Apply a 3D Gaussian blur over axes (0,2) while keeping axis 1 intact."""
    kernel_1d = gaussian_kernel1d(kernel_size, sigma).to(tensor.device)
    kernel_1d = kernel_1d.view(1, 1, 1, 1, kernel_size)

    # Reshape tensor for Conv3d: (batch=1, channels=D, H, W) → (1, D, H, W)
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, D, H, W)
    tensor_padded = apply_nearest_neighbor_padding(tensor, kernel_size, axis=0)
    modified = F.conv3d(tensor_padded, kernel_1d).permute((0, 1, 3, 4, 2))
    tensor_padded = apply_nearest_neighbor_padding(modified, kernel_size, axis=0)
    modified = F.conv3d(tensor_padded, kernel_1d).permute((0, 1, 4, 2, 3))
    return modified.squeeze(0).squeeze(0)


class GaussianOperator(Operator):
    """
    A modified operator that applies a Gaussian blur to projections
    during forward and backward projection.

    Parameters:
    -----------
    A : Operator
        An instance of the base `Operator` class that provides forward
        and backward projection functionality.
    sigma : float
        The standard deviation of the Gaussian filter.
    n_sigma : int, optional (default=3)
        Determines the kernel size as `2 * n_sigma * sigma + 1`.
    """

    def __init__(self, A, sigma, n_sigma=3):
        # Copy all attributes from original A
        for attr, value in A.__dict__.items():
            setattr(self, attr, value)

        self.sigma = sigma
        self.n_sigma = n_sigma
        self.kernel_size = int(n_sigma * sigma) * 2 + 1

        self._transpose = BackprojectionOperator(self)

    def conv_np(self, volume):
        return ndi.gaussian_filter(volume, (self.sigma, 0, self.sigma), mode="nearest")

    def conv_pytorch(self, volume):
        return gaussian_blur_sep_nn(
            volume, kernel_size=self.kernel_size, sigma=self.sigma
        )

    def _fp(self, volume, out=None):
        if out is not None and isinstance(out, np.ndarray):
            warnings.warn("np.ndarray inplace operations not supported]")

        vlink = _to_link(self.astra_compat_vg, volume)

        if out is not None:
            plink = _to_link(self.astra_compat_pg, out)
        else:
            if self.additive:
                plink = vlink.new_zeros(self.range_shape)
            else:
                plink = vlink.new_empty(self.range_shape)

        direct_fp(self.astra_projector, vlink, plink, additive=self.additive)
        if isinstance(plink.data, np.ndarray):
            plink.data[:] = self.conv_np(plink.data)
        elif isinstance(plink.data, torch.Tensor):
            plink.data[:] = self.conv_pytorch(plink.data)

        if isinstance(volume, Data):
            return ts.data(self.projection_geometry, plink.data)
        else:
            return plink.data

    def _bp(self, projection, out=None):
        if out is not None and isinstance(out, np.ndarray):
            warnings.warn("np.ndarray inplace operations not supported]")

        plink = _to_link(self.astra_compat_pg, projection)

        if out is not None:
            vlink = _to_link(self.astra_compat_vg, out)
        else:
            if self.additive:
                vlink = plink.new_zeros(self.domain_shape)
            else:
                vlink = plink.new_empty(self.domain_shape)

        if isinstance(plink.data, np.ndarray):
            plink.data[:] = self.conv_np(plink.data)
        elif isinstance(plink.data, torch.Tensor):
            plink.data[:] = self.conv_pytorch(plink.data)
        direct_bp(
            self.astra_projector,
            vlink,
            plink,
            additive=self.additive,
        )

        if isinstance(projection, Data):
            return ts.data(self.volume_geometry, vlink.data)
        else:
            return vlink.data


def proj2vec(angles, tx=None, ty=None, pixel=1):
    """
    Compute projection geometry vectors for a series of projection angles.

    Parameters
    ----------
    angles : array_like
        Array of projection angles (in radians).
    tx : array_like or None, optional
        Array of horizontal shifts of the detector origin in the detector
        plane for each angle. If None, defaults to zero.
    ty : array_like or None, optional
        Array of vertical shifts of the detector origin in the detector
        plane for each angle. If None, defaults to zero.
    pixel : float, optional
        Pixel size in physical units. Default is 1.

    Returns
    -------
    ray_dir : ndarray, shape (N, 3)
        Ray direction vectors for each projection angle.
    det_pos : ndarray, shape (N, 3)
        Detector origin positions for each projection angle.
    det_v : ndarray, shape (N, 3)
        Detector vertical basis vectors for each angle.
    det_u : ndarray, shape (N, 3)
        Detector horizontal basis vectors for each angle.
    """
    angles = np.asarray(angles)
    N = len(angles)
    if tx is None:
        tx = np.zeros(N)
    if ty is None:
        ty = np.zeros(N)

    sa, ca = np.sin(angles), np.cos(angles)

    ray_dir = np.column_stack((np.zeros(N), -ca, sa))
    det_u = pixel * np.column_stack((np.zeros(N), sa, ca))
    det_v = pixel * np.tile(np.array([1, 0, 0]), (N, 1))
    det_pos = tx[:, None] * det_u + ty[:, None] * det_v

    return ray_dir, det_pos, det_v, det_u


def translate_operator(A: ts.Operator.Operator, tx, ty):
    angles = A.astra_compat_pg.angles
    ray_dir, det_pos, det_v, det_u = proj2vec(angles, tx, ty)
    vec_pg = ts.parallel_vec(
        shape=A.astra_compat_pg.det_shape,
        ray_dir=ray_dir,
        det_pos=det_pos,
        det_v=det_v,
        det_u=det_u,
    )
    return ts.operator(A.astra_compat_vg, vec_pg)

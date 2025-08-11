import numpy as np

# from scipy import ndimage as ndi
# from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt

# import ncxtutils
# from .astrawrapper import (
#     astraA,
#     astraA_vec,
#     astraCGNE,
#     astraCGNE_multi,
#     astraAT,
#     astraAT_vec,
# )
# from astrawrapper import astraCGNE_multi_pnp_single_iter, astraCGNE_pnp_single_iter
# from astrawrapper import astraCGNE_deconv,astraCGNE_multi_deconv,astraCGNE_multi_deconv_gauss
from .cached_data import cached_psf, cached_proj, cached_volfunc

from .mvtomo.base import get_parallel_operator
from .mvtomo.algorithm_single import CGNE as CGNEs
from .mvtomo.algorithms_mv import CGNE as CGNEm
from .mvtomo.modifier import GaussianOperator

# import tomopy

from .utils import add_noise, get_psnr

# from utils import (
#     x_translation,
#     get_psnr,
#     extract_roi,
#     peak_signal_noise_ratio,
#     structural_similarity,
# )


# def add_noise(x, I, ff=1, seed=None):
#     I = ff * I
#     I_im = I * np.exp(-x)
#     data = I_im[x > 1e-3]
#     perc = np.percentile(data, (1, 5, 50))
#     # print(f'Data range: min: {np.min(data):.2f} 1% {perc[0]:.2f}, 5% {perc[1]:.2f} 50% {perc[2]:.2f} ')

#     np.random.seed(seed)
#     I_im = np.random.poisson(I_im)
#     I_im[I_im < 1] = 1
#     return -np.log(1.0 * I_im / I)


class InteriorPhantomPSF:
    def __init__(self, volume, roi_width, pixel_size):
        self.phantom_um = volume
        self.phantom_pix = pixel_size * volume
        self.pixel_size = pixel_size
        self.roi_width = roi_width

        shape = np.array(volume.shape)
        print(f"Shape {shape}")
        print(f"Height {pixel_size*shape[0]:.2f}")
        print(f"In slice {pixel_size*shape[1:]}")
        print(f"Pixel size {1000*pixel_size:.2f} nm")

        self.height = shape[0]
        self.full_width = int(np.ceil(np.linalg.norm(shape[1:])))
        self.offset = [0, 0, 0]

    @property
    def x0(self):
        return 0.0 * self.phantom_pix

    def make_astra(self, angles):
        self.angles = angles
        A = get_parallel_operator(
            self.phantom_pix.shape, (self.height, self.full_width), angles
        )
        self.proj_astra = A(self.phantom_pix)

    #     def get_proj(self, OZW, angles):
    #         return cached_proj(
    #             self.phantom_pix,
    #             angles,
    #             OZW=OZW,
    #             pix_nm=int(1000 * self.pixel_size),
    #             BW=300,
    #         )

    def tag(self):
        shapestr = "x".join([str(x) for x in self.phantom_pix.shape])
        roistr = f"roi{self.roi_width}"
        px_str = f"px{int(1000*self.pixel_size)}"
        return "_".join([shapestr, roistr, px_str])

    def z_equalizing_ff(self, n_images):
        self.make_astra(n_images)
        att_coeff = np.exp(self.proj_astra)
        att_im = np.mean(att_coeff, 2)
        return np.repeat(att_im[:, :, np.newaxis], self.proj_astra.shape[2], axis=2)

    def roi_mask(self, width):
        shape = np.array(self.phantom_pix.shape)
        center_roi = (shape - 1) / 2 + self.offset

        X, Y, Z = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )
        R2 = (Y - center_roi[1]) ** 2 + (Z - center_roi[2]) ** 2
        CYL = abs(X - center_roi[0])

        mask_rad = R2 < (width / 2) ** 2
        mask_height = CYL < (width / 2)

        return mask_rad * mask_height

    def center_distance(self):
        shape = np.array(self.phantom_pix.shape)
        center_roi = (shape - 1) / 2 + self.offset

        X, Y, Z = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing="ij"
        )
        R2 = (Y - center_roi[1]) ** 2 + (Z - center_roi[2]) ** 2

        return np.sqrt(R2)

    #     def oracle_rec(self, OZW, n_angles, noneg):
    #         angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    #         proj = cached_proj(
    #             self.phantom_pix,
    #             angles,
    #             OZW=OZW,
    #             pix_nm=int(1000 * self.pixel_size),
    #             BW=300,
    #         )

    #         def recfunc():
    #             return self.oracle_cgne_roi(
    #                 proj, angles, roi_width=self.roi_width, n_iters=100, noneg=noneg
    #             )

    #         name = f"rec_ozw{OZW}_n{n_angles}_nn{int(noneg)}" + self.tag()
    #         return cached_volfunc(recfunc, name)

    #     def oracle_rec_mv(self, OZW_full, OZW_roi, n_angles, noneg):
    #         angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    #         proj_full = cached_proj(
    #             self.phantom_pix,
    #             angles,
    #             OZW=OZW_full,
    #             pix_nm=int(1000 * self.pixel_size),
    #             BW=300,
    #         )
    #         proj = cached_proj(
    #             self.phantom_pix,
    #             angles,
    #             OZW=OZW_roi,
    #             pix_nm=int(1000 * self.pixel_size),
    #             BW=300,
    #         )
    #         crop = (proj.shape[2] - self.roi_width) // 2
    #         proj_trunk = proj[:, :, crop:-crop]

    #         def recfunc():
    #             return self.oracle_cgne_roi_multiview(
    #                 proj_full,
    #                 angles,
    #                 proj_trunk,
    #                 angles,
    #                 roi_width=self.roi_width,
    #                 n_iters=100,
    #                 noneg=noneg,
    #             )

    #         name = f"rec_mv_ozw{OZW_full}_{OZW_roi}_n{n_angles}_nn{int(noneg)}" + self.tag()
    #         return cached_volfunc(recfunc, name)

    #     def FBP_poisson(self, OZW, n_angles, intensity, seed=1):
    #         angles = np.linspace(0, np.pi, n_angles, endpoint=False)
    #         proj = cached_proj(
    #             self.phantom_pix,
    #             angles,
    #             OZW=OZW,
    #             pix_nm=int(1000 * self.pixel_size),
    #             BW=300,
    #         )

    #         ff_stack = self.z_equalizing_ff(n_angles)
    #         proj_poisson = add_noise(proj, intensity, ff=ff_stack, seed=seed)

    #         def recfunc():
    #             recon = tomopy.recon(
    #                 proj_poisson,
    #                 -angles,
    #                 algorithm=tomopy.astra,
    #                 options={"proj_type": "cuda", "method": "FBP_CUDA"},
    #                 sinogram_order=True,
    #             )
    #             crop = [
    #                 int((a - b) / 2) for a, b in zip(recon.shape, self.phantom_pix.shape)
    #             ]
    #             return recon[:, crop[1] : -crop[1], crop[2] : -crop[2]]

    #         name = f"FBP_ozw{OZW}_n{n_angles}_I{intensity}({seed})" + self.tag()
    #         return cached_volfunc(recfunc, name)

    def oracle_rec_poisson(
        self, OZW, n_angles, noneg, intensity, seed, n_max=500, cache=True, plot=False
    ):
        str_ozw = f"ozw{OZW}"
        str_na = f"n{n_angles}"
        str_I = f"I{intensity}"
        name = f"rec_{str_ozw}_{str_na}_nn{int(noneg)}_{str_I}({seed})" + self.tag()

        def recfunc():
            angles = np.linspace(0, np.pi, n_angles, endpoint=False)
            proj = cached_proj(
                self.phantom_pix,
                angles,
                OZW=OZW,
                pix_nm=int(1000 * self.pixel_size),
                BW=300,
            )

            ff_stack = self.z_equalizing_ff(n_angles)
            proj_poisson = add_noise(proj, intensity, ff=ff_stack, seed=seed)

            nn_step = 1 if noneg else None
            rec_cgne = CGNEs(self.x0, proj_poisson, angles, nn_step=nn_step)
            mask = self.roi_mask(self.roi_width)

            def metric_roi(x, oracle):
                return get_psnr(x, oracle, mask=mask)

            rec_cgne.set_metric(metric_roi)
            vol_cgne, psnr_cgne = rec_cgne(
                n_max, oracle=self.phantom_pix, stop_at_best=True
            )
            if plot:
                f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(13, 3))
                ax1.plot(psnr_cgne)
                ax2.imshow(vol_cgne[self.height // 2])
                ax3.imshow(self.phantom_pix[self.height // 2])
            return vol_cgne

        if not cache:
            return recfunc()
        return cached_volfunc(recfunc, name)

    def oracle_rec_interior_poisson(
        self, OZW, n_angles, noneg, intensity, seed, n_max=500, cache=True, plot=False
    ):
        str_ozw = f"ozw{OZW}"
        str_na = f"n{n_angles}"
        str_I = f"I{intensity}"
        name = (
            f"rec_interior_{str_ozw}_{str_na}_nn{int(noneg)}_{str_I}({seed})"
            + self.tag()
        )

        def recfunc():
            angles = np.linspace(0, np.pi, n_angles, endpoint=False)
            proj = cached_proj(
                self.phantom_pix,
                angles,
                OZW=OZW,
                pix_nm=int(1000 * self.pixel_size),
                BW=300,
            )

            ff_stack = self.z_equalizing_ff(n_angles)
            proj_poisson = add_noise(proj, intensity, ff=ff_stack, seed=seed)

            crop = (proj.shape[2] - self.roi_width) // 2
            proj_trunk = np.ascontiguousarray(proj_poisson[:, :, crop:-crop]).astype(
                "float32"
            )

            nn_step = 1 if noneg else None
            rec_cgne = CGNEs(self.x0, proj_trunk, angles, nn_step=nn_step)
            mask = self.roi_mask(self.roi_width)

            def metric_roi(x, oracle):
                return get_psnr(x, oracle, mask=mask)

            rec_cgne.set_metric(metric_roi)
            vol_cgne, psnr_cgne = rec_cgne(
                n_max, oracle=self.phantom_pix, stop_at_best=True
            )
            if plot:
                f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(13, 3))
                ax1.plot(psnr_cgne)
                ax2.imshow(vol_cgne[self.height // 2])
                ax3.imshow(self.phantom_pix[self.height // 2])
            return vol_cgne

        if not cache:
            return recfunc()
        return cached_volfunc(recfunc, name)

    def oracle_rec_mv_poisson(
        self,
        OZW_full,
        OZW_roi,
        n_angles_full,
        n_angles_roi,
        intensity_full,
        intensity_roi,
        noneg,
        seed,
        n_max=500,
        cache=True,
        plot=False,
    ):
        str_ozw = f"ozw{OZW_full}_{OZW_roi}"
        str_na = f"n{n_angles_full}_{n_angles_roi}"
        str_I = f"I{intensity_full}_{intensity_roi}"
        name = f"rec_mv_{str_ozw}_{str_na}_nn{int(noneg)}_{str_I}({seed})" + self.tag()

        def recfunc():
            angles_full = np.linspace(0, np.pi, n_angles_full, endpoint=False)
            angles_roi = np.linspace(0, np.pi, n_angles_roi, endpoint=False)
            common_kwargs = {"pix_nm": int(1000 * self.pixel_size), "BW": 300}

            proj_full = cached_proj(
                self.phantom_pix, angles_full, OZW=OZW_full, **common_kwargs
            )
            ff_stack = self.z_equalizing_ff(n_angles_full)
            proj_full_poisson = add_noise(
                proj_full, intensity_full, ff=ff_stack, seed=seed
            )

            proj = cached_proj(
                self.phantom_pix, angles_roi, OZW=OZW_roi, **common_kwargs
            )
            ff_stack = self.z_equalizing_ff(n_angles_roi)
            proj_poisson = add_noise(proj, intensity_roi, ff=ff_stack, seed=seed)
            crop = (proj.shape[2] - self.roi_width) // 2
            proj_trunk = np.ascontiguousarray(proj_poisson[:, :, crop:-crop]).astype(
                "float32"
            )

            kwargs = {
                "y": [proj_full_poisson, proj_trunk],
                "angles": [angles_full, angles_roi],
            }
            rec_cgne = CGNEm(self.x0, **kwargs)
            # define masked PSNR as metric
            mask = self.roi_mask(self.roi_width)

            def metric_roi(x, oracle):
                return get_psnr(x, oracle, mask=mask)

            rec_cgne.set_metric(metric_roi)

            vol_cgne, psnr_cgne = rec_cgne(
                n_max, oracle=self.phantom_pix, stop_at_best=True
            )
            if plot:
                f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(13, 3))
                ax1.plot(psnr_cgne)
                ax2.imshow(vol_cgne[self.height // 2])
                ax3.imshow(self.phantom_pix[self.height // 2])
            return vol_cgne

        if not cache:
            return recfunc()
        return cached_volfunc(recfunc, name)

    def oracle_rec_mv_poisson_deco_gauss(
        self,
        OZW_full,
        OZW_roi,
        n_angles_full,
        n_angles_roi,
        intensity_full,
        intensity_roi,
        noneg,
        seed,
        n_max=500,
        cache=True,
        plot=False,
    ):
        str_ozw = f"ozw{OZW_full}_{OZW_roi}"
        str_na = f"n{n_angles_full}_{n_angles_roi}"
        str_I = f"I{intensity_full}_{intensity_roi}"
        name = (
            f"rec_mv_deco_gauss_{str_ozw}_{str_na}_nn{int(noneg)}_{str_I}({seed})"
            + self.tag()
        )

        def recfunc():
            angles_full = np.linspace(0, np.pi, n_angles_full, endpoint=False)
            angles_roi = np.linspace(0, np.pi, n_angles_roi, endpoint=False)
            pix_nm = int(1000 * self.pixel_size)
            common_kwargs = {"pix_nm": pix_nm, "BW": 300}

            proj_full = cached_proj(
                self.phantom_pix, angles_full, OZW=OZW_full, **common_kwargs
            )
            ff_stack = self.z_equalizing_ff(n_angles_full)
            proj_full_poisson = add_noise(
                proj_full, intensity_full, ff=ff_stack, seed=seed
            )

            proj = cached_proj(
                self.phantom_pix, angles_roi, OZW=OZW_roi, **common_kwargs
            )
            ff_stack = self.z_equalizing_ff(n_angles_roi)
            proj_poisson = add_noise(proj, intensity_roi, ff=ff_stack, seed=seed)
            crop = (proj.shape[2] - self.roi_width) // 2
            proj_trunk = proj_poisson[:, :, crop:-crop]

            sigma_full = int(OZW_full) / pix_nm / 2.355
            sigma_roi = int(OZW_roi) / pix_nm / 2.355
            print(f"sigma_full {sigma_full:.2f}")
            print(f"sigma_roi {sigma_roi:.2f}")

            kwargs = {
                "y": [proj_full_poisson, proj_trunk],
                "angles": [angles_full, angles_roi],
            }
            rec_cgne = CGNEm(self.x0, **kwargs)
            rec_cgne.A[0] = GaussianOperator(rec_cgne.A[0], sigma_full)
            rec_cgne.A[1] = GaussianOperator(rec_cgne.A[1], sigma_roi)

            # define masked PSNR as metric
            mask = self.roi_mask(self.roi_width)

            def metric_roi(x, oracle):
                return get_psnr(x, oracle, mask=mask)

            rec_cgne.set_metric(metric_roi)

            vol_cgne, psnr_cgne = rec_cgne(
                n_max, oracle=self.phantom_pix, stop_at_best=True
            )
            if plot:
                f, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(13, 3))
                ax1.plot(psnr_cgne)
                ax2.imshow(vol_cgne[self.height // 2])
                ax3.imshow(self.phantom_pix[self.height // 2])
            return vol_cgne

        if not cache:
            return recfunc()
        return cached_volfunc(recfunc, name)

        cached_rec = cached_volfunc(None, name, make_if_missing=False)
        if cached_rec is not None:
            return cached_rec

        angles_full = np.linspace(0, np.pi, n_angles_full, endpoint=False)
        angles_roi = np.linspace(0, np.pi, n_angles_roi, endpoint=False)

        pix_nm = int(1000 * self.pixel_size)
        BW = 300
        proj_full = cached_proj(
            self.phantom_pix,
            angles_full,
            OZW=OZW_full,
            pix_nm=pix_nm,
            BW=BW,
        )
        proj = cached_proj(
            self.phantom_pix,
            angles_roi,
            OZW=OZW_roi,
            pix_nm=pix_nm,
            BW=BW,
        )

        ff_stack = self.z_equalizing_ff(n_angles_full)
        proj_full_poisson = add_noise(proj_full, intensity_full, ff=ff_stack, seed=seed)

        ff_stack = self.z_equalizing_ff(n_angles_roi)
        proj_poisson = add_noise(proj, intensity_roi, ff=ff_stack, seed=seed)

        crop = (proj.shape[2] - self.roi_width) // 2
        proj_trunk = proj_poisson[:, :, crop:-crop]

        sigma_full = int(OZW_full) / pix_nm / 2.355
        sigma_roi = int(OZW_roi) / pix_nm / 2.355
        print(f"sigma_full {sigma_full:.2f}")
        print(f"sigma_roi {sigma_roi:.2f}")

    #     def oracle_rec_mv_poisson_pnp(
    #         self,
    #         OZW_full,
    #         OZW_roi,
    #         n_angles_full,
    #         n_angles_roi,
    #         intensity_full,
    #         intensity_roi,
    #         noneg,
    #         ccoeff,
    #         seed,
    #         sigma=None,
    #     ):
    #         if sigma is None:
    #             sigma = np.round((int(OZW_roi) / 20) / 2.355, 2)
    #         sigma_str = f"_s{sigma}"
    #         ccoeff_str = str(ccoeff).replace(".", "")

    #         str_ozw = f"ozw{OZW_full}_{OZW_roi}"
    #         str_na = f"n{n_angles_full}_{n_angles_roi}"
    #         str_I = f"I{intensity_full}_{intensity_roi}"
    #         name = (
    #             f"rec_mv_pnp_{str_ozw}_{str_na}_nn{int(noneg)}{sigma_str}_c{ccoeff_str}_{str_I}({seed})"
    #             + self.tag()
    #         )
    #         cached_rec = cached_volfunc(None, name, make_if_missing=False)
    #         if cached_rec is not None:
    #             return cached_rec

    #         angles_full = np.linspace(0, np.pi, n_angles_full, endpoint=False)
    #         angles_roi = np.linspace(0, np.pi, n_angles_roi, endpoint=False)

    #         proj_full = cached_proj(
    #             self.phantom_pix,
    #             angles_full,
    #             OZW=OZW_full,
    #             pix_nm=int(1000 * self.pixel_size),
    #             BW=300,
    #         )
    #         proj = cached_proj(
    #             self.phantom_pix,
    #             angles_roi,
    #             OZW=OZW_roi,
    #             pix_nm=int(1000 * self.pixel_size),
    #             BW=300,
    #         )

    #         ff_stack = self.z_equalizing_ff(n_angles_full)
    #         proj_full_poisson = add_noise(proj_full, intensity_full, ff=ff_stack, seed=seed)

    #         ff_stack = self.z_equalizing_ff(n_angles_roi)
    #         proj_poisson = add_noise(proj, intensity_roi, ff=ff_stack, seed=seed)

    #         crop = (proj.shape[2] - self.roi_width) // 2
    #         proj_trunk = proj_poisson[:, :, crop:-crop]

    #         def recfunc():
    #             return self.oracle_cgne_roi_multiview(
    #                 proj_full_poisson,
    #                 angles_full,
    #                 proj_trunk,
    #                 angles_roi,
    #                 roi_width=self.roi_width,
    #                 n_iters=100,
    #                 noneg=noneg,
    #             )

    #         name_x0 = (
    #             f"rec_mv_{str_ozw}_{str_na}_nn{int(noneg)}_{str_I}({seed})" + self.tag()
    #         )
    #         x0 = cached_volfunc(recfunc, name_x0)

    #         coeff = ccoeff * self.pnp_coeff(
    #             proj_full_poisson, angles_full, proj_trunk, angles_roi
    #         )

    #         def agent(x):
    #             x_out = ndi.gaussian_filter(x, sigma)
    #             x_out[x_out < 0] = 0
    #             return x_out

    #         def recfunc_pnp():
    #             return self.oracle_cgne_roi_multiview_pnp(
    #                 proj_full_poisson,
    #                 angles_full,
    #                 proj_trunk,
    #                 angles_roi,
    #                 roi_width=self.roi_width,
    #                 agent=agent,
    #                 coeff=coeff,
    #                 n_iters=100,
    #                 x0=x0,
    #             )

    #         return cached_volfunc(recfunc_pnp, name)

    #     # CGNE ITERATIONS WITH ORACLE KNOWLEDGE
    #     def oracle_cgne_roi(
    #         self,
    #         projections,
    #         angles,
    #         roi_width,
    #         n_iters=10,
    #         n_inner=2,
    #         noneg=True,
    #         x0=None,
    #         stop_at_best=True,
    #     ):
    #         if x0 is not None:
    #             x = 1.0 * x0
    #         else:
    #             x = np.zeros(self.phantom_pix.shape)

    #         roi_mask = self.roi_mask(roi_width)
    #         x_best = None

    #         bar = trange(n_iters, leave=False, disable=False)

    #         psnr = []
    #         psnr_full = []

    #         for i in bar:
    #             x = astraCGNE(
    #                 x, projections, angles, n_iter=n_inner, disable_tqdm=True, chatty=False
    #             )
    #             if noneg:
    #                 x[x < 0] /= 2

    #             psnr_full.append(get_psnr(x, self.phantom_pix))
    #             psnr.append(get_psnr(x, self.phantom_pix, roi_mask == 1))

    #             if np.argmax(psnr) == i:
    #                 x_best = np.copy(x)

    #             cut = i - 3
    #             if np.argmax(psnr) < cut:
    #                 print(f"Break iter at {i}. Max at {np.argmax(psnr)}")
    #                 break

    #         f, ax1 = plt.subplots(ncols=1)
    #         ax1.plot(psnr_full, label="psnr_all")
    #         ax1.plot(psnr, label="psnr_roi")
    #         ax1.legend()
    #         return x_best

    #     def oracle_cgne_roi_multiview(
    #         self,
    #         projections,
    #         angles,
    #         projections2,
    #         angles2,
    #         roi_width,
    #         roi_weight=1,
    #         n_iters=10,
    #         noneg=True,
    #         x0=None,
    #         stop_at_best=True,
    #     ):
    #         if x0 is not None:
    #             x = 1.0 * x0
    #         else:
    #             x = np.zeros(self.phantom_pix.shape)
    #         print(f"x0 norm ", np.linalg.norm(x))
    #         tx = x_translation(self.offset[1:], angles2)
    #         ty = 0 * tx + self.offset[0]
    #         roi_mask = self.roi_mask(roi_width)
    #         x_best = None

    #         bar = trange(n_iters, leave=False, disable=False)

    #         psnr = []
    #         psnr_full = []

    #         for i in bar:
    #             x = astraCGNE_multi(
    #                 x,
    #                 projections,
    #                 angles,
    #                 projections2,
    #                 angles2,
    #                 Tx2=tx,
    #                 Ty2=ty,
    #                 weight2=roi_weight,
    #                 n_iter=2,
    #                 disable_tqdm=True,
    #             )

    #             if noneg:
    #                 x[x < 0] /= 2

    #             psnr_full.append(get_psnr(x, self.phantom_pix))
    #             psnr.append(get_psnr(x, self.phantom_pix, roi_mask == 1))

    #             if np.argmax(psnr) == i:
    #                 x_best = np.copy(x)

    #             cut = i - 1
    #             if np.argmax(psnr) < cut and stop_at_best:
    #                 print(
    #                     f"Break iter at {i}. Max at {np.argmax(psnr)}/{np.argmax(psnr_full)}"
    #                 )
    #                 break

    #         f, ax1 = plt.subplots(ncols=1)
    #         ax1.plot(psnr_full, label="psnr_all")
    #         ax1.plot(psnr, label="psnr_roi")
    #         ax1.legend()
    #         return x_best

    #     def oracle_cgne_roi_pnp(
    #         self,
    #         projections,
    #         angles,
    #         roi_width,
    #         agent,
    #         coeff,
    #         n_iters=10,
    #         x0=None,
    #         stop_at_best=True,
    #     ):
    #         if x0 is not None:
    #             x = 1.0 * x0
    #         else:
    #             x = np.zeros(self.phantom_pix.shape)

    #         print(f"x0 norm ", np.linalg.norm(x))
    #         roi_mask = self.roi_mask(roi_width)
    #         x_best = None

    #         bar = trange(n_iters, leave=False, disable=False)

    #         psnr = []
    #         psnr_full = []
    #         psnr_full.append(get_psnr(x, self.phantom_pix))
    #         psnr.append(get_psnr(x, self.phantom_pix, roi_mask == 1))

    #         v = None
    #         u = None

    #         for i in bar:
    #             x, u, v = astraCGNE_pnp_single_iter(
    #                 x,
    #                 u,
    #                 v,
    #                 projections,
    #                 angles,
    #                 agent=agent,
    #                 coeff=coeff * 0.1,
    #                 chatty=False,
    #             )

    #             psnr_full.append(get_psnr(x, self.phantom_pix))
    #             psnr.append(get_psnr(x, self.phantom_pix, roi_mask == 1))

    #             if np.argmax(psnr) == i:
    #                 x_best = np.copy(x)

    #             cut = i - 5
    #             if np.argmax(psnr) < cut and stop_at_best:
    #                 print(
    #                     f"Break iter at {i}. Max at {np.argmax(psnr)}/{np.argmax(psnr_full)}"
    #                 )
    #                 break

    #         f, ax1 = plt.subplots(ncols=1)
    #         ax1.plot(psnr_full, label=f"psnr_all {psnr_full[-1]:.3f}")
    #         ax1.plot(psnr, label=f"psnr_roi{psnr[-1]:.3f}")

    #         ax1.legend()
    #         return x_best

    #     def oracle_cgne_roi_multiview_pnp(
    #         self,
    #         projections,
    #         angles,
    #         projections2,
    #         angles2,
    #         roi_width,
    #         agent,
    #         coeff,
    #         n_iters=10,
    #         x0=None,
    #         stop_at_best=True,
    #     ):
    #         if x0 is not None:
    #             x = 1.0 * x0
    #         else:
    #             x = np.zeros(self.phantom_pix.shape)

    #         print(f"x0 norm ", np.linalg.norm(x))
    #         # I dont use this now , can be passed to iterfunc if needed
    #         # tx = x_translation(self.offset[1:], angles2)
    #         # ty = 0*tx+self.offset[0]
    #         roi_mask = self.roi_mask(roi_width)
    #         x_best = None

    #         bar = trange(n_iters, leave=False, disable=False)

    #         psnr = []
    #         psnr_full = []
    #         psnr_full.append(get_psnr(x, self.phantom_pix))
    #         psnr.append(get_psnr(x, self.phantom_pix, roi_mask == 1))

    #         v = None
    #         u = None

    #         for i in bar:
    #             x, u, v = astraCGNE_multi_pnp_single_iter(
    #                 x,
    #                 u,
    #                 v,
    #                 projections,
    #                 angles,
    #                 projections2,
    #                 angles2,
    #                 agent=agent,
    #                 coeff=coeff * 0.1,
    #                 weight1=1,
    #                 weight2=1,
    #                 chatty=False,
    #             )

    #             psnr_full.append(get_psnr(x, self.phantom_pix))
    #             psnr.append(get_psnr(x, self.phantom_pix, roi_mask == 1))

    #             if np.argmax(psnr) == len(psnr) - 1:
    #                 x_best = np.copy(x)

    #             cut = i - 5
    #             if np.argmax(psnr) < cut and stop_at_best:
    #                 print(
    #                     f"Break iter at {i}. Max at {np.argmax(psnr)}/{np.argmax(psnr_full)}"
    #                 )
    #                 break

    #         f, ax1 = plt.subplots(ncols=1)
    #         ax1.plot(psnr_full, label=f"psnr_all {psnr_full[-1]:.3f}")
    #         ax1.plot(psnr, label=f"psnr_roi{psnr[-1]:.3f}")

    #         ax1.legend()
    #         return x_best

    #     # OLD OR REDUNDANT CHECK BEFORE USE

    #     def oracle_rec_mv_poisson_warm(
    #         self, OZW_full, OZW_roi, n_angles_full, n_angles_roi, noneg, intensity, seed
    #     ):
    #         angles_full = np.linspace(0, np.pi, n_angles_full, endpoint=False)
    #         angles_roi = np.linspace(0, np.pi, n_angles_roi, endpoint=False)

    #         proj_full = cached_proj(
    #             self.phantom_pix,
    #             angles_full,
    #             OZW=OZW_full,
    #             pix_nm=int(1000 * self.pixel_size),
    #             BW=300,
    #         )
    #         proj = cached_proj(
    #             self.phantom_pix,
    #             angles_roi,
    #             OZW=OZW_roi,
    #             pix_nm=int(1000 * self.pixel_size),
    #             BW=300,
    #         )

    #         ff_stack = self.z_equalizing_ff(n_angles_full)
    #         proj_full_poisson = add_noise(proj_full, intensity, ff=ff_stack, seed=seed)

    #         ff_stack = self.z_equalizing_ff(n_angles_roi)
    #         proj_poisson = add_noise(proj, intensity, ff=ff_stack, seed=seed)

    #         crop = (proj.shape[2] - self.roi_width) // 2
    #         proj_trunk = proj_poisson[:, :, crop:-crop]

    #         x0 = self.oracle_rec_poisson(
    #             OZW_full, n_angles_full, noneg=noneg, intensity=intensity, seed=seed
    #         )

    #         def recfunc():
    #             return self.oracle_cgne_roi_multiview(
    #                 proj_full_poisson,
    #                 angles_full,
    #                 proj_trunk,
    #                 angles_roi,
    #                 roi_width=self.roi_width,
    #                 n_iters=100,
    #                 noneg=noneg,
    #                 x0=x0,
    #             )

    #         name = (
    #             f"rec_mv__warm_ozw{OZW_full}_{OZW_roi}_n{n_angles_full}_{n_angles_roi}_nn{int(noneg)}_I{intensity}({seed})"
    #             + self.tag()
    #         )
    #         return x0, cached_volfunc(recfunc, name)

    #     def oracle_rec_mv_poisson_weight(
    #         self,
    #         OZW_full,
    #         OZW_roi,
    #         n_angles_full,
    #         n_angles_roi,
    #         noneg,
    #         intensity,
    #         seed,
    #         roi_weight=None,
    #     ):
    #         if roi_weight is None:
    #             roi_weight = n_angles_full / n_angles_roi

    #         # check if cached and skip loading projections
    #         name = (
    #             f"rec_mv_weight_ozw{OZW_full}_{OZW_roi}_n{n_angles_full}_{n_angles_roi}_nn{int(noneg)}_I{intensity}_w{roi_weight}_({seed})"
    #             + self.tag()
    #         )
    #         cached_rec = cached_volfunc(None, name, make_if_missing=False)
    #         if cached_rec is not None:
    #             return cached_rec

    #         angles_full = np.linspace(0, np.pi, n_angles_full, endpoint=False)
    #         angles_roi = np.linspace(0, np.pi, n_angles_roi, endpoint=False)

    #         proj_full = cached_proj(
    #             self.phantom_pix,
    #             angles_full,
    #             OZW=OZW_full,
    #             pix_nm=int(1000 * self.pixel_size),
    #             BW=300,
    #         )
    #         proj = cached_proj(
    #             self.phantom_pix,
    #             angles_roi,
    #             OZW=OZW_roi,
    #             pix_nm=int(1000 * self.pixel_size),
    #             BW=300,
    #         )

    #         ff_stack = self.z_equalizing_ff(n_angles_full)
    #         proj_full_poisson = add_noise(proj_full, intensity, ff=ff_stack, seed=seed)

    #         ff_stack = self.z_equalizing_ff(n_angles_roi)
    #         proj_poisson = add_noise(proj, intensity, ff=ff_stack, seed=seed)

    #         crop = (proj.shape[2] - self.roi_width) // 2
    #         proj_trunk = proj_poisson[:, :, crop:-crop]

    #         def recfunc():
    #             return self.oracle_cgne_roi_multiview(
    #                 proj_full_poisson,
    #                 angles_full,
    #                 proj_trunk,
    #                 angles_roi,
    #                 roi_width=self.roi_width,
    #                 n_iters=100,
    #                 noneg=noneg,
    #                 roi_weight=roi_weight,
    #             )

    #         return cached_volfunc(recfunc, name)

    #     def make_full(self, angles):
    #         self.angles = angles
    #         self._proj_full = astraA(self.phantom_pix, angles, self.height, self.full_width)

    #     def make_roi(self, angles, roi_width, offset=[0, 0, 0]):
    #         self.angles_roi = angles
    #         self.offset = offset
    #         self.roi_width = roi_width

    #         tx = x_translation(offset[1:], angles)
    #         ty = 0 * tx + offset[0]

    #         self._proj_roi = astraA_vec(
    #             self.phantom_pix, angles, tx, ty, roi_width, roi_width
    #         )

    #     def nullspace(self, mode="full", n_iter=100, disable_tqdm=False):
    #         x0 = 1.0 * self.phantom_pix

    #         common_opt = {"n_iter": n_iter, "disable_tqdm": disable_tqdm, "chatty": False}

    #         if mode == "full":
    #             return astraCGNE(x0, 0.0 * self._proj_full, self.angles, **common_opt)

    #         if mode == "interior":
    #             tx = x_translation(self.offset[1:], self.angles_roi)
    #             ty = 0 * tx + self.offset[0]
    #             return astraCGNE(
    #                 x0, 0.0 * self._proj_roi, self.angles_roi, Tx=tx, Ty=ty, **common_opt
    #             )

    #         if mode == "combined":
    #             tx = x_translation(self.offset[1:], self.angles_roi)
    #             ty = 0 * tx + self.offset[0]

    #             return astraCGNE_multi(
    #                 x0,
    #                 0.0 * self._proj_full,
    #                 self.angles,
    #                 0.0 * self._proj_roi,
    #                 self.angles_roi,
    #                 Tx2=tx,
    #                 Ty2=ty,
    #                 **common_opt,
    #             )

    #     # def add_noise(self, intensity =None):
    #     #     if intensity is None:
    #     #         self.proj_full =   1.0*self._proj_full
    #     #         self.proj_roi =    1.0*self._proj_roi
    #     #     else:
    #     #         self.proj_full =   add_noise(self.proj_full, intensity)
    #     #         self.proj_roi =   add_noise(self.proj_roi, intensity)

    #     def oracle_cgne(self, projections, angles, n_iters=10, n_inner=2, noneg=True):
    #         x = np.zeros(self.phantom_pix.shape)
    #         x_best = None
    #         bar = trange(n_iters, leave=False, disable=False)
    #         psnr = []
    #         for i in bar:
    #             x = astraCGNE(
    #                 x, projections, angles, n_iter=n_inner, disable_tqdm=True, chatty=False
    #             )
    #             if noneg:
    #                 x[x < 0] /= 2
    #             psnr.append(get_psnr(x, self.phantom_pix))
    #             if np.argmax(psnr) == i:
    #                 x_best = np.copy(x)

    #             cut = i - 3
    #             if np.argmax(psnr) < cut:
    #                 print(f"Break iter at {i}. Max at {np.argmax(psnr)}")
    #                 break
    #         f, ax1 = plt.subplots(ncols=1)
    #         ax1.plot(psnr, label="psnr_all")
    #         ax1.legend()
    #         return x_best

    #     def oracle_cgne_deconv(
    #         self, projections, angles, kernel, x0=None, n_iters=10, n_inner=2, noneg=True
    #     ):
    #         if x0 is not None:
    #             x = 1.0 * x0
    #         else:
    #             x = np.zeros(self.phantom_pix.shape)
    #         x_best = None
    #         bar = trange(n_iters, leave=False, disable=False)
    #         psnr = []
    #         for i in bar:
    #             x = astraCGNE_deconv(
    #                 x,
    #                 projections,
    #                 kernel,
    #                 angles,
    #                 n_iter=n_inner,
    #                 disable_tqdm=True,
    #                 chatty=False,
    #             )
    #             if noneg:
    #                 x[x < 0] /= 2
    #             psnr.append(get_psnr(x, self.phantom_pix))
    #             if np.argmax(psnr) == i:
    #                 x_best = np.copy(x)

    #             cut = i - 3
    #             if np.argmax(psnr) < cut:
    #                 print(f"Break iter at {i}. Max at {np.argmax(psnr)}")
    #                 break
    #         f, ax1 = plt.subplots(ncols=1)
    #         ax1.plot(psnr, label="psnr_all")
    #         ax1.legend()
    #         return x_best

    #     def rec_roi(self, n_iters=10, n_inner=2, noneg=True):
    #         x = np.zeros(self.phantom_pix.shape)
    #         tx = x_translation(self.offset[1:], self.angles_roi)
    #         ty = 0 * tx + self.offset[0]
    #         roi_mask = self.roi_mask(self.roi_width)
    #         x_best = None

    #         bar = trange(n_iters, leave=False, disable=False)
    #         psnr_all = []
    #         psnr_in = []
    #         psnr_out = []

    #         for i in bar:
    #             x = astraCGNE(
    #                 x,
    #                 self.proj_roi,
    #                 self.angles_roi,
    #                 Tx=tx,
    #                 Ty=ty,
    #                 n_iter=n_inner,
    #                 disable_tqdm=True,
    #                 chatty=False,
    #             )
    #             if noneg:
    #                 x[x < 0] /= 2

    #             psnr_all.append(get_psnr(x, self.phantom_pix))
    #             psnr_in.append(get_psnr(x, self.phantom_pix, roi_mask == 1))
    #             psnr_out.append(get_psnr(x, self.phantom_pix, roi_mask == 0))

    #             # if np.argmax(psnr)==i:
    #             #     x_psnr = np.copy(x)
    #             # if np.argmax(ssim)==i:
    #             #     x_ssim = np.copy(x)

    #             # cut = i-3
    #             # if np.argmax(psnr)<cut and np.argmax(ssim)<cut:
    #             #     print(f'Break iter at {i}. Max at {np.argmax(psnr)} {np.argmax(ssim)}' )
    #             #     break

    #         f, ax1 = plt.subplots(ncols=1)
    #         ax1.plot(psnr_all, label="psnr_all")
    #         ax1.plot(psnr_in, label="psnr_in")
    #         ax1.plot(psnr_out, label="psnr_out")
    #         ax1.legend()
    #         return x

    #     def rec_combine(self, n_iters=10, n_inner=2, noneg=True, roi_weight=1):
    #         x = np.zeros(self.phantom_pix.shape)
    #         x_psnr = None
    #         x_ssim = None

    #         tx = x_translation(self.offset[1:], self.angles)
    #         ty = 0 * tx + self.offset[0]

    #         bar = trange(n_iters, leave=False, disable=False)
    #         psnr = []
    #         ssim = []
    #         psnr_full = []
    #         ssim_full = []
    #         data_range = np.percentile(self.phantom_pix, 99)

    #         shape = (self.roi_width, self.roi_width, self.roi_width)
    #         phantom_roi = extract_roi(self.phantom_pix, self.offset, shape)

    #         for i in bar:
    #             x = astraCGNE_multi(
    #                 x,
    #                 self.proj_full,
    #                 self.angles,
    #                 self.proj_roi,
    #                 self.angles_roi,
    #                 Tx2=tx,
    #                 Ty2=ty,
    #                 weight2=roi_weight,
    #                 n_iter=n_inner,
    #                 disable_tqdm=True,
    #             )
    #             if noneg:
    #                 x[x < 0] /= 2

    #             rec_roi = extract_roi(x, self.offset, shape)
    #             psnr.append(
    #                 peak_signal_noise_ratio(rec_roi, phantom_roi, data_range=data_range)
    #             )
    #             ssim.append(
    #                 structural_similarity(rec_roi, phantom_roi, data_range=data_range)
    #             )
    #             psnr_full.append(
    #                 peak_signal_noise_ratio(x, self.phantom_pix, data_range=data_range)
    #             )
    #             ssim_full.append(
    #                 structural_similarity(x, self.phantom_pix, data_range=data_range)
    #             )
    #             if np.argmax(psnr) == i:
    #                 x_psnr = np.copy(x)
    #             if np.argmax(ssim) == i:
    #                 x_ssim = np.copy(x)

    #             cut = i - 3
    #             if np.argmax(psnr) < cut and np.argmax(ssim) < cut:
    #                 print(f"Break iter at {i}. Max at {np.argmax(psnr)} {np.argmax(ssim)}")
    #                 break

    #         f, (ax1, ax2) = plt.subplots(ncols=2)
    #         ax1.plot(psnr)
    #         ax2.plot(ssim)
    #         ax1.plot(psnr_full)
    #         ax2.plot(ssim_full)
    #         return x_psnr, x_ssim

    #     def pnp_coeff(
    #         self, projections, angles, projections2=None, angles2=None, Tx=None, Ty=None
    #     ):

    #         height1, _, width1 = projections.shape
    #         dAgent = np.random.random(self.phantom_pix.shape) > 0.5
    #         n_slices, n_rows, n_cols = dAgent.shape

    #         A1dAgent = astraA(dAgent, angles, height1, width1)
    #         ATAdAgent1 = astraAT(A1dAgent, angles, n_rows, n_cols, n_slices)
    #         coeff_full = np.sum(ATAdAgent1) / np.sum(dAgent)
    #         if angles2 is None:
    #             return coeff_full

    #         tx = 0.0 * angles2
    #         ty = 0 * tx
    #         height2, _, width2 = projections2.shape

    #         A2dAgent = astraA_vec(dAgent, angles2, tx, ty, height2, width2)
    #         ATAdAgent2 = astraAT_vec(A2dAgent, angles2, tx, ty, n_rows, n_cols, n_slices)

    #         coeff_roi = np.sum(ATAdAgent2) / np.sum(dAgent)
    #         coeff_joint = np.sum(ATAdAgent1 + ATAdAgent2) / np.sum(dAgent)

    #         print(f"Cfull {coeff_full:.3f}")
    #         print(f"Croi {coeff_roi:.3f}")
    #         print(f"Cjoint {coeff_joint:.3f}")
    #         print(f"Csum {coeff_full+coeff_roi:.3f}")
    #         return coeff_joint

    # # deconv test
    #     def oracle_rec_mv_poisson_deco(
    #         self,
    #         OZW_full,
    #         OZW_roi,
    #         n_angles_full,
    #         n_angles_roi,
    #         intensity_full,
    #         intensity_roi,
    #         noneg,
    #         seed,
    #     ):
    #         str_ozw = f"ozw{OZW_full}_{OZW_roi}"
    #         str_na = f"n{n_angles_full}_{n_angles_roi}"
    #         str_I = f"I{intensity_full}_{intensity_roi}"
    #         name = f"rec_mv_deco_{str_ozw}_{str_na}_nn{int(noneg)}_{str_I}({seed})" + self.tag()
    #         cached_rec = cached_volfunc(None, name, make_if_missing=False)
    #         if cached_rec is not None:
    #             return cached_rec

    #         angles_full = np.linspace(0, np.pi, n_angles_full, endpoint=False)
    #         angles_roi = np.linspace(0, np.pi, n_angles_roi, endpoint=False)

    #         pix_nm = int(1000 * self.pixel_size)
    #         BW = 300
    #         proj_full = cached_proj(
    #             self.phantom_pix,
    #             angles_full,
    #             OZW=OZW_full,
    #             pix_nm=pix_nm,
    #             BW=BW,
    #         )
    #         proj = cached_proj(
    #             self.phantom_pix,
    #             angles_roi,
    #             OZW=OZW_roi,
    #             pix_nm=pix_nm,
    #             BW=BW,
    #         )

    #         ff_stack = self.z_equalizing_ff(n_angles_full)
    #         proj_full_poisson = add_noise(proj_full, intensity_full, ff=ff_stack, seed=seed)

    #         ff_stack = self.z_equalizing_ff(n_angles_roi)
    #         proj_poisson = add_noise(proj, intensity_roi, ff=ff_stack, seed=seed)

    #         crop = (proj.shape[2] - self.roi_width) // 2
    #         proj_trunk = proj_poisson[:, :, crop:-crop]

    #         full_width = int(np.ceil(np.linalg.norm(self.phantom_pix.shape[1:])))
    #         psf_len =int(2**np.ceil(np.log2(full_width)))
    #         psf_full = cached_psf(OZW_full, psf_len,pix_nm , BW)[psf_len//2]
    #         psf_roi = cached_psf(OZW_roi, psf_len,pix_nm , BW)[psf_len//2]

    #         def recfunc():
    #             return self.oracle_cgne_roi_multiview_deconvolve(
    #                 proj_full_poisson,
    #                 psf_full,
    #                 angles_full,
    #                 proj_trunk,
    #                 psf_roi,
    #                 angles_roi,
    #                 roi_width=self.roi_width,
    #                 n_iters=200,
    #                 noneg=noneg,
    #             )

    #         return cached_volfunc(recfunc, name)

    #     def oracle_cgne_roi_multiview_deconvolve(
    #         self,
    #         projections,
    #         kernel,
    #         angles,
    #         projections2,
    #         kernel2,
    #         angles2,
    #         roi_width,
    #         roi_weight=1,
    #         n_iters=10,
    #         noneg=True,
    #         x0=None,
    #         stop_at_best=True,
    #     ):
    #         if x0 is not None:
    #             x = 1.0 * x0
    #         else:
    #             x = np.zeros(self.phantom_pix.shape)
    #         print(f"x0 norm ", np.linalg.norm(x))
    #         tx = x_translation(self.offset[1:], angles2)
    #         ty = 0 * tx + self.offset[0]
    #         roi_mask = self.roi_mask(roi_width)
    #         x_best = None

    #         bar = trange(n_iters, leave=False, disable=False)

    #         psnr = []
    #         psnr_full = []

    #         for i in bar:
    #             x = astraCGNE_multi_deconv(
    #                 x,
    #                 projections,
    #                 kernel,
    #                 angles,
    #                 projections2,
    #                 kernel2,
    #                 angles2,
    #                 Tx2=tx,
    #                 Ty2=ty,
    #                 weight2=roi_weight,
    #                 n_iter=2,
    #                 disable_tqdm=True,
    #             )

    #             if noneg:
    #                 x[x < 0] /= 2

    #             psnr_full.append(get_psnr(x, self.phantom_pix))
    #             psnr.append(get_psnr(x, self.phantom_pix, roi_mask == 1))

    #             if np.argmax(psnr) == i:
    #                 x_best = np.copy(x)

    #             cut = i - 1
    #             if np.argmax(psnr) < cut and stop_at_best:
    #                 print(
    #                     f"Break iter at {i}. Max at {np.argmax(psnr)}/{np.argmax(psnr_full)}"
    #                 )
    #                 break

    #         f, ax1 = plt.subplots(ncols=1)
    #         ax1.plot(psnr_full, label="psnr_all")
    #         ax1.plot(psnr, label="psnr_roi")
    #         ax1.legend()
    #         return x_best


#         def recfunc():
#             return self.oracle_cgne_roi_multiview_deconvolve_gauss(
#                 proj_full_poisson,
#                 sigma_full,
#                 angles_full,
#                 proj_trunk,
#                 sigma_roi,
#                 angles_roi,
#                 roi_width=self.roi_width,
#                 n_iters=200,
#                 noneg=noneg,
#             )

#         return cached_volfunc(recfunc, name)

#     def oracle_cgne_roi_multiview_deconvolve_gauss(
#         self,
#         projections,
#         sigma,
#         angles,
#         projections2,
#         sigma2,
#         angles2,
#         roi_width,
#         roi_weight=1,
#         n_iters=10,
#         noneg=True,
#         x0=None,
#         stop_at_best=True,
#     ):
#         if x0 is not None:
#             x = 1.0 * x0
#         else:
#             x = np.zeros(self.phantom_pix.shape)
#         print(f"x0 norm ", np.linalg.norm(x))
#         tx = x_translation(self.offset[1:], angles2)
#         ty = 0 * tx + self.offset[0]
#         roi_mask = self.roi_mask(roi_width)
#         x_best = None

#         bar = trange(n_iters, leave=False, disable=False)

#         psnr = []
#         psnr_full = []

#         for i in bar:
#             x = astraCGNE_multi_deconv_gauss(
#                 x,
#                 projections,
#                 sigma,
#                 angles,
#                 projections2,
#                 sigma2,
#                 angles2,
#                 Tx2=tx,
#                 Ty2=ty,
#                 weight2=roi_weight,
#                 n_iter=2,
#                 disable_tqdm=True,
#             )

#             if noneg:
#                 x[x < 0] /= 2

#             psnr_full.append(get_psnr(x, self.phantom_pix))
#             psnr.append(get_psnr(x, self.phantom_pix, roi_mask == 1))

#             if np.argmax(psnr) == i:
#                 x_best = np.copy(x)

#             cut = i - 1
#             if np.argmax(psnr) < cut and stop_at_best:
#                 print(
#                     f"Break iter at {i}. Max at {np.argmax(psnr)}/{np.argmax(psnr_full)}"
#                 )
#                 break

#         f, ax1 = plt.subplots(ncols=1)
#         ax1.plot(psnr_full, label="psnr_all")
#         ax1.plot(psnr, label="psnr_roi")
#         ax1.legend()
#         return x_best

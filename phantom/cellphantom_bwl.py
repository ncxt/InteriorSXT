import numpy as np
from scipy import ndimage as ndi
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt

# import ncxtutils
# from astrawrapper import astraA,astraA_vec,astraCGNE,astraCGNE_multi
# from .astrawrapper import astraA,astraA_vec,astraCGNE_multi ,astraCGNE
from .utils import get_psnr, x_translation, add_noise

# from utils import x_translation, get_psnr
# from cellphantom_psf import add_noise

import tomosipo as ts
from .mvtomo.base import get_parallel_operator
from .mvtomo.modifier import translate_operator
from .mvtomo.algorithm_single import CGNE as CGNEs
from .mvtomo.algorithms_mv import CGNE as CGNEm


def make_bwl(x, FWHM):
    sigma = FWHM / 2.355
    I_im = 1 * np.exp(-x)
    I_im = ndi.gaussian_filter(I_im, (sigma, 0, sigma))
    return -np.log(1.0 * I_im)


class InteriorPhantom:
    def __init__(self, volume, roi_width, pixel_size, FWHM):
        self.phantom_um = volume
        self.phantom_pix = pixel_size * volume
        self.FWHM = FWHM

        self.phantom_oracle = pixel_size * volume
        self.pixel_size = pixel_size
        self.roi_width = roi_width

        shape = np.array(volume.shape)

        self.height = shape[0]
        self.full_width = int(np.ceil(np.linalg.norm(shape[1:])))
        self.offset = [0, 0, 0]

        self._proj_full = None
        self._proj_roi = None

    @property
    def x0(self):
        return 0.0 * self.phantom_pix

    def make_full(self, angles):
        self.angles = angles

        A = get_parallel_operator(
            self.phantom_pix.shape, (self.height, self.full_width), angles
        )

        self._proj_full_ref = A(self.phantom_pix)
        self._proj_full = make_bwl(self._proj_full_ref, self.FWHM)

    def make_roi(self, angles, roi_width, offset=[0, 0, 0]):
        self.angles_roi = angles
        self.offset = offset
        self.roi_width = roi_width

        tx = x_translation(offset[1:], angles)
        ty = 0 * tx + offset[0]

        A_base = get_parallel_operator(
            self.phantom_pix.shape, (self.height, roi_width), angles
        )
        A = translate_operator(A_base, tx, ty)
        self._proj_roi_ref = A(self.phantom_pix)
        self._proj_roi = make_bwl(self._proj_roi_ref, self.FWHM)

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

    def z_equalizing_ff(self, images):
        att_coeff = np.exp(images)
        att_im = np.mean(att_coeff, 2)
        return np.repeat(att_im[:, :, np.newaxis], images.shape[2], axis=2)

    # def center_distance(self):
    #     shape = np.array(self.phantom_pix.shape)
    #     center_roi = (shape-1)/2+self.offset

    #     X,Y,Z = np.meshgrid(np.arange(shape[0]),
    #                 np.arange(shape[1]),
    #                 np.arange(shape[2]), indexing='ij')
    #     R2 = (Y-center_roi[1])**2+(Z-center_roi[2])**2
    #     return np.sqrt(R2)

    def add_noise(self, I):
        if self._proj_full is not None:
            ff_stack = self.z_equalizing_ff(self._proj_full)
            # print(np.percentile(ff_stack, (1,99)))
            self._proj_full = add_noise(self._proj_full, I, ff=ff_stack)
        if self._proj_roi is not None:
            ff_stack = self.z_equalizing_ff(self._proj_roi)
            self._proj_roi = add_noise(self._proj_roi, I, ff=ff_stack)

    # def full_rec_oracle(self, n_step = 50, max_outer = 5, plot = False):
    #     x = self.x0
    #     x_best = None
    #     psnr = []

    #     bar = trange(max_outer, leave=False, disable=False)
    #     for i in bar:
    #         x = astraCGNE(x, self._proj_full, self.angles, n_iter=n_step)
    #         psnr.append(get_psnr(x,self.phantom_oracle))
    #         if np.argmax(psnr)==i:
    #             x_best = np.copy(x)
    #         cut = i
    #         if np.argmax(psnr)<cut:
    #             print(f'Break iter at {i}. Max at {np.argmax(psnr)}' )
    #             break

    #     if plot:
    #         f, ax1= plt.subplots(ncols =1 )
    #         # ax1.plot(psnr_full,label = 'psnr_all')
    #         ax1.plot(psnr,label = 'psnr_roi')
    #         ax1.legend()
    #     return x_best

    def mv_rec_oracle(self, n_step=100, plot=False):
        kwargs = {
            "y": [self._proj_full, self._proj_roi],
            "angles": [self.angles, self.angles_roi],
        }
        rec_cgne = CGNEm(self.x0, **kwargs)
        # define masked PSNR as metric
        mask = self.roi_mask(self.roi_width)

        def metric_roi(x, oracle):
            return get_psnr(x, oracle, mask=mask)

        rec_cgne.set_metric(metric_roi)

        vol_cgne, psnr_cgne = rec_cgne(
            n_step, oracle=self.phantom_pix, stop_at_best=True
        )

        if plot:
            f, (ax1, ax2, ax3) = plt.subplots(ncols=3)
            ax1.plot(psnr_cgne)
            ax2.imshow(vol_cgne[self.height // 2])
            ax3.imshow(self.phantom_pix[self.height // 2])

        return vol_cgne

    def full_rec(self, n_iter=10):
        rec_cgne = CGNEs(self.x0, self._proj_full, self.angles)
        return rec_cgne(n_iter)

    def roi_rec(self, n_iter=10):
        # return astraCGNE(self.x0, self._proj_roi, self.angles_roi, n_iter=n_iter)
        rec_cgne = CGNEs(self.x0, self._proj_roi, self.angles_roi)
        return rec_cgne(n_iter)

    def mv_rec(self, n_iter=10):
        kwargs = {
            "y": [self._proj_full, self._proj_roi],
            "angles": [self.angles, self.angles_roi],
        }
        rec_cgne = CGNEm(self.x0, **kwargs)
        return rec_cgne(n_iter)

    def nullspace(self, mode="full", n_iter=100, disable_tqdm=False, plot=False):
        x0 = 1.0 * self.phantom_pix
        tx = x_translation(self.offset[1:], self.angles_roi)
        ty = 0 * tx + self.offset[0]

        if mode == "full":
            rec_cgne = CGNEs(x0, 0.0 * self._proj_full, self.angles)

        if mode == "interior":
            rec_cgne = CGNEs(x0, 0.0 * self._proj_roi, self.angles_roi)
            rec_cgne.A = translate_operator(rec_cgne.A, tx, ty)

        if mode == "combined":
            kwargs = {
                "y": [0.0 * self._proj_full, 0.0 * self._proj_roi],
                "angles": [self.angles, self.angles_roi],
            }
            rec_cgne = CGNEm(x0, **kwargs)
            rec_cgne.A[1] = translate_operator(rec_cgne.A[1], tx, ty)

        nullspace = rec_cgne(n_iter)
        if plot:
            f, (ax1, ax2) = plt.subplots(ncols=2)
            ax1.imshow(nullspace[self.height // 2])
            ax2.semilogy(rec_cgne.loss)

        return nullspace

    # def check_nullspace(self, x, mode = 'full'):
    #     if mode == 'full':
    #         height,_,width = self._proj_full.shape
    #         return astraA(x, self.angles, height, width)

    #     if mode == 'interior':
    #         tx = x_translation(self.offset[1:], self.angles_roi)
    #         ty = 0*tx+self.offset[0]
    #         height,_,width = self._proj_roi.shape
    #         return astraA_vec(x, self.angles_roi, tx,ty,height, width)

    # def oracle_cgne_roi(self, projections, angles, roi_width, n_iters = 10,n_inner = 2, noneg = True):
    #     x = np.zeros(self.phantom_pix.shape)
    #     tx = x_translation(self.offset[1:], angles)
    #     ty = 0*tx+self.offset[0]
    #     roi_mask = self.roi_mask(roi_width)
    #     x_best = None

    #     bar = trange(n_iters, leave=False, disable=False)

    #     psnr = []
    #     psnr_full = []

    #     for i in bar:
    #         x = astraCGNE(x,  projections,angles,n_iter = n_inner,
    #                        disable_tqdm = True, chatty = False)
    #         if noneg:
    #             x[x<0]/=2

    #         psnr_full.append(get_psnr(x,self.phantom_pix))
    #         psnr.append(get_psnr(x,self.phantom_pix,roi_mask==1))

    #         if np.argmax(psnr)==i:
    #             x_best = np.copy(x)

    #         cut = i-3
    #         if np.argmax(psnr)<cut:
    #             print(f'Break iter at {i}. Max at {np.argmax(psnr)}' )
    #             break

    #     f, ax1= plt.subplots(ncols =1 )
    #     ax1.plot(psnr_full,label = 'psnr_all')
    #     ax1.plot(psnr,label = 'psnr_roi')
    #     ax1.legend()
    #     return x_best

    # def oracle_cgne_roi_multiview(self, projections, angles,projections2, angles2, roi_width,roi_weight = 1, n_iters = 10,n_inner = 2, noneg = True):
    #     x = np.zeros(self.phantom_pix.shape)
    #     tx = x_translation(self.offset[1:], angles)
    #     ty = 0*tx+self.offset[0]
    #     roi_mask = self.roi_mask(roi_width)
    #     x_best = None

    #     bar = trange(n_iters, leave=False, disable=False)

    #     psnr = []
    #     psnr_full = []

    #     for i in bar:
    #         x = astraCGNE_multi(x,
    #                 projections, angles,
    #                 projections2, angles2,
    #                 Tx2=tx,Ty2=ty,weight2=roi_weight,
    #                 n_iter = n_inner,disable_tqdm = True)

    #         if noneg:
    #             x[x<0]/=2

    #         psnr_full.append(get_psnr(x,self.phantom_pix))
    #         psnr.append(get_psnr(x,self.phantom_pix,roi_mask==1))

    #         if np.argmax(psnr)==i:
    #             x_best = np.copy(x)

    #         cut = i-3
    #         if np.argmax(psnr)<cut:
    #             print(f'Break iter at {i}. Max at {np.argmax(psnr)}' )
    #             break

    #     f, ax1= plt.subplots(ncols =1 )
    #     ax1.plot(psnr_full,label = 'psnr_all')
    #     ax1.plot(psnr,label = 'psnr_roi')
    #     ax1.legend()
    #     return x_best

    # def rec_roi(self, n_iters = 10,n_inner = 2, noneg = True):
    #     x = np.zeros(self.phantom_pix.shape)
    #     tx = x_translation(self.offset[1:], self.angles_roi)
    #     ty = 0*tx+self.offset[0]
    #     roi_mask = self.roi_mask(self.roi_width)
    #     x_best = None

    #     bar = trange(n_iters, leave=False, disable=False)
    #     psnr_all = []
    #     psnr_in = []
    #     psnr_out = []

    #     for i in bar:
    #         x = astraCGNE(x,  self.proj_roi,self.angles_roi,Tx=tx, Ty=ty,
    #                       n_iter = n_inner,
    #                        disable_tqdm = True, chatty = False)
    #         if noneg:
    #             x[x<0]/=2

    #         psnr_all.append(get_psnr(x,self.phantom_pix))
    #         psnr_in.append(get_psnr(x,self.phantom_pix,roi_mask==1))
    #         psnr_out.append(get_psnr(x,self.phantom_pix,roi_mask==0))

    #         # if np.argmax(psnr)==i:
    #         #     x_psnr = np.copy(x)
    #         # if np.argmax(ssim)==i:
    #         #     x_ssim = np.copy(x)

    #         # cut = i-3
    #         # if np.argmax(psnr)<cut and np.argmax(ssim)<cut:
    #         #     print(f'Break iter at {i}. Max at {np.argmax(psnr)} {np.argmax(ssim)}' )
    #         #     break

    #     f, ax1= plt.subplots(ncols =1 )
    #     ax1.plot(psnr_all,label = 'psnr_all')
    #     ax1.plot(psnr_in,label = 'psnr_in')
    #     ax1.plot(psnr_out,label = 'psnr_out')
    #     ax1.legend()
    #     return x

    # def rec_combine(self, n_iters = 10,n_inner = 2, noneg = True, roi_weight=1):
    #     x = np.zeros(self.phantom_pix.shape)
    #     x_psnr = None
    #     x_ssim = None

    #     tx = x_translation(self.offset[1:], self.angles)
    #     ty = 0*tx+self.offset[0]

    #     bar = trange(n_iters, leave=False, disable=False)
    #     psnr = []
    #     ssim = []
    #     psnr_full = []
    #     ssim_full = []
    #     data_range = np.percentile(self.phantom_pix,99)

    #     shape = (self.roi_width,self.roi_width,self.roi_width)
    #     phantom_roi = extract_roi(self.phantom_pix,self.offset, shape )

    #     for i in bar:
    #         x = astraCGNE_multi(x,
    #                 self.proj_full, self.angles,
    #                 self.proj_roi, self.angles_roi,
    #                 Tx2=tx,Ty2=ty,weight2=roi_weight,
    #                 n_iter = n_inner,disable_tqdm = True)
    #         if noneg:
    #             x[x<0]/=2

    #         rec_roi = extract_roi(x,self.offset, shape )
    #         psnr.append(peak_signal_noise_ratio(rec_roi, phantom_roi, data_range=data_range))
    #         ssim.append(structural_similarity(rec_roi, phantom_roi, data_range=data_range))
    #         psnr_full.append(peak_signal_noise_ratio(x, self.phantom_pix, data_range=data_range))
    #         ssim_full.append(structural_similarity(x, self.phantom_pix, data_range=data_range))
    #         if np.argmax(psnr)==i:
    #             x_psnr = np.copy(x)
    #         if np.argmax(ssim)==i:
    #             x_ssim = np.copy(x)

    #         cut = i-3
    #         if np.argmax(psnr)<cut and np.argmax(ssim)<cut:
    #             print(f'Break iter at {i}. Max at {np.argmax(psnr)} {np.argmax(ssim)}' )
    #             break

    #     f, (ax1,ax2)= plt.subplots(ncols =2 )
    #     ax1.plot(psnr)
    #     ax2.plot(ssim)
    #     ax1.plot(psnr_full)
    #     ax2.plot(ssim_full)
    #     return x_psnr,x_ssim

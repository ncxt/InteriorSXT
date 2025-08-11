import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
from scipy import ndimage as ndi
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm
from scipy import special

from tqdm.auto import tqdm



def sigma2fwhm(sigma):
    return (2 * np.sqrt(2 * np.log(2)))*sigma

def fwhm2sigma(FWHM):
    return FWHM/ (2 * np.sqrt(2 * np.log(2)))

def integrate_quadrant(quandrant):
    pow_signal = np.zeros(quandrant.shape)
    n0,n1 = quandrant.shape

    for i in range(n0):
        for j in range(n1):
            el10 = pow_signal[i - 1, j] if i > 0 else 0
            el01 = pow_signal[i, j - 1] if j > 0 else 0
            el11 = pow_signal[i - 1, j - 1] if i > 0 and j > 0 else 0

            el_ij = quandrant[i, j] ** 2
            pow_signal[i, j] = el10 + el01 - el11 + el_ij

    return pow_signal

def airy(r, FWHM):
    v = 2 * np.pi * r*0.61/FWHM
    sel = v!=0
    h2 = r*0+1.0
    h2[sel] = (2 * special.jv(1, v[sel]) / v[sel])**2
    h2/=np.sum(h2)
    return h2



def airy_OTF(k,FWHM):
    y = 0*k
    rho = k*FWHM/(2*0.61)
    rho_sel = rho[np.abs(rho)<1]
    y[np.abs(rho)<1]=(2/np.pi)*(np.arccos(rho_sel)-rho_sel*np.sqrt(1-rho_sel**2))
    return y

def airy_Power(k,FWHM):
    ks = k*FWHM/(2*0.61)



    tmax = -(2/np.pi)**2*((64/45)-2/3*np.pi)
    retval=k*0.0+tmax
    
    x = ks[ks<1]
    t1 = -(2/3)*np.sqrt(1 - x**2) * (2 + x**2)* np.arccos(x) 
    t2 = -(1/45)*x*(60 - 5* x**2 + 9*x**4) 
    t3 = x* np.arccos(x)**2
    retval[ks<1] =  (2/np.pi)**2 * (t1+t2+t3+2/3*np.pi )
    retval[ks>1]=tmax 
    retval/=tmax

    return retval
    # return retval*(2*0.61)/FWHM

def Fradon_full(L,n_angles, R):
    wt = np.arange(-n_angles // 2, n_angles // 2)
    wu = np.arange(-L // 2, L // 2)

    A = n_angles / (2 * np.pi)
    pr0 = 2 * np.pi * R / L

    WU, WT = np.meshgrid(wu, wt, indexing="ij")
    prefix = 2 * np.pi * np.exp(-1j * WT * (np.pi / 2 + 0))
    bess = special.jv(WT, pr0 * WU)
    signal = A * np.abs(prefix * bess)
    return signal

def Fradon_quad(L,n_angles, R):
    wt = np.arange(n_angles // 2)
    wu = np.arange(L // 2)

    A = n_angles / (2 * np.pi)
    pr0 = 2 * np.pi * R / L

    WU, WT = np.meshgrid(wu, wt, indexing="ij")
    prefix = 2 * np.pi * np.exp(-1j * WT * (np.pi / 2 + 0))
    bess = special.jv(WT, pr0 * WU)
    signal = A * np.abs(prefix * bess)
    return signal

def Fradon_quad_180(L,n_angles, R):
    wt = np.arange(n_angles)
    wu = np.arange(L // 2)

    A = n_angles / np.pi
    pr0 = 2 * np.pi * R / L

    WU, WT = np.meshgrid(wu, wt, indexing="ij")
    prefix = 2 * np.pi * np.exp(-1j * WT * (np.pi / 2 + 0))
    bess = special.jv(WT, pr0 * WU)
    signal = A * np.abs(prefix * bess)
    return signal


class Sampling():
    def __init__(self):
        
        self.func_psf = None
        self.psf = None
        self.OTF = None
        self.power = None
        self.power_inv = None    

    def init_gauss(self,  FWHM, cutoff_th = 0.98):
        sigma = fwhm2sigma(FWHM)
        sigmak = 1 / (2 * np.pi * sigma)
        Ak = 1 / np.sqrt(2 * np.pi * sigma**2)

        self.func_psf = lambda x: ndi.gaussian_filter(x, sigma)
        self.psf = lambda x: norm.pdf(x, scale = sigma)
        self.OTF = lambda x: Ak * norm.pdf(x, scale=sigmak)
        const = 1 / (4 * np.sqrt(np.pi) * sigmak)
        self.power = lambda x:  special.erf(x / sigmak)
        self.power_inv = lambda x: special.erfinv(x) * sigmak        

    def init_diff(self, FWHM, cutoff_th = 0.98):

        R = int(10*FWHM)
        r = np.linspace(-R, R, 2 * R + 1)
        v = 2 * np.pi * r*0.61/FWHM
        sel = v!=0
        h2 = r*0+1.0
        h2[sel] = (2 * special.jv(1, v[sel]) / v[sel])**2
        h2/=np.sum(h2)

        self.func_psf = lambda x: ndi.convolve1d(x, h2, mode='nearest')
        self.psf = lambda x: airy(x, FWHM)
        self.OTF = lambda x: airy_OTF(x, FWHM)
        self.power = lambda x: airy_Power(x,FWHM)

        k = np.linspace(0,0.5)
        self.power_inv = lambda x: np.interp(x, self.power(k),k)


    def plot_peak(self,L, cutoff_th = 0.98):
        x = np.arange(-L//2,L//2)
        y = x*0.0
        y[L//2]=1.0
        y = self.func_psf(y)

        k = 1.0 * np.arange(L // 2 + 1) / L
        Fy = np.abs(np.fft.rfft(y))
        intP = cumulative_trapezoid(Fy**2, k, initial=0)

        f, ax= plt.subplots(ncols = 3, figsize = (8,3))
        ax[0].plot(x, y,'.')
        ymax = np.max(y)
        ax[0].set_xlim(-3*1/ymax,3*1/ymax)
        ax[1].plot(k, Fy,'.')
        ax[0].plot(x, self.psf(x))
        ax[1].plot(k, self.OTF(k))

        ax[2].plot(k, intP/intP[-1],'.')
        ax[2].plot(k,self.power(k))

        ax[0].set_title('Signal')
        ax[1].set_title('OTF')
        ax[2].set_title('Power')
        
        ax[0].set_xlabel('x')
        ax[1].set_xlabel('k')
        ax[2].set_xlabel('k')
        
        ax[2].axvline(self.power_inv(cutoff_th))

    def radon(self,L, R, oversample = 1.2):
        cutoff = self.power_inv(0.98)
        print('cutoff',cutoff)

        # cutoff values from Rattey and Lindgren
        cutoff_l = int(2*L*cutoff)
        cutoff_a = int(2*np.pi*cutoff * R + 2) #for 180
        # oversample for plotting purpose
        n_angles = int(oversample*cutoff_a) #for 180
        print(f'L {L} nangles {n_angles}')
        print(f'cutoff_l {cutoff_l}')
        print(f'cutoff_a {cutoff_a}')

        signal_quad = Fradon_quad_180(L,n_angles, R)

        k = np.arange(0, L // 2) / L
        signal_otf_quad = signal_quad * self.OTF(k)[:, np.newaxis]

        eps = 1e-6

        f, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 5))
        aspect = 8 / 13 * n_angles / (L//2)
        signal_kwargs = {
        "norm": LogNorm(vmin=1, vmax=n_angles),
        "aspect": aspect,
        "origin": "lower",
        }
        ax[0, 0].imshow(eps + signal_quad, **signal_kwargs)
        ax[0, 1].imshow(eps + signal_otf_quad, **signal_kwargs)

        power_radon = integrate_quadrant(signal_quad)
        power_otf = integrate_quadrant(signal_otf_quad)
        power_otf /= np.max(power_otf)
        print(power_radon.shape)

        p_cutoff = (min(cutoff_l//2,L//2-1),cutoff_a)
        print('p_cutoff',p_cutoff)
        pnorm = power_otf[p_cutoff] / power_radon[p_cutoff]
        power_radon = 1.0 * power_radon * pnorm

        ax[1, 0].imshow(integrate_quadrant(signal_quad), aspect=aspect, origin="lower")
        ax[1, 1].imshow(integrate_quadrant(signal_otf_quad), aspect=aspect, origin="lower")

        levels = [0.9, 0.99, 0.999]
        colors = ["C1", "C1", "C1"]
        ax[1, 0].contour(power_radon, [1], colors="C1")
        ax[1, 1].contour(power_otf, levels, colors=colors)

        for axis in ax.ravel():
            axis.axhline(cutoff_l//2)
            axis.axvline(cutoff_a)

        xx = np.arange(cutoff_a)
        yy = xx*L/(2*np.pi*R)

        ax[0,0].plot(xx,yy,'r')

        ax[0,0].set_title('F(Radon)')
        ax[0,1].set_title('F(Radon)*OTF')
        ax[1,0].set_title('Integrated power')
        ax[1,1].set_title('Integrated power')

        plt.tight_layout()
        return k,signal_quad,signal_otf_quad

    def angular_sampling(self,L, R, oversample = 1.2, plot = False):
        cutoff = self.power_inv(0.999)

        # cutoff values from Rattey and Lindgren
        cutoff_l = int(2*L*cutoff)
        cutoff_a = int(2*np.pi*cutoff * R + 2) #for 180
        # oversample for plotting purpose
        n_angles = int(oversample*cutoff_a) #for 180
        print(f'L {L} nangles {n_angles}')
        print(f'cutoff_l {cutoff_l}')
        print(f'cutoff_a {cutoff_a}')

        signal_quad = Fradon_quad_180(L,n_angles, R)
        k = np.arange(0, L // 2) / L
        signal_otf_quad = signal_quad * self.OTF(k)[:, np.newaxis]

        power_ideal =   integrate_quadrant(signal_quad)
        power_otf =  integrate_quadrant(signal_otf_quad)

        p_cutoff = (min(cutoff_l//2,L//2-1),cutoff_a)
        print('p_cutoff',p_cutoff)
        if plot:
            f, ax = plt.subplots(figsize=(8, 5))
            # plt.plot(power_ideal[-1,:])
            ax.plot(power_otf[-1,:]/power_otf[p_cutoff],label = 'Full Width')
            ax.plot(power_otf[L//4,:]/power_otf[p_cutoff],label = 'Bin2')
            ax.plot(power_otf[L//8,:]/power_otf[p_cutoff],label = 'Bin4')
            ax.legend()
            ax.set_title('')
            ax.axvline(cutoff_a)
        return np.arange(power_otf.shape[1]),power_otf[-1,:]/power_otf[p_cutoff]

    def power_otf(self,ai, L, R):
        cutoff = self.power_inv(0.999)
        n_angles = int(2*np.pi*cutoff * R + 2) #for 180
        # oversample for plotting purpose
        signal_quad = Fradon_quad_180(L,n_angles, R)
        k = np.arange(0, L // 2) / L
        signal_otf_quad = signal_quad * self.OTF(k)[:, np.newaxis]
        power_otf =  integrate_quadrant(signal_otf_quad)
        if ai>n_angles:
            return 1
        return power_otf[-1,ai]/np.max(power_otf)


    def wiener(self,L, R,signal_power, mumean, dose, oversample = 1.2):
        cutoff = self.power_inv(0.99)

        # cutoff values from Rattey and Lindgren
        cutoff_l = int(2*L*cutoff)
        cutoff_a = int(2*np.pi*cutoff * R + 2) #for 180
        # oversample for plotting purpose
        n_angles = int(oversample*cutoff_a) #for 180
        print(f'L {L} nangles {n_angles}')
        print(f'cutoff_l {cutoff_l}')
        print(f'cutoff_a {cutoff_a}')

        signal_quad = Fradon_quad_180(L,n_angles, R)
        k = np.arange(0, L // 2) / L
        signal_otf_quad = signal_quad * self.OTF(k)[:, np.newaxis]
        OTF2 = self.OTF(k)**2

        wt = np.arange(n_angles)
        wu = np.arange(L // 2)

        WC_direct = np.zeros(signal_quad.shape)
        WC_psf = np.zeros(signal_quad.shape)
        WC_noise = np.zeros(signal_quad.shape)
        WC_noise2 = np.zeros(signal_quad.shape)

        for i in tqdm(wt):
            I0 = dose / (i + 1)
            N = np.exp(mumean) / I0


            R_2 = (signal_quad) ** 2
            HR_2 = (signal_otf_quad) ** 2

            #     pure OTF inv
            SNR =  OTF2[:, np.newaxis]*signal_power / N
            WC = 1 / (1 + 1 / (SNR))
            WC_2 = (WC[:, np.newaxis] * signal_otf_quad) ** 2

            #     radon+OTF inv
            SNR = signal_quad**2 * signal_power / N
            WC = 1 / (1 + 1 / (SNR))
            WCb_2 = (WC * signal_otf_quad) ** 2

            for j in wu:
                WC_direct[j, i] = np.sum(R_2[: j + 1, : i + 1])
                WC_psf[j, i] = np.sum(HR_2[: j + 1, : i + 1])
                WC_noise[j, i] = np.sum(WC_2[: j + 1, : i + 1])
                WC_noise2[j, i] = np.sum(WCb_2[: j + 1, : i + 1])

        WC_psf = WC_psf / np.max(WC_psf)
        WC_noise = WC_noise / np.max(WC_noise)
        WC_noise2 = WC_noise2 / np.max(WC_noise2)

        p_cutoff = (cutoff_l//2,cutoff_a)
        pnorm = WC_psf[p_cutoff] / WC_direct[p_cutoff]
        WC_direct = 1.0 * WC_direct * pnorm

        f, ax = plt.subplots(ncols=3, figsize=(13, 5))
        eps = 1e-3
        aspect = 8 / 13 * n_angles / (L//2)
        ax[0].imshow(WC_psf, aspect=aspect, origin="lower")
        ax[1].imshow(WC_noise, aspect=aspect, origin="lower")
        ax[2].imshow(WC_noise2, aspect=aspect, origin="lower")

        levels = [0.9, 0.99, 0.999]
        colors = ["C1", "C1", "C1"]
        ax[0].contour(WC_psf, levels, colors=colors)
        ax[1].contour(WC_noise, levels, colors=colors)
        ax[2].contour(WC_noise2, levels, colors=colors)

        ax[0].set_title('Power OTF')
        ax[1].set_title('Power Deconv')
        ax[2].set_title('Power Joint')

        return WC_noise,WC_noise2

    def wiener_power_otf(self,xx, L, R,signal_power, mumean, dose):
        n_angles = int(max(xx)+1) 

        signal_quad = Fradon_quad_180(L,n_angles, R)
        k = np.arange(0, L // 2) / L
        signal_otf_quad = signal_quad * self.OTF(k)[:, np.newaxis]
        OTF2 = self.OTF(k)**2

        retval = xx*0

        for i,ai in enumerate(tqdm(xx)):
            I0 = dose / (ai + 1)
            N = np.exp(mumean) / I0
            #     pure OTF inv
            SNR =  OTF2[:, np.newaxis]*signal_power / N
            WC = 1 / (1 + 1 / (SNR))
            WC_2 = (WC[:, np.newaxis] * signal_otf_quad) ** 2
            retval[i]=np.sum(WC_2[:, : ai + 1])

        return retval/np.max(retval)
        










# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:54:20 2017

@author: sajid

Based on the MATLAB code by Michael Wojcik

M. van Heela, and M. Schatzb, "Fourier shell correlation threshold
criteria," Journal of Structural Biology 151, 250-262 (2005)

"""

# importing required libraries

import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt

def spinavej(x):
    """
    read the shape and dimensions of the input image
    """
    shape = np.shape(x)
    dim = np.size(shape)
    """
    Depending on the dimension of the image 2D/3D, create an array of integers 
    which increase with distance from the center of the array
    """
    if dim == 2:
        nr, nc = shape
        nrdc = np.floor(nr / 2) + 1
        ncdc = np.floor(nc / 2) + 1
        r = np.arange(nr) - nrdc + 1
        c = np.arange(nc) - ncdc + 1
        [R, C] = np.meshgrid(r, c)
        index = np.round(np.sqrt(R ** 2 + C ** 2)) + 1

    elif dim == 3:
        nr, nc, nz = shape
        nrdc = np.floor(nr / 2) + 1
        ncdc = np.floor(nc / 2) + 1
        nzdc = np.floor(nz / 2) + 1
        r = np.arange(nr) - nrdc + 1
        c = np.arange(nc) - ncdc + 1
        z = np.arange(nc) - nzdc + 1
        [R, C, Z] = np.meshgrid(r, c, z)
        index = np.round(np.sqrt(R ** 2 + C ** 2 + Z ** 2)) + 1
    else:
        print("input is neither a 2d or 3d array")
    """
    The index array has integers from 1 to maxindex arranged according to distance
    from the center
    """

    maxindex = np.max(index)
    output = np.zeros(int(maxindex), dtype=complex)
    """
    In the next step the output is generated. The output is an array of length
    maxindex. The elements in this array corresponds to the sum of all the elements
    in the original array correponding to the integer position of the output array 
    divided by the number of elements in the index array with the same value as the
    integer position. 
    
    Depening on the size of the input array, use either the pixel or index method.
    By-pixel method for large arrays and by-index method for smaller ones.
    """
    if nr >= 512:
        # print("performed by pixel method")
        sumf = np.zeros(int(maxindex), dtype=complex)
        count = np.zeros(int(maxindex), dtype=complex)
        for ri in range(nr):
            for ci in range(nc):
                sumf[int(index[ri, ci]) - 1] = sumf[int(index[ri, ci]) - 1] + x[ri, ci]
                count[int(index[ri, ci]) - 1] = count[int(index[ri, ci]) - 1] + 1
        output = sumf / count
        return output
    else:
        # print("performed by index method")
        indices = []
        running_sum = 0
        n_sum = []
        for i in np.arange(int(maxindex)):
            indices.append(np.where(index == i + 1))
        for i in np.arange(int(maxindex)):
            output[i] = sum(x[indices[i]]) / len(indices[i][0])
            running_sum += len(indices[i][0])
            n_sum.append(len(indices[i][0]))

        return output

def FSC(i1, i2, disp=0, SNRt=0.1, real = True):
    """
    Check whether the inputs dimensions match and the images are square
    """
    if np.shape(i1) != np.shape(i2):
        print("input images must have the same dimensions")
    if np.shape(i1)[0] != np.shape(i1)[1]:
        print("input images must be squares")
    I1 = fft.fftshift(fft.fft2(i1))
    I2 = fft.fftshift(fft.fft2(i2))
    """
    I1 and I2 store the DFT of the images to be used in the calcuation for the FSC
    """
    C = spinavej(np.multiply(I1, np.conj(I2)))
    C1 = spinavej(np.multiply(I1, np.conj(I1)))
    C2 = spinavej(np.multiply(I2, np.conj(I2)))

    if real:
        FSC = C.real / np.sqrt(abs(np.multiply(C1, C2)))
    else:
        FSC = abs(C) / np.sqrt(abs(np.multiply(C1, C2)))

    """
    T is the SNR threshold calculated accoring to the input SNRt, if nothing is given
    a default value of 0.1 is used.
    
    x2 contains the normalized spatial frequencies
    """
    r = np.arange(1 + np.shape(i1)[0] / 2)
    n = 2 * np.pi * r
    n[0] = 1
    eps = np.finfo(float).eps
    t1 = np.divide(np.ones(np.shape(n)), n + eps)
    t2 = SNRt + 2 * np.sqrt(SNRt) * t1 + np.divide(np.ones(np.shape(n)), np.sqrt(n))
    t3 = SNRt + 2 * np.sqrt(SNRt) * t1 + 1
    T = np.divide(t2, t3)
    x1 = np.arange(np.shape(C)[0]) / (np.shape(i1)[0] / 2)
    x2 = r / (np.shape(i1)[0] / 2)
    """
    If the disp input is set to 1, an output plot is generated. 
    """
    if disp != 0:
        print(f"len x1 {len(x1)}")
        plt.plot(x1, FSC, ".-", label="FSC")
        plt.plot(x2, T, "--", label="Threshold SNR = " + str(SNRt))
        plt.xlim(0, 1)
        plt.legend()
        plt.xlabel("Spatial Frequency/Nyquist")
        plt.show()

    return x1, FSC

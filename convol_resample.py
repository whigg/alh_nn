import numpy as np
from scipy.interpolate import interp1d

def convol_resample( x, y, x_out, fwhm=0.25, k=4, cg_f=10.):

    # x, y: orginal data
    # x_out: values at which to resample the convolved spectra
    # fwhm: full width half maximum of the convolution function
    # k: super Gaussian order
    # cg_f: factor to resample on a common grid

    # step 1: bring x and y on a common grid (cg)
    dx = np.mean( np.abs( x[1:] - x[0:-1] ) ) / cg_f
    x_cg = np.arange(x[0]+dx, x[-1], dx)
#    x_cg = np.arange(x[0]+dx, x[-1], dx) :original

    f_y_cg = interp1d( x, y)
    y_cg = f_y_cg(x_cg)
    
    # step 2: construct super gaussian (see http://www.atmos-meas-tech-discuss.net/amt-2016-307)
    x_cg_p = x_cg - x_cg[ np.int(x_cg.shape[0]/2) ]
    
    inv_w = ( 2. * np.power(np.log(2),1./k) ) / fwhm

    v = np.exp(-1 * np.power(x_cg_p*inv_w,k))
    v = v / np.sum(v)

    v_min = 1e-20
    idx = np.where( (v-v_min)>0 )

    x_cg_p = x_cg_p[idx]
    v = v[idx]
    
    # step 3: perform the convolution
    y_cg_convol = np.convolve(y_cg, v, mode='same')

    # step 4: interpolate to the output grid

    f_y_cg_convol = interp1d(x_cg, y_cg_convol)
    y_out = f_y_cg_convol(x_out)
    
    return [y_out, x_cg, y_cg, y_cg_convol, x_cg_p, v]


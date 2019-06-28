#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 29 08:38:07 2018

@author: nanda
"""

import scipy.interpolate
import numpy as np
import netCDF4 as nc4
import tables, time
from scipy.interpolate import interp1d
from scipy import interpolate
import sys

#%%

def spline( x_points, y_points, x):

    tck = interpolate.splrep(x_points, y_points)
    return interpolate.splev(x, tck)

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def convolve_isrf( WAVEL_START, WAVEL_END, pixel, 
                  to_convolve, wav, instr_wavl_grid, weights,
                  isrfData = '/nobackup_1/users/nanda/projects/alh_nn/scripts/alh_nn_code/S5P_OPER_AUX_SF_UVN_00000101T000000_99991231T235959_20180320T084215.nc', 
                  band = 'band_6'):
    
    # get tropomi slit function from file:
    with nc4.Dataset(isrfData, 'r') as f:
        
        node = f[band]
        
        idxStartWav = find_nearest(node['wavelength'][pixel, :],WAVEL_START-2)
        idxEndWav = find_nearest(node['wavelength'][pixel, :],WAVEL_END+2)
        
        centralWav          = node['wavelength'][pixel, idxStartWav:idxEndWav]
        deltaWavl           = node['delta_wavelength'][:]
        slit_function_vals  = node['isrf'][pixel,idxStartWav:idxEndWav,:]
    
    lambda_c_interpolant = scipy.interpolate.interp1d(centralWav, slit_function_vals, 
                                                      axis = 0,fill_value=0.0, 
                                                      bounds_error=False)
    lin_interp_instrGrid_isrf = lambda_c_interpolant(instr_wavl_grid)

    integrated = []
    wavl_final = []
    
    """ the following code needs to be severely optimised """
    
    for ix in range(len(instr_wavl_grid)):
        
        instrwavl_nominal = (instr_wavl_grid[ix, None] + deltaWavl).reshape(-1)
        isrf_lambda = lin_interp_instrGrid_isrf[ix, :]
        interp_isrf_lambda = spline(instrwavl_nominal, isrf_lambda, wav)
        
        integrated_value= np.dot(interp_isrf_lambda, to_convolve*weights)
        
        if integrated_value != 0:
            
            integrated.append(np.dot(interp_isrf_lambda, to_convolve*weights) )
            wavl_final.append(instr_wavl_grid[ix])
        
    
    integrated = np.array(integrated)
    wavl_final = np.array(wavl_final)
    
    return wavl_final, integrated
















# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 13:35:12 2015

@author: Swadhin Nanda, KNMI
"""

""" imports """
import numpy as np


""" Basic function/definition to find a value in an array """
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]


def extractsurfalbedo_lite(dataLat, dataLon, data_wavelengths, lerData, month,wavelengths,latitude,longitude, printing=True):
    """
    Example

    Supply 'typedata' as either: 'Mode_LER' or 'Minimum_LER'
    Supply 'wavelength' as wavelength for the surface albedo you are looking for. The program will find the closest wavelength to it in the data
    Supply 'latitude' and 'longitude' as the coordinates you are looking for. The program will find the closest coordinates to it in the data.
    Supply 'month' as the month you want the data for. eg. 12
    Supply 'acceptableflags' as the list of flags acceptable to you, eg. [0.,1.,2.]
    
    """    
    
    """ find index of the wavelength, latitude, longitude that is the closest to the user-specified input"""
    latindex, valat  = find_nearest(dataLat,latitude)
    lonindex, valon  = find_nearest(dataLon,longitude)
    
    if printing:
        print('surface albedo obtained at center coordinates lat/lon: {}/{}'.format(valat, valon))
    
    wavindices = []
    for wavelength in wavelengths:
        waveindex, val   = find_nearest(data_wavelengths,wavelength)
        wavindices.append(waveindex)
    
    surfalbedo = []
    for i in wavindices:
        surfalbedo.append(lerData[latindex,lonindex,i,month-1])

    return surfalbedo

def extractsurfalbedoUncertainty_lite(dataLat, dataLon, data_wavelengths, leruncertaintyData, month,wavelengths,latitude,longitude, printing=True):
    """
    Example

    Supply 'typedata' as either: 'Mode_LER' or 'Minimum_LER'
    Supply 'wavelength' as wavelength for the surface albedo you are looking for. The program will find the closest wavelength to it in the data
    Supply 'latitude' and 'longitude' as the coordinates you are looking for. The program will find the closest coordinates to it in the data.
    Supply 'month' as the month you want the data for. eg. 12
    Supply 'acceptableflags' as the list of flags acceptable to you, eg. [0.,1.,2.]
    
    """    
    
    """ find index of the wavelength, latitude, longitude that is the closest to the user-specified input"""
    latindex, valat  = find_nearest(dataLat,latitude)
    lonindex, valon  = find_nearest(dataLon,longitude)
    
    if printing:
        print('surface albedo obtained at center coordinates lat/lon: {}/{}'.format(valat, valon))
    
    wavindices = []
    for wavelength in wavelengths:
        waveindex, val   = find_nearest(data_wavelengths,wavelength)
        wavindices.append(waveindex)
    
    uncertainty = []
    for i in wavindices:
        uncertainty.append(leruncertaintyData[latindex,lonindex,i,month-1])

    return uncertainty




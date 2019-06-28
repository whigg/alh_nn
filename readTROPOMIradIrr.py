# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 11:25:23 2017

@author: nanda
"""

def readTROPOMIradIrrData_nc(f,g,scanline,pixel,band):
    
    """
    examples:
    
    """
    
    """ irradiance """
    
    
    wavelength = f['BAND{0}_IRRADIANCE'.format(int(band))]['STANDARD_MODE']['INSTRUMENT']['calibrated_wavelength'][0,int(pixel),:]
    irradiance = f['BAND{0}_IRRADIANCE'.format(int(band))]['STANDARD_MODE']['OBSERVATIONS']['irradiance'][0,0,int(pixel),:]
    signalToNoiseIrr= f['BAND{0}_IRRADIANCE'.format(int(band))]['STANDARD_MODE']['OBSERVATIONS']['irradiance_noise'][0,0,int(pixel),:]
    QL_irr = f['BAND{0}_IRRADIANCE'.format(int(band))]['STANDARD_MODE']['OBSERVATIONS']['quality_level'][0, 0,int(pixel),:]

    QL_irr = QL_irr[wavelength[:]<1e35]
    wvl_irr = wavelength[wavelength[:]<1e35]
    irr = irradiance[irradiance[:]<1e35] * 6.02214179*10E23
    snr_irr = 10**(signalToNoiseIrr[irradiance[:]<1e35]/10.)
    
    
    """ radiance """
    
    wavelength = g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['INSTRUMENT']['nominal_wavelength'][0,int(pixel),:]
    radiance = g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['OBSERVATIONS']['radiance'][0,int(scanline),int(pixel),:]
    signalToNoiseRad= g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['OBSERVATIONS']['radiance_noise'][0,int(scanline),int(pixel),:]
    QL_rad = g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['OBSERVATIONS']['quality_level'][0, int(scanline),int(pixel),:]

    latB = g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['GEODATA']['latitude_bounds'][0,int(scanline),int(pixel),:]
    lonB = g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['GEODATA']['longitude_bounds'][0,int(scanline),int(pixel),:]
    
    sza = g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['GEODATA']['solar_zenith_angle'][0,int(scanline),int(pixel)]
    saa = g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['GEODATA']['solar_azimuth_angle'][0,int(scanline),int(pixel)]
    vza = g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['GEODATA']['viewing_zenith_angle'][0,int(scanline),int(pixel)]
    vaa = g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['GEODATA']['viewing_azimuth_angle'][0,int(scanline),int(pixel)]

    lat = g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['GEODATA']['latitude'][0,int(scanline),int(pixel)]
    lon = g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['GEODATA']['longitude'][0,int(scanline),int(pixel)]
    
    dTime = g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['OBSERVATIONS']['time'][:] 
    dTime = dTime + g['BAND{0}_RADIANCE'.format(int(band))]['STANDARD_MODE']['OBSERVATIONS']['delta_time'][0,int(scanline)]/1000.

    geodata = [sza,vza,saa,vaa,lat,lon]
    
    QL_rad = QL_rad[wavelength[:]<1e35]
    wvl_rad = wavelength[wavelength[:]<1e35]
    rad = radiance[radiance[:]<1e35] * 6.02214179*10E23
    snr_rad = 10**(signalToNoiseRad[radiance[:]<1e35]/10.)
    
        
    return wvl_irr, irr, snr_irr, wvl_rad, rad, snr_rad, geodata, dTime, latB, lonB












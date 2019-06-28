#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:08:56 2017

@author: nanda
"""

import netCDF4
import numpy as np
import datetime, calendar, sys
from scipy.interpolate import interp1d
from scipy import spatial

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx,array[idx]



def to_Cartesian(LAT, LON):
    R = 6367 # radius of the Earth in kilometers
    
    LAT = np.deg2rad(LAT)
    LON = np.deg2rad(LON)
    
    x = R * np.cos(LAT) * np.cos(LON)
    y = R * np.cos(LAT) * np.sin(LON)
    z = R * np.sin(LAT)
    return x, y, z

def assemble_tree(filename, filename_surfacePressure):
    
    data = netCDF4.Dataset(filename)
    data_surfacePressure = netCDF4.Dataset(filename_surfacePressure)
      
    lat     = data.variables['latitudes'][:]
    lon     = data.variables['longitudes'][:]
    a       = data.variables['hyam'][:]
    b       = data.variables['hybm'][:]
    t       = data.variables['t'][:]
    
    
    #get reference time
    reftime_string = data.variables['time'].units.rsplit()[-2:]
    reftime_year = int(reftime_string[0].rsplit('-')[0])
    reftime_month = int(reftime_string[0].rsplit('-')[1])
    reftime_day = int(reftime_string[0].rsplit('-')[2])
    reftime_hour = int(reftime_string[1].rsplit(':')[0])
    reftime_min = int(reftime_string[1].rsplit(':')[1])
    reftime_sec = int(reftime_string[1].rsplit(':')[2])
    
    reftime = datetime.datetime(reftime_year, reftime_month, reftime_day,
                                reftime_hour,reftime_min,reftime_sec)
    
    time_data    = [reftime + datetime.timedelta(hours=int(i)) for i in data.variables['time'][:]]
    sp      = data_surfacePressure.variables['sp'][:]
    
    data.close()
    data_surfacePressure.close()
    
    meteo_coord_array = np.array([lat, lon]).T
    tree = spatial.KDTree(meteo_coord_array)
    
    return tree, a, b, t, sp, lat, lon, time_data

def calculate_levels(tree, inp, a, b, t, sp, lat, lon, time_data):
    
    longitude_converted = inp[1] % 360

    value = np.array([inp[0], longitude_converted])
    
    dist, index = tree.query(value)
    
    inpTime = datetime.datetime.utcfromtimestamp(inp[2])
    inpTime = inpTime.replace(year=time_data[0].year)
    time_data    = np.array([calendar.timegm(i.timetuple()) for i in time_data])
    timeToInterpolate = calendar.timegm(inpTime.timetuple())

    if timeToInterpolate > time_data.max():
        timeToInterpolate = time_data.max()
        print('meteo data is interpolated at time maxMeteoTime')
        print('pixel time = ', inpTime.hour,':',inpTime.minute,':',inpTime.second)
        print('latest available data = ', time_data.max()/3600)

    else:
        print('meteo data is interpolated at time {0}:{1}:{2}'.format(inpTime.hour,inpTime.minute,inpTime.second))    
    
    """
    Half pressure is calculated as P = a + b * SurfacePressure, i.e., the boundaries of the layers.
    source: https://rda.ucar.edu/datasets/ds115.4/docs/levels.hybrid.html
    
    """

    tempProf = np.vstack(t[:,:,index])
    nT =  interp1d(time_data,tempProf,axis=0)(timeToInterpolate)        
    
    nSP = sp[:,index]

    nSP_TIME = np.interp(timeToInterpolate,time_data,nSP)
    
    nP  = a + b * nSP_TIME
    
    print('latitude: ', lat[index], 'longitude: ', lon[index], 'surface pressure: ', nSP_TIME)

    
    return lat[index], lon[index], nP[::-1]/100, nT[::-1], nSP_TIME/100
    


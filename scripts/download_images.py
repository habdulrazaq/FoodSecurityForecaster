#!/usr/bin/env python
# coding: utf-8

import ee
import geemap
from ee import ImageCollection

import os
import numpy as np
import matplotlib.pyplot as plt


def download_map(county_name):
    ee.Authenticate()
    ee.Initialize()
    Map = geemap.Map()
    counties_shp = '../raw_data/gadm41_SSD_shp/gadm41_SSD_2.shp'
    ee_shape = geemap.shp_to_ee(counties_shp)
    ee_shape.getInfo()
    county_ee = ee_shape.filter(ee.Filter.eq("NAME_2", county_name))
    Map.addLayer(county_ee.geometry())

    start_date = '2022-04-20'
    end_date = '2022-05-04'
    MODIS_layer = 'MOD13A1'  #(high to low resolution: MOD13Q1 MOD13A1 MOD13A2)
    image_collection = ee.ImageCollection(f'MODIS/006/{MODIS_layer}').filter(ee.Filter.date(start_date,end_date)).select(['NDVI']).toBands()

    region = county_ee.geometry()

    sampled_pixels  = image_collection.sample(region, scale=300 , numPixels=5000)

    pixel_values = sampled_pixels.getInfo()['features'][3]['properties']['2022_04_23_NDVI']
    print(pixel_values)

test_name = 'Bahr al Jabal'

download_map(test_name)

#image_name = '2022_04_23_NDVI'
#pixel_value_2022_04_23_NDVI_list = []
#
#i=0
#for i in range(number_of_pixels - 1):
#    pixel_value = output.getInfo()['features'][i]['properties'][image_name]
#    pixel_value_2022_04_23_NDVI_list.append(pixel_value)
#
#pixel_value_2022_04_23_NDVI_list.shape()
#
#
## # Create Histogram
#
## In[ ]:
#
#
## histogram for NDVI layer for image of 2022-04-23
#plt.hist(pixel_value_2022_04_23_NDVI_list, bins=32)
#



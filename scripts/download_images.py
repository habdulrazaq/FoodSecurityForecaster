#!/usr/bin/env python
# coding: utf-8

import ee
from ee import ImageCollection
import geemap
import numpy as np
import pickle

def download_map(country_code, state_name, modis_collection='006/MOD13A1', num_pixels=5000):
    ee.Initialize()
    Map = geemap.Map()
    counties_shp = f'../raw_data/gadm41_{country_code}_shp/gadm41_{country_code}_2.shp'
    ee_shape = geemap.shp_to_ee(counties_shp)
    state_ee = ee_shape.filter(ee.Filter.eq("NAME_2", state_name))
    Map.addLayer(state_ee.geometry())

    start_date = '2021-01-01'
    end_date = '2022-12-30'
    image_collection = ee.ImageCollection(f'MODIS/{modis_collection}').filter(ee.Filter.date(start_date,end_date)).toBands()

    region = state_ee.geometry()

    sampled_pixels = [
        sample['properties'] for sample in image_collection.sample(region, scale=300 , numPixels=num_pixels).getInfo()['features']
    ]

    return {k[11:] : np.array([sample[k] for sample in sampled_pixels]) for k in sampled_pixels[0]}

test_name = 'Bahr al Jabal'
results = download_map('SSD', test_name, num_pixels=5000)
for k, v in results.items():
    print(k, v)

import matplotlib.pyplot as plt

for i, (k, v) in enumerate(results.items()):
    plt.subplot(6, 2, i + 1)
    plt.title(k)
    plt.hist(v, bins='auto')
plt.show()

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



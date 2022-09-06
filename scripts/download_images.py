#!/usr/bin/env python
# coding: utf-8

import ee
from ee import ImageCollection
import geemap
import numpy as np
import pickle

def download_map(country_code,
                 state_name,
                 date_range=('2021-01-01','2022-12-30'),
                 modis_collection='006/MOD13A1',
                 num_pixels=5000):
    Map = geemap.Map()
    states_shp = f'../raw_data/gadm41_{country_code}_shp/gadm41_{country_code}_2.shp'
    ee_shape = geemap.shp_to_ee(states_shp)
    state_ee = ee_shape.filter(ee.Filter.eq("NAME_2", state_name))
    Map.addLayer(state_ee.geometry())

    image_collection = ee.ImageCollection(f'MODIS/{modis_collection}').filter(ee.Filter.date(*date_range)).toBands()

    region = state_ee.geometry()

    sampled_pixels = [
        sample['properties'] for sample in image_collection.sample(region, scale=300 , numPixels=num_pixels).getInfo()['features']
    ]

    return {k[11:] : np.array([sample[k] for sample in sampled_pixels]) for k in sampled_pixels[0]}

def load_all(country_code='SSD', modis_collection='006/MOD13A1'):
    ee.Initialize()
    states_shp = f'../raw_data/gadm41_{country_code}_shp/gadm41_{country_code}_2.shp'
    ee_shape = geemap.shp_to_ee(states_shp)
    print(ee_shape)

load_all()

#test_name = 'Bahr al Jabal'
#results = download_map('SSD', test_name, num_pixels=5000)
#for k, v in results.items():
#    print(k, v)
#
#import matplotlib.pyplot as plt
#
#for i, (k, v) in enumerate(results.items()):
#    plt.subplot(6, 2, i + 1)
#    plt.title(k)
#    plt.hist(v, bins='auto')
#plt.show()

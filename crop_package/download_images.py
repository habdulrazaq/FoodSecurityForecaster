#!/usr/bin/env python
# coding: utf-8

import collections
from datetime import date, timedelta
import os
from socket import timeout
import sys

import ee
from ee import ImageCollection
import geemap
import numpy as np
import pandas as pd

def download_map(country_code,
                 state_name,
                 date_range=('2021-01-01','2022-12-30'),
                 modis_collection='006/MOD13A1',
                 num_pixels=1000):
    Map = geemap.Map()
    states_shp = f'../raw_data/gadm41_{country_code}_shp/gadm41_{country_code}_2.shp'
    ee_shape = geemap.shp_to_ee(states_shp)
    state_ee = ee_shape.filter(ee.Filter.eq("NAME_2", state_name))
    Map.addLayer(state_ee.geometry())

    image_collection = ee.ImageCollection(f'MODIS/{modis_collection}').filter(ee.Filter.date(*date_range)).toBands()

    region = state_ee.geometry()

    samples = image_collection.sample(region, scale=300 , numPixels=num_pixels)

    features = samples.getInfo()['features']

    data = []
    for feature_set in features:
        for k, v in feature_set['properties'].items():
            date, property_name = k[:10], k[11:]
            date = date.replace('_', '-')
            data.append({'band': property_name, 'date': date, 'value': v})

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
    df.attrs['state_name'] = state_name
    df['band'] = df['band'].astype("category")

    return df

def load_all(country_code='SSD', date_range=('2010-01-01', '2018-01-01'), modis_collection='006/MOD13A1', num_pixels=1000):
    states_shp = f'../raw_data/gadm41_{country_code}_shp/gadm41_{country_code}_2.shp'
    ee_shape = geemap.shp_to_ee(states_shp)
    geemap.common.ee_to_csv(ee_shape, "tmp.csv", timeout=1000)
    state_names = pd.read_csv('tmp.csv')['NAME_2']
    os.remove('tmp.csv')
    for state_name in state_names:
        print(f'working on:{state_name}')
        df = download_map(country_code, state_name, date_range, modis_collection, num_pixels)
        df.to_pickle(f'../raw_data/raw_pixels/{state_name}.zip')
        print(f"Downloaded data for {state_name}...")

if __name__ == "__main__":
    ee.Initialize()
    load_all(sys.argv[1])

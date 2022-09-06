#!/usr/bin/env python
# coding: utf-8

from datetime import date, timedelta
import os

import ee
from ee import ImageCollection
import geemap
import numpy as np
import pandas as pd

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

    return pd.DataFrame(sampled_pixels)

def iter_dates(start, stop, step):
    date = start
    while date < stop:
        yield date
        new_date = date + timedelta(days=step)
        if new_date.year == date.year:
            date = new_date
        else:
            #reset to 1st Jan. when a new year starts
            date = date(new_date.year, 1, 1)

def load_all(country_code='SSD', date_range=('2010-01-01', '2018-01-01'), modis_collection='006/MOD13A1', num_pixels=1000):
    if isinstance(date_range[0], str):
        date_range = tuple(map(date.fromisoformat, date_range))
    states_shp = f'../raw_data/gadm41_{country_code}_shp/gadm41_{country_code}_2.shp'
    ee_shape = geemap.shp_to_ee(states_shp)
    geemap.common.ee_to_csv(ee_shape, 'tmp.csv')
    state_names = pd.read_csv('tmp.csv')['NAME_2']
    os.remove('tmp.csv')
    for start_date in iter_dates(*date_range, 16):
        for state_name in state_names:
            end_date = start_date + timedelta(days=1)
            df = download_map(country_code, state_name, (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')), modis_collection, num_pixels)
            df.attrs['state_name'] = state_name
            df.attrs['date'] = start_date
            print(f"Downloaded data for {state_name} on {start_date}...")
            df.to_pickle(f'../raw_data/raw_pixels/{state_name}_{start_date}.zst')

if __name__ == "__main__":
    ee.Initialize()
    load_all()

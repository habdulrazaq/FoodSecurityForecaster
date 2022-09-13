#!/usr/bin/env python
# coding: utf-8

import collections
from curses import start_color
from datetime import date, timedelta
from http.client import CONTINUE
import os
from socket import timeout
import sys
from tracemalloc import start

import ee
from ee import ImageCollection
import geemap
import numpy as np
import pandas as pd

state_names = pd.read_csv('../raw_data/state_names_IND_working_states.csv')['NAME_1,C,22']

# MYD11A2 (2002-2022) 8-day Aqua Land Surface Temperature and Emissivity 1km
# MOD09A1   surface spectral reflectance of
# https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MCD12Q1

def download_map(country_code,
                 state_name,
                 date_range=('2002-01-01','2022-12-30'),
                 modis_collection='006/MOD11A2',
                 num_pixels=1000):
    Map = geemap.Map()
    lower_case_country_code = country_code.lower
    states_shp = f'../raw_data/IND/admin1/ind.shp'

    ee_shape = geemap.shp_to_ee(states_shp)
    state_ee = ee_shape.filter(ee.Filter.eq("NAME1_", state_name))


    Map.addLayer(state_ee.geometry())
       #  .select(['sur_refl_b01','sur_refl_b02', 'sur_refl_b03', 'sur_refl_b04' ,'sur_refl_b05', 'sur_refl_b06', 'sur_refl_b07'])\
    image_collection = ee.ImageCollection(f'MODIS/{modis_collection}') \
                         .filter(ee.Filter.date(*date_range)) \
                         .toBands()

    region = state_ee.geometry()

    samples = image_collection.sample(region, scale=250 , numPixels=num_pixels)

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


def load_all(country_code='SSD', date_range=('2000-01-01', '2015-01-01'), modis_collection='006/MOD11A2', num_pixels=1000):
    lower_case_country_code = country_code.lower
    states_shp = f'../raw_data/IND/admin1/ind.shp'
                  #working surface refl. layers: [8,9,10,13,15, 16, 17,18,20,22-25,27, 29-30, 32-36,38-43,46,48-49]
                  #working temperature refl. layers: [10 Georgia,16 kansas,25 missouri, 29 nebraska, 36 oklahoma, 41 south dakora, 42 tennessee,43 texas]
    for state_name in state_names[0:]:

        list_df = []
        year_lst = [2003,2004,] #2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2017,2018,2019,2020,2021
        for start_year in range(2003,2013):
            print(f"working on {start_year}")

            print(f'working on {state_name}...')

            df = download_map(country_code, state_name, (f'{start_year}-01-01', f'{start_year+1}-01-01'), modis_collection, num_pixels)
            list_df.append(df)

        exit(0)

        df = pd.concat(list_df)
        df.to_pickle(f'../raw_data/raw_pixels/INDIA/temperature_1000p_scale250_2003-2012/{state_name}.zip')
        print(f"Downloaded data for {state_name}...")


if __name__ == "__main__":
    print(pd.__version__)
    ee.Initialize()
    load_all(sys.argv[1], num_pixels=1000)

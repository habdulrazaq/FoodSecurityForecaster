#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import sys

def build_histograms(country_code='SSD', num_bins=30):

    data_path = f'../raw_data/raw_pixels/USA/h'

    df_list = []

    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        print(file_path)
        df = pd.read_pickle(file_path)
        df_list.append(df)

    df_list = sorted(df_list, key = lambda df: df.attrs['state_name'])

    stacked_df = pd.concat(df_list)
    band_groups = stacked_df.groupby('band')
    q = np.linspace(0, 1, num_bins + 1)
    band_quantiles = [np.quantile(band_group['value'], q) for band, band_group in band_groups]

    for bins in band_quantiles:
        bins[0] = -float('inf')
        bins[-1] = float('inf')

    years = stacked_df['date'].dt.year.unique()

    num_bands = stacked_df['band'].nunique()
    num_years = len(years)
    num_counties = len(df_list)
    num_samples = 46

    X = np.zeros((num_years, num_counties, num_samples, num_bins, num_bands))

    for county_index, df in enumerate(df_list):
        for year_index, (year, year_group) in enumerate(df.groupby(df['date'].dt.year)):
            for sample_index, (sample_date, sample_group) in enumerate(year_group.groupby('date')):
                for band_index, (band_name, band_group) in enumerate(sample_group.groupby('band')):
                    hist = np.histogram(band_group['value'], bins=band_quantiles[band_index])[0]
                    X[year_index,county_index,sample_index,:,band_index] = hist

    county_names = np.array([df.attrs['state_name'] for df in df_list])
    np.savez_compressed(f'../data/USA_other_states_data_MOD09A1_h.npz', X=X, county_names=county_names)

if __name__ == "__main__":
    build_histograms(sys.argv[1])

#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os
import sys

def build_histograms(country_code='IND', num_bins=30):

    data_path = f'../raw_data/raw_pixels/USA/2000-2022_CORN'

    df_list = []

    for file in os.listdir(data_path):
        file_path = os.path.join(data_path, file)
        print(file_path)
        df = pd.read_pickle(file_path)
        df_list.append(df)
        break

    df_list = sorted(df_list, key = lambda df: df.attrs['state_name'])

    stacked_df = pd.concat(df_list)
    band_groups = stacked_df.groupby('band')
    q = np.linspace(0, 1, num_bins + 1)
    band_quantiles = [np.quantile(band_group['value'], q) for band, band_group in band_groups]

    for bins in band_quantiles:
        bins[0] = -float('inf')
        bins[-1] = float('inf')

    years = stacked_df['date'].dt.year.drop_duplicates()

    num_bands = stacked_df['band'].nunique()
    num_years = years.nunique()
    num_counties = len(df_list)
    num_samples = stacked_df['date'].groupby(stacked_df['date'].dt.year).nunique().max()

    X = np.zeros((num_years, num_counties, num_samples, num_bins, num_bands))

    for county_index, df in enumerate(df_list):
        for year_index, (year, year_group) in enumerate(df.groupby(df['date'].dt.year)):
            print(year, year_index)
            for sample_index, (sample_date, sample_group) in enumerate(year_group.groupby('date')):
                for band_index, (band_name, band_group) in enumerate(sample_group.groupby('band')):
                    hist = np.histogram(band_group['value'], bins=band_quantiles[band_index])[0]
                    X[year_index,county_index,sample_index,:,band_index] = hist

    debug = True
    if debug:
        import matplotlib.pyplot as plt
        for i in range(X.shape[0]):
            plt.subplot(5, 5, i + 1)
            plt.imshow(np.log(X[i,0,:,:,0] + 0.01))
        plt.show()

    first_samples = np.sum(np.sum(X[0,0,:,0], axis=-1) != 0)
    last_samples  = np.sum(np.sum(X[-1,0,:,0], axis=-1) != 0)

    if first_samples != num_samples or last_samples != num_samples:
        first_year = np.copy(X[0,:,:first_samples])
        X[0] = 0
        X[0,:,num_samples - first_samples:] = first_year
        X = np.concatenate((X[:-1,:,last_samples:], X[1:,:,:last_samples]), axis=2)

    if debug:
        import matplotlib.pyplot as plt
        for i in range(X.shape[0]):
            plt.subplot(5, 5, i + 1)
            plt.imshow(np.log(X[i,0,:,:,0] + 0.01))
        plt.show()
        exit(0)

    county_names = np.array([df.attrs['state_name'] for df in df_list])
<<<<<<< Updated upstream:crop_package/pixels_to_histograms_USA_INDIA.py
<<<<<<< HEAD:crop_package/pixels_to_histograms_USA_INDIA.py
    np.savez_compressed(f'../data/USA_19_soybeans_states_data_2013-2022_MOD09A1.npz', X=X, county_names=county_names)
=======
    np.savez_compressed('../data/INDIA_MOD09A1.npz', X=X, county_names=county_names, years=years.to_numpy())
>>>>>>> 3f0734057fca272d5ba8a59b6361ec90a31c5b59:crop_package/pixels_to_histograms_india.py
=======
    np.savez_compressed('../data/INDIA_WHEAT_states_data_MOD09A1.npz', X=X, county_names=county_names, years=years.to_numpy())
>>>>>>> Stashed changes:crop_package/pixels_to_histograms_india.py

if __name__ == "__main__":
    build_histograms(sys.argv[1])

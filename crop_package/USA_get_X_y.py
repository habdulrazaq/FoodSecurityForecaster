import pandas as pd
import numpy as np


def get_X_y(country_code='USA'):

    # 0. import df
    X_data = np.load(f'../data/kansas_MOD09A1_band5to12_30bins.npz')
    X = X_data['X'][:-1, :, :, :, :]

    df_y = pd.read_csv(
        f'../raw_data/Crop yield data/COUNTY_level_annual/soybeans_usa.csv')

    df_y['STATE'] = df_y['STATE'].str.lower()

    state_names = [name.lower() for name in X_data['county_names']]

    X_counties = set(state_names)
    y_counties = set(df_y['STATE'])

    X_in_y = np.array([name in y_counties for name in state_names])

    X = X[:, X_in_y]

    year_groups = df_y[df_y['STATE'].apply(lambda s: s in X_counties)] \
                    [df_y['YEAR'] != 2011] \
                      .sort_values('STATE') \
                      .groupby('YEAR')

    y = np.zeros(X.shape[:2])
    for i, (year, group) in enumerate(year_groups):
        print(group)
        y[i] = group['YIELD']

    return X, y


def split_years(*arrays, test_size=1):
    return sum(((array[:-test_size], array[-test_size:]) for array in arrays),
               ())

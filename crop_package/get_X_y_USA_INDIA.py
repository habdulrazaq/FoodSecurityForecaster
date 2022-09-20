import pandas as pd
import numpy as np
import tensorflow as tf

def get_X_y(country_code='USA'):

    # 0. import df
    X_data = np.load(f'../data/USA_30states_data_MOD09A1.npz')
    X = X_data['X']

    if country_code == 'USA':
        year_idx = np.arange(X.shape[0])
        X = X[(year_idx != 15) & (year_idx != 16)]

    print(X.shape)

    df_y = pd.read_csv(f'../raw_data/Crop yield data/COUNTY_level_annual/india_SUGARCANE.csv')

    df_y['STATE'] = df_y['STATE'].str.lower()

    state_names = [name.lower() for name in X_data['county_names']]

    X_counties = set(state_names)
    y_counties = set(df_y['STATE'])

    X_in_y = np.array([name in y_counties for name in state_names])

    if False:
        for name in X_data['county_names']:
            if name.lower() in y_counties:
                print(name)
        exit(0)

    X = X[:, X_in_y]

    if country_code == 'USA':
        df_y = df_y[(df_y['YEAR'] != 2016) & (df_y['YEAR'] != 2017)]

    year_groups = df_y[df_y['STATE'].apply(lambda s: s in X_counties)] \
                      .sort_values('STATE') \
                      .groupby('YEAR')

    y = np.zeros(X.shape[:2])  #### change to remove 0 y va2022ls
    for i, (year, group) in enumerate(year_groups):

        y[i] = group['YIELD']

    return X, y


def split_years(*arrays, test_size=1):
    return sum(((array[:-test_size], array[-test_size:]) for array in arrays),
               ())

if __name__ == "__main__":
    get_X_y()

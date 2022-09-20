import pandas as pd
import numpy as np
import tensorflow as tf

def get_X_y(X_only=False, max_masked=0.5):
    """
    Args:
        X_only: Whether to only return X
        max_masked: maximum proportion of samples to mask from the end of the year
    """

<<<<<<< Updated upstream:crop_package/get_X_y_USA_INDIA.py
<<<<<<< HEAD:crop_package/get_X_y_USA_INDIA.py
    # 0. import df
    X_data = np.load(f'../data/USA_30states_data_MOD09A1.npz')
    X = X_data['X']

    if country_code == 'USA':
        year_idx = np.arange(X.shape[0])
        X = X[(year_idx != 15) & (year_idx != 16)]

    print(X.shape)
=======
    X_data = np.load('../data/INDIA_MOD09A1.npz')
=======
    X_data = np.load('../data/INDIA_RICE_states_data_MOD09A1.npz')
>>>>>>> Stashed changes:crop_package/USA_get_X_y.py
    X = X_data['X']

    #(num_years,num_counties,num_samples,num_bins,num_bands)
    print("X shape:", X.shape)
>>>>>>> 3f0734057fca272d5ba8a59b6361ec90a31c5b59:crop_package/USA_get_X_y.py

    if X_only:
        info = {
            'years': X_data['years'],
            'county_names': X_data['county_names'],
        }
        return X, info

    df_y = pd.read_csv('../raw_data/Crop yield data/COUNTY_level_annual/india_RICE.csv')

    df_y['STATE'] = df_y['STATE'].str.lower()

    state_names = [name.lower() for name in X_data['county_names']]

    X_counties = set(state_names)
    X_years = set(X_data['years'])

    y_counties = set([name.lower() for name in df_y['STATE']])
    y_years = set(df_y['YEAR'])

    X_county_in_y = np.array([name in y_counties for name in state_names])
    X_year_in_y = np.array([year in y_years for year in X_data['years']])

<<<<<<< HEAD:crop_package/get_X_y_USA_INDIA.py
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
=======
    X = X[X_year_in_y]

    X = X[:,X_county_in_y]

    df_y = df_y[df_y['STATE'].apply(lambda s: s in X_counties) & df_y['YEAR'].apply(lambda s: s in X_years)]

    year_groups = df_y.sort_values('STATE') \
                      .groupby('YEAR')

    y = np.zeros(X.shape[:2] + (1,))
    for i, (year, group) in enumerate(year_groups):
        y[i,:,0] = group['YIELD']

    info = {
        'years': [year for year in X_data['years'] if year in y_years],
        'county_names': [name for name in X_data['county_names'] if name.lower() in y_counties],
    }
>>>>>>> 3f0734057fca272d5ba8a59b6361ec90a31c5b59:crop_package/USA_get_X_y.py

    if max_masked:
        xs = []
        ys = []
        for i in range(int(X.shape[2] * max_masked)):
            X_i = np.copy(X)
            if i != 0:
                X_i[:,:,-i:] = 0
            xs.append(X_i)
            ys.append(y)
        X = np.stack(xs, axis=2)
        y = np.stack(ys, axis=2)

    return X, y, info

def split_years(*arrays, test_size=1):
<<<<<<< HEAD:crop_package/get_X_y_USA_INDIA.py
    return sum(((array[:-test_size], array[-test_size:]) for array in arrays),
               ())

if __name__ == "__main__":
    get_X_y()
=======
    return sum(((array[:-test_size], array[-test_size:]) for array in arrays), ())
>>>>>>> 3f0734057fca272d5ba8a59b6361ec90a31c5b59:crop_package/USA_get_X_y.py

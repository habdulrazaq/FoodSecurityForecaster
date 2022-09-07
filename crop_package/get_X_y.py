import pandas as pd
import numpy as np

def get_X_y(country_code='SSD'):

    # 0. import df
    X_data = np.load(f'../data/{country_code}_data.npz')
    X = X_data['X']

    df_y = pd.read_csv(f'../raw_data/Crop yield data/COUNTY_level_annual/{country_code} processed.csv')

    X_counties = set(X_data['county_names'])
    y_counties = set(df_y['County'])

    X_in_y = np.array([name in y_counties for name in X_data['county_names']])

    X = X[:,X_in_y]

    year_groups = df_y[df_y['County'].apply(lambda s: s in X_counties)] \
                      .sort_values('County') \
                      .groupby('Year')

    y = np.zeros(X.shape[:2])

    for i, (year, group) in enumerate(year_groups):
        y[i] = group['Yield']

    return X, y

def split_years(*arrays, test_size=1):
    return sum(((array[:-test_size], array[-test_size:]) for array in arrays), ())

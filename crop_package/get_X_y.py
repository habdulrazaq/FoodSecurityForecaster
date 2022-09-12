import pandas as pd
import numpy as np

def get_X_y(country_code='USA'):

    X_data = np.load(f'../data/{country_code}_data.npz')
    X = X_data['X']

    df_y = pd.read_csv(f'../raw_data/Crop yield data/COUNTY_level_annual/{country_code} processed.csv')

    X_years = set(X_data['years'])
    X_counties = set(X_data['county_names'])

    y_years = set(y_data['years'])
    y_counties = set(df_y['County'])

    X_county_in_y = np.array([name in y_counties for name in X_data['county_names']])
    X_year_in_y = np.array([name in y_years for name in X_data['years']])

    X = X[X_year_in_y,X_county_in_y]

    year_groups = df_y[df_y['County'].isin(X_counties) & df_y['Year'].isin(X_years)] \
                      .sort_values('County') \
                      .groupby('Year')

    y = np.zeros(X.shape[:2])
    for i, (year, group) in enumerate(year_groups):
        y[i] = group['Yield']

    return X, y

def split_years(*arrays, test_size=1):
    return sum(((array[:-test_size], array[-test_size:]) for array in arrays), ())

from webbrowser import get
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from histograms import hi

def get_X_y(country_code='SSD', county='Juba'):

    # 0. import df
    df_X = df_X_test
    df_y = pd.read_csv(f'../raw_data/Crop yield data/COUNTY_level_annual/{country_code}.csv')
    df_combined = df_y.merge(df_X, on='Year')

    # 1. cross-validate
    tscv = TimeSeriesSplit(n_splits=2, max_train_size=None, test_size=2)
    for (train_idx, test_idx) in tscv.split(df_combined):
        X_y_train, X_y_test = df_combined[train_idx], df_combined[test_idx]

    # 2. get X and y for each county for each country
    X_train = X_y_train[X_y_train['County'].str.strip() == f'{county}']['tensors']
    X_test = X_y_test[X_y_test['County'].str.strip() == f'{county}']['tensors']

    y_train = X_y_train[X_y_train['County'].str.strip() == f'{county}']['Yield']
    y_test = X_y_test[X_y_test['County'].str.strip() == f'{county}']['Yield']
    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    return X_train, X_test, y_train, y_test

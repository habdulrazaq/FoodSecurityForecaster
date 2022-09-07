from webbrowser import get
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def get_X_y(country_code='SSD', county='Juba'):

    # 0. import df
    X_data = np.load(f'../data/{country_code}_data.npz')
    X = X_data['X']

    y = np.zeros(X.shape[:2])
    df_y = pd.read_csv(f'../raw_data/Crop yield data/COUNTY_level_annual/{country_code}.csv')
    print([name for name in X_data['county_names'] if name in df_y['County'].str.strip()])
    exit(0)
    df_y = df_y.sort_values('County').sort_values('Year', kind='stable')
    print(list(df_y['County'].drop_duplicates()))
    for i, (year, year_group) in enumerate(df_y.groupby('Year')):
        print(year_group['County'].nunique())
        #y[i] = year_group['Yield']
    exit(0)
    import matplotlib.pyplot as plt
    plt.imshow(y)
    plt.show()
    exit(0)
    df_combined = df_y.merge(X, on='Year')

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



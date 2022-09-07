from webbrowser import get
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

#################################################
# DELETE X.shape() = 23, 30, 1) #bins=30 #DNVI layer
X_2010 = np.random.uniform(0, 9000, (23, 30))
X_2011 = np.random.uniform(0, 9000, (23, 30))
X_2012 = np.random.uniform(0, 9000, (23, 30))
X_2013 = np.random.uniform(0, 9000, (23, 30))
X_2014 = np.random.uniform(0, 9000, (23, 30))
X_2015 = np.random.uniform(0, 9000, (23, 30))
X_2016 = np.random.uniform(0, 9000, (23, 30))
X_2017 = np.random.uniform(0, 9000, (23, 30))
df_X_test = np.vstack((X_2010, X_2011, X_2012, X_2013, X_2014, X_2015, X_2016, X_2017))
df_X_test = pd.DataFrame(df_X_test)
df_X_test['Year'] = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]
print(df_X_test)
##################################################

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

get_X_y()

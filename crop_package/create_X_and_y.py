import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


#from crop_package.CNN_LSTM_Draft import CNN_LSTM

#synthetic pixel dataframe for Juba
# hist_2022_5bins = pd.read_csv('../raw_data/juba_example_histogram.csv')
# df = pd.DataFrame(hist_2022_5bins)
# print(df)

# 1. import y
df = pd.read_csv('../raw_data/Crop yield data/COUNTY_level_annual/south_sudan_2010_2017.csv')
print(df)

# 2. y for each county all years' data
y_Juba_2010 = df[df['County'].str.strip() == 'Juba']['Yield']
#print(y_Juba_2010)
y_Juba_2010.shape


# 3. split y
y_Juba_2010 = y_Juba_2010.to_numpy()
tscv = TimeSeriesSplit(n_splits=2, max_train_size=None, test_size=2)


for (train_idx, test_idx) in tscv.split(y_Juba_2010):
    y_train, y_test = y_Juba_2010[train_idx], y_Juba_2010[test_idx]
    print(y_train, y_test)


# 4. import feature    X.shape() = 23, 30, 1)
X_2010 = np.random.uniform(0, 9000, (23, 30))
X_2011 = np.random.uniform(0, 9000, (23, 30))
X_2012 = np.random.uniform(0, 9000, (23, 30))
X_2013 = np.random.uniform(0, 9000, (23, 30))
X_2014 = np.random.uniform(0, 9000, (23, 30))
X_2015 = np.random.uniform(0, 9000, (23, 30))
X_2016 = np.random.uniform(0, 9000, (23, 30))
X_2017 = np.random.uniform(0, 9000, (23, 30))
print(X_2010)

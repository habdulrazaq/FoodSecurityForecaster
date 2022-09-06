import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


#from crop_package.CNN_LSTM_Draft import CNN_LSTM

#synthetic pixel dataframe for Juba
# hist_2022_5bins = pd.read_csv('../raw_data/juba_example_histogram.csv')
# df = pd.DataFrame(hist_2022_5bins)
# print(df)

# 1. import y
yield_data_SSD = pd.read_csv('../raw_data/Crop yield data/COUNTY_level_annual/south_sudan_2010_2017.csv')
y = pd.DataFrame(yield_data_SSD)['Yield']
print(y)

# 2. split data using timeseries apprach
tscv = TimeSeriesSplit(n_splits=5, max_train_size=None, test_size=0.3)
tscv.split(y)

for train_index, test_index in tscv.split(y):
    print("TRAIN:", train_index, "TEST:", test_index)

# 3. import feature/X (23 images/columns and 32 rows/bins)
X = np.random.uniform(0, 9000, (23, 32))

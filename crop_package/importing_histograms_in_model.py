import pandas as pd

from crop_package.CNN_LSTM_Draft import CNN_LSTM

hist_2022_5bins = pd.read_csv('../raw_data/juba_example_histogram.csv')

df = pd.DataFrame(hist_2022_5bins)
print(df)

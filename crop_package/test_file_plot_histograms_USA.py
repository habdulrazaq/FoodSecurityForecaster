import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

penny_data = pd.read_pickle('../raw_data/raw_pixels/USA/Arizona.zip')

penny_df = penny_data[penny_data['band'] == 'NDVI']

value_min, value_max = penny_df['value'].min(), penny_df['value'].max()

histograms = []
dates = []
for date, group in penny_df.groupby('date'):
    hist = np.histogram(group['value'], bins=10, range=(0,9000))[0]
    histograms.append(hist)
    dates.append(date)

penny_histogram_df = pd.DataFrame(histograms, index=dates)

print(penny_histogram_df)


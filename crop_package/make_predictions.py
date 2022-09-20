import sys

import numpy as np
import pandas as pd
from tensorflow import keras

import get_X_y_USA_INDIA

def make_predictions(model_path, out_path):
    X, y = get_X_y_USA_INDIA.get_X_y()
    X_flat = X.reshape((-1,) + X.shape[2:])
    model = keras.models.load_model(model_path)
    y_pred_flat = model.predict(X_flat).flatten()

    years = [year for year in range(2001, 2023) if year not in (2016, 2017)]
    county_names = [name.strip() for name in open('../data/soybean_states.txt')]
    num_years = len(years)
    num_counties = len(county_names)
    years = [year for year in years for _ in range(num_counties)]
    county_names = [county for _ in range(num_years) for county in county_names]
    print(len(y_pred_flat))
    print(len(years))
    print(len(county_names))
    df = pd.DataFrame({
        'years': years,
        'county_names': county_names,
        'predictions': y_pred_flat
    })
    df.to_pickle(out_path)

if __name__ == "__main__":
    make_predictions(sys.argv[1], sys.argv[2])

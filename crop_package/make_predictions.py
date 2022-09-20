import sys

import numpy as np
import pandas as pd
from tensorflow import keras

import USA_get_X_y

def make_predictions(model_path, out_path):
<<<<<<< HEAD
    X, y = USA_get_X_y.get_X_y()
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
=======
    X, info = USA_get_X_y.get_X_y(X_only=True)
    X_flat = X.reshape((-1,) + X.shape[2:])
    model = keras.models.load_model(model_path)
    y_pred_flat = model.predict(X_flat).flatten()
    county_names = np.repeat(info['county_names'][None,:], X.shape[0], axis=0).flatten()
    years = np.repeat(info['years'][:,None], X.shape[1], axis=1).flatten()
>>>>>>> 3f0734057fca272d5ba8a59b6361ec90a31c5b59
    df = pd.DataFrame({
        'years': years,
        'county_names': county_names,
        'predictions': y_pred_flat
    })
    df.to_pickle(out_path)

if __name__ == "__main__":
    make_predictions(sys.argv[1], sys.argv[2])

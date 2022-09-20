import sys

import numpy as np
import pandas as pd
from tensorflow import keras

import USA_get_X_y

def make_predictions(model_path, out_path):
    X, info = USA_get_X_y.get_X_y(X_only=True)
    X_flat = X.reshape((-1,) + X.shape[2:])
    model = keras.models.load_model(model_path)
    y_pred_flat = model.predict(X_flat).flatten()
    county_names = np.repeat(info['county_names'][None,:], X.shape[0], axis=0).flatten()
    years = np.repeat(info['years'][:,None], X.shape[1], axis=1).flatten()
    df = pd.DataFrame({
        'years': years,
        'county_names': county_names,
        'predictions': y_pred_flat
    })
    df.to_pickle(out_path)

if __name__ == "__main__":
    make_predictions(sys.argv[1], sys.argv[2])

import numpy as np
from sklearn.dummy import DummyRegressor

class BaselineModel(DummyRegressor):
    def fit(self, X, y):
        #X shape = (years, counties, datapoints, bins, bands)
        #y shape = (years, counties)
        self.baseline_value = y[-1]
    def predict(self, X):
        return np.repeat(self.baseline_value[None,:], axis=0, repeats=X.shape[0])

import USA_fit_model
from sklearn.model_selection import RandomizedSearchCV
from scipy import stats


def fit_model_RandomizedSearchCV():

    model = USA_fit_model()
    grid = {''}
    search = RandomizedSearchCV(model,
                                grid,
                                scoring=

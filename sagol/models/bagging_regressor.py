import numpy as np
from sklearn.ensemble import BaggingRegressor


def train_bagging_regressor(x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> BaggingRegressor:
    # Parallelize training on all CPUs.
    mdl = BaggingRegressor(**kwargs, n_jobs=-1)
    mdl.fit(x_train, y_train)
    return mdl

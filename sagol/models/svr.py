import numpy as np

from sklearn.svm import SVR


def train_svr(x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> SVR:
    mdl = SVR(**kwargs)
    mdl.fit(x_train, y_train)
    return mdl

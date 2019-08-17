import numpy as np

from sklearn.svm import SVR


def train_svr(x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> SVR:
    print("train size is now" ,len(x_train))
    print("first data is ", x_train[0])
    print("first label is ", y_train[0])
    mdl = SVR(**kwargs)
    mdl.fit(x_train, y_train)
    return mdl

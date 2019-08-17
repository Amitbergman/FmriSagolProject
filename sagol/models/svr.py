import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def train_svr(x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> SVR:
    print("train size is now" ,len(x_train))
    print("first data is ", x_train[0])
    print("first label is ", y_train[0])
    mdl = SVR(**kwargs)
    mdl.fit(x_train, y_train)
    return mdl

def run_svr_grid_search(data):
    # TODO: get parameters for the grid search, the ones here are copied from:
    #https://scikit-learn.org/stable/auto_examples/plot_kernel_ridge_regression.html#sphx-glr-auto-examples-plot-kernel-ridge-regression-py
    #TODO: understand how the gridSearch uses test set (currently it does not, I guess it does k fold cross validation)
    x_train, x_test, y_train, y_test = generate_X_and_Y_from_data_and_feature(data, 'FPES')
    print('number of data points in train set is ', len(x_train))
    print('number of features is ', len(x_train[0]))
    svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=5,
                    param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                                "gamma": np.logspace(-2, 2, 5)})
    svr.fit(x_train, y_train)
    # TODO: understand the meaning of the score when we talk about regression and not classification
    print("best estimator is",svr.best_estimator_)
    print("the best score is", svr.best_score_)
    return svr.best_estimator_


    
#exapmle: data = create_subject_experiment_data
#and then: X,Y = generate_X_and_Y_from_data_and_feature(data, 'FPES')
# it takes all the images in the data and makes X of them and the Y is the correspondind label based on the feature
def generate_X_and_Y_from_data_and_feature(data, feature):
    X = []
    Y = []
    for i in range (len(data.subjects_data)):
        subject_y = data.subjects_data[i].features_data[feature]
        for task in data.subjects_data[i].tasks_data.values():
            X.append(task)
            Y.append(subject_y)
    return train_test_split(X, Y)

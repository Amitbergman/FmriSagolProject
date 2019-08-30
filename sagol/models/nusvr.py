from sklearn.svm import NuSVR
import numpy as np
import logbook
from sklearn.model_selection import GridSearchCV
from sagol.models.utils import get_model_params

logger = logbook.Logger(__name__)


def train_nusvr(x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> (NuSVR, dict):
    mdl = NuSVR(**kwargs)
    mdl.degree = 2

    should_grid_search = False
    param_grid = {}

    if 'C' not in kwargs:
        should_grid_search = True
        param_grid['C'] = [0.1, 1, 10, 100, 250]
    if 'gamma' not in kwargs:
        should_grid_search = True
        param_grid['gamma'] = [1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.25]
    if 'kernel' not in kwargs:
        should_grid_search = True
        param_grid['kernel'] = ['rbf', 'linear', 'poly']

    if should_grid_search:
        param_grid['degree'] = [2]

        logger.info(f'Performing grid search. Using param_grid: {param_grid} '
                    f'This may take a while...')
        gs = GridSearchCV(estimator=mdl, param_grid=param_grid)
        gs.fit(x_train, y_train)
        mdl = gs.best_estimator_
    mdl.fit(x_train, y_train)
    return mdl, get_model_params(model_name='nusvr', model=mdl)

from sklearn.svm import NuSVR
import numpy as np
import logbook

from sklearn.model_selection import GridSearchCV

logger = logbook.Logger(__name__)


def train_nusvr(x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> NuSVR:
    mdl = NuSVR(gamma='scale', kernel='rbf' )

    should_grid_search = False
    param_grid = {}

    if 'C' not in kwargs:
        should_grid_search = True
        param_grid['C'] = [0.1, 1, 10, 100, 250]
    if 'kernel' not in kwargs:
        should_grid_search = True
        param_grid['kernel'] = ['rbf', 'linear', 'poly']
    if should_grid_search:
        param_grid['degree'] = [2]
        logger.info(f'Performing grid search. Using param_grid: {param_grid} '
                    f'This may take a while...')
        gs = GridSearchCV(estimator=mdl, param_grid=param_grid, verbose=12)
        gs.fit(x_train, y_train)
        return gs.best_estimator_
    else:
        mdl.fit(x_train, y_train)
        return mdl

import logbook
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV

logger = logbook.Logger(__name__)


def train_bagging_regressor(x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> BaggingRegressor:
    # Parallelize training on all CPUs.
    mdl = BaggingRegressor(n_jobs=-1, verbose=True, **kwargs)

    if 'n_estimators' in kwargs:
        mdl.fit(x_train, y_train)
        return mdl
    else:
        param_grid = {
            'n_estimators': [5, 10, 20, 50, 100],
        }
        logger.info(
            f'n_estimators for `bagging_regressor` not passed, performing grid search. Using param_grid: {param_grid} '
            f'This may take a while...')
        gs = GridSearchCV(estimator=mdl, param_grid=param_grid)
        gs.fit(x_train, y_train)
        return gs.best_estimator_

import logbook
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
from sagol.models.utils import get_model_params

logger = logbook.Logger(__name__)


def train_bagging_regressor(x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> (BaggingRegressor, dict):
    # Parallelize training on all CPUs.
    mdl = BaggingRegressor(n_jobs=-1, **kwargs)

    if 'n_estimators' not in kwargs:
        param_grid = {
            'n_estimators': [5, 10, 20, 50, 100],
        }
        logger.info(
            f'n_estimators for `bagging_regressor` not passed, performing grid search. Using param_grid: {param_grid} '
            f'This may take a while...')
        gs = GridSearchCV(estimator=mdl, param_grid=param_grid)
        gs.fit(x_train, y_train)
        mdl = gs.best_estimator_

    mdl.fit(x_train, y_train)
    return mdl, get_model_params(model_name='bagging_regressor', model=mdl)

import logbook
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.utils.testing import ignore_warnings

from sagol.models.utils import get_model_params

logger = logbook.Logger(__name__)


@ignore_warnings(category=ConvergenceWarning)
def train_lasso(x_train: np.ndarray, y_train: np.ndarray, **kwargs) -> (Lasso, dict):
    mdl = Lasso(**kwargs)

    should_grid_search = False
    param_grid = {}

    if 'alpha' not in kwargs:
        should_grid_search = True
        param_grid['alpha'] = np.linspace(0.03, 0.015)

    if should_grid_search:
        logger.info(f'Performing grid search. Using param_grid: {param_grid} '
                    f'This may take a while...')

        gs = GridSearchCV(estimator=mdl, param_grid=param_grid)
        gs.fit(x_train, y_train)
        mdl = gs.best_estimator_

    mdl.fit(x_train, y_train)
    return mdl, get_model_params(model_name='lasso', model=mdl)


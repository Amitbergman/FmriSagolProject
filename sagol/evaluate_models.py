from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from attr import attrs, attrib
from matplotlib.figure import Figure


@attrs
class Models:
    ylabels: List[str] = attrib()
    roi_paths: Optional[List[str]] = attrib()
    # (x, y, z)
    shape: tuple = attrib()
    # {'svr' : <model>, 'bagging_regressor': <model>}
    models: dict = attrib(default={})
    # {'svr' : 0.52, 'bagging_regressor': 0.48}
    scores: dict = attrib(default={})
    # {'svr' : <graph>, 'bagging_regressor': <graph>}
    residual_plots: dict = attrib(default={})


def create_residual_plot(model, model_name: str, x_test: np.array, y_test: np.array) -> Figure:
    fig = plt.figure()
    plt.xlabel('True', figure=fig)
    plt.ylabel('Predicted', figure=fig)
    plt.title(f'Model: {model_name} - True VS Predicted, optimal is aligned with red line', figure=fig)

    y_predict = model.predict(x_test)
    # Add data points.
    plt.scatter(y_test, y_predict, color='black', figure=fig)
    # Add optimal line - a perfect prediction.
    plt.plot(y_test, y_test, 'r--', figure=fig)
    return fig


def evalute_models(models: Models, x_test: np.ndarray, y_test: np.ndarray) -> Models:
    for model_name, model in models.models.items():
        models.scores[model_name] = model.score(x_test, y_test)
        models.residual_plots[model_name] = create_residual_plot(model, model_name, x_test, y_test)
    return models

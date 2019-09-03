import pickle
from typing import List, Optional

import logbook
import matplotlib.pyplot as plt
import numpy as np
from attr import attrs, attrib
from matplotlib.figure import Figure

from sagol.models.cnn_3d import DEFAULT_BATCH_SIZE

logger = logbook.Logger(__name__)


@attrs
class Models:
    ylabels: List[str] = attrib(default=None)
    roi_paths: Optional[List[str]] = attrib(default=None)
    # (x, y, z)
    shape: tuple = attrib(default=None)
    # {'svr' : <model>, 'bagging_regressor': <model>}
    models: dict = attrib(default={})
    # {'svr' : 0.52, 'bagging_regressor': 0.48}
    train_scores: dict = attrib(default={})
    # {'svr' : 0.98, 'bagging_regressor': 0.92}
    test_scores: dict = attrib(default={})
    # {'svr' : <graph>, 'bagging_regressor': <graph>}
    residual_plots: dict = attrib(default={})
    # {'svr': dict{str:value}, 'bagging_regressor': dict{str:value}}
    parameters: dict = attrib(default={})
    # {0: 'Hariri_2ndLev_AngryVsNeu', 1: 'Hariri_2ndLev_AngryVsShapes'}
    reverse_contrast_mapping = attrib(default={})

    @property
    def available_models(self):
        return list(self.models.keys())

    def save_model(self, model_name: str, file_path: str):
        assert model_name in self.models, f'Could not find model {model_name}. Available models: {self.available_models}'

        metadata = {
            'model_name': model_name,
            'ylabels': self.ylabels,
            'roi_paths': self.roi_paths,
            'shape': self.shape,
            'reverse_contrast_mapping': self.reverse_contrast_mapping,
            'train_score': self.train_scores[model_name],
            'parameters': self.parameters[model_name]
        }
        model_with_metadata = {**metadata, **{'model': self.models[model_name]}}

        with open(file_path, 'wb') as mdl_f:
            mdl_f.write(pickle.dumps(model_with_metadata))
        logger.info(f'Saved model {model_name} to {file_path}.')

    def load_model(self, model_file_path: str, force: bool = False):
        with open(model_file_path, 'rb') as model_file:
            model_with_metadata = pickle.loads(model_file.read())

        ylabels = model_with_metadata['ylabels']
        roi_paths = model_with_metadata['roi_paths']
        shape = model_with_metadata['shape']
        reverse_contrast_mapping = model_with_metadata['reverse_contrast_mapping']
        train_score = model_with_metadata['train_score']
        parameters = model_with_metadata['parameters']
        assert not self.ylabels or set(ylabels) == set(self.ylabels), 'ylabels of the loaded model do not match.'
        assert not self.roi_paths or set(roi_paths) == set(self.roi_paths), 'ROIs of the loaded model do not match.'
        assert not self.shape or shape == self.shape, 'ylabels of the loaded model do not match.'
        assert not self.reverse_contrast_mapping or reverse_contrast_mapping == self.reverse_contrast_mapping,\
            'Contrast selection do not match.'

        model_name = model_with_metadata['model_name']
        assert model_name not in self.models or force, f'Model {model_name} already exists. Use `force` to override.'

        self.models[model_name] = model_with_metadata['model']
        self.ylabels = ylabels
        self.roi_paths = roi_paths
        self.shape = shape
        self.reverse_contrast_mapping = reverse_contrast_mapping
        self.train_scores[model_name] = train_score
        self.parameters[model_name] = parameters

    def set_models(self, trained_models):
        self.ylabels = trained_models.ylabels
        self.roi_paths = trained_models.roi_paths
        for name, model in trained_models.models.items():
            self.models[name] = model
            self.train_scores[name] = trained_models.train_scores[name]
            if name in trained_models.test_scores:
                self.test_scores[name] = trained_models.test_scores[name]
            else:
                if name in self.test_scores:
                    del self.test_scores[name]
            if name in trained_models.residual_plots:
                self.residual_plots[name] = trained_models.residual_plots[name]
            else:
                if name in self.residual_plots:
                    del self.residual_plots[name]
            self.parameters[name] = trained_models.parameters[name]

    def test(self, model_name, X_test, y_test):
        score = self.models[model_name].score(X_test, y_test)
        self.test_scores[model_name] = score
        fig = create_residual_plot(self.models[model_name], model_name, X_test, y_test)
        self.residual_plots[model_name] = fig
        return score, fig

    def get_train_score(self, model_name, as_str=False):
        if model_name in self.train_scores:
            return '%.2f' % self.train_scores[model_name] if as_str else self.train_scores[model_name]
        else:
            return '' if as_str else None

    def get_test_score(self, model_name, as_str=False):
        if model_name in self.test_scores:
            return '%.2f' % self.test_scores[model_name] if as_str else self.test_scores[model_name]
        else:
            return '' if as_str else None


def _create_residual_plot_private(model_name, y_test, y_predict):
    fig = plt.figure(figsize=(4, 3))
    plt.xlabel('True', figure=fig)
    plt.ylabel('Predicted', figure=fig)
    plt.title(f'Model: {model_name} - True VS Predicted, optimal is aligned with red line', figure=fig)

    # Add data points.
    plt.scatter(y_test, y_predict, color='black', figure=fig)
    # Add optimal line - a perfect prediction.
    plt.plot(y_test, y_test, 'r--', figure=fig)
    return fig


def create_residual_plot(model, model_name: str, x_test: np.array, y_test: np.array) -> Figure:
    return _create_residual_plot_private(model_name, y_test, model.predict(x_test))


def create_residual_plot_cnn(model, model_name: str, x_test, y_test, batch_size=-1) -> Figure:
    return _create_residual_plot_private(model_name, y_test, model.predict(x_test, batch_size=batch_size))


def evaluate_models(models: Models, x_test, y_test, x_test_3d, y_test_3d) -> Models:
    for model_name, model in models.models.items():
        if not model:
            continue
        if model_name == 'cnn':
            models.test_scores[model_name] = model.score(x_test_3d, y_test_3d, batch_size=DEFAULT_BATCH_SIZE)
            models.residual_plots[model_name] = create_residual_plot_cnn(model, model_name, x_test_3d, y_test_3d,
                                                                         batch_size=DEFAULT_BATCH_SIZE)
        else:
            models.test_scores[model_name] = model.score(x_test, y_test)
            models.residual_plots[model_name] = create_residual_plot(model, model_name, x_test, y_test)
    return models

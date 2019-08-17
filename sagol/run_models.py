from typing import List, Optional

import numpy as np
from attr import attrs, attrib
from sklearn.model_selection import train_test_split

from sagol.load_data import ExperimentData, FlattenedExperimentData
from sagol.models.svr import train_svr
from sagol.rois import apply_roi_masks

AVAILABLE_MODELS = ['svr']


@attrs
class Models:
    ylabel: str = attrib()
    roi_paths: Optional[List[str]] = attrib()
    # (x, y, z)
    shape: tuple = attrib()
    # {'svr' : <model>, 'cnn': <model>}
    models: dict = attrib()

    def save(self):
        return

    @classmethod
    def load(cls, ylabels, rois, shape):
        raise NotImplementedError()


def generate_samples_for_model(experiment_data: FlattenedExperimentData, task_name: str, constrast_name: str,
                               ylabel: str):
    samples = [{
        'x': subject_data.tasks_data[task_name][constrast_name],
        'y': subject_data.features_data[ylabel]
    } for subject_data in experiment_data.subjects_data if task_name in subject_data.tasks_data]
    X = [sample['x'] for sample in samples]
    Y = [sample['y'] for sample in samples]
    return X, Y


def get_or_create_models(experiment_data: ExperimentData, task_name: str, contrast_name: str, ylabel: str,
                         roi_paths: Optional[List[str]], model_params: Optional[dict] = None) -> Models:
    model_params = model_params or {}

    masked_experiment_data = apply_roi_masks(experiment_data, roi_paths)

    pre_computed_models = get_pre_computed_models()
    return pre_computed_models or generate_models(masked_experiment_data, task_name, contrast_name, ylabel, roi_paths,
                                                  model_params)


def get_pre_computed_models() -> Optional[Models]:
    return


def generate_models(experiment_data_roi_masked: FlattenedExperimentData, task_name: str, constrast_name: str,
                    ylabel: str, roi_paths: Optional[List[str]], model_params: Optional[dict] = None) -> Models:
    model_params = model_params or {}
    models = {}

    X, Y = generate_samples_for_model(experiment_data_roi_masked, task_name, constrast_name, ylabel)
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    for model_name in AVAILABLE_MODELS:
        models[model_name] = train_model(x_train, y_train, model_name=model_name,
                                         **model_params.get(model_name, {}))
        print(f'Trained {model_name} model, score on train data: {models[model_name].score(x_train, y_train)}.')

    models = Models(ylabel=ylabel, roi_paths=roi_paths,
                    shape=experiment_data_roi_masked.shape, models=models)
    models.save()

    scores = evalute_models(models, x_test, y_test)
    print(scores)
    return models


def evalute_models(models: Models, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    scores = {}
    for model_name, model in models.models.items():
        scores[model_name] = model.score(x_test, y_test)
    return scores


def train_model(x_train: np.ndarray, y_train: np.ndarray, model_name: str, **kwargs):
    if model_name == 'svr':
        return train_svr(x_train, y_train, **kwargs)
    else:
        raise NotImplementedError(f'Model: {model_name} is not supported.')

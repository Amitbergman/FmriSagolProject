from typing import List, Optional

import numpy as np
from attr import attrs, attrib

from sagol.load_data import ExperimentData, SubjectExperimentData

AVAILABLE_MODELS = ['svr']


def get_available_rois() -> List[str]:
    return


def get_mask_from_roi(roi_name) -> np.array:
    return


def apply_roi_masks(experiment_data: ExperimentData, rois) -> ExperimentData:
    return


@attrs
class Models:
    ylabels: List[str] = attrib()
    rois: Optional[List[str]] = attrib()
    # (x, y, z)
    shape: tuple = attrib()
    # {'svr' : <model>, 'cnn': <model>}
    models: dict = attrib()

    def save(self):
        raise NotImplementedError

    @classmethod
    def load(cls, ylabels, rois, shape):
        raise NotImplementedError


def get_or_create_models(experiment_data: ExperimentData, ylabels, rois) -> Models:
    if rois:
        experiment_data = apply_roi_masks(experiment_data, rois)

    pre_computed_models = get_pre_computed_models(ylabels, rois, shape)
    return pre_computed_models or generate_models(experiment_data, ylabels, rois)


def get_pre_computed_models(ylabels, rois, shape) -> Optional[Models]:
    return


def generate_models(experiment_data_roi_masked: List[SubjectExperimentData], ylabels, rois) -> Models:
    models = {}
    for model_name in AVAILABLE_MODELS:
        models[model_name] = train_model(experiment_data_roi_masked, model_name=model_name)

    models = Models(ylabels=ylabels, rois=rois, shape=experiment_data_roi_masked.shape, models=model)
    models.save()
    return models


def train_model(experiment_data: ExperimentData, model_name: str):
    if model_name == 'svr':
        train_svr(experiment_data)
    else:
        raise NotImplementedError(f'Model: {model_name} is not supported.')

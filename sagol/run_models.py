from typing import List, Optional

from attr import attrs, attrib

from sagol.load_data import ExperimentData
from sagol.rois import apply_roi_masks

AVAILABLE_MODELS = ['svr']


@attrs
class Models:
    ylabels: List[str] = attrib()
    roi_paths: Optional[List[str]] = attrib()
    # (x, y, z)
    shape: tuple = attrib()
    # {'svr' : <model>, 'cnn': <model>}
    models: dict = attrib()

    def save(self):
        raise NotImplementedError

    @classmethod
    def load(cls, ylabels, rois, shape):
        raise NotImplementedError


def get_or_create_models(experiment_data: ExperimentData, ylabels, roi_paths: Optional[List[str]]) -> Models:
    masked_experiment_data = apply_roi_masks(experiment_data, roi_paths)

    pre_computed_models = get_pre_computed_models(ylabels, roi_paths, experiment_data.shape)
    return pre_computed_models or generate_models(masked_experiment_data, ylabels, roi_paths)


def get_pre_computed_models(ylabels, rois, shape) -> Optional[Models]:
    return


def generate_models(experiment_data_roi_masked: ExperimentData, ylabels) -> Models:
    models = {}
    for model_name in AVAILABLE_MODELS:
        models[model_name] = train_model(experiment_data_roi_masked, model_name=model_name)

    models = Models(ylabels=ylabels, roi_paths=experiment_data_roi_masked.roi_paths,
                    shape=experiment_data_roi_masked.shape, models=models)
    models.save()
    return models


def train_model(experiment_data: ExperimentData, model_name: str):
    if model_name == 'svr':
        train_svr(experiment_data)
    else:
        raise NotImplementedError(f'Model: {model_name} is not supported.')

from typing import Union, List, Optional

import logbook
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sagol.evaluate_models import evaluate_models, Models
from sagol.load_data import FlattenedExperimentData, ExperimentData, ExperimentDataAfterSplit, \
    ExperimentDataAfterSplit3D
from sagol.models.bagging_regressor import train_bagging_regressor
from sagol.models.nusvr import train_nusvr
from sagol.models.svr import train_svr
from sagol.pre_processing import generate_subjects_ylabel, one_hot_encode_contrasts, get_one_hot_from_index
from sagol.rois import apply_roi_masks

AVAILABLE_MODELS = {'svr': {'kernel': str, 'C': float, 'gamma': float},
                    'bagging_regressor': {'n_estimators': int},
                    'nusvr': {'kernel': str, 'C': float, 'gamma': float}}

logger = logbook.Logger(__name__)


def generate_samples_for_model(experiment_data: Union[FlattenedExperimentData, ExperimentData],
                               tasks_and_contrasts: Optional[dict], ylabels: List[str],
                               weights: Optional[List] = None) -> (np.ndarray, np.ndarray, dict):
    """
    :param tasks_and_contrasts: A dictionary of {<task_name>: [<contrast_name>, <contrast_name2>]}
    Pass `None` to fetch all tasks and all contrasts. Pass None/[] inside a `task_name` to fetch all contrast for
    that specific task.
    """
    assert ylabels
    # Whether the fMRI data is flattened or 3D.
    is_3d = isinstance(experiment_data, ExperimentData)
    logger.info(f'Generating samples, is_3d: {is_3d}.')

    tasks_and_contrasts = tasks_and_contrasts or {}

    # We need to differentiate images of the same brain "cut" belonging to different contrast.
    # To do so, we use one hot encoding of the task + contrast combination.
    contrast_hot_encoding_mapping = {}
    current_contrast_index = 0

    use_all_tasks = not bool(tasks_and_contrasts)
    if use_all_tasks:
        task_names = []
    else:
        task_names = tasks_and_contrasts.keys()

    subjects_ylabel_data = generate_subjects_ylabel(experiment_data, ylabels, weights)

    X, Y = [], []
    for subject_index, subject_data in tqdm(enumerate(experiment_data.subjects_data)):
        for task_name, task_data in subject_data.tasks_data.items():
            if use_all_tasks or task_name in task_names:
                use_all_contrasts = not bool(tasks_and_contrasts.get(task_name))

                for contrast_name, fmri_data in task_data.items():
                    if use_all_contrasts or contrast_name in tasks_and_contrasts.get(task_name, []):
                        task_contrast_name = f'{task_name}.{contrast_name}'
                        if task_contrast_name not in contrast_hot_encoding_mapping:
                            contrast_hot_encoding_mapping[task_contrast_name] = current_contrast_index
                            current_contrast_index += 1
                        # Add the contrast as the last feature in the data.
                        if is_3d:
                            X.append([torch.from_numpy(fmri_data), task_contrast_name])
                        else:
                            X.append(np.concatenate((fmri_data, [contrast_hot_encoding_mapping[task_contrast_name]])))
                        Y.append(subjects_ylabel_data[subject_index])

    # Because the numbers of the contrasts are meaningless it is necessary to convert them to one hot codes so they will not be
    # treated as numerical features.
    if is_3d:
        number_of_contrasts = current_contrast_index
        for entry in X:
            entry[1] = get_one_hot_from_index(contrast_hot_encoding_mapping[entry[1]], number_of_contrasts)
    else:
        X = one_hot_encode_contrasts(np.array(X))

    # Returning the inverse dictionary to allow getting the task+contrasts out of their index.
    one_hot_encoding_mapping = {ind: task_contrast_name for task_contrast_name, ind in
                                contrast_hot_encoding_mapping.items()}

    return X, Y, one_hot_encoding_mapping


def get_or_create_models(experiment_data: ExperimentData, tasks_and_contrasts: Optional[dict], ylabels: List[str],
                         roi_paths: Optional[List[str]], ylabel_to_weight: Optional[dict] = None,
                         model_params: Optional[dict] = None, train_only: bool = False,
                         model_names: Optional[List[str]] = None) -> Models:
    model_params = model_params or {}

    pre_computed_models = get_pre_computed_models()
    if pre_computed_models:
        return pre_computed_models
    else:
        models, data, data_3d = split_data_and_generate_models(experiment_data, tasks_and_contrasts,
                                                               ylabels, roi_paths,
                                                               ylabel_to_weight=ylabel_to_weight,
                                                               model_params=model_params,
                                                               train_only=train_only,
                                                               model_names=model_names)
        return models


def get_pre_computed_models() -> Optional[Models]:
    return


def split_data_and_generate_models(experiment_data: ExperimentData, tasks_and_contrasts: dict,
                                   ylabels: List[str], roi_paths: Optional[List[str]],
                                   ylabel_to_weight: Optional[dict] = None,
                                   model_params: Optional[dict] = None, train_only: bool = False,
                                   model_names: Optional[List[str]] = None) -> (
        Models, ExperimentDataAfterSplit, ExperimentDataAfterSplit3D):
    experiment_data_after_split, experiment_data_after_split_3d = generate_experiment_data_after_split(
        experiment_data, tasks_and_contrasts, ylabels, roi_paths, ylabel_to_weight, train_only)

    return generate_models(experiment_data_after_split, experiment_data_after_split_3d, ylabels, roi_paths,
                           model_names=model_names, model_params=model_params, train_only=train_only)


def generate_models(experiment_data_after_split: ExperimentDataAfterSplit,
                    experiment_data_after_split_3d: ExperimentDataAfterSplit3D,
                    ylabels: List[str], roi_paths: Optional[List[str]], model_params: Optional[dict] = None,
                    train_only: bool = False, model_names: Optional[List[str]] = None) -> (
        Models, ExperimentDataAfterSplit, ExperimentDataAfterSplit3D):

    x_train = experiment_data_after_split.x_train
    y_train = experiment_data_after_split.y_train
    x_test = experiment_data_after_split.x_test
    y_test = experiment_data_after_split.y_test

    model_names = model_names or AVAILABLE_MODELS.keys()
    model_params = model_params or {}
    models = {}
    train_scores = {}

    for model_name in model_names:
        models[model_name] = train_model(x_train, y_train, model_name=model_name,
                                         **model_params.get(model_name, {}))
        train_score = models[model_name].score(x_train, y_train)
        train_scores[model_name] = train_score

        logger.info(f'Trained {model_name} model, score on train data: {train_score}.')

    models = Models(ylabels=ylabels, roi_paths=roi_paths, train_scores=train_scores,
                    shape=experiment_data_after_split.shape, models=models)

    if not train_only:
        models = evaluate_models(models, x_test, y_test)
        logger.info(f'Model scores: {models.test_scores}')

    return models, experiment_data_after_split, experiment_data_after_split_3d


def generate_experiment_data_after_split(experiment_data: ExperimentData, tasks_and_contrasts: dict,
                                         ylabels: Optional[List[str]], roi_paths: Optional[List[str]],
                                         ylabel_to_weight: Optional[dict] = None, train_only: bool = False) -> (
        ExperimentDataAfterSplit, ExperimentDataAfterSplit3D):
    weights = generate_ylabel_weights(ylabels, ylabel_to_weight)

    experiment_data_roi_masked = apply_roi_masks(experiment_data, roi_paths)

    X, Y, one_hot_encoding_mapping = generate_samples_for_model(experiment_data_roi_masked, tasks_and_contrasts,
                                                                ylabels, weights=weights)
    X_3d, Y_3d, one_hot_encoding_mapping_3d = generate_samples_for_model(experiment_data, tasks_and_contrasts,
                                                                         ylabels, weights=weights)
    # Make the same train-test split for both the flattened and 3D data.
    if train_only:
        x_train, y_train, x_test, y_test = X, Y, [], []
        x_train_3d, x_test_3d = X_3d, []
    else:
        x_train, x_test, y_train, y_test, train_idx, test_idx = train_test_split(X, Y, np.arange(len(X)))
        x_train_3d, x_test_3d, _, _ = X_3d[train_idx], X_3d[test_idx], Y_3d[train_idx], Y_3d[test_idx]

    experiment_data_after_split = ExperimentDataAfterSplit(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        flattened_vector_index_to_voxel=experiment_data_roi_masked.flattened_vector_index_to_voxel,
        flattened_vector_index_to_rois=experiment_data_roi_masked.flattened_vector_index_to_rois,
        shape=experiment_data_roi_masked.shape
    )

    experiment_data_after_split_3d = ExperimentDataAfterSplit3D(
        x_train=x_train_3d,
        y_train=y_train,
        x_test=x_test_3d,
        y_test=y_test,
        shape=experiment_data_roi_masked.shape
    )

    return experiment_data_after_split, experiment_data_after_split_3d


def generate_ylabel_weights(ylabels: List[str], ylabel_to_weight: Optional[dict]) -> List[float]:
    weights = []
    if ylabel_to_weight:
        assert len(ylabel_to_weight) == len(ylabels), 'Weights must be provided for all ylabels.'
        sum_of_weights = sum(ylabel_to_weight.values()) != 1
        if sum_of_weights:
            logger.info('Weights were not normalized, normalizing the weights such that the sum is 1.')
            ylabel_to_weight = {k: v / sum_of_weights for k, v in ylabel_to_weight.items()}

        weights = [ylabel_to_weight[ylabel] for ylabel in ylabels]
    return weights


def train_model(x_train: np.ndarray, y_train: np.ndarray, model_name: str, **kwargs):
    if model_name == 'svr':
        return train_svr(x_train, y_train, **kwargs)
    elif model_name == 'bagging_regressor':
        return train_bagging_regressor(x_train, y_train, **kwargs)
    elif model_name == 'nusvr':
        return train_nusvr(x_train, y_train, **kwargs)
    else:
        raise NotImplementedError(f'Model: {model_name} is not supported.')

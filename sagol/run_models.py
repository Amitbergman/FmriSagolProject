from typing import List, Optional

import numpy as np
from attr import attrs, attrib
from sklearn.model_selection import train_test_split

from sagol.load_data import ExperimentData, FlattenedExperimentData
from sagol.models.bagging_regressor import train_bagging_regressor
from sagol.models.svr import train_svr
from sagol.pre_processing import one_hot_encode_contrasts
from sagol.rois import apply_roi_masks

AVAILABLE_MODELS = ['svr', 'bagging_regressor']


@attrs
class Models:
    ylabels: List[str] = attrib()
    roi_paths: Optional[List[str]] = attrib()
    # (x, y, z)
    shape: tuple = attrib()
    # {'svr' : <model>, 'multiple_regressor': <model>}
    models: dict = attrib()


def generate_samples_for_model(experiment_data: FlattenedExperimentData, tasks_and_contrasts: Optional[dict],
                               ylabels: List[str], weights: Optional[List] = None) -> (np.ndarray, np.ndarray, dict):
    """
    :param tasks_and_contrasts: A dictionary of {<task_name>: [<contrast_name>, <contrast_name2>]}
    Pass `None` to fetch all tasks and all contrasts. Pass None/[] inside a `task_name` to fetch all contrast for
    that specific task.
    """
    assert ylabels

    tasks_and_contrasts = tasks_and_contrasts or {}
    X, Y = [], []

    # We need to differentiate images of the same brain "cut" belonging to different contrast.
    # To do so, we use one hot encoding of the task + contrast combination.
    contrast_hot_encoding_mapping = {}
    current_contrast_index = 0

    use_all_tasks = not bool(tasks_and_contrasts)
    if use_all_tasks:
        task_names = []
    else:
        task_names = tasks_and_contrasts.keys()

    for subject_data in experiment_data.subjects_data:
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
                        X.append(np.concatenate((fmri_data, [contrast_hot_encoding_mapping[task_contrast_name]])))
                        if len(ylabels) == 1:
                            Y.append(subject_data.features_data[ylabels[0]])
                        else:
                            Y.append([subject_data.features_data[label] for label in ylabels])

    # Because the numbers of the contrasts are meaningless it is necessary to convert them to one hot codes so they will not be
    # treated as numerical features.
    X = one_hot_encode_contrasts(np.array(X))

    # Returning the inverse dictionary to allow getting the task+contrasts out of their index.
    one_hot_encoding_mapping = {ind: task_contrast_name for task_contrast_name, ind in
                                contrast_hot_encoding_mapping.items()}

    # In case of multiple labels, use a weighted sum.
    Y = [np.average(y, weights=weights) for y in Y]

    return X, Y, one_hot_encoding_mapping


def get_or_create_models(experiment_data: ExperimentData, tasks_and_contrasts: Optional[dict], ylabels: List[str],
                         roi_paths: Optional[List[str]], ylabel_to_weight: Optional[dict] = None,
                         model_params: Optional[dict] = None) -> Models:
    model_params = model_params or {}

    masked_experiment_data = apply_roi_masks(experiment_data, roi_paths)

    pre_computed_models = get_pre_computed_models()
    return pre_computed_models or generate_models(masked_experiment_data, tasks_and_contrasts, ylabels, roi_paths,
                                                  ylabel_to_weight=ylabel_to_weight, model_params=model_params)


def get_pre_computed_models() -> Optional[Models]:
    return


def generate_models(experiment_data_roi_masked: FlattenedExperimentData, tasks_and_contrasts: dict,
                    ylabels: List[str], roi_paths: Optional[List[str]], ylabel_to_weight: Optional[dict] = None,
                    model_params: Optional[dict] = None) -> Models:
    model_params = model_params or {}
    models = {}

    weights = generate_ylabel_weights(ylabels, ylabel_to_weight)

    X, Y, one_hot_encoding_mapping = generate_samples_for_model(experiment_data_roi_masked, tasks_and_contrasts,
                                                                ylabels, weights=weights)
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    for model_name in AVAILABLE_MODELS:
        models[model_name] = train_model(x_train, y_train, model_name=model_name,
                                         **model_params.get(model_name, {}))
        print(f'Trained {model_name} model, score on train data: {models[model_name].score(x_train, y_train)}.')

    models = Models(ylabels=ylabels, roi_paths=roi_paths,
                    shape=experiment_data_roi_masked.shape, models=models)

    scores = evalute_models(models, x_test, y_test)
    print(scores)
    return models


def generate_ylabel_weights(ylabels: List[str], ylabel_to_weight: Optional[dict]) -> List[float]:
    weights = []
    if ylabel_to_weight:
        assert len(ylabel_to_weight) == len(ylabels), 'Weights must be provided for all ylabels.'
        sum_of_weights = sum(ylabel_to_weight.values()) != 1
        if sum_of_weights:
            print('Weights were not normalized, normalizing the weights such that the sum is 1.')
            ylabel_to_weight = {k: v / sum_of_weights for k, v in ylabel_to_weight.items()}

        weights = [ylabel_to_weight[ylabel] for ylabel in ylabels]
    return weights


def evalute_models(models: Models, x_test: np.ndarray, y_test: np.ndarray) -> dict:
    scores = {}
    for model_name, model in models.models.items():
        scores[model_name] = model.score(x_test, y_test)
    return scores


def train_model(x_train: np.ndarray, y_train: np.ndarray, model_name: str, **kwargs):
    if model_name == 'svr':
        return train_svr(x_train, y_train, **kwargs)
    elif model_name == 'bagging_regressor':
        return train_bagging_regressor(x_train, y_train, **kwargs)
    else:
        raise NotImplementedError(f'Model: {model_name} is not supported.')

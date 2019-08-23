from typing import List, Optional

import numpy as np
from attr import attrs, attrib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sagol.load_data import ExperimentData, FlattenedExperimentData
from sagol.models.svr import train_svr
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
        return

    @classmethod
    def load(cls, ylabels, rois, shape):
        raise NotImplementedError()


def group_by_contrast(X, y, first_index_of_contrast: int):
    if len(X) != len(y):
        raise RuntimeError("X and y must be of the same length!")
    X = np.array(X)
    y = np.array(y)
    T = [np.where(x[first_index_of_contrast:] == 1)[0][0] for x in X]  # Extracting contrasts as numbers out of X data
    contrasts = sorted(set(T))  # Getting the unique contrasts
    num_of_contrasts = len(contrasts)

    # Needed for the case that the contrasts' numbers are sparsed in the range they are in
    contrast_to_index = {contrasts[i]: i for i in range(num_of_contrasts)}

    # Bucket sorting
    counters = [0] * num_of_contrasts
    for t in T:
        counters[contrast_to_index[t]] += 1
    X_grouped, y_grouped = np.ndarray(X.shape), np.ndarray(y.shape)
    insertion_indexes = [0]
    for i in range(1, num_of_contrasts):
        insertion_indexes.append(insertion_indexes[i - 1] + counters[i - 1])
    for i in range(len(T)):
        insertion_index = insertion_indexes[contrast_to_index[T[i]]]
        X_grouped[insertion_index] = X[i]
        y_grouped[insertion_index] = y[i]
        insertion_indexes[contrast_to_index[T[i]]] += 1

    # Returning the number of instances in each group too
    return X_grouped, y_grouped, counters


def centeralize_1d_data(X, y, axis: str, first_index_of_contrast=None):
    if len(X) != len(y):
        raise RuntimeError("X and y must be of the same length!")
    if axis == 'contrast':
        if first_index_of_contrast is None:
            raise RuntimeError(
                "You must pass as parameter the first index of contrast if you want to normalize by this axis!")
        X_centeralized, y_centeralized, counters = group_by_contrast(X, y, first_index_of_contrast)
        group = 0
        first_index_of_group = 0
        mean_vector = np.zeros(len(X_centeralized[0][:first_index_of_contrast]))

        # This loop calculates the mean vector of each group and reduces it from each vector in the group
        for i in range(len(X_centeralized) + 1):
            if i == first_index_of_group + counters[group]:
                mean_vector /= counters[group]
                mean_vector = np.concatenate(
                    (mean_vector, np.zeros(len(X_centeralized[0][first_index_of_contrast:]), dtype=float)))
                for j in range(first_index_of_group, i):
                    X_centeralized[j] -= mean_vector
                group += 1
                first_index_of_group = i
                mean_vector = np.zeros(len(X_centeralized[0][:first_index_of_contrast]))
            if i == len(X_centeralized):
                break
            mean_vector += X_centeralized[i][:first_index_of_contrast]
    else:
        raise RuntimeError('No such axis: ' + axis + '!')

    # Notcie that the new X and y might not be shuffled
    return X_centeralized, y_centeralized


def one_hot_encode_contrasts(X):
    X = X.reshape(len(X),
                  len(X[0]))  # Reshaping to get a matrix like numpy array is necessary for the one hot encoding ahead
    contrast_data = X[:, -1:]  # Getting contrast data only

    # Binary encoding
    one_hot_encoder = OneHotEncoder(sparse=False, categories='auto')
    one_hot_encoded = one_hot_encoder.fit_transform(contrast_data)  # Getting the the one hot encoded data

    # Deleting old contrast int-column and replace it by one-hot-encoded column
    X = np.delete(X, -1, axis=1)
    X = np.concatenate((X, one_hot_encoded), axis=1)

    return X


def generate_samples_for_model(experiment_data: FlattenedExperimentData, tasks_and_contrasts: Optional[dict],
                               ylabels: List[str]):
    """
    :param ylabels:
    :param tasks_and_contrasts: A dictionary of {<task_name>: [<contrast_name>, <contrast_name2>]}
    Pass `None` to fetch all tasks and all contrasts. Pass None/[] inside a `task_name` to fetch all contrast for
    that specific task.
    """
    assert ylabels
    X, Y = [], []
    contrast_hot_encoding_mapping = {}
    current_contrast_index = 0

    task_names = tasks_and_contrasts.keys()
    use_all_tasks = not bool(tasks_and_contrasts)

    for subject_data in experiment_data.subjects_data:
        for task_name, task_data in subject_data.tasks_data.items():
            if use_all_tasks or task_name in task_names:
                use_all_contrasts = not bool(task_data.keys())

                for contrast_name, fmri_data in task_data.items():
                    if use_all_contrasts or contrast_name in tasks_and_contrasts.get(task_name, []):
                        task_contrast_name = f'{task_name}.{contrast_name}'
                        if task_contrast_name not in contrast_hot_encoding_mapping:
                            contrast_hot_encoding_mapping[task_contrast_name] = current_contrast_index
                            current_contrast_index += 1
                        X.append(np.concatenate((fmri_data, [contrast_hot_encoding_mapping[task_contrast_name]])))
                        if len(ylabels) == 1:
                            Y.append(subject_data.features_data[ylabels[0]])
                        else:
                            Y.append([subject_data.features_data[label] for label in ylabels])

    # Because the numbers of the contrasts are meaningless it is necessary to convert them to one hot codes so they will not be
    # treated as numerical features
    X = one_hot_encode_contrasts(np.array(X))

    # Returning the inverse dictionary to allow getting the task+contrasts out of their index.
    one_hot_encoding_mapping = {ind: tasks_and_contrasts for task_contrast_name, ind in
                                contrast_hot_encoding_mapping.items()}

    return X, Y, one_hot_encoding_mapping


def get_or_create_models(experiment_data: ExperimentData, tasks_and_contrasts: dict, ylabels: List[str],
                         roi_paths: Optional[List[str]], model_params: Optional[dict] = None) -> Models:
    model_params = model_params or {}

    masked_experiment_data = apply_roi_masks(experiment_data, roi_paths)

    pre_computed_models = get_pre_computed_models()
    return pre_computed_models or generate_models(masked_experiment_data, tasks_and_contrasts, ylabels, roi_paths,
                                                  model_params)


def get_pre_computed_models() -> Optional[Models]:
    return


def generate_models(experiment_data_roi_masked: FlattenedExperimentData, tasks_and_contrasts: dict,
                    ylabels: List[str], roi_paths: Optional[List[str]], model_params: Optional[dict] = None) -> Models:
    model_params = model_params or {}
    models = {}

    X, Y, one_hot_encoding_mapping = generate_samples_for_model(experiment_data_roi_masked, tasks_and_contrasts,
                                                                ylabels)
    x_train, x_test, y_train, y_test = train_test_split(X, Y)

    for model_name in AVAILABLE_MODELS:
        models[model_name] = train_model(x_train, y_train, model_name=model_name,
                                         **model_params.get(model_name, {}))
        print(f'Trained {model_name} model, score on train data: {models[model_name].score(x_train, y_train)}.')

    models = Models(ylabels=ylabels, roi_paths=roi_paths,
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

import logbook

from sagol.load_data import FlattenedExperimentData, ExperimentDataAfterSplit
from sagol.evaluate_models import Models
import copy
import numpy as np
import nibabel as nib
from nilearn import plotting

logger = logbook.Logger(__name__)


def deduce_by_leave_one_roi_out(models: Models, flattened_experiment_data: ExperimentDataAfterSplit):
    for model_name, model in models.models.items():
        score_on_all_rois = model.score(flattened_experiment_data.x_test, flattened_experiment_data.y_test)
        logger.info(f'score on all rois for {model_name} is {score_on_all_rois}')
        for roi_path in models.roi_paths:
            # Leaving the ROI out by masking it out of the current data.
            leftout_roi_indices = _get_roi_indices(roi_path, flattened_experiment_data.flattened_vector_index_to_rois)
            current_x_train = zero_indexes_in_data(flattened_experiment_data.x_train, leftout_roi_indices)
            current_x_test = zero_indexes_in_data(flattened_experiment_data.x_test, leftout_roi_indices)

            model.fit(current_x_train, flattened_experiment_data.y_train)
            score = model.score(current_x_test, flattened_experiment_data.y_test)

            logger.info(f'score in model {model_name} without roi {roi_path} is {score}')


def _get_roi_indices(roi_path, d):
    return [k for k, v in d.items() if roi_path in v]


def zero_indexes_in_data(data, indexes_to_zero):
    res = copy.copy(data)
    for data_point in res:
        for ind in indexes_to_zero:
            data_point[ind] = 0
    return res


def deduce_by_coefs(models: Models, first_index_of_contrast):
    models_importances = {}
    for name, model in models.models.items():
        models_importances[name] = {models.flattened_vector_index_to_voxel[index]: value
                                    for index, value in enumerate(model.coef_[:first_index_of_contrast])}
    return models_importances


def deduce_from_bagging_regressor(models: Models, first_index_of_contrast):
    models_importances = {}
    for name, model in models.models.items():
        feature_importances = np.mean([reg.feature_importances_ for reg in model.estimators_], axis=0)
        models_importances[name] = {models.flattened_vector_index_to_voxel[index]: value
                                    for index, value in enumerate(feature_importances[:first_index_of_contrast])}
    return models_importances


def plot_brain_image_from_nifty(nifty_path):
    data = nib.load(nifty_path)

    plotting.plot_roi(data,
                      title="plot_roi")
    plotting.show()

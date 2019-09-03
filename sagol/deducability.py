from sagol.load_data import FlattenedExperimentData, ExperimentDataAfterSplit
from sagol.evaluate_models import Models
import copy
import numpy as np
import nibabel as nib
from nilearn import plotting
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def deduce_by_leave_one_roi_out(models: Models, flattened_experiment_data: ExperimentDataAfterSplit):
    # will return the score without roi1, without roi2
    map_roi_to_score_without_it = {}
    for model_name, model in models.models.items():
        score_on_all_rois = model.score(flattened_experiment_data.x_test,
                                        flattened_experiment_data.y_test)
        print(f"score on all rois for current model is {score_on_all_rois}")
        for roi_path in models.roi_paths:
            list_of_indexes = get_indexes_of_roi(roi_path, flattened_experiment_data.flattened_vector_index_to_rois)
            current_x_train = zero_indexes_in_data(flattened_experiment_data.x_train, list_of_indexes)
            current_x_test = zero_indexes_in_data(flattened_experiment_data.x_test, list_of_indexes)
            model.fit(current_x_train, flattened_experiment_data.y_train)
            print(
                f"score in model {model_name} without roi {roi_path} is {model.score(current_x_test, flattened_experiment_data.y_test)}")
            map_roi_to_score_without_it[roi_path] = model.score(current_x_test, flattened_experiment_data.y_test)

    return map_roi_to_score_without_it

def get_indexes_of_roi(roi_path, d):
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

def plot_brain_image_from_nifty(nifty_path, path_to_save):
    data = nib.load(nifty_path)

    display = plotting.plot_roi(data,
                      title="plot_roi")
    display.savefig(path_to_save)




from sagol.load_data import FlattenedExperimentData, ExperimentDataAfterSplit
from sagol.run_models import *
import copy

@attrs
class Models:
    ylabels: List[str] = attrib()
    roi_paths: Optional[List[str]] = attrib()
    # (x, y, z)
    shape: tuple = attrib()
    # {'svr' : <model>, 'multiple_regressor': <model>}
    models: dict = attrib()

def deduce_by_leave_one_roi_out(models:Models, flattened_experiment_data: ExperimentDataAfterSplit):
#will return the score without roi1, without roi2
    for model_name, model in models.models.items():
        score_on_all_rois = model.score(flattened_experiment_data.original_x_test, flattened_experiment_data.original_y_test)
        print(f"score on all rois for current model is {score_on_all_rois}")
        for roi_path in models.roi_paths:
            list_of_indexes = get_indexes_of_roi(roi_path, flattened_experiment_data.flattened_vector_index_to_rois)
            current_x_train = zero_indexes_in_data(flattened_experiment_data.original_x_train, list_of_indexes)
            current_x_test = zero_indexes_in_data(flattened_experiment_data.original_x_test, list_of_indexes)
            model.fit(current_x_train, flattened_experiment_data.original_y_train)
            print(f"score in model {model_name} without roi {roi_path} is {model.score(current_x_test, flattened_experiment_data.original_y_test)}")     

def get_indexes_of_roi(roi_path, d):
    return [k for k,v in d.items() if roi_path in v]

def zero_indexes_in_data(data, indexes_to_zero):
    res = copy.copy(data)
    for data_point in res:
        for ind in indexes_to_zero:
            data_point[ind] = 0
    return res


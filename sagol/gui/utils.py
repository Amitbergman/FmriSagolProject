from sagol.load_data import create_subject_experiment_data, ExperimentDataAfterSplit, ExperimentDataAfterSplit3D, \
    combine_train_and_test_data, update_test_data
from sagol.gui.globals import STATE
from sagol.run_models import apply_roi_masks_and_generate_samples_for_model

def load_test_data(excel_paths, nifty_dir, combine_train_and_test=True):
    test_experiment_data = create_subject_experiment_data(excel_paths, [nifty_dir])
    trained_models = STATE['trained_models']
    contrast_mapping = {name: ind for ind, name in trained_models.reverse_contrast_mapping.items()}
    tasks_and_contrasts = STATE['tasks_and_contrasts'] if 'tasks_and_contrasts' in STATE else None
    experiment_data_roi_masked, X, y, X_3d, y_3d, _ = apply_roi_masks_and_generate_samples_for_model(
        experiment_data=test_experiment_data, roi_paths=trained_models.roi_paths, contrast_mapping=contrast_mapping,
        tasks_and_contrasts=tasks_and_contrasts, ylabels=trained_models.ylabels, weights=STATE['weights'])
    if len(X) == 0:
        return False
    if 'experiment_data' not in STATE:
        STATE['experiment_data'] = test_experiment_data
        STATE['experiment_data_after_split'] = \
            ExperimentDataAfterSplit(x_train=None, x_test=X, y_train=None, y_test=y,
                                     flattened_vector_index_to_voxel=experiment_data_roi_masked.flattened_vector_index_to_voxel,
                                     flattened_vector_index_to_rois=experiment_data_roi_masked.flattened_vector_index_to_rois,
                                     shape=experiment_data_roi_masked.shape)
        STATE['experiment_data_after_split_3d'] = ExperimentDataAfterSplit3D(x_train=None, x_test=X_3d, y_train=None,
                                                                             y_test=y_3d, shape=experiment_data_roi_masked.shape)
    else:
        if combine_train_and_test:
            combine_train_and_test_data(STATE['experiment_data_after_split'])
            combine_train_and_test_data(STATE['experiment_data_after_split_3d'])
        update_test_data(STATE['experiment_data_after_split'], X, y)
        update_test_data(STATE['experiment_data_after_split_3d'], X_3d, y_3d)
    return True

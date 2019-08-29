import os
import re
from collections import defaultdict
from typing import List, Optional

import logbook
import nibabel as nib
import numpy as np
import pandas as pd
from attr import attrs, attrib
from tqdm import tqdm

from sagol.rois import apply_roi_masks
from sagol.run_models import generate_samples_for_model
from sklearn.model_selection import train_test_split
import torch

SUBJECT_NAME_REGEX = re.compile('sub-(\d+).*')

logger = logbook.Logger(__name__)


@attrs
class SubjectExperimentData:
    subject_id: int = attrib()
    # Subject general data (e.g Age) and scores on questionnaires (e.g FPES).
    features_data: dict = attrib()
    # fMRI data for each task and contrast.
    tasks_data: dict = attrib()


@attrs
class ExperimentData:
    subjects_data: List[SubjectExperimentData] = attrib()
    # (x, y, z)
    shape: tuple = attrib()
    # Holds the available contrasts for each task.
    tasks_metadata: dict = attrib()
    roi_paths: Optional[List[str]] = attrib(default=None)

    @property
    def available_features(self):
        return sorted(self.subjects_data[0].features_data.keys())


@attrs
class FlattenedExperimentData:
    subjects_data: List[SubjectExperimentData] = attrib()
    # {0: 1762, 1: 1763, 2: 1764 ..., 25: 16584}
    flattened_vector_index_to_voxel: dict = attrib()
    # Same as `flattened_vector_index_to_voxel`, but allows tracking back the relevant ROIs.
    flattened_vector_index_to_rois: dict = attrib()
    # (x, y, z)
    shape: tuple = attrib()
        
@attrs
class ExperimentDataAfterSplit:
    original_x_test: np.ndarray = attrib()
    original_x_train: np.ndarray = attrib()
    original_y_train: np.ndarray = attrib()
    original_y_test: np.ndarray = attrib()
    flattened_vector_index_to_voxel: dict = attrib()
    flattened_vector_index_to_rois: dict = attrib()
    shape: tuple = attrib()

@attrs
class ExperimentDataAfterSplit3D:
    original_x_test: torch.Tensor = attrib()
    original_x_train: torch.Tensor = attrib()
    original_y_train: torch.Tensor = attrib()
    original_y_test: torch.Tensor = attrib()
    shape: tuple = attrib()


def convert_nifty_to_image_array(path: str) -> np.array:
    data = nib.load(path).get_fdata()
    return np.array(np.nan_to_num(data))


def merge_subject_dfs(dfs: List[pd.DataFrame], column_to_merge_on='Sub') -> pd.DataFrame:
    first_df, *rest = dfs
    data = first_df.to_dict(orient='records')
    data = {datum[column_to_merge_on]: datum for datum in data}
    for df in rest:
        data_to_update = df.to_dict(orient='record')
        data_to_update = {datum[column_to_merge_on]: datum for datum in data_to_update}
        for subject_id, subject_data in data_to_update.items():
            if subject_id not in data:
                data[subject_id] = subject_data
            else:
                data[subject_id].update(subject_data)

    return pd.DataFrame(data=data.values())


def create_subject_experiment_data(excel_paths: List[str], nifty_dirs: List[str]) -> ExperimentData:
    assert excel_paths
    assert nifty_dirs
    logger.info('Creating subjects experiment data.')

    tasks_data = defaultdict(lambda: defaultdict(dict))
    tasks_metadata = defaultdict(list)
    experiment_data = []
    dfs = []

    logger.info('Loading excel data.')
    for excel_path in excel_paths:
        # Read excel file with multiple sheets, allows for merging the data belonging to the same subject later.
        if excel_path.endswith('.xlsx') or excel_path.endswith('.xls'):
            xls = pd.ExcelFile(excel_path)
            for sheet in xls.sheet_names:
                dfs.append(pd.read_excel(excel_path, sheet_name=sheet))
        else:
            raise Exception()

    features_df = merge_subject_dfs(dfs)
    # Transform NaN/NaT to None
    features_df = features_df.where((pd.notnull(features_df)), None)
    logger.info('Done loading excel data.')

    # Extract fMRI data
    for nifty_dir in nifty_dirs:
        task_name = os.path.basename(nifty_dir)
        logger.info(f'Loading data for task: {task_name}')
        for contrast_name in tqdm(filter(lambda p: os.path.isdir(os.path.join(nifty_dir, p)), os.listdir(nifty_dir))):
            contrast_folder = os.path.join(nifty_dir, contrast_name)
            if not contrast_folder.startswith('.'):
                tasks_metadata[task_name].append(contrast_name)
            for fname in sorted(filter(lambda f: f.endswith('.nii'), os.listdir(contrast_folder))):
                pth = os.path.join(contrast_folder, fname)
                subject_num = int(SUBJECT_NAME_REGEX.findall(os.path.basename(pth))[0])
                tasks_data[subject_num][task_name][contrast_name] = convert_nifty_to_image_array(pth)
    logger.info('Done loading fMRI data.')

    logger.info('Merging subjects data.')
    subjects = sorted(list(features_df['Sub']))

    for subject in subjects:
        features_data = features_df[features_df.Sub == subject].to_dict(orient='records')[0]
        features_data.pop('Sub')
        experiment_data.append(SubjectExperimentData(
            subject_id=int(subject),
            tasks_data=dict(tasks_data[subject]),
            features_data=features_data

        ))

    # Assuming all scans were done using the same scanning option and have the same shape
    example_subject_data = tasks_data[subjects[0]]
    example_tasks_data = example_subject_data[list(example_subject_data.keys())[0]]
    shape = example_tasks_data[list(example_tasks_data.keys())[0]].shape

    logger.info('Done creating subjects experiment data.')
    return ExperimentData(subjects_data=experiment_data, tasks_metadata=dict(tasks_metadata), shape=shape)


def get_experiment_data_after_split(experiment_data: ExperimentData, roi_paths: Optional[List[str]],
                                    tasks_and_contrasts: Optional[dict], ylabels: List[str], weights: Optional[List]):
    flattened_data = apply_roi_masks(experiment_data, roi_paths)
    X, y, one_hot_encoding_mapping = generate_samples_for_model(experiment_data=flattened_data,
                                                                tasks_and_contrasts=tasks_and_contrasts,
                                                                ylabels=ylabels, weights=weights)
    X_3d, y_3d, one_hot_encoding_mapping_3d = generate_samples_for_model(experiment_data=experiment_data,
                                                                         tasks_and_contrasts=tasks_and_contrasts,
                                                                         ylabels=ylabels, weights=weights)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(X, y, np.arange(len(X)))
    X_3d_train, X_3d_test, y_3d_train, y_3d_test = X_3d[train_idx], X_3d[test_idx], y_3d[train_idx], y_3d[test_idx]
    experiment_data_after_split = ExperimentDataAfterSplit(
        original_x_train=X_train,
        original_y_train=y_train,
        original_x_test=X_test,
        original_y_test=y_test,
        flattened_vector_index_to_voxel=flattened_data.flattened_vector_index_to_voxel,
        flattened_vector_index_to_rois=flattened_data.flattened_vector_index_to_rois,
        shape=flattened_data.shape
    )
    experiment_data_after_split_3d = ExperimentDataAfterSplit3D(
        original_x_train=X_3d_train,
        original_y_train=y_3d_train,
        original_x_test=X_3d_test,
        original_y_test=y_3d_test,
        shape=experiment_data.shape
    )
    return experiment_data_after_split, experiment_data_after_split_3d

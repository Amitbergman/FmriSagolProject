import copy
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Union, Optional

import numpy as np

from sagol import config
from sagol.load_data import ExperimentData, convert_nifty_to_image_array, FlattenedExperimentData

# Flattened
ROIS_TO_VOXELS = defaultdict(list)
VOXEL_TO_ROIS = defaultdict(list)


def get_available_rois() -> List[str]:
    return [str(path) for path in config.ROIS_DIR.iterdir()]


def get_rois_and_voxels_mappings() -> (dict, dict):
    print('Creating ROIs-Voxel mappings.')
    global ROIS_TO_VOXELS
    global VOXEL_TO_ROIS

    if not ROIS_TO_VOXELS or not VOXEL_TO_ROIS:
        for roi_path in get_available_rois():
            flattened_mask = get_mask_from_roi(roi_path).flatten()
            for i, val in enumerate(flattened_mask):
                if val != 0:
                    ROIS_TO_VOXELS[roi_path].append(i)
                    VOXEL_TO_ROIS[i].append(roi_path)

    return dict(ROIS_TO_VOXELS), dict(VOXEL_TO_ROIS)


def get_mask_from_roi(roi_path: Union[Path, str]) -> np.array:
    roi_path = str(roi_path)
    assert roi_path.endswith('nii')
    nifty = convert_nifty_to_image_array(os.path.join(config.ROIS_DIR, roi_path))
    # Make ROIs binary.
    # TODO: ask ofir why there were values that are not binary.
    return np.where(nifty == 0, 0, 1)


def _apply_roi_mask_on_flattened_data(flattened_data: np.ndarray, voxels: List[int]) -> np.ndarray:
    masked_data = []
    for voxel in sorted(voxels):
        masked_data.append(flattened_data[voxel])
    return np.array(masked_data)


def _create_vector_index_to_model_mapping(roi_paths: Optional[str]):
    vector_index_to_voxel = {}
    if roi_paths:
        relevant_voxels = set()
        rois_to_voxels, voxels_to_rois = get_rois_and_voxels_mappings()
        for roi_path in roi_paths:
            relevant_voxels.update(rois_to_voxels[roi_path])
        for i, voxel_index in enumerate(sorted(relevant_voxels)):
            vector_index_to_voxel[i] = voxel_index
    return vector_index_to_voxel


def apply_roi_masks(experiment_data: ExperimentData, roi_paths: Optional[List[str]]) -> FlattenedExperimentData:
    global VOXEL_TO_ROIS
    flattened_vector_index_to_rois = {}

    flattened_vector_index_to_voxel = _create_vector_index_to_model_mapping(roi_paths)
    for vector_index, voxel_index in flattened_vector_index_to_voxel.items():
        flattened_vector_index_to_rois[vector_index] = VOXEL_TO_ROIS.get(voxel_index)

    relevant_voxels = sorted(flattened_vector_index_to_voxel.values())

    subjects_data = copy.deepcopy(experiment_data.subjects_data)

    print(f'Applying ROIs.')
    for subject_data in subjects_data:
        for task_name, task_data in subject_data.tasks_data.items():
            for contrast_name, fmri_data in task_data.items():
                subject_data.tasks_data[task_name][contrast_name] = _apply_roi_mask_on_flattened_data(
                    fmri_data.flatten(), voxels=relevant_voxels)

    return FlattenedExperimentData(subjects_data=subjects_data,
                                   flattened_vector_index_to_voxel=flattened_vector_index_to_voxel,
                                   flattened_vector_index_to_rois=flattened_vector_index_to_rois,
                                   shape=experiment_data.shape)

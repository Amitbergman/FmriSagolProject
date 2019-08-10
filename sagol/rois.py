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


def get_available_rois() -> List[str]:
    return [str(path) for path in config.ROIS_DIR.iterdir()]


def get_rois_to_voxels_mapping() -> dict:
    global ROIS_TO_VOXELS
    if not ROIS_TO_VOXELS:
        for roi_path in get_available_rois():
            flattened_mask = get_mask_from_roi(roi_path).flatten()
            for i, val in enumerate(flattened_mask):
                if val != 0:
                    ROIS_TO_VOXELS[roi_path].append(i)

    return dict(ROIS_TO_VOXELS)


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
        rois_to_voxels = get_rois_to_voxels_mapping()
        for roi_path in roi_paths:
            relevant_voxels.update(rois_to_voxels[roi_path])
        for i, voxel_index in enumerate(sorted(relevant_voxels)):
            vector_index_to_voxel[i] = voxel_index
    return vector_index_to_voxel


def apply_roi_masks(experiment_data: ExperimentData, roi_paths: Optional[List[str]]) -> FlattenedExperimentData:
    flattened_vector_index_to_voxel = _create_vector_index_to_model_mapping(roi_paths)
    relevant_voxels = sorted(flattened_vector_index_to_voxel.values())

    subjects_data = copy.deepcopy(experiment_data.subjects_data)
    for subject_data in subjects_data:
        for task_name, fmri_data in subject_data.tasks_data.items():
            subject_data.tasks_data[task_name] = _apply_roi_mask_on_flattened_data(
                subject_data.tasks_data[task_name].flatten(),
                voxels=relevant_voxels)

    return FlattenedExperimentData(subjects_data=subjects_data,
                                   flattened_vector_index_to_voxel=flattened_vector_index_to_voxel,
                                   shape=experiment_data.shape)

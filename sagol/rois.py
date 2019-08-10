import copy
import os
from pathlib import Path

import numpy as np
from typing import List, Union

from sagol import config
from sagol.load_data import ExperimentData, convert_nifty_to_image_array


def get_available_rois() -> List[str]:
    return [str(path) for path in config.ROIS_DIR.iterdir()]


def get_mask_from_roi(roi_path: Union[Path, str]) -> np.array:
    roi_path = str(roi_path)
    assert roi_path.endswith('nii')
    nifty = convert_nifty_to_image_array(os.path.join(config.ROIS_DIR, roi_path))
    # Make ROIs binary.
    # TODO: ask ofir why there were values that are not binary.
    return np.where(nifty == 0, 0, 1)


def _apply_roi_mask(fmri_scan: np.ndarray, mask: np.ndarray = None):
    assert isinstance(mask, np.ndarray)
    assert fmri_scan.shape == mask.shape

    return np.multiply(fmri_scan, mask)


def apply_roi_masks(experiment_data: ExperimentData, roi_paths: List[str]) -> ExperimentData:
    combined_masks = sum(*[get_mask_from_roi(roi_path) for roi_path in roi_paths])
    combined_masks = np.where(combined_masks == 0, 0, 1)
    assert combined_masks.shape == experiment_data.shape

    subjects_data = copy.deepcopy(experiment_data.subjects_data)
    for subject_data in subjects_data:
        for task_name, fmri_data in subject_data.tasks_data.items():
            subject_data.tasks_data[task_name] = _apply_roi_mask(subject_data.tasks_data[task_name],
                                                                 mask=combined_masks)

    return ExperimentData(subjects_data=subjects_data, shape=experiment_data.shape, roi_paths=roi_paths)

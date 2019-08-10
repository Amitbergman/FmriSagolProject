import os
import re
from collections import defaultdict
from typing import List

import nibabel as nib
import numpy as np
import pandas as pd
from attr import attrs, attrib

SUBJECT_NAME_REGEX = re.compile('sub-(\d+).*')


@attrs
class SubjectExperimentData:
    subject_id: int = attrib()
    features_data: dict = attrib()
    tasks_data: dict = attrib()


@attrs
class ExperimentData:
    subjects_data: List[SubjectExperimentData] = attrib()
    # (x, y, z)
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


def create_subject_experiment_data(excel_paths: List[str], nifty_tasks) -> ExperimentData:
    tasks_data = defaultdict(dict)
    experiment_data = []
    dfs = []

    for excel_path in excel_paths:
        # Read excel file with multiple sheets, merging the data belonging to the same subject
        if excel_path.endswith('.xlsx') or excel_path.endswith('.xls'):
            xls = pd.ExcelFile(excel_path)
            for sheet in xls.sheet_names:
                dfs.append(pd.read_excel(excel_path, sheet_name=sheet))
        else:
            raise Exception()

    features_df = merge_subject_dfs(dfs)
    # Transform NaN/NaT to None
    features_df = features_df.where((pd.notnull(features_df)), None)

    subjects = sorted(list(features_df['Sub']))

    # Extract fMRI data
    for nifty_dir, task_name in nifty_tasks:
        for fname in sorted(filter(lambda f: f.endswith('.nii'), os.listdir(nifty_dir))):
            pth = os.path.join(nifty_dir, fname)
            subject_num = int(SUBJECT_NAME_REGEX.findall(os.path.basename(pth))[0])
            tasks_data[subject_num][task_name] = convert_nifty_to_image_array(pth)

    for subject in subjects:
        features_data = features_df[features_df.Sub == subject].to_dict(orient='records')[0]
        features_data.pop('Sub')
        experiment_data.append(SubjectExperimentData(
            subject_id=int(subject),
            tasks_data=tasks_data[subject],
            features_data=features_data

        ))

    # Assuming all scans were done using the same scanning option and have the same shape
    example_tasks_data = tasks_data[subjects[0]]
    shape = example_tasks_data[list(example_tasks_data.keys())[0]].shape

    return ExperimentData(subjects_data=experiment_data, shape=shape)

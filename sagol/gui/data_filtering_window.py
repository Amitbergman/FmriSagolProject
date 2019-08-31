import os
import tkinter as tk

from sagol.gui.globals import STATE, MAIN_WINDOW
from sagol.gui.utils import clear_frame
from sagol.load_data import create_subject_experiment_data


def open_data_filtering_window(contrasts_selector):
    selected_task_names = [contrasts_selector.get(idx) for idx in contrasts_selector.curselection()]

    clear_frame(MAIN_WINDOW)

    some_label = tk.Label(text='Loading data...')
    some_label.pack()

    STATE['experiment_data'] = create_subject_experiment_data(
        excel_paths=STATE['excel_paths'],
        nifty_dirs=[os.path.join(STATE['root_dir'], task) for task in selected_task_names])

    print(STATE['experiment_data'].shape)

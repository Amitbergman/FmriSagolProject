import os
import tkinter as tk
from tkinter import filedialog

from sagol.gui.globals import STATE
from sagol.load_data import create_subject_experiment_data
from sagol.rois import get_available_rois
from sagol.run_models import generate_experiment_data_after_split, generate_ylabel_weights


def load_initial_window(parent):
    excels_button = tk.Button(parent,
                              text="Choose excels",
                              fg="red",
                              command=lambda: open_excel_selector(parent))
    excels_button.pack(side=tk.LEFT)

    root_dir_button = tk.Button(parent,
                                text="Choose root direcotry",
                                command=lambda: on_root_button_click(parent))
    root_dir_button.pack(side=tk.LEFT)


def on_root_button_click(parent):
    open_root_dir_selector(parent)
    contast_selector = display_tasks_selector(parent)
    create_load_data_button(parent, contast_selector)


def on_load_data_click(parent, contrasts_selector):
    selected_task_names = [contrasts_selector.get(idx) for idx in contrasts_selector.curselection()]

    STATE['experiment_data'] = create_subject_experiment_data(
        excel_paths=STATE['excel_paths'],
        nifty_dirs=[os.path.join(STATE['root_dir'], task) for task in selected_task_names])

    roi_selector = display_roi_selector(parent)

    choose_roi_button = tk.Button(parent,
                                 text="Choose ROIs",
                                 command=lambda: on_choose_roi_click(parent, roi_selector))
    choose_roi_button.pack(side=tk.LEFT)


def on_choose_roi_click(parent, roi_selector):
    roi_paths = [roi_selector.get(idx) for idx in roi_selector.curselection()]
    STATE['roi_paths'] = roi_paths

    ylabels_selector = display_ylabel_selector(parent)




def open_excel_selector(parent):
    excel_paths = filedialog.askopenfilenames(initialdir="/", title="Select excels",
                                              filetypes=(
                                                  ("Excel Files", "*.xls*"), ("Comma Separated Files", "*.csv"),
                                                  ("All files", "*.*")))
    STATE['excel_paths'] = excel_paths

    excel_paths_label = tk.Label(parent, text='\n'.join(excel_paths))
    excel_paths_label.pack()


def open_root_dir_selector(parent):
    root_dir = filedialog.askdirectory(initialdir="/", title="Select root directory")
    STATE['root_dir'] = root_dir

    task_names = [dir_name for dir_name in os.listdir(root_dir) if
                  not dir_name.startswith('.') and os.path.isdir(os.path.join(root_dir, dir_name))]
    STATE['task_names'] = task_names

    root_dir_label = tk.Label(parent, text=root_dir)
    root_dir_label.pack()


def display_tasks_selector(parent):
    contrast_selector = tk.Listbox(parent, selectmode=tk.MULTIPLE)
    for task in STATE['task_names']:
        contrast_selector.insert(tk.END, task)
    contrast_selector.pack()
    return contrast_selector


def create_load_data_button(parent, contrasts_selector):
    load_data_button = tk.Button(parent,
                                 text="Load data",
                                 fg="green",
                                 command=lambda: on_load_data_click(parent, contrasts_selector))
    load_data_button.pack(side=tk.LEFT)


def display_roi_selector(parent):
    available_rois = get_available_rois()
    roi_selector = tk.Listbox(parent, selectmode=tk.MULTIPLE, width=max(len(roi)for roi in available_rois))

    for roi in get_available_rois():
        roi_selector.insert(tk.END, roi)
    roi_selector.pack()
    return roi_selector


def display_ylabel_selector(parent):
    available_ylabels = STATE['experiment_data'].available_ylabels
    y_label_selector = tk.Listbox(parent, selectmode=tk.MULTIPLE, width=max(len(ylabel)for ylabel in available_ylabels))

    for ylabel in available_ylabels:
        y_label_selector.insert(tk.END, ylabel)
    y_label_selector.pack()
    choose_roi_button = tk.Button(parent,
                                  text="Choose Y labels",
                                  command=lambda: on_choose_ylabel_click(parent, y_label_selector))
    choose_roi_button.pack(side=tk.LEFT)
    return y_label_selector


def on_choose_ylabel_click(parent, ylabels_selector):
    selected_ylabels = [ylabels_selector.get(idx) for idx in ylabels_selector.curselection()]
    STATE['ylabels'] = selected_ylabels
    print(STATE['ylabels'])
    print(STATE['roi_paths'])

